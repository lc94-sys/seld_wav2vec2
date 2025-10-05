# Query and Retrieval Notebook
# Demonstrates hybrid search and answer generation

import yaml
import json
import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
import boto3

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class HybridRetriever:
    def __init__(self, config):
        self.config = config
        
        # Initialize embedding client
        self.embedding_endpoint_name = config['models']['embedding']['endpoint_name']
        embedding_creds = config['models']['embedding']['credentials']
        self.embedding_client = boto3.client(
            'sagemaker-runtime',
            region_name=embedding_creds['region'],
            aws_access_key_id=embedding_creds['accessKeyId'],
            aws_secret_access_key=embedding_creds['secretAccessKey'],
            aws_session_token=embedding_creds['sessionToken']
        )
        
        # Initialize LLM client
        self.llm_endpoint_name = config['models']['llm']['endpoint_name']
        llm_creds = config['models']['llm']['credentials']
        self.llm_client = boto3.client(
            'sagemaker-runtime',
            region_name=llm_creds['region'],
            aws_access_key_id=llm_creds['accessKeyId'],
            aws_secret_access_key=llm_creds['secretAccessKey'],
            aws_session_token=llm_creds['sessionToken']
        )
        
        self.load_indexes()
        
    def load_indexes(self):
        """Load FAISS and BM25 indexes"""
        faiss_path = os.path.join(self.config['storage']['faiss_index'], 'faiss.index')
        self.faiss_index = faiss.read_index(faiss_path)
        
        embeddings_path = os.path.join(self.config['storage']['faiss_index'], 'embeddings.npy')
        self.embeddings = np.load(embeddings_path)
        
        bm25_path = os.path.join(self.config['storage']['bm25_index'], 'bm25.pkl')
        with open(bm25_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        metadata_path = os.path.join(self.config['storage']['faiss_index'], 'chunk_metadata.json')
        with open(metadata_path, 'r') as f:
            self.chunks = json.load(f)
        
        print("✓ Indexes loaded successfully")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint"""
        # Request parameters
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        # Obtain response and read output data
        response = self.embedding_client.invoke_endpoint(
            EndpointName=self.embedding_endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0])
    
    def hybrid_search(self, query: str, entitlement: str, org_id: str = None, 
                     tags: List[str] = None, top_k: int = None) -> List[Dict]:
        """Perform hybrid search with filtering"""
        if top_k is None:
            top_k = self.config['retrieval']['hybrid']['top_k']
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Vector search (FAISS)
        vector_scores, vector_indices = self.faiss_index.search(query_embedding, top_k * 2)
        vector_scores = vector_scores[0]
        vector_indices = vector_indices[0]
        
        # Keyword search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize scores
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-10:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)
        
        vector_scores_norm = normalize(vector_scores)
        bm25_scores_norm = normalize(bm25_scores)
        
        # Compute hybrid scores
        vector_weight = self.config['retrieval']['hybrid']['vector_weight']
        bm25_weight = self.config['retrieval']['hybrid']['bm25_weight']
        
        hybrid_scores = {}
        for idx, score in zip(vector_indices, vector_scores_norm):
            hybrid_scores[idx] = score * vector_weight
        
        for idx, score in enumerate(bm25_scores_norm):
            if idx in hybrid_scores:
                hybrid_scores[idx] += score * bm25_weight
            else:
                hybrid_scores[idx] = score * bm25_weight
        
        # Sort and filter
        sorted_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_indices:
            chunk = self.chunks[idx].copy()
            
            # Apply filters
            if chunk['entitlement'] != entitlement:
                continue
            if org_id and chunk['orgId'] != org_id:
                continue
            if tags and not any(t in chunk['metadata']['tags'] for t in tags):
                continue
            
            chunk['score'] = float(score)
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer using Llama LLM"""
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}" 
            for chunk in context_chunks
        ])
        
        prompt = f"""Based on the following procedure documents, provide a clear and accurate answer.

Context:
{context}

Question: {query}

Answer:"""
        
        # Request parameters for LLM
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config['models']['llm']['max_tokens'],
                "temperature": self.config['models']['llm']['temperature']
            }
        }
        body = json.dumps(params)
        
        # Obtain response and read output data
        response = self.llm_client.invoke_endpoint(
            EndpointName=self.llm_endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        # Extract answer from response
        answer = output_data[0]['generated_text'] if isinstance(output_data, list) else output_data['generated_text']
        
        return {
            'answer': answer,
            'sources': [
                {'title': c['title'], 'doc_id': c['doc_id'], 'score': c['score']}
                for c in context_chunks
            ]
        }
    
    def query(self, query: str, entitlement: str, org_id: str = None, 
              tags: List[str] = None, top_k: int = 5) -> Dict:
        """Complete query pipeline"""
        chunks = self.hybrid_search(query, entitlement, org_id=org_id, tags=tags, top_k=top_k)
        
        if not chunks:
            return {'answer': 'No relevant information found.', 'sources': []}
        
        return self.generate_answer(query, chunks)

# Initialize retriever
retriever = HybridRetriever(config)

# Example queries
queries = [
    {
        'query': 'How do I process a cancellation?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['cancellation']
    },
    {
        'query': 'What are the steps for creating a new booking?',
        'entitlement': 'agent_sales',
        'org_id': 'org_123'
    },
    {
        'query': 'How to update a reservation?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['update']
    }
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q['query']}")
    print(f"Entitlement: {q['entitlement']}")
    print(f"{'='*60}\n")
    
    result = retriever.query(
        q['query'], 
        q['entitlement'],
        org_id=q.get('org_id'),
        tags=q.get('tags')
    )
    
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for src in result['sources']:
        print(f"  - {src['title']} (score: {src['score']:.3f})")
