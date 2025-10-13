# Semantic Chunking and Indexing Notebook
# Performs semantic chunking and builds FAISS + BM25 indexes

import yaml
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import boto3
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create output directories
Path(config['storage']['chunks']).mkdir(parents=True, exist_ok=True)
Path(config['storage']['faiss_index']).mkdir(parents=True, exist_ok=True)
Path(config['storage']['bm25_index']).mkdir(parents=True, exist_ok=True)

class SemanticChunker:
    def __init__(self, config):
        self.config = config
        self.endpoint_name = config['models']['embedding']['endpoint_name']
        
        # Initialize boto3 client
        endpoint_creds = config['models']['embedding']['credentials']
        self.client = boto3.client(
            'sagemaker-runtime',
            region_name=endpoint_creds['region'],
            aws_access_key_id=endpoint_creds['accessKeyId'],
            aws_secret_access_key=endpoint_creds['secretAccessKey'],
            aws_session_token=endpoint_creds['sessionToken']
        )
        
        self.breakpoint_threshold = config['chunking']['breakpoint_threshold']
        self.min_chunk_size = config['chunking']['min_chunk_size']
        self.max_chunk_size = config['chunking']['max_chunk_size']
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint"""
        # Request parameters
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        # Obtain response and read output data
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        # Ensure it's a numpy array with correct dtype
        embedding = np.array(output_data[0], dtype='float32')
        return embedding
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Perform semantic chunking based on sentence similarity"""
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings
        embeddings = [self.get_embedding(s) for s in sentences]
        
        # Calculate similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1), 
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Find breakpoints
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk_sentences = sentences[breakpoints[i]:breakpoints[i+1]]
            chunk_text = ' '.join(chunk_sentences)
            
            if len(chunk_text) >= self.min_chunk_size:
                if len(chunk_text) > self.max_chunk_size:
                    # Split large chunks
                    while len(chunk_text) > self.max_chunk_size:
                        chunks.append(chunk_text[:self.max_chunk_size])
                        chunk_text = chunk_text[self.max_chunk_size:]
                    if chunk_text:
                        chunks.append(chunk_text)
                else:
                    chunks.append(chunk_text)
        
        return chunks

class HybridIndexer:
    def __init__(self, config):
        self.config = config
        self.endpoint_name = config['models']['embedding']['endpoint_name']
        
        # Initialize boto3 client
        endpoint_creds = config['models']['embedding']['credentials']
        self.client = boto3.client(
            'sagemaker-runtime',
            region_name=endpoint_creds['region'],
            aws_access_key_id=endpoint_creds['accessKeyId'],
            aws_secret_access_key=endpoint_creds['secretAccessKey'],
            aws_session_token=endpoint_creds['sessionToken']
        )
        
        self.dimension = config['indexing']['vector_store']['dimension']
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint"""
        # Request parameters
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        # Obtain response and read output data
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0])
    
    def build_indexes(self, chunks: List[Dict]):
        """Build FAISS and BM25 indexes"""
        print("Building indexes...")
        
        # Get embeddings for FAISS
        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = self.get_embedding(chunk['content'])
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks")
        
        # Build FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        faiss_index = faiss.IndexFlatIP(self.dimension)
        faiss_index.add(embeddings_array)
        
        # Build BM25 index
        tokenized_corpus = [chunk['content'].lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(
            tokenized_corpus,
            k1=self.config['indexing']['keyword_store']['k1'],
            b=self.config['indexing']['keyword_store']['b']
        )
        
        # Save indexes
        faiss.write_index(
            faiss_index, 
            os.path.join(self.config['storage']['faiss_index'], 'faiss.index')
        )
        np.save(
            os.path.join(self.config['storage']['faiss_index'], 'embeddings.npy'),
            embeddings_array
        )
        with open(os.path.join(self.config['storage']['bm25_index'], 'bm25.pkl'), 'wb') as f:
            pickle.dump(bm25_index, f)
        with open(os.path.join(self.config['storage']['faiss_index'], 'chunk_metadata.json'), 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"✓ FAISS index created with {faiss_index.ntotal} vectors")
        print(f"✓ BM25 index created")
        print(f"✓ Indexes saved")

# Step 1: Load and chunk documents
chunker = SemanticChunker(config)
processed_docs_path = config['storage']['processed_docs']

all_chunks = []
chunk_id_counter = 0

for filename in os.listdir(processed_docs_path):
    if filename.endswith('_processed.json'):
        with open(os.path.join(processed_docs_path, filename), 'r') as f:
            doc = json.load(f)
        
        text_chunks = chunker.semantic_chunk(doc['content'])
        
        for chunk_text in text_chunks:
            chunk_obj = {
                'chunk_id': f"chunk_{chunk_id_counter:06d}",
                'doc_id': doc['doc_id'],
                'content': chunk_text,
                'entitlement': doc['metadata']['entitlement'],
                'metadata': doc['metadata']['metadata'],
                'orgId': doc['metadata']['orgId'],
                'title': doc['metadata']['title'],
                'summary': doc['metadata']['summary']
            }
            all_chunks.append(chunk_obj)
            chunk_id_counter += 1
        
        print(f"✓ Chunked {filename}: {len(text_chunks)} chunks")

# Save chunks
chunks_file = os.path.join(config['storage']['chunks'], 'all_chunks.json')
with open(chunks_file, 'w') as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nTotal chunks created: {len(all_chunks)}\n")

# Step 2: Build indexes
indexer = HybridIndexer(config)
indexer.build_indexes(all_chunks)

print("\n✓ Processing complete!")
