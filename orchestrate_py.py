"""
Core Hybrid Retriever Orchestration Logic
"""

import os
import json
import pickle
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

import yaml
import numpy as np
import faiss

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from src.core.ai_models import AIModels


class HybridRetriever:
    """Hybrid Retriever combining FAISS vector search and BM25 keyword search"""
    
    def __init__(self, config_path: str = "config.yml", base_path: str = None):
        """Initialize the retriever with configuration
        
        Args:
            config_path: Path to config file (default: 'config.yml' in current directory)
            base_path: Explicit path to vector_store directory (optional)
        """
        # Simple approach: just use paths relative to where the service is started
        # This assumes you'll always start the service from the project root
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Current working directory: {os.getcwd()}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set base path for vector store
        # Allow explicit base_path or default to 'vector_store' in current directory
        if base_path:
            self.base_path = base_path
        else:
            self.base_path = 'vector_store'
        
        # Verify vector_store directory exists
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Vector store directory not found at {self.base_path}. Current working directory: {os.getcwd()}")
        
        # Initialize AI models
        self.ai_models = AIModels(self.config)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Session storage
        self.sessions = {}
        
        # Load indexes
        self.load_indexes()
    
    def load_indexes(self):
        """Load FAISS and BM25 indexes from vector_store directory"""
        # Simple file loading - just read from the expected paths
        faiss_index_path = os.path.join(self.base_path, 'faiss_index', 'faiss.index')
        embeddings_path = os.path.join(self.base_path, 'faiss_index', 'embeddings.npy')
        bm25_path = os.path.join(self.base_path, 'bm25_index', 'bm25.pkl')
        metadata_path = os.path.join(self.base_path, 'chunks', 'chunk_metadata.json')
        
        # Load FAISS index
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load embeddings
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        
        # Load BM25 index
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        with open(bm25_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load chunk metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Chunk metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.chunks = json.load(f)
        
        print("✓ Indexes loaded successfully")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text - wrapper for AI models"""
        return self.ai_models.get_embedding(text)
    
    def call_inference_endpoint(self, prompt: str) -> str:
        """Call LLM inference endpoint - wrapper for AI models"""
        return self.ai_models.call_inference_endpoint(prompt)
    
    def hybrid_search(self, query: str, entitlement: List[str], org_id: str = None,
                     tags: List[str] = None, top_k: int = None) -> List[Dict]:
        """Perform hybrid search with filtering"""
        if top_k is None:
            top_k = self.config['retrieval']['hybrid']['top_k']
            print(f"Top K: {top_k}")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        faiss.normalize_L2(query_embedding)
        
        retrieval_multiplier = 5
        initial_top_k = min(top_k * retrieval_multiplier, len(self.chunks))
        
        # Vector search (FAISS)
        vector_scores, vector_indices = self.faiss_index.search(query_embedding, initial_top_k)
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
            chunk_entitlements = chunk['entitlement']
            
            chunk['score'] = float(score)
            results.append(chunk)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def create_session(self, user_id: str, entitlement: str, org_id: str = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'entitlement': entitlement,
            'org_id': org_id,
            'created_at': datetime.now().isoformat(),
            'conversation_history': [],
            'last_activity': datetime.now().isoformat()
        }
        print(f"✓ Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def get_conversation_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        history = session['conversation_history']
        return history[-limit:] if limit else history
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       conversation_history: List[Dict] = None) -> Dict:
        """Generate answer using Llama LLM"""
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for turn in conversation_history:
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer']}\n\n"
            history_context += "---\n\n"
        
        prompt = f"""<|begin_of_text|>You are a strict information extractor and document interpreter.

ONLY use the following context from official documentation to answer the question. Your answer MUST be directly extracted from the provided context. Do NOT rephrase, summarize, or add any information not present in the context.

If the information is not available in the context, respond ONLY with 'The requested information is not available in the provided documents.' Do NOT elaborate, summarize, or suggest additional information.

MUST cite which source you are using by including [1], [2], etc. in your answer. Your answer must consist ONLY of direct quotes from the provided context. Cite each quote using [1], [2], etc. as appropriate.

Context:
{context}

{history_context}

Question: {query}

Answer:"""
        
        # Extract answer from response
        answer = self.call_inference_endpoint(prompt)
        
        return {
            'answer': answer,
            'sources': [
                {'title': c['title'], 'doc_id': c['doc_id'], 'score': c['score']}
                for c in context_chunks
            ]
        }
    
    def extract_context_terms(self, history: List[Dict]) -> List[str]:
        """Extract context terms from conversation history"""
        terms = set()
        
        for turn in history[-2:]:  # Last 2 turns
            query_words = turn.get('query', '').lower().split()
            
            stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                         'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                         'through', 'during', 'how', 'when', 'why', 'what', 'where', 'who',
                         'there', 'can', 'i', 'you', 'do', 'does', 'did', 'will', 'would',
                         'could', 'should', 'may', 'might', 'must', 'shall', 'me', 'my'}
            
            for word in query_words:
                if len(word) > 3 and word not in stop_words:
                    terms.add(word)
        
        return list(terms)
    
    def is_follow_up_question(self, query: str) -> bool:
        """Check if query is a follow-up question"""
        follow_up_indicators = [
            'it', 'that', 'this', 'these', 'those',
            'the same', 'also', 'too', 'as well',
            'more about', 'what about', 'how about',
            'script', 'template', 'example',
            'mentioned', 'discussed', 'said'
        ]
        
        query_lower = query.lower()
        
        if len(query.split()) <= 5:
            return True
        
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        if query_lower.startswith(('and ', 'but ', 'also ', 'what about', 'how about')):
            return True
        
        return False
    
    def query(self, query: str, entitlement: str, org_id: str = None,
              tags: List[str] = None, top_k: int = 5) -> Dict:
        """Query the knowledge base"""
        chunks = self.hybrid_search(
            query=query,
            entitlement=[entitlement],
            org_id=org_id,
            tags=tags,
            top_k=top_k
        )
        
        if not chunks:
            return {
                'answer': 'No relevant information found.',
                'sources': []
            }
        
        return self.generate_answer(query, chunks)
    
    def query_with_session(self, session_id: str, query: str,
                          tags: List[str] = None, top_k: int = 5,
                          history_limit: int = 3) -> Dict:
        """Query with session context"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session['last_activity'] = datetime.now().isoformat()
        
        history = self.get_conversation_history(session_id, limit=history_limit)
        
        chunks_dict = {}  # Track unique chunks by ID
        
        original_chunks = self.hybrid_search(
            query,
            session['entitlement'],
            org_id=session['org_id'],
            tags=tags,
            top_k=top_k
        )
        
        for chunk in original_chunks:
            chunks_dict[chunk['chunk_id']] = chunk
        
        if history and self.is_follow_up_question(query):
            # Extract context terms dynamically
            context_terms = self.extract_context_terms(history)
            
            if context_terms:
                contextual_query = f"{query} {' '.join(context_terms[:3])}"  # Limit to top 3 terms
                print(f"Context-enhanced search: {contextual_query}")
                
                context_chunks = self.hybrid_search(
                    contextual_query,
                    session['entitlement'],
                    org_id=session['org_id'],
                    tags=tags,
                    top_k=top_k
                )
                
                for chunk in context_chunks:
                    if chunk['chunk_id'] not in chunks_dict:
                        chunks_dict[chunk['chunk_id']] = chunk
        
        all_chunks = list(chunks_dict.values())
        all_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_chunks = all_chunks[:top_k]
        
        if not final_chunks:
            response = {
                'answer': 'No relevant information found for your query.',
                'sources': [],
                'session_id': session_id
            }
        else:
            response = self.generate_answer(query, final_chunks, conversation_history=history)
            response['session_id'] = session_id
        
        session['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': response['answer'],
        })
        
        return response
    
    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"Session {session_id} cleared")
    
    def export_session(self, session_id: str, filepath: str):
        """Export session to file"""
        session = self.get_session(session_id)
        if session:
            with open(filepath, 'w') as f:
                json.dump(session, f, indent=2)
            print(f"Session exported to: {filepath}")