"""
Core Hybrid Retriever Orchestration Logic - OpenSearch Version
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

import yaml
import numpy as np

from src.core.ai_models import AIModels
from src.core.native_hybrid_opensearch import NativeHybridOpenSearch


class HybridRetriever:
    """Hybrid Retriever combining OpenSearch vector search and BM25 keyword search"""
    
    def __init__(self, config_path: str = "config.yml", base_path: str = None):
        """Initialize the retriever with configuration
        
        Args:
            config_path: Path to config file (default: 'config.yml' in current directory)
            base_path: Explicit path to vector_store directory (optional)
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}. Current working directory: {os.getcwd()}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set base path for vector store
        if base_path:
            self.base_path = base_path
        else:
            self.base_path = 'vector_store'
        
        # Verify vector_store directory exists
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Vector store directory not found at {self.base_path}. Current working directory: {os.getcwd()}")
        
        # Initialize AI models
        self.ai_models = AIModels(self.config)
        
        # Initialize native hybrid OpenSearch vector store (no BM25 needed)
        self.vector_store = NativeHybridOpenSearch(self.config, self.base_path)
        
        # Initialize memory
        self.memory = {
            "chat_history": [],
            "memory_key": "chat_history",
            "output_key": "answer"
        }
        
        # Session storage
        self.sessions = {}
        
        print("✓ HybridRetriever with OpenSearch initialized successfully")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text - wrapper for AI models"""
        return self.ai_models.get_embedding(text)
    
    def call_inference_endpoint(self, prompt: str) -> str:
        """Call LLM inference endpoint - wrapper for AI models"""
        return self.ai_models.call_inference_endpoint(prompt)
    
    def hybrid_search(self, query: str, entitlement: List[str], org_id: str = None,
                     tags: List[str] = None, top_k: int = None) -> List[Dict]:
        """Perform hybrid search with filtering using OpenSearch + BM25"""
        if top_k is None:
            top_k = self.config['retrieval']['hybrid']['top_k']
            print(f"Top K: {top_k}")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Get weights from config
        vector_weight = self.config['retrieval']['hybrid']['vector_weight']
        bm25_weight = self.config['retrieval']['hybrid']['bm25_weight']
        
        # Perform hybrid search using OpenSearch vector store
        results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            entitlement=entitlement,
            org_id=org_id,
            tags=tags,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        return results
    
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
        # Build context from chunks - these are the documents actually used by LLM
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Build conversation history context if available
        history_context = ""
        if conversation_history:
            history_context = "\nPrevious conversation:\n"
            for turn in conversation_history:
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer']}\n\n"
        
        # Simple, direct prompt for Llama 3.1 8B
        prompt = f"""Use ONLY the documents below to answer the question. Include ALL relevant information from the documents. Do not summarize. Do not add any information not in the documents.

Documents:
{context}
{history_context}
Question: {query}

Answer from documents only:"""
        
        # Get answer from LLM
        answer = self.call_inference_endpoint(prompt)
        
        return {
            'answer': answer,
            'sources': [
                {'title': c['title'], 'doc_id': c['doc_id'], 'score': c['score']}
                for c in context_chunks
            ],
            'used_chunks': context_chunks  # Add the actual chunks used for evaluation
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
        # Get all retrieved chunks
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
                'sources': [],
                'used_chunks': []  # Return empty list when no chunks
            }
        
        # Generate answer and get the response with used chunks
        response = self.generate_answer(query, chunks)
        
        # Add the chunks that were used (passed to generate_answer)
        response['used_chunks'] = chunks  # These are the chunks the LLM actually used
        return response
    
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
        
        # Get all chunks from initial search
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
        
        # Get all retrieved chunks and select top_k
        all_chunks = list(chunks_dict.values())
        all_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # These are the chunks that will actually be used by the LLM
        final_chunks = all_chunks[:top_k]  # ← THESE ARE YOUR USED CHUNKS!
        
        if not final_chunks:
            response = {
                'answer': 'No relevant information found for your query.',
                'sources': [],
                'session_id': session_id,
                'used_chunks': []  # No chunks used
            }
        else:
            response = self.generate_answer(query, final_chunks, conversation_history=history)
            response['session_id'] = session_id
            response['used_chunks'] = final_chunks  # Add the chunks that were actually used
        
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
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components"""
        health = {
            'opensearch_connected': self.vector_store.health_check(),
            'embedding_model': False,
            'llm_model': False
        }
        
        # Check embedding model
        if self.ai_models.embedding_model:
            try:
                test_embedding = self.get_embedding("test")
                health['embedding_model'] = test_embedding is not None
            except:
                health['embedding_model'] = False
        
        # Check LLM model
        if self.ai_models.llm_model:
            try:
                test_response = self.call_inference_endpoint("Hello")
                health['llm_model'] = test_response and not test_response.startswith("Error:")
            except:
                health['llm_model'] = False
        
        return health