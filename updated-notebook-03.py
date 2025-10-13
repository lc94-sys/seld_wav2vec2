# Query and Retrieval Notebook
# Demonstrates hybrid search and answer generation
# Can run as: python notebook_03_indexing.py (CLI mode)
# Or run as: streamlit run notebook_03_indexing.py (Web UI mode)

import yaml
import json
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss
import boto3
import uuid
from datetime import datetime
import sys

# Check if running in Streamlit mode
try:
    import streamlit as st
    import pandas as pd
    STREAMLIT_AVAILABLE = True
    # Check if we're actually running in streamlit
    IN_STREAMLIT = hasattr(st, 'runtime') and hasattr(st.runtime, 'exists')
except ImportError:
    STREAMLIT_AVAILABLE = False
    IN_STREAMLIT = False

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
        
        # Session management
        self.sessions = {}  # session_id -> session data
        
        self.load_indexes()
        
    def load_indexes(self):
        """Load FAISS and BM25 indexes"""
        # Load FAISS index
        faiss_path = os.path.join(self.config['storage']['faiss_index'], 'faiss.index')
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Run notebook 02 first.")
        
        self.faiss_index = faiss.read_index(faiss_path)
        print(f"✓ FAISS index loaded: {type(self.faiss_index)}")
        
        # Load embeddings (optional - for reference)
        embeddings_path = os.path.join(self.config['storage']['faiss_index'], 'embeddings.npy')
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            print(f"✓ Embeddings loaded: shape {self.embeddings.shape}")
        
        # Load BM25 index
        bm25_path = os.path.join(self.config['storage']['bm25_index'], 'bm25.pkl')
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run notebook 02 first.")
            
        with open(bm25_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        print(f"✓ BM25 index loaded: {type(self.bm25_index)}")
        
        # Load chunk metadata
        metadata_path = os.path.join(self.config['storage']['faiss_index'], 'chunk_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Chunk metadata not found at {metadata_path}. Run notebook 02 first.")
            
        with open(metadata_path, 'r') as f:
            self.chunks = json.load(f)
        print(f"✓ Chunk metadata loaded: {len(self.chunks)} chunks")
        
        print("\n✓ All indexes loaded successfully")
    
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
        
        # Ensure it's a numpy array with correct dtype
        embedding = np.array(output_data[0], dtype='float32')
        return embedding
    
    def hybrid_search(self, query: str, entitlement: str, org_id: str = None, 
                     tags: List[str] = None, top_k: int = None) -> List[Dict]:
        """Perform hybrid search with filtering"""
        if top_k is None:
            top_k = self.config['retrieval']['hybrid']['top_k']
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Debug: Check index type
        if not isinstance(self.faiss_index, faiss.Index):
            raise TypeError(f"faiss_index is not a FAISS Index object. Got: {type(self.faiss_index)}")
        
        # ⚡ CRITICAL FIX: Retrieve MORE results initially to account for filtering
        # This ensures we have enough results AFTER filtering
        retrieval_multiplier = 10  # Increased from 5 to 10
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
        
        # Sort by score (BEFORE filtering)
        sorted_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # ⚡ CRITICAL FIX: Collect ALL accessible results first, THEN take top K
        # This ensures we don't stop early and miss high-scoring universal docs
        accessible_results = []
        
        for idx, score in sorted_indices:
            chunk = self.chunks[idx].copy()
            
            # Apply entitlement filter
            chunk_entitlements = chunk['entitlement']
            
            # Ensure chunk_entitlements is always a list
            if isinstance(chunk_entitlements, str):
                chunk_entitlements = [chunk_entitlements]
            
            # Check if universal access or user has matching entitlement
            has_access = (
                'universal' in chunk_entitlements or 
                entitlement in chunk_entitlements
            )
            
            if not has_access:
                continue
            
            # Apply org filter
            if org_id and chunk['orgId'] != org_id:
                continue
            
            # Apply tag filter
            if tags and not any(t in chunk['metadata']['tags'] for t in tags):
                continue
            
            chunk['score'] = float(score)
            accessible_results.append(chunk)
        
        # ⚡ KEY FIX: Sort all accessible results by score again
        # This ensures the highest scoring accessible documents come first
        # regardless of whether they're universal or specific
        accessible_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top K from sorted accessible results
        return accessible_results[:top_k]
    
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
    
    # ============================================
    # NEW HELPER METHODS FOR CONTEXT-AWARE RETRIEVAL
    # ============================================
    
    def _is_follow_up_question(self, query: str) -> bool:
        """
        Detect if query is likely a follow-up question
        This is a simple heuristic - no LLM needed
        """
        follow_up_indicators = [
            'it', 'that', 'this', 'these', 'those',
            'the same', 'also', 'too', 'as well',
            'more about', 'what about', 'how about',
            'script', 'template', 'example',
            'mentioned', 'discussed', 'said'
        ]
        
        query_lower = query.lower()
        
        # Check for short queries (often follow-ups)
        if len(query.split()) <= 5:
            return True
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in query_lower:
                return True
        
        # Check if query starts with certain patterns
        if query_lower.startswith(('and ', 'but ', 'also ', 'what about', 'how about')):
            return True
        
        return False
    
    def _extract_context_terms(self, history: List[Dict]) -> List[str]:
        """
        Extract key terms from recent conversation history dynamically
        No predefined domain terms needed
        """
        terms = set()
        
        for turn in history[-2:]:  # Last 2 turns
            # Extract nouns and important words from previous queries
            query_words = turn.get('query', '').lower().split()
            
            # Filter out common stop words and short words
            stop_words = {'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
                         'through', 'during', 'how', 'when', 'why', 'what', 'where', 'who',
                         'there', 'can', 'i', 'you', 'do', 'does', 'did', 'will', 'would',
                         'could', 'should', 'may', 'might', 'must', 'shall', 'me', 'my'}
            
            # Add meaningful words (length > 3, not stop words)
            for word in query_words:
                if len(word) > 3 and word not in stop_words:
                    terms.add(word)
            
            # Extract terms from the document titles that were previously relevant
            for source in turn.get('sources', []):
                title_words = source.get('title', '').lower().split()
                for word in title_words:
                    if len(word) > 3 and word not in stop_words:
                        terms.add(word)
        
        return list(terms)
    
    # ============================================
    # ENHANCED generate_answer WITH CONTEXT AWARENESS
    # ============================================
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       conversation_history: List[Dict] = None) -> Dict:
        """Enhanced answer generation with better context awareness"""
        
        # Build document context with better formatting
        context = "\n\n".join([
            f"Document: {chunk['title']}\nContent:\n{chunk['content']}" 
            for chunk in context_chunks
        ])
        
        # Build conversation history with entity extraction
        history_context = ""
        key_entities = []
        
        if conversation_history:
            history_context = "Previous conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer'][:500]}...\n\n"
                
                # Extract key entities from previous queries
                prev_query_lower = turn['query'].lower()
                if 'military' in prev_query_lower:
                    key_entities.append('military customer')
                if 'script' in prev_query_lower or 'template' in prev_query_lower:
                    key_entities.append('script/template')
                if 'cancel' in prev_query_lower:
                    key_entities.append('cancellation')
            
            history_context += "---\n\n"
            
            # Add entity context hint
            if key_entities:
                history_context += f"Key topics from conversation: {', '.join(set(key_entities))}\n\n"
        
        # Enhanced prompt with explicit instruction about context
        prompt = f"""{history_context}You are a helpful assistant answering questions based on procedure documents.
    
IMPORTANT INSTRUCTIONS:
1. If the user asks about "it", "that", "this" or uses other pronouns, refer back to the previous conversation topic.
2. If the user asks for a "script" without specifying which one, assume they mean a script related to the most recent topic discussed.
3. Pay close attention to the conversation history to maintain context.
4. The user is currently asking a follow-up question if their query is short or uses pronouns.

Relevant Documents:
{context}

Current Question: {query}

Considering the full conversation context, provide a clear and accurate answer:"""
        
        # Request parameters for LLM with adjusted settings
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config['models']['llm']['max_tokens'],
                "temperature": 0.3,  # Lower temperature for consistency
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
                {'title': c['title'], 'doc_id': c['doc_id'], 'score': c.get('score', 0)}
                for c in context_chunks
            ]
        }
    
    def query(self, query: str, entitlement: str, org_id: str = None, 
              tags: List[str] = None, top_k: int = 5) -> Dict:
        """Complete query pipeline without session"""
        chunks = self.hybrid_search(query, entitlement, org_id=org_id, tags=tags, top_k=top_k)
        
        if not chunks:
            return {'answer': 'No relevant information found.', 'sources': []}
        
        return self.generate_answer(query, chunks)
    
    # ============================================
    # ENHANCED query_with_session WITH DUAL SEARCH
    # ============================================
    
    def query_with_session(self, session_id: str, query: str, 
                          tags: List[str] = None, top_k: int = 5,
                          history_limit: int = 3) -> Dict:
        """
        Enhanced query with session - using dual search with dynamic term extraction
        """
        # Get session
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Update last activity
        session['last_activity'] = datetime.now().isoformat()
        
        # Get conversation history
        history = self.get_conversation_history(session_id, limit=history_limit)
        
        # Collect all unique chunks
        chunks_dict = {}  # Use dict to track unique chunks by ID
        
        # 1. Always search with original query
        print(f"\n🔍 Searching with original query: '{query}'")
        original_chunks = self.hybrid_search(
            query,
            session['entitlement'],
            org_id=session['org_id'],
            tags=tags,
            top_k=top_k
        )
        
        for chunk in original_chunks:
            chunks_dict[chunk['chunk_id']] = chunk
        
        print(f"  Found {len(original_chunks)} chunks from original search")
        
        # 2. If this might be a follow-up, do additional searches
        if history and self._is_follow_up_question(query):
            print(f"  ℹ️ Detected follow-up question")
            
            # Extract context terms dynamically
            context_terms = self._extract_context_terms(history)
            
            if context_terms:
                # Create an enhanced query
                contextual_query = f"{query} {' '.join(context_terms[:3])}"  # Limit to top 3 terms
                print(f"  🔍 Context-enhanced search: '{contextual_query}'")
                
                context_chunks = self.hybrid_search(
                    contextual_query,
                    session['entitlement'],
                    org_id=session['org_id'],
                    tags=tags,
                    top_k=top_k
                )
                
                # Add new chunks (avoid duplicates)
                new_chunks_added = 0
                for chunk in context_chunks:
                    if chunk['chunk_id'] not in chunks_dict:
                        chunks_dict[chunk['chunk_id']] = chunk
                        new_chunks_added += 1
                
                print(f"  Added {new_chunks_added} new unique chunks from context search")
        
        # Convert back to list and sort by score
        all_chunks = list(chunks_dict.values())
        all_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_chunks = all_chunks[:top_k]
        
        print(f"  📚 Final: Using top {len(final_chunks)} chunks for answer generation")
        
        if not final_chunks:
            response = {
                'answer': 'No relevant information found for your query.',
                'sources': [],
                'session_id': session_id
            }
        else:
            # Generate answer with history
            response = self.generate_answer(query, final_chunks, conversation_history=history)
            response['session_id'] = session_id
        
        # Store in conversation history
        session['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': response['answer'],
            'sources': response['sources']
        })
        
        return response
    
    def clear_session(self, session_id: str):
        """Clear/end a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"✓ Session {session_id} cleared")
    
    def export_session(self, session_id: str, filepath: str):
        """Export session to JSON file"""
        session = self.get_session(session_id)
        if session:
            with open(filepath, 'w') as f:
                json.dump(session, f, indent=2)
            print(f"✓ Session exported to: {filepath}")

# ============================================
# STREAMLIT UI (if running in streamlit mode)
# ============================================

if IN_STREAMLIT:
    # Page config
    st.set_page_config(
        page_title="RAG Chatbot Tester",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .user-message { background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px; }
        .assistant-message { background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px; }
        .source-box { background-color: #fff3e0; padding: 5px; border-radius: 5px; margin: 2px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize retriever (cached)
    @st.cache_resource
    def init_retriever():
        return HybridRetriever(config)
    
    # User profiles
    USER_PROFILES = {
        'Alice (Support Agent)': {
            'user_id': 'agent_001',
            'entitlement': 'agent_support',
            'org_id': 'org_123',
            'avatar': '👩‍💼'
        },
        'Bob (Sales Agent)': {
            'user_id': 'agent_002', 
            'entitlement': 'agent_sales',
            'org_id': 'org_123',
            'avatar': '👨‍💼'
        },
        'Carol (Manager)': {
            'user_id': 'manager_001',
            'entitlement': 'agent_manager',
            'org_id': 'org_123',
            'avatar': '👩‍💻'
        }
    }
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.current_session_id = None
        st.session_state.current_user = None
        st.session_state.retriever = init_retriever()
    
    # Sidebar
    with st.sidebar:
        st.title("🎮 Control Panel")
        
        # User selection
        st.subheader("👤 User Selection")
        selected_user_name = st.selectbox(
            "Select User Profile",
            options=list(USER_PROFILES.keys())
        )
        
        if st.button("🔄 Switch User", type="primary"):
            user_profile = USER_PROFILES[selected_user_name]
            st.session_state.current_user = user_profile
            
            if st.session_state.current_session_id:
                st.session_state.retriever.clear_session(st.session_state.current_session_id)
            
            st.session_state.current_session_id = st.session_state.retriever.create_session(
                user_id=user_profile['user_id'],
                entitlement=user_profile['entitlement'],
                org_id=user_profile['org_id']
            )
            
            st.session_state.messages = []
            st.success(f"✅ Switched to {selected_user_name}")
            st.rerun()
        
        if st.session_state.current_user:
            st.info(f"""
            **Current User:** {selected_user_name}
            **Entitlement:** {st.session_state.current_user['entitlement']}
            **Session:** {st.session_state.current_session_id[:8] if st.session_state.current_session_id else 'None'}...
            """)
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Top K Results", 1, 10, 5)
        history_limit = st.slider("History Limit", 1, 5, 3)
        
        # Quick actions
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            if st.session_state.current_session_id:
                session = st.session_state.retriever.get_session(st.session_state.current_session_id)
                if session:
                    session['conversation_history'] = []
            st.rerun()
    
    # Main interface
    st.title("🤖 RAG Chatbot Tester")
    
    if not st.session_state.current_user:
        st.warning("👈 Please select a user from the sidebar")
        st.stop()
    
    # Sample questions
    st.caption("**Quick Test Questions:**")
    cols = st.columns(4)
    sample_questions = [
        "How do I handle military customers?",
        "Is there a script I can provide?",
        "What about cancellations?",
        "Show me the refund procedure"
    ]
    
    for col, question in zip(cols, sample_questions):
        if col.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.divider()
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"📄 **{src['title']}** (Score: {src['score']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    result = st.session_state.retriever.query_with_session(
                        session_id=st.session_state.current_session_id,
                        query=prompt,
                        top_k=top_k,
                        history_limit=history_limit
                    )
                    
                    st.write(result['answer'])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', [])
                    })
                    
                    if result.get('sources'):
                        with st.expander("📚 Sources"):
                            for src in result['sources']:
                                st.markdown(f"📄 **{src['title']}** (Score: {src['score']:.3f})")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================
# CLI MODE (if not running in streamlit)
# ============================================

elif __name__ == "__main__":
    # Initialize retriever
    retriever = HybridRetriever(config)
    
    print("\n" + "="*70)
    print("🎮 INTERACTIVE CLI MODE")
    print("="*70)
    
    # If streamlit is available, offer to run in UI mode
    if STREAMLIT_AVAILABLE:
        print("\n💡 Tip: You can also run this with a web UI using:")
        print("   streamlit run notebook_03_indexing.py\n")
    
    class Colors:
        """ANSI color codes"""
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
    
    # Pre-configured users for testing
    USER_PROFILES = {
        '1': {
            'user_id': 'agent_001',
            'name': 'Alice (Support Agent)',
            'entitlement': 'agent_support',
            'org_id': 'org_123'
        },
        '2': {
            'user_id': 'agent_002',
            'name': 'Bob (Sales Agent)',
            'entitlement': 'agent_sales',
            'org_id': 'org_123'
        },
        '3': {
            'user_id': 'manager_001',
            'name': 'Carol (Manager)',
            'entitlement': 'agent_manager',
            'org_id': 'org_123'
        }
    }
    
    def show_user_menu():
        """Display available users"""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}👥 Select User:{Colors.ENDC}")
        for key, profile in USER_PROFILES.items():
            print(f"  {Colors.BOLD}{key}.{Colors.ENDC} {profile['name']} ({profile['entitlement']})")
        print(f"  {Colors.BOLD}q.{Colors.ENDC} Quit")
    
    def interactive_chat():
        """Main interactive chat function"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🤖 Interactive Testing Mode{Colors.ENDC}")
        print(f"{Colors.CYAN}Type your questions naturally. Commands:{Colors.ENDC}")
        print(f"  {Colors.CYAN}/history{Colors.ENDC} - View conversation")
        print(f"  {Colors.CYAN}/switch{Colors.ENDC}  - Change user")
        print(f"  {Colors.CYAN}/clear{Colors.ENDC}   - Reset session")
        print(f"  {Colors.CYAN}/quit{Colors.ENDC}    - Exit\n")
        
        current_user = None
        current_session = None
        
        while True:
            # User selection
            if current_user is None:
                show_user_menu()
                choice = input(f"\n{Colors.BOLD}Choose user (1-3, q): {Colors.ENDC}").strip()
                
                if choice == 'q':
                    print(f"{Colors.CYAN}👋 Goodbye!{Colors.ENDC}")
                    break
                
                if choice not in USER_PROFILES:
                    print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                    continue
                
                current_user = USER_PROFILES[choice]
                current_session = retriever.create_session(
                    user_id=current_user['user_id'],
                    entitlement=current_user['entitlement'],
                    org_id=current_user['org_id']
                )
                print(f"\n{Colors.GREEN}✅ Session started for {current_user['name']}{Colors.ENDC}")
                print(f"{Colors.GREEN}Session ID: {current_session[:8]}...{Colors.ENDC}\n")
            
            # Get query
            query = input(f"{Colors.BOLD}{current_user['name']}:{Colors.ENDC} ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query == '/quit':
                print(f"{Colors.CYAN}👋 Goodbye!{Colors.ENDC}")
                break
            
            elif query == '/switch':
                current_user = None
                current_session = None
                continue
            
            elif query == '/history':
                history = retriever.get_conversation_history(current_session)
                print(f"\n{Colors.CYAN}📜 Conversation History ({len(history)} messages):{Colors.ENDC}")
                for i, turn in enumerate(history, 1):
                    print(f"\n{Colors.BOLD}Turn {i}:{Colors.ENDC}")
                    print(f"  Q: {turn['query']}")
                    print(f"  A: {turn['answer'][:100]}...")
                continue
            
            elif query == '/clear':
                retriever.clear_session(current_session)
                current_session = retriever.create_session(
                    user_id=current_user['user_id'],
                    entitlement=current_user['entitlement'],
                    org_id=current_user['org_id']
                )
                print(f"{Colors.GREEN}✅ Session cleared{Colors.ENDC}")
                continue
            
            # Process query
            try:
                print(f"{Colors.CYAN}🔍 Searching...{Colors.ENDC}")
                
                result = retriever.query_with_session(
                    session_id=current_session,
                    query=query,
                    top_k=3,
                    history_limit=2
                )
                
                print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 Assistant:{Colors.ENDC}")
                print(f"{Colors.GREEN}{result['answer']}{Colors.ENDC}\n")
                
                if result.get('sources'):
                    print(f"{Colors.CYAN}📚 Sources:{Colors.ENDC}")
                    for i, src in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {src['title']} (score: {src['score']:.3f})")
                
                print()  # Empty line for spacing
                
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {str(e)}{Colors.ENDC}\n")
    
    # Choose mode
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Select Mode:{Colors.ENDC}")
    print(f"  {Colors.BOLD}1.{Colors.ENDC} Run Manual Tests (predefined queries)")
    print(f"  {Colors.BOLD}2.{Colors.ENDC} Interactive Chat (test with real queries)")
    print(f"  {Colors.BOLD}3.{Colors.ENDC} Skip testing")
    
    mode = input(f"\n{Colors.BOLD}Choice (1-3): {Colors.ENDC}").strip()
    
    if mode == '2':
        interactive_chat()
    
    elif mode == '1':
        # Run manual tests focused on query_with_session
        print("\n" + "="*70)
        print("TESTING query_with_session WITH CONTEXT-AWARE RETRIEVAL")
        print("="*70)
        
        # Create test session
        session_id = retriever.create_session(
            user_id="agent_001",
            entitlement="agent_support",
            org_id="org_123"
        )
        
        # Test sequence
        test_sequence = [
            ("How do I handle military customers?", "Initial context"),
            ("Is there a script I can provide?", "Follow-up with pronoun"),
            ("What about cancellations?", "Context switch"),
            ("Can you give me the script for that?", "Reference to new context"),
        ]
        
        for i, (query, description) in enumerate(test_sequence, 1):
            print(f"\n{Colors.YELLOW}Test {i}: {description}{Colors.ENDC}")
            print(f"Query: {query}")
            
            result = retriever.query_with_session(
                session_id=session_id,
                query=query,
                top_k=5,
                history_limit=2
            )
            
            print(f"\nAnswer: {result['answer'][:300]}...")
            print(f"Sources: {[s['title'] for s in result['sources'][:3]]}")
    
    else:
        print(f"\n{Colors.CYAN}Skipping tests. Retriever is ready.{Colors.ENDC}")
    
    print("\n" + "="*70)
    print("NOTEBOOK COMPLETE")
    print("="*70)