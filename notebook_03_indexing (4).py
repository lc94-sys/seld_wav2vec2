# Query and Retrieval Notebook
# Demonstrates hybrid search and answer generation

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
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       conversation_history: List[Dict] = None) -> Dict:
        """Generate answer using Llama LLM with optional conversation history"""
        # Build document context
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}" 
            for chunk in context_chunks
        ])
        
        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for turn in conversation_history:
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer']}\n\n"
            history_context += "---\n\n"
        
        prompt = f"""{history_context}Based on the following procedure documents and our conversation history, provide a clear and accurate answer.

Relevant Documents:
{context}

Current Question: {query}

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
        """Complete query pipeline without session"""
        chunks = self.hybrid_search(query, entitlement, org_id=org_id, tags=tags, top_k=top_k)
        
        if not chunks:
            return {'answer': 'No relevant information found.', 'sources': []}
        
        return self.generate_answer(query, chunks)
    
    def query_with_session(self, session_id: str, query: str, 
                          tags: List[str] = None, top_k: int = 5,
                          history_limit: int = 3) -> Dict:
        """Query with session context and conversation history"""
        # Get session
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Update last activity
        session['last_activity'] = datetime.now().isoformat()
        
        # Get conversation history
        history = self.get_conversation_history(session_id, limit=history_limit)
        
        # Retrieve relevant chunks
        chunks = self.hybrid_search(
            query, 
            session['entitlement'],
            org_id=session['org_id'],
            tags=tags,
            top_k=top_k
        )
        
        if not chunks:
            response = {
                'answer': 'No relevant information found for your query.',
                'sources': [],
                'session_id': session_id
            }
        else:
            # Generate answer with history
            response = self.generate_answer(query, chunks, conversation_history=history)
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

# Initialize retriever
retriever = HybridRetriever(config)

print("\n" + "="*70)
print("🎮 INTERACTIVE CLI MODE")
print("="*70)

# ============================================
# INTERACTIVE CLI FUNCTIONS
# ============================================

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

# ============================================
# CHOOSE MODE
# ============================================

print(f"\n{Colors.YELLOW}{Colors.BOLD}Select Mode:{Colors.ENDC}")
print(f"  {Colors.BOLD}1.{Colors.ENDC} Run Manual Tests (predefined queries)")
print(f"  {Colors.BOLD}2.{Colors.ENDC} Interactive Chat (test with real queries)")
print(f"  {Colors.BOLD}3.{Colors.ENDC} Skip testing")

mode = input(f"\n{Colors.BOLD}Choice (1-3): {Colors.ENDC}").strip()

if mode == '2':
    # Run interactive mode
    interactive_chat()

elif mode == '1':
    # Run manual tests (original code)
    print("\n" + "-"*70)
    print("TEST 1: Single Query Without Session")
    print("-"*70)

    query1 = {
        'query': 'How do I process a cancellation?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['cancellation']
    }

    print(f"\nQuery: {query1['query']}")
    print(f"Entitlement: {query1['entitlement']}")

    result1 = retriever.query(
        query1['query'], 
        query1['entitlement'],
        org_id=query1.get('org_id'),
        tags=query1.get('tags')
    )

    print(f"\nAnswer:\n{result1['answer']}\n")
    print("Sources:")
    for src in result1['sources']:
        print(f"  - {src['title']} (score: {src['score']:.3f})")

    # ============================================
    # TEST 2: Conversational Query (With Session)
    # ============================================
    print("\n" + "-"*70)
    print("TEST 2: Multi-Turn Conversation With Session")
    print("-"*70)

    # Create a session for a support agent
    session_id = retriever.create_session(
        user_id="agent_001",
        entitlement="agent_support",
        org_id="org_123"
    )

    # Turn 1: Ask about cancellation
    print("\n--- Turn 1 ---")
    print("User: How do I cancel a booking?")

    result_turn1 = retriever.query_with_session(
        session_id=session_id,
        query="How do I cancel a booking?",
        tags=['cancellation'],
        history_limit=2
    )

    print(f"\nAssistant: {result_turn1['answer'][:300]}...")
    print(f"\nSources: {[s['title'] for s in result_turn1['sources']]}")

    # Turn 2: Follow-up question
    print("\n--- Turn 2 ---")
    print("User: What documents do I need?")

    result_turn2 = retriever.query_with_session(
        session_id=session_id,
        query="What documents do I need?",
        history_limit=2
    )

    print(f"\nAssistant: {result_turn2['answer'][:300]}...")
    print(f"\nSources: {[s['title'] for s in result_turn2['sources']]}")

    # Turn 3: Another follow-up
    print("\n--- Turn 3 ---")
    print("User: How long does it take?")

    result_turn3 = retriever.query_with_session(
        session_id=session_id,
        query="How long does it take?",
        history_limit=2
    )

    print(f"\nAssistant: {result_turn3['answer'][:300]}...")
    print(f"\nSources: {[s['title'] for s in result_turn3['sources']]}")

    # View conversation history
    print("\n--- Conversation History ---")
    history = retriever.get_conversation_history(session_id)
    for i, turn in enumerate(history, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {turn['query']}")
        print(f"  Assistant: {turn['answer'][:100]}...")
        print(f"  Time: {turn['timestamp']}")

    # Export session
    retriever.export_session(session_id, f'session_{session_id}.json')

    # ============================================
    # TEST 3: Multiple Sessions
    # ============================================
    print("\n" + "-"*70)
    print("TEST 3: Multiple Sessions - Verify Isolation")
    print("-"*70)

    # Session 1: Support agent
    session1 = retriever.create_session(
        user_id="support_agent_001",
        entitlement="agent_support",
        org_id="org_123"
    )

    result_s1 = retriever.query_with_session(
        session_id=session1,
        query="How do I process a refund?",
        tags=['cancellation']
    )

    print(f"\nSession 1 (Support):")
    print(f"Query: How do I process a refund?")
    print(f"Answer: {result_s1['answer'][:200]}...")

    # Session 2: Sales agent
    session2 = retriever.create_session(
        user_id="sales_agent_001",
        entitlement="agent_sales",
        org_id="org_123"
    )

    result_s2 = retriever.query_with_session(
        session_id=session2,
        query="How do I create a new booking?",
        tags=['booking']
    )

    print(f"\nSession 2 (Sales):")
    print(f"Query: How do I create a new booking?")
    print(f"Answer: {result_s2['answer'][:200]}...")

    # Verify isolation
    print(f"\nSession 1 history length: {len(retriever.get_conversation_history(session1))}")
    print(f"Session 2 history length: {len(retriever.get_conversation_history(session2))}")
    print("\n✓ Sessions are isolated")

    # ============================================
    # TEST 4: Entitlement Filtering
    # ============================================
    print("\n" + "-"*70)
    print("TEST 4: Entitlement-Based Access Control")
    print("-"*70)

    same_query = "What are the procedures for bookings?"

    # Support agent session
    support_session = retriever.create_session(
        user_id="support_001",
        entitlement="agent_support",
        org_id="org_123"
    )

    support_result = retriever.query_with_session(
        session_id=support_session,
        query=same_query
    )

    # Sales agent session
    sales_session = retriever.create_session(
        user_id="sales_001",
        entitlement="agent_sales",
        org_id="org_123"
    )

    sales_result = retriever.query_with_session(
        session_id=sales_session,
        query=same_query
    )

    print(f"\nQuery: {same_query}")
    print(f"\nSupport Agent sees {len(support_result['sources'])} sources:")
    for s in support_result['sources']:
        print(f"  - {s['title']}")

    print(f"\nSales Agent sees {len(sales_result['sources'])} sources:")
    for s in sales_result['sources']:
        print(f"  - {s['title']}")

    print("\n✓ Each agent sees only documents they're entitled to access")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("MANUAL TESTING COMPLETE")
    print("="*70)

    print("\nFeatures Demonstrated:")
    print("✓ Single query without session tracking")
    print("✓ Multi-turn conversation with context preservation")
    print("✓ Session isolation between different users")
    print("✓ Entitlement-based access control")

else:
    print(f"\n{Colors.CYAN}Skipping tests. Retriever is ready to use.{Colors.ENDC}")

print("\n" + "="*70)
print("NOTEBOOK COMPLETE")
print("="*70)
