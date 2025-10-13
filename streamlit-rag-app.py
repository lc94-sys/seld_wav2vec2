# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import yaml
import json
import os
import sys
from datetime import datetime
import pandas as pd

# Add your project directory to path if needed
# sys.path.append('.')

# Import your existing retriever code
# You'll need to modify this import based on your file structure
from notebook_03_indexing import HybridRetriever

# Page config
st.set_page_config(
    page_title="RAG Chatbot Tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChat { max-height: 600px; }
    .user-message { background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px; }
    .assistant-message { background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px; }
    .source-box { background-color: #fff3e0; padding: 5px; border-radius: 5px; margin: 2px; }
    .debug-info { background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Initialize retriever (cached to prevent reloading)
@st.cache_resource
def init_retriever():
    config = load_config()
    return HybridRetriever(config)

# User profiles for testing
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
    st.session_state.show_debug = False
    st.session_state.retriever = init_retriever()
    st.session_state.search_history = []

# Sidebar
with st.sidebar:
    st.title("🎮 Control Panel")
    
    # User selection
    st.subheader("👤 User Selection")
    selected_user_name = st.selectbox(
        "Select User Profile",
        options=list(USER_PROFILES.keys()),
        key="user_selector"
    )
    
    # Change user and create new session
    if st.button("🔄 Switch User", type="primary"):
        user_profile = USER_PROFILES[selected_user_name]
        st.session_state.current_user = user_profile
        
        # Clear old session if exists
        if st.session_state.current_session_id:
            st.session_state.retriever.clear_session(st.session_state.current_session_id)
        
        # Create new session
        st.session_state.current_session_id = st.session_state.retriever.create_session(
            user_id=user_profile['user_id'],
            entitlement=user_profile['entitlement'],
            org_id=user_profile['org_id']
        )
        
        # Clear chat history
        st.session_state.messages = []
        st.session_state.search_history = []
        
        st.success(f"✅ Switched to {selected_user_name}")
        st.rerun()
    
    # Display current user info
    if st.session_state.current_user:
        st.info(f"""
        **Current User:** {st.session_state.current_user['avatar']} {selected_user_name}
        **Entitlement:** {st.session_state.current_user['entitlement']}
        **Org ID:** {st.session_state.current_user['org_id']}
        **Session:** {st.session_state.current_session_id[:8] if st.session_state.current_session_id else 'None'}...
        """)
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    # Retrieval settings
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
    history_limit = st.slider("Conversation History Limit", min_value=1, max_value=5, value=3)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        use_query_rewriting = st.checkbox("Enable Query Rewriting", value=True)
        st.session_state.show_debug = st.checkbox("Show Debug Info", value=False)
        
        # Temperature control
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    st.divider()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.search_history = []
        if st.session_state.current_session_id:
            # Clear session history but keep session
            session = st.session_state.retriever.get_session(st.session_state.current_session_id)
            if session:
                session['conversation_history'] = []
        st.rerun()
    
    if st.button("📥 Export Session"):
        if st.session_state.current_session_id:
            filepath = f"session_{st.session_state.current_session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.session_state.retriever.export_session(st.session_state.current_session_id, filepath)
            st.success(f"Exported to {filepath}")
    
    st.divider()
    
    # Search History
    if st.session_state.search_history:
        st.subheader("🔍 Recent Searches")
        for i, search in enumerate(reversed(st.session_state.search_history[-5:]), 1):
            st.text(f"{i}. {search['query'][:30]}...")
            if st.session_state.show_debug:
                st.caption(f"   Rewritten: {search.get('rewritten', 'N/A')[:30]}...")

# Main chat interface
st.title("🤖 RAG Chatbot Tester")

# Check if user is selected
if not st.session_state.current_user:
    st.warning("👈 Please select a user from the sidebar to start chatting")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🗂️ Session Data"])

with tab1:
    # Sample questions for quick testing
    st.caption("**Quick Test Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    sample_questions = [
        "How do I handle military customers?",
        "Is there a script I can provide?",
        "What about cancellations?",
        "Show me the refund procedure"
    ]
    
    for col, question in zip([col1, col2, col3, col4], sample_questions):
        if col.button(question, key=f"sample_{question[:10]}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.divider()
    
    # Display chat messages
    message_container = st.container()
    with message_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"""
                            <div class="source-box">
                            📄 <b>{source['title']}</b> (Score: {source['score']:.3f})
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.show_debug and message.get("debug"):
                    with st.expander("🐛 Debug Info"):
                        st.json(message["debug"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Make the query with session
                    result = st.session_state.retriever.query_with_session(
                        session_id=st.session_state.current_session_id,
                        query=prompt,
                        top_k=top_k,
                        history_limit=history_limit
                    )
                    
                    # Display the answer
                    st.write(result['answer'])
                    
                    # Prepare message data
                    assistant_message = {
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', [])
                    }
                    
                    # Add debug info if enabled
                    if st.session_state.show_debug:
                        # Get the last entry from conversation history to see rewritten query
                        session = st.session_state.retriever.get_session(st.session_state.current_session_id)
                        if session and session['conversation_history']:
                            last_entry = session['conversation_history'][-1]
                            assistant_message['debug'] = {
                                'original_query': prompt,
                                'search_query': last_entry.get('search_query', prompt),
                                'num_sources': len(result.get('sources', [])),
                                'timestamp': last_entry.get('timestamp')
                            }
                    
                    # Add to messages
                    st.session_state.messages.append(assistant_message)
                    
                    # Track search history
                    st.session_state.search_history.append({
                        'query': prompt,
                        'rewritten': assistant_message.get('debug', {}).get('search_query', prompt),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Show sources
                    if result.get('sources'):
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                st.markdown(f"📄 **{source['title']}** (Score: {source['score']:.3f})")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)

with tab2:
    st.subheader("📊 Session Analytics")
    
    if st.session_state.current_session_id:
        session = st.session_state.retriever.get_session(st.session_state.current_session_id)
        
        if session and session['conversation_history']:
            # Conversation metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Queries", len(session['conversation_history']))
            col2.metric("Unique Sources", len(set(
                src['title'] for turn in session['conversation_history'] 
                for src in turn.get('sources', [])
            )))
            col3.metric("Session Duration", 
                       f"{(datetime.now() - datetime.fromisoformat(session['created_at'])).seconds // 60} min")
            col4.metric("Avg Sources/Query", 
                       f"{sum(len(turn.get('sources', [])) for turn in session['conversation_history']) / len(session['conversation_history']):.1f}")
            
            # Source usage chart
            st.subheader("📚 Most Used Sources")
            source_counts = {}
            for turn in session['conversation_history']:
                for source in turn.get('sources', []):
                    title = source['title']
                    source_counts[title] = source_counts.get(title, 0) + 1
            
            if source_counts:
                df_sources = pd.DataFrame(
                    source_counts.items(), 
                    columns=['Document', 'Usage Count']
                ).sort_values('Usage Count', ascending=False)
                st.bar_chart(df_sources.set_index('Document'))
            
            # Query timeline
            st.subheader("⏱️ Query Timeline")
            timeline_data = []
            for turn in session['conversation_history']:
                timeline_data.append({
                    'Time': datetime.fromisoformat(turn['timestamp']).strftime('%H:%M:%S'),
                    'Query': turn['query'][:50] + '...' if len(turn['query']) > 50 else turn['query'],
                    'Sources': len(turn.get('sources', []))
                })
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline, use_container_width=True)
        else:
            st.info("No conversation history yet. Start chatting to see analytics!")
    else:
        st.warning("Please select a user to view analytics")

with tab3:
    st.subheader("🗂️ Raw Session Data")
    
    if st.session_state.current_session_id:
        session = st.session_state.retriever.get_session(st.session_state.current_session_id)
        
        if session:
            # Session metadata
            with st.expander("Session Metadata", expanded=True):
                st.json({
                    'session_id': session['session_id'],
                    'user_id': session['user_id'],
                    'entitlement': session['entitlement'],
                    'org_id': session['org_id'],
                    'created_at': session['created_at'],
                    'last_activity': session['last_activity'],
                    'total_turns': len(session['conversation_history'])
                })
            
            # Conversation history
            with st.expander("Full Conversation History"):
                if session['conversation_history']:
                    for i, turn in enumerate(session['conversation_history'], 1):
                        st.markdown(f"**Turn {i}**")
                        st.json(turn)
                        st.divider()
                else:
                    st.info("No conversation history yet")
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy Session JSON"):
                    st.code(json.dumps(session, indent=2), language='json')
            
            with col2:
                if st.button("💾 Download Session"):
                    json_str = json.dumps(session, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"session_{session['session_id'][:8]}.json",
                        mime="application/json"
                    )
    else:
        st.warning("Please select a user to view session data")

# Footer
st.divider()
st.caption("🔧 RAG Chatbot Tester | Built with Streamlit")
