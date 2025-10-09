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
    original_chunks = self.hybrid_search(
        query,
        session['entitlement'],
        org_id=session['org_id'],
        tags=tags,
        top_k=top_k
    )
    
    for chunk in original_chunks:
        chunks_dict[chunk['chunk_id']] = chunk
    
    # 2. If this might be a follow-up, do additional searches
    if history and self._is_follow_up_question(query):
        # Extract context terms dynamically
        context_terms = self._extract_context_terms(history)
        
        if context_terms:
            # Create an enhanced query
            contextual_query = f"{query} {' '.join(context_terms[:3])}"  # Limit to top 3 terms
            print(f"Context-enhanced search: {contextual_query}")
            
            context_chunks = self.hybrid_search(
                contextual_query,
                session['entitlement'],
                org_id=session['org_id'],
                tags=tags,
                top_k=top_k
            )
            
            # Add new chunks (avoid duplicates)
            for chunk in context_chunks:
                if chunk['chunk_id'] not in chunks_dict:
                    chunks_dict[chunk['chunk_id']] = chunk
    
    # Convert back to list and sort by score
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