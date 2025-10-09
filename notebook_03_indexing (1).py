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
