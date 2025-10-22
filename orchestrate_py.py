"""
Simplified FastAPI Router with Single Query Endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

from src.core.orchestrate import HybridRetriever


# Initialize router
router = APIRouter(prefix="/api/v1", tags=["retriever"])

# Initialize retriever (singleton pattern)
_retriever_instance = None

def get_retriever() -> HybridRetriever:
    """Get or create retriever instance"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


# Single Request Model
class QueryRequest(BaseModel):
    session_id: str
    query: str
    user_id: str
    entitlement: str
    org_id: Optional[str] = None
    tags: Optional[List[str]] = []
    top_k: Optional[int] = 5
    history_limit: Optional[int] = 3


# Response Model
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


# Single API Endpoint
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Single endpoint to query the knowledge base with session management
    
    This endpoint:
    - Creates a session if it doesn't exist
    - Performs hybrid search
    - Maintains conversation history
    - Returns the answer with sources
    """
    try:
        retriever = get_retriever()
        
        # Check if session exists, if not create it
        session = retriever.get_session(request.session_id)
        if not session:
            # Create new session
            retriever.create_session(
                user_id=request.user_id,
                entitlement=request.entitlement,
                org_id=request.org_id
            )
            # Override with provided session_id
            retriever.sessions[request.session_id] = retriever.sessions.popitem()[1]
            retriever.sessions[request.session_id]['session_id'] = request.session_id
        
        # Query with session
        response = retriever.query_with_session(
            session_id=request.session_id,
            query=request.query,
            tags=request.tags,
            top_k=request.top_k,
            history_limit=request.history_limit
        )
        
        return QueryResponse(**response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        retriever = get_retriever()
        return {
            "status": "healthy",
            "indexes_loaded": retriever.faiss_index is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
