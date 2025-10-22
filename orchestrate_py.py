"""
FastAPI Router for Hybrid Retriever API
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
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
        _retriever_instance = HybridRetriever(config_path="config.yml")
    return _retriever_instance


# Pydantic Models for Request/Response
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to search for")
    entitlement: str = Field(..., description="User entitlement level")
    org_id: Optional[str] = Field(None, description="Organization ID")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags for filtering")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=20)


class SessionQueryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="The query to search for")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags for filtering")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=20)
    history_limit: Optional[int] = Field(3, description="Number of historical interactions to consider", ge=0, le=10)


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    entitlement: str = Field(..., description="User entitlement level")
    org_id: Optional[str] = Field(None, description="Organization ID")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    entitlement: str
    org_id: Optional[str]
    created_at: str
    conversation_history: List[Dict]
    last_activity: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: Optional[str] = None


class ExportSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to export")
    filepath: str = Field(..., description="Path to save the exported session")


class HealthResponse(BaseModel):
    status: str
    service: str
    indexes_loaded: bool


# API Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check(retriever: HybridRetriever = Depends(get_retriever)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Hybrid Retriever API",
        indexes_loaded=retriever.faiss_index is not None and retriever.bm25_index is not None
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """
    Query the knowledge base
    
    This endpoint performs a hybrid search combining vector similarity (FAISS) 
    and keyword matching (BM25) to find relevant documents.
    """
    try:
        response = retriever.query(
            query=request.query,
            entitlement=request.entitlement,
            org_id=request.org_id,
            tags=request.tags,
            top_k=request.top_k
        )
        return QueryResponse(**response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/create", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """
    Create a new conversation session
    
    Sessions maintain conversation context and history for improved responses
    to follow-up questions.
    """
    try:
        session_id = retriever.create_session(
            user_id=request.user_id,
            entitlement=request.entitlement,
            org_id=request.org_id
        )
        session = retriever.get_session(session_id)
        return SessionResponse(**session)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """Get session details including conversation history"""
    session = retriever.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return SessionResponse(**session)


@router.post("/session/query", response_model=QueryResponse)
async def query_with_session(
    request: SessionQueryRequest,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """
    Query with session context
    
    This endpoint uses conversation history to provide better context-aware
    responses, especially for follow-up questions.
    """
    try:
        response = retriever.query_with_session(
            session_id=request.session_id,
            query=request.query,
            tags=request.tags,
            top_k=request.top_k,
            history_limit=request.history_limit
        )
        return QueryResponse(**response)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """Clear/delete a session and its history"""
    retriever.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.post("/session/export")
async def export_session(
    request: ExportSessionRequest,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """Export session history to a JSON file"""
    try:
        retriever.export_session(request.session_id, request.filepath)
        return {"message": f"Session exported to {request.filepath}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions(
    retriever: HybridRetriever = Depends(get_retriever)
):
    """List all active sessions"""
    return {
        "sessions": list(retriever.sessions.keys()),
        "count": len(retriever.sessions)
    }


@router.get("/sessions/{user_id}")
async def get_user_sessions(
    user_id: str,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """Get all sessions for a specific user"""
    user_sessions = [
        session for session in retriever.sessions.values()
        if session['user_id'] == user_id
    ]
    return {
        "user_id": user_id,
        "sessions": user_sessions,
        "count": len(user_sessions)
    }
