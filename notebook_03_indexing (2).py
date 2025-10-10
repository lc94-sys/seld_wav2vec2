# app.py
import os
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

HF_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")  # e.g. https://<id>.<region>.endpoints.huggingface.cloud/<path>
HF_TOKEN = os.getenv("HF_TOKEN")

class EmbedRequest(BaseModel):
    # accept a single string or a list of strings
    inputs: Union[str, List[str]]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

app = FastAPI()
client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def _startup():
    # one shared async client for connection pooling
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    global client
    client = httpx.AsyncClient(timeout=timeout, limits=limits)

@app.on_event("shutdown")
async def _shutdown():
    global client
    if client:
        await client.aclose()

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not HF_ENDPOINT or not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Endpoint or token not configured")
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    payload = {"inputs": texts}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = await client.post(HF_ENDPOINT, headers=headers, json=payload)  # type: ignore
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"http error: {e!s}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    # Normalize: TEI/Endpoints may return either a list or {"embeddings": [...]}
    vectors = data.get("embeddings", data) if isinstance(data, dict) else data
    return {"embeddings": vectors}


# src/core/app_factory.py
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.rightApp.ar_router import router as right_router  # your router

@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    app.state.http = httpx.AsyncClient(timeout=timeout, limits=limits)
    yield
    await app.state.http.aclose()

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(right_router, prefix="/v1", tags=["embeddings"])
    return app

# Optional: create a module-level app for uvicorn "module:app"
app = create_app()


# src/api/rightApp/ar_router.py
import os
from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import httpx

router = APIRouter()

HF_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
HF_TOKEN = os.getenv("HF_TOKEN")

class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http

@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, http: httpx.AsyncClient = Depends(get_http)):
    if not HF_ENDPOINT or not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Endpoint or token not configured")
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    r = await http.post(HF_ENDPOINT, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": texts})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    vectors = data.get("embeddings", data) if isinstance(data, dict) else data
    return {"embeddings": vectors}

# main.py (repo root)
from src.core.app_factory import app  # or create_app() if you prefer

# If you insist on in-file run:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


