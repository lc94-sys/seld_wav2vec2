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

