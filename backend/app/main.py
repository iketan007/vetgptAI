"""
VetGPT backend — FastAPI service exposing a RAG-powered pet care chat endpoint.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health          -> liveness check
    POST /chat            -> { "message": str, "history": [...] } -> answer + sources
    POST /reindex          -> rebuilds the FAISS index from data/knowledge_base
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config
from app import rag

app = FastAPI(
    title="VetGPT API",
    description="AI-powered veterinary chatbot backend (RAG: LangChain + FAISS + Llama 2)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class Source(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        answer, docs = rag.answer_query(req.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    sources = [Source(content=d.page_content, metadata=d.metadata) for d in docs]
    return ChatResponse(answer=answer, sources=sources)


@app.post("/reindex")
def reindex():
    try:
        rag.build_index()
        # Force the in-memory vectorstore to reload on next query
        rag._vectorstore = None
        return {"status": "reindexed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reindex failed: {exc}")
