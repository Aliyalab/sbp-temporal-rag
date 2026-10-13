"""FastAPI backend. Run with:  uvicorn src.api:app --port 8000

Set SBP_DATA_FILE to the passage file (default: data/processed/passages.json).
Questions are never written to logs by this code.
"""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Settings
from src.engine import RagEngine
from src.schemas import AskResponse
from src.security import RateLimiter
from src.store import DataError, load_passages

settings = Settings()
DATA_FILE = os.environ.get("SBP_DATA_FILE", "data/processed/passages.json")
limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

app = FastAPI(title="SBP Temporal RAG", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # your Streamlit page only
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars * 4)
    transaction_date: date


@lru_cache(maxsize=1)
def get_engine() -> RagEngine:
    return RagEngine(load_passages(DATA_FILE, settings), settings)


@app.get("/health")
def health() -> dict:
    try:
        return {"status": "ok", "passages": len(get_engine().passages)}
    except DataError:
        return {"status": "no_data", "passages": 0}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    try:
        engine = get_engine()
    except DataError:
        raise HTTPException(status_code=503, detail="The document data is not loaded.")
    return engine.ask(req.question, req.transaction_date)
