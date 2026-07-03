"""
FastAPI backend expose endpoint /query cho RAG pipeline.
Chạy: uvicorn app.api:app --reload

Lưu ý: retriever được load 1 lần lúc startup (không load lại mỗi request)
vì việc load embedding model + BM25 index khá tốn thời gian.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generation.llm_client import generate
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import get_retriever

app = FastAPI(title="ML Research RAG API")


class QueryRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    source: str
    page: int | None = None
    section: str | None = None
    content_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


@app.on_event("startup")
def _load_retriever_on_startup():
    """Load retriever (embedding model, BM25 index, Chroma collection) 1 lần khi server khởi động."""
    get_retriever()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Nhận câu hỏi, trả về câu trả lời kèm nguồn trích dẫn."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    retriever = get_retriever()
    chunks = retriever.retrieve(request.question)

    user_prompt = build_user_prompt(request.question, chunks)
    answer = generate(SYSTEM_PROMPT, user_prompt)

    sources = [
        SourceInfo(
            source=c.source,
            page=c.page,
            section=c.section,
            content_type=c.content_type,
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}