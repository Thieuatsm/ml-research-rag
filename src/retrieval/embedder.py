"""
Embedder: sinh vector embedding cho các Chunk và lưu vào Chroma vector store.
Cũng lưu song song toàn bộ chunk ra file JSONL (data/processed/chunks.jsonl)
để retriever.py có thể rebuild BM25 index mà không cần đọc lại từ Chroma.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    DATA_PROCESSED_DIR,
)
from src.ingestion.chunker import Chunk

_model_cache: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load (và cache) model embedding BAAI/bge-small-en-v1.5."""
    global _model_cache
    if _model_cache is None:
        print(f"[embedder] Đang load embedding model: {EMBEDDING_MODEL}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache


def embed_chunks(chunks: list[Chunk], batch_size: int = 32):
    """Encode danh sách chunk thành vectors, trả về list[list[float]]."""
    model = load_embedding_model()
    texts = [c.text for c in chunks]
    # bge models khuyến nghị thêm prefix cho passage khi embed để tối ưu retrieval
    prefixed_texts = [f"passage: {t}" for t in texts] if "bge" in EMBEDDING_MODEL.lower() else texts
    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def save_chunks_jsonl(chunks: list[Chunk], out_path: Path | None = None) -> Path:
    """Lưu toàn bộ chunk (text + metadata) ra JSONL để BM25 index rebuild lại sau này."""
    out_path = out_path or (DATA_PROCESSED_DIR / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "content_type": c.content_type,
                        "source": c.source,
                        "page": c.page,
                        "section": c.section,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[embedder] Đã lưu {len(chunks)} chunk vào {out_path}")
    return out_path


def load_chunks_jsonl(path: Path | None = None) -> list[Chunk]:
    """Đọc lại chunks đã lưu từ JSONL (dùng khi retriever cần rebuild BM25 mà không re-parse tài liệu)."""
    path = path or (DATA_PROCESSED_DIR / "chunks.jsonl")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=d["chunk_id"],
                    text=d["text"],
                    content_type=d["content_type"],
                    source=d["source"],
                    page=d.get("page"),
                    section=d.get("section"),
                )
            )
    return chunks


def build_vector_store(chunks: list[Chunk], reset: bool = True):
    """
    Tạo/khởi tạo Chroma collection, add chunks kèm metadata + embeddings.
    reset=True: xóa collection cũ trước khi build lại (tránh trùng lặp khi rebuild toàn bộ).
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # collection chưa tồn tại lần đầu chạy -> bỏ qua

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = embed_chunks(chunks)

    # Chroma giới hạn batch size khi add, chia nhỏ để an toàn với dataset lớn
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=batch_embeddings,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "content_type": c.content_type,
                    "source": c.source,
                    "page": c.page or -1,
                    "section": c.section or "",
                }
                for c in batch
            ],
        )

    print(f"[embedder] Đã build vector store với {len(chunks)} chunk tại {CHROMA_PERSIST_DIR}")
    save_chunks_jsonl(chunks)
    return collection


def get_collection():
    """Lấy lại Chroma collection đã build (dùng ở retriever.py, không tạo mới)."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(CHROMA_COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    """Embed 1 câu query để tìm kiếm (dùng prefix 'query:' theo khuyến nghị của bge models)."""
    model = load_embedding_model()
    prefixed = f"query: {query}" if "bge" in EMBEDDING_MODEL.lower() else query
    return model.encode([prefixed], normalize_embeddings=True)[0].tolist()
