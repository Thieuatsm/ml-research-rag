"""
Config trung tâm cho toàn bộ pipeline RAG.
Đọc biến môi trường từ .env, cung cấp default hợp lý cho từng thành phần.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EVAL_RESULTS_DIR = ROOT_DIR / "eval_results"

# --- Chunking ---
CHUNK_SIZE_TOKENS = 400          # kích thước chunk mặc định cho text thường
CHUNK_OVERLAP_TOKENS = 50
# Các content_type được phân loại khi ingest, dùng để chunk khác nhau theo từng loại
CONTENT_TYPES = ["text", "code", "equation", "table"]

# --- Embedding / Vector store ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", str(DATA_PROCESSED_DIR / "chroma_db")
)
CHROMA_COLLECTION_NAME = "ml_research_docs"

# --- Retrieval ---
TOP_K_RETRIEVE = 20      # số chunk lấy ra trước khi rerank
TOP_K_RERANK = 5         # số chunk cuối cùng đưa vào prompt sau rerank
HYBRID_ALPHA = 0.5        # trọng số kết hợp dense vs BM25 (0 = chỉ BM25, 1 = chỉ dense)

# --- Generation (100% free stack) ---
# "ollama": chạy local khi dev, không giới hạn, không tốn phí
# "groq":   free tier, dùng khi deploy demo public (HF Spaces free CPU không chạy nổi Ollama)
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_GENERATION_TOKENS = 800

# --- Evaluation ---
EVAL_SET_PATH = ROOT_DIR / "eval_results" / "eval_questions.jsonl"
