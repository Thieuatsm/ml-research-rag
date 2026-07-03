"""
Script chạy toàn bộ pipeline ingestion: parse -> chunk -> embed -> build vector store.

Sử dụng:
    python scripts/build_index.py

Yêu cầu: đã đặt file .pdf/.md vào thư mục data/raw/ trước khi chạy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW_DIR
from src.ingestion.parser import parse_directory
from src.ingestion.chunker import chunk_document
from src.retrieval.embedder import build_vector_store


def main():
    if not DATA_RAW_DIR.exists() or not any(DATA_RAW_DIR.iterdir()):
        print(
            f"[build_index] Thư mục {DATA_RAW_DIR} rỗng. "
            f"Hãy đặt file .pdf/.md vào đó trước khi chạy script này."
        )
        return

    print(f"[build_index] Bước 1/3: Parse tài liệu từ {DATA_RAW_DIR}...")
    blocks = parse_directory(DATA_RAW_DIR)
    print(f"[build_index] Đã parse {len(blocks)} content block.")

    print("[build_index] Bước 2/3: Content-aware chunking...")
    chunks = chunk_document(blocks)
    print(f"[build_index] Đã tạo {len(chunks)} chunk.")

    print("[build_index] Bước 3/3: Embedding + build vector store...")
    build_vector_store(chunks)

    print("[build_index] Hoàn tất! Chạy `python app/demo.py` để thử demo.")


if __name__ == "__main__":
    main()
