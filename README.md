# ML Research Assistant (RAG)

Hệ thống RAG (Retrieval-Augmented Generation) cho Q&A trên tài liệu kỹ thuật ML/AI
(papers arXiv + docs thư viện như PyTorch/Hugging Face), với trọng tâm là xử lý tốt
**công thức toán, code snippet và bảng biểu** — những phần RAG generic thường xử lý kém.


## Điểm nhấn kỹ thuật

- **Content-aware chunking**: phân loại từng đoạn nội dung (text / code / equation / table)
  và áp dụng chiến lược chunk riêng cho từng loại, thay vì fixed-size chunking đồng nhất.
- **Hybrid retrieval**: kết hợp BM25 (sparse) + dense embedding, hợp nhất bằng
  Reciprocal Rank Fusion, sau đó re-rank bằng cross-encoder.
- **Grounded generation**: prompt bắt buộc model trích nguồn (tên tài liệu + trang) và
  từ chối trả lời khi context không đủ, nhằm giảm hallucination.
- **Evaluation có số liệu**: đo bằng RAGAS (Faithfulness, Context Precision/Recall,
  Answer Relevancy), có ablation study so sánh naive vs content-aware chunking,
  dense-only vs hybrid+rerank.

## Kiến trúc

```
Raw docs (PDF/MD)
  → Parser (content-type classification)
  → Content-aware Chunker
  → Embedding (BAAI/bge-small-en-v1.5) → Chroma vector store
                                        → BM25 index
  → Query → Hybrid Search (RRF) → Re-rank (cross-encoder) → Top-k chunks
  → LLM Generation (Claude API / Ollama local) → Answer + citations
```

## Cấu trúc thư mục

```
ml-research-rag/
├── data/
│   ├── raw/              # PDFs, docs gốc (không commit — xem .gitignore)
│   └── processed/        # chunks đã xử lý, Chroma DB
├── src/
│   ├── config.py          # config trung tâm
│   ├── ingestion/         # parser.py, chunker.py
│   ├── retrieval/         # embedder.py, retriever.py
│   ├── generation/        # prompts.py, llm_client.py
│   └── evaluation/        # evaluate.py (RAGAS)
├── app/
│   ├── api.py             # FastAPI backend
│   └── demo.py             # Gradio demo (deploy lên HF Spaces)
├── notebooks/              # thử nghiệm, so sánh chunking strategies
├── eval_results/           # eval set + bảng kết quả ablation
├── requirements.txt
├── .env.example
└── README.md
```

## Sử dụng

```bash
# 1. Đặt PDF/Markdown vào data/raw/, sau đó chạy pipeline ingestion đầy đủ
#    (parse -> content-aware chunking -> embedding -> build Chroma vector store)
python scripts/build_index.py

# 2. Chạy demo Gradio (local, dùng Ollama)
python app/demo.py

# Hoặc chạy API backend
uvicorn app.api:app --reload

# 3. Tạo eval set: copy eval_results/eval_questions.example.jsonl
#    thành eval_results/eval_questions.jsonl và tự viết câu hỏi + ground truth

# 4. Chạy evaluation + ablation study (dense-only vs hybrid+rerank)
python -m src.evaluation.evaluate
```

## Kết quả (sẽ cập nhật sau khi có số liệu)

| Config | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| Naive chunking + dense-only | TBD | TBD | TBD | TBD |
| Content-aware chunking + hybrid + rerank | TBD | TBD | TBD | TBD |

## Tech stack

Python, PyMuPDF, sentence-transformers, ChromaDB, rank-bm25,
Ollama (Qwen2.5-3B, free/local) + Groq (free tier, demo public),
RAGAS, FastAPI, Gradio. **Không phát sinh chi phí.**
