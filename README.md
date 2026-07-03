---
title: ML Research Assistant RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
---
# ML Research Assistant (RAG)

Hệ thống RAG (Retrieval-Augmented Generation) cho Q&A trên tài liệu kỹ thuật ML/AI
(papers arXiv + docs thư viện như PyTorch/Hugging Face), với trọng tâm là xử lý tốt
**công thức toán, code snippet và bảng biểu** — những phần RAG generic thường xử lý kém.

> 🚧 Đang xây dựng — checklist tiến độ ở dưới.

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

## Cài đặt

```bash
git clone <repo-url>
cd ml-research-rag
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env

# Cài Ollama (free, chạy local): https://ollama.com
ollama pull qwen2.5:3b-instruct
```

> Toàn bộ stack mặc định **100% free**: Ollama (local, dev) + Groq free tier
> (khi deploy demo public lên HF Spaces) + embedding/reranker chạy local.
> Không cần thẻ tín dụng, không cần Anthropic/OpenAI API key.

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

## Checklist tiến độ

- [x] Setup khung project
- [x] Ingestion: parser (PDF/Markdown, phân loại content-type)
- [x] Ingestion: content-aware chunker (+ baseline naive chunker để so sánh)
- [x] Embedding + Chroma vector store
- [x] Hybrid retrieval (BM25 + dense + RRF)
- [x] Re-ranking (cross-encoder)
- [x] Generation (prompt + LLM client: Ollama/Groq)
- [x] FastAPI backend
- [x] Gradio demo
- [x] Script CLI build_index.py (parse -> chunk -> embed, chạy 1 lệnh)
- [x] Evaluation harness (RAGAS) + ablation study (dense-only vs hybrid+rerank)
- [ ] Tạo eval set thật (30-50 câu hỏi) — xem `eval_results/eval_questions.example.jsonl` để biết format
- [ ] Chạy thử trên tài liệu thật, điền bảng kết quả vào README
- [ ] Deploy demo lên Hugging Face Spaces (đổi `GENERATION_BACKEND=groq` trong `.env` trước khi deploy)

> Toàn bộ code đã triển khai đầy đủ và kiểm tra cú pháp/logic thuần Python (parser,
> chunker, RRF). Các phần cần model thật (embedding, Ollama, reranker) chưa được chạy
> thử trong môi trường phát triển này — hãy `pip install -r requirements.txt`, cài
> Ollama, rồi chạy `python scripts/build_index.py` trên máy bạn để kiểm tra end-to-end.

## Tech stack

Python, PyMuPDF, sentence-transformers, ChromaDB, rank-bm25,
Ollama (Qwen2.5-3B, free/local) + Groq (free tier, demo public),
RAGAS, FastAPI, Gradio. **Không phát sinh chi phí.**
