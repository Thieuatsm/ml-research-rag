"""
Evaluation: đo chất lượng pipeline RAG bằng RAGAS trên eval set tự tạo
(eval_results/eval_questions.jsonl — mỗi dòng: {"question": ..., "ground_truth": ...}).

Metrics chính:
- Context Precision / Context Recall (đánh giá retrieval)
- Faithfulness (câu trả lời có bám sát context không, đo hallucination)
- Answer Relevancy

LLM giám khảo (judge) dùng Ollama local (free) — chất lượng đánh giá sẽ thấp hơn
dùng GPT-4/Claude làm giám khảo, đây là trade-off đã cân nhắc để giữ pipeline 100% free.
Nếu sau này có credit, chỉ cần đổi RAGAS_JUDGE_BACKEND sang "groq" trong config.
"""
import csv
import json

from src.config import EVAL_SET_PATH, EVAL_RESULTS_DIR, OLLAMA_MODEL
from src.generation.llm_client import generate
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import get_retriever


def load_eval_set(path=None) -> list[dict]:
    """Đọc eval set từ EVAL_SET_PATH (jsonl). Mỗi dòng: {"question": ..., "ground_truth": ...}."""
    path = path or EVAL_SET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy eval set tại {path}. "
            f"Tạo file này với mỗi dòng dạng "
            f'{{"question": "...", "ground_truth": "..."}} (xem eval_results/eval_questions.example.jsonl)'
        )
    eval_set = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_set.append(json.loads(line))
    return eval_set


def run_pipeline_on_eval_set(
    eval_set: list[dict], use_bm25: bool = True, use_rerank: bool = True
) -> list[dict]:
    """
    Với mỗi câu hỏi, chạy full pipeline (retrieve -> generate), lưu lại
    question/answer/contexts/ground_truth theo đúng format RAGAS yêu cầu.
    use_bm25/use_rerank cho phép chạy các config khác nhau phục vụ ablation.
    """
    retriever = get_retriever()
    results = []

    for item in eval_set:
        question = item["question"]
        chunks = retriever.retrieve(question, use_bm25=use_bm25, use_rerank=use_rerank)
        contexts = [c.text for c in chunks]

        user_prompt = build_user_prompt(question, chunks)
        answer = generate(SYSTEM_PROMPT, user_prompt)

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
            }
        )
        print(f"[evaluate] Đã xử lý: {question[:60]}...")

    return results


def compute_ragas_metrics(results: list[dict]) -> dict:
    """
    Tính các metric RAGAS trên kết quả pipeline.
    Dùng Ollama (OLLAMA_MODEL) làm LLM giám khảo + embedding model của project
    làm embeddings cho RAGAS — toàn bộ free, chạy local.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    from src.config import EMBEDDING_MODEL

    dataset = Dataset.from_list(results)

    judge_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    scores = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    return scores.to_pandas().mean(numeric_only=True).to_dict()


def run_ablation(eval_set: list[dict] | None = None) -> None:
    """
    So sánh 2 cấu hình retrieval trên cùng 1 eval set:
      1. "dense_only_no_rerank": baseline đơn giản (chỉ dense search, không BM25, không rerank)
      2. "hybrid_rerank": pipeline đầy đủ của project (dense + BM25 + RRF + cross-encoder rerank)
    Ghi kết quả ra eval_results/ablation_results.csv để đưa vào README/CV.

    Ghi chú: để ablation cả phần content-aware chunking vs naive chunking, cần build thêm
    1 Chroma collection riêng từ chunk_document_naive() (xem src/ingestion/chunker.py) và
    trỏ HybridRetriever vào collection đó — có thể mở rộng thêm nếu muốn so sánh đầy đủ hơn.
    """
    eval_set = eval_set or load_eval_set()
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = {
        "dense_only_no_rerank": {"use_bm25": False, "use_rerank": False},
        "hybrid_rerank": {"use_bm25": True, "use_rerank": True},
    }

    rows = []
    for config_name, kwargs in configs.items():
        print(f"\n[evaluate] === Đang chạy config: {config_name} ===")
        results = run_pipeline_on_eval_set(eval_set, **kwargs)
        metrics = compute_ragas_metrics(results)
        metrics["config"] = config_name
        rows.append(metrics)
        print(f"[evaluate] Kết quả {config_name}: {metrics}")

    out_path = EVAL_RESULTS_DIR / "ablation_results.csv"
    fieldnames = ["config"] + [k for k in rows[0].keys() if k != "config"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[evaluate] Đã lưu bảng ablation vào {out_path}")


if __name__ == "__main__":
    # Entry point: python -m src.evaluation.evaluate
    run_ablation()
