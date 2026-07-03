"""
Retriever: hybrid search (BM25 sparse + dense embedding) kết hợp bằng
Reciprocal Rank Fusion (RRF), sau đó re-rank bằng cross-encoder.
"""
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import TOP_K_RETRIEVE, TOP_K_RERANK, RERANKER_MODEL
from src.retrieval.embedder import get_collection, embed_query, load_chunks_jsonl

_TOKEN_RE = re.compile(r"[a-zA-Z0-9À-ỹ]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho BM25: lowercase, tách theo từ (hỗ trợ ký tự có dấu tiếng Việt)."""
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """
    Gói toàn bộ trạng thái cần thiết cho retrieval (BM25 index, chunk list,
    Chroma collection, reranker model) để không phải load lại mỗi lần gọi retrieve().
    Khởi tạo 1 lần khi app start (xem app/api.py, app/demo.py).
    """

    def __init__(self):
        print("[retriever] Đang load chunks + build BM25 index...")
        self.chunks = load_chunks_jsonl()
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}

        tokenized_corpus = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("[retriever] Đang kết nối Chroma collection...")
        self.collection = get_collection()

        self._reranker: CrossEncoder | None = None  # lazy load, chỉ load khi thực sự cần rerank

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"[retriever] Đang load reranker: {RERANKER_MODEL}")
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def dense_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[str]:
        """Tìm kiếm dense trên Chroma, trả về danh sách chunk_id theo thứ tự liên quan giảm dần."""
        query_embedding = embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.chunks)),
        )
        return results["ids"][0] if results.get("ids") else []

    def bm25_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[str]:
        """Tìm kiếm sparse bằng BM25, trả về danh sách chunk_id theo thứ tự score giảm dần."""
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]
        return [self.chunks[i].chunk_id for i in top_indices]

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[str]], k: int = 60
    ) -> list[str]:
        """
        Kết hợp nhiều ranked list (mỗi list là danh sách chunk_id theo thứ tự liên quan)
        thành 1 list duy nhất bằng công thức RRF:
            score(doc) = sum over lists of 1 / (k + rank_trong_list)
        k=60 là giá trị mặc định phổ biến trong các paper RRF gốc.
        """
        scores: dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    def rerank(self, query: str, candidate_ids: list[str], top_k: int = TOP_K_RERANK) -> list[str]:
        """Re-rank candidate_ids bằng cross-encoder, trả về top_k chunk_id cuối cùng."""
        if not candidate_ids:
            return []
        pairs = [(query, self.chunk_by_id[cid].text) for cid in candidate_ids]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]

    def retrieve(self, query: str, use_bm25: bool = True, use_rerank: bool = True):
        """
        Pipeline retrieval đầy đủ: dense search (+ BM25 search nếu use_bm25=True)
        -> RRF -> rerank (nếu use_rerank=True).
        use_bm25=False, use_rerank=False tương đương "dense-only" — dùng làm baseline
        so sánh trong ablation study (xem src/evaluation/evaluate.py).
        Trả về list[Chunk] kèm metadata để cite nguồn.
        """
        dense_ids = self.dense_search(query, top_k=TOP_K_RETRIEVE)

        if use_bm25:
            bm25_ids = self.bm25_search(query, top_k=TOP_K_RETRIEVE)
            fused_ids = self.reciprocal_rank_fusion([dense_ids, bm25_ids])
        else:
            fused_ids = dense_ids

        if use_rerank:
            final_ids = self.rerank(query, fused_ids[:TOP_K_RETRIEVE], top_k=TOP_K_RERANK)
        else:
            final_ids = fused_ids[:TOP_K_RERANK]

        return [self.chunk_by_id[cid] for cid in final_ids if cid in self.chunk_by_id]


# --- Singleton tiện dụng để import trực tiếp hàm retrieve() từ nơi khác ---
_retriever_instance: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def retrieve(query: str) -> list:
    """Hàm tiện dụng: lấy singleton retriever và trả kết quả retrieve(query)."""
    return get_retriever().retrieve(query)
