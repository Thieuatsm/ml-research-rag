"""
Prompt templates cho generation.
Yêu cầu bắt buộc: model phải cite nguồn (tên source + page) và từ chối trả lời
nếu context không đủ để giảm hallucination.
"""

SYSTEM_PROMPT = """Bạn là trợ lý nghiên cứu ML/AI. Chỉ trả lời dựa trên các đoạn \
context được cung cấp bên dưới. Với mỗi thông tin bạn đưa ra, hãy chú thích \
nguồn theo dạng [source: <tên tài liệu>, trang <số trang>].

Nếu context không chứa đủ thông tin để trả lời câu hỏi, hãy trả lời rõ ràng: \
"Tôi không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này." \
Không được tự bịa thông tin ngoài context."""


def _format_chunk(chunk, index: int) -> str:
    """Format 1 chunk kèm metadata nguồn để model có thể trích dẫn chính xác."""
    page_info = f", trang {chunk.page}" if getattr(chunk, "page", None) else ""
    section_info = f" (mục: {chunk.section})" if getattr(chunk, "section", None) else ""
    header = f"[Đoạn {index}] Nguồn: {chunk.source}{page_info}{section_info}"
    return f"{header}\n{chunk.text}"


def build_user_prompt(question: str, context_chunks: list) -> str:
    """
    Ghép các context_chunks (kèm metadata source/page) và câu hỏi thành
    prompt cuối cùng gửi cho LLM.
    """
    if not context_chunks:
        context_block = "(Không tìm thấy đoạn context nào liên quan.)"
    else:
        context_block = "\n\n".join(
            _format_chunk(chunk, i + 1) for i, chunk in enumerate(context_chunks)
        )

    return f"""Dưới đây là các đoạn tài liệu liên quan:

{context_block}

---

Câu hỏi: {question}

Hãy trả lời câu hỏi trên chỉ dựa vào các đoạn tài liệu ở trên, nhớ trích nguồn \
theo đúng định dạng [source: ..., trang ...]."""
