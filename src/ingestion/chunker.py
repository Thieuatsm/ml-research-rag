"""
Chunker: nhận danh sách ContentBlock từ parser.py, sinh ra các Chunk cuối cùng
để đưa vào embedding. Chiến lược chunk khác nhau theo content_type
(content-aware chunking — điểm nhấn kỹ thuật chính của project).

Ghi chú về "token": để không phụ thuộc thêm thư viện tokenizer riêng (tiktoken...),
ta xấp xỉ 1 token ~ 0.75 từ (heuristic phổ biến cho văn bản tiếng Anh/kỹ thuật).
Với mục đích chunking, xấp xỉ này đủ tốt — không cần chính xác tuyệt đối.
"""
import uuid
from dataclasses import dataclass

from src.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS
from src.ingestion.parser import ContentBlock


@dataclass
class Chunk:
    chunk_id: str
    text: str
    content_type: str
    source: str
    page: int | None = None
    section: str | None = None


def _words_per_chunk(token_budget: int) -> int:
    """Xấp xỉ số từ tương ứng với token budget (1 token ~ 0.75 từ)."""
    return max(1, int(token_budget * 0.75))


def _make_chunk(text: str, content_type: str, source: str, page=None, section=None) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4())[:8],
        text=text.strip(),
        content_type=content_type,
        source=source,
        page=page,
        section=section,
    )


def chunk_text_blocks(blocks: list[ContentBlock]) -> list[Chunk]:
    """
    Chunk các block loại 'text' theo kích thước từ + overlap (sliding window),
    dùng CHUNK_SIZE_TOKENS / CHUNK_OVERLAP_TOKENS từ config.
    Mỗi block được xử lý độc lập để không trộn nội dung giữa các section khác nhau.
    """
    chunk_size_words = _words_per_chunk(CHUNK_SIZE_TOKENS)
    overlap_words = _words_per_chunk(CHUNK_OVERLAP_TOKENS)

    chunks: list[Chunk] = []
    for block in blocks:
        words = block.text.split()
        if len(words) <= chunk_size_words:
            chunks.append(
                _make_chunk(block.text, "text", block.source, block.page, block.section)
            )
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            piece = " ".join(words[start:end])
            chunks.append(
                _make_chunk(piece, "text", block.source, block.page, block.section)
            )
            if end == len(words):
                break
            start = end - overlap_words  # lùi lại để tạo overlap giữa các chunk liên tiếp

    return chunks


def chunk_code_blocks(blocks: list[ContentBlock]) -> list[Chunk]:
    """
    Giữ nguyên mỗi code block thành 1 chunk riêng (không cắt giữa function),
    trừ khi block quá dài (>2x chunk_size) thì mới cắt theo ranh giới dòng trống
    để tránh 1 chunk quá lớn làm loãng embedding.
    """
    max_words = _words_per_chunk(CHUNK_SIZE_TOKENS) * 2
    chunks: list[Chunk] = []

    for block in blocks:
        words = block.text.split()
        if len(words) <= max_words:
            chunks.append(
                _make_chunk(block.text, "code", block.source, block.page, block.section)
            )
            continue

        # Code quá dài: cắt theo đoạn trống (thường là ranh giới function/class)
        parts = block.text.split("\n\n")
        buffer = ""
        for part in parts:
            if len((buffer + "\n\n" + part).split()) > max_words and buffer:
                chunks.append(
                    _make_chunk(buffer, "code", block.source, block.page, block.section)
                )
                buffer = part
            else:
                buffer = f"{buffer}\n\n{part}" if buffer else part
        if buffer.strip():
            chunks.append(
                _make_chunk(buffer, "code", block.source, block.page, block.section)
            )

    return chunks


def chunk_equation_blocks(blocks: list[ContentBlock], all_blocks: list[ContentBlock]) -> list[Chunk]:
    """
    Giữ công thức cùng ngữ cảnh câu trước/sau (không tách rời công thức khỏi
    phần giải thích) — vì công thức đứng một mình gần như vô nghĩa với retrieval.
    Tìm block liền trước/sau (cùng content_type='text') trong all_blocks để ghép thêm.
    """
    chunks: list[Chunk] = []
    for block in blocks:
        idx = all_blocks.index(block) if block in all_blocks else -1
        context_parts = [block.text]

        if idx > 0 and all_blocks[idx - 1].content_type == "text":
            prev_words = all_blocks[idx - 1].text.split()
            context_parts.insert(0, " ".join(prev_words[-40:]))  # lấy ~40 từ cuối của đoạn trước

        if 0 <= idx < len(all_blocks) - 1 and all_blocks[idx + 1].content_type == "text":
            next_words = all_blocks[idx + 1].text.split()
            context_parts.append(" ".join(next_words[:40]))  # lấy ~40 từ đầu của đoạn sau

        merged_text = "\n".join(context_parts)
        chunks.append(
            _make_chunk(merged_text, "equation", block.source, block.page, block.section)
        )

    return chunks


def chunk_table_blocks(blocks: list[ContentBlock]) -> list[Chunk]:
    """Bảng biểu giữ nguyên nguyên khối — cắt bảng giữa chừng sẽ phá vỡ ngữ nghĩa hàng/cột."""
    return [
        _make_chunk(block.text, "table", block.source, block.page, block.section)
        for block in blocks
    ]


def chunk_document(blocks: list[ContentBlock]) -> list[Chunk]:
    """
    Điều phối: gọi hàm chunk phù hợp theo content_type rồi gộp kết quả.
    Thứ tự trả về giữ theo thứ tự xuất hiện trong tài liệu gốc để dễ debug/đọc lại.
    """
    text_blocks = [b for b in blocks if b.content_type == "text"]
    code_blocks = [b for b in blocks if b.content_type == "code"]
    equation_blocks = [b for b in blocks if b.content_type == "equation"]
    table_blocks = [b for b in blocks if b.content_type == "table"]

    all_chunks: list[Chunk] = []
    all_chunks.extend(chunk_text_blocks(text_blocks))
    all_chunks.extend(chunk_code_blocks(code_blocks))
    all_chunks.extend(chunk_equation_blocks(equation_blocks, blocks))
    all_chunks.extend(chunk_table_blocks(table_blocks))

    return all_chunks


def chunk_document_naive(blocks: list[ContentBlock]) -> list[Chunk]:
    """
    Baseline để so sánh trong ablation study: fixed-size chunking đồng nhất,
    KHÔNG phân biệt content_type. Dùng để chứng minh content-aware chunking
    tốt hơn baseline này trong evaluation (xem src/evaluation/evaluate.py).
    """
    chunk_size_words = _words_per_chunk(CHUNK_SIZE_TOKENS)
    overlap_words = _words_per_chunk(CHUNK_OVERLAP_TOKENS)

    full_text = "\n\n".join(b.text for b in blocks)
    words = full_text.split()
    source = blocks[0].source if blocks else "unknown"

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        piece = " ".join(words[start:end])
        chunks.append(_make_chunk(piece, "text", source))
        if end == len(words):
            break
        start = end - overlap_words

    return chunks
