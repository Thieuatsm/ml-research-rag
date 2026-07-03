"""
Parser: đọc file PDF/Markdown thô, trả về danh sách ContentBlock kèm
content_type (text / code / equation / table) và metadata (source, page, section).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContentBlock:
    text: str
    content_type: str      # "text" | "code" | "equation" | "table"
    source: str             # tên file gốc
    page: int | None = None
    section: str | None = None


# --- Heuristics phân loại nội dung ---

# Ký hiệu LaTeX / toán học thường gặp trong công thức
_EQUATION_MARKERS = re.compile(
    r"(\\frac|\\sum|\\int|\\alpha|\\beta|\\theta|\\nabla|\\partial|"
    r"\\mathbb|\\mathcal|\\left|\\right|\$\$|\\\[|\\\])"
)
# Một dòng toàn ký hiệu toán ngắn, ít chữ thường (heuristic đơn giản, không hoàn hảo)
_SHORT_MATH_LINE = re.compile(r"^[A-Za-z0-9\s\+\-\*/=\^_\(\)\.,\\{}\[\]]{1,80}$")

_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
# Nhận diện dòng code khi không có fence: thụt lề >=4 space, hoặc chứa từ khóa lập trình phổ biến
_CODE_KEYWORDS = re.compile(
    r"^\s*(def |class |import |from |for |while |if __name__|return |#include|public |private )"
)

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")  # dòng dạng markdown table "| a | b |"


def classify_block(raw_text: str) -> str:
    """
    Phân loại 1 đoạn text thô thành content_type dựa trên heuristic đơn giản.
    Đây không phải NLP classifier chính xác tuyệt đối — mục tiêu là tách được
    phần lớn code/equation/table ra khỏi text thường để chunk khác nhau.
    """
    stripped = raw_text.strip()
    if not stripped:
        return "text"

    lines = stripped.splitlines()

    # Table: nhiều dòng dạng "| ... | ... |"
    table_lines = sum(1 for ln in lines if _TABLE_ROW.match(ln))
    if table_lines >= 2 or (len(lines) <= 3 and table_lines >= 1):
        return "table"

    # Code: có từ khóa lập trình hoặc thụt lề đồng nhất kiểu code
    code_lines = sum(1 for ln in lines if _CODE_KEYWORDS.match(ln))
    if code_lines >= 1 or stripped.startswith(("def ", "class ", "import ", "from ")):
        return "code"

    # Equation: chứa nhiều ký hiệu LaTeX
    if _EQUATION_MARKERS.search(stripped):
        return "equation"

    # Equation heuristic bổ sung: đoạn rất ngắn, mật độ ký hiệu toán cao
    if len(stripped) < 120:
        symbol_count = sum(stripped.count(c) for c in "=^_+-*/\\")
        if symbol_count >= 3 and symbol_count / max(len(stripped), 1) > 0.08:
            return "equation"

    return "text"


def parse_markdown(file_path: Path) -> list[ContentBlock]:
    """
    Parse 1 file Markdown thành danh sách ContentBlock.
    - Tách code fence (```...```) thành content_type="code" riêng biệt.
    - Phần còn lại chia theo đoạn văn (blank line), mỗi đoạn qua classify_block().
    - Heading (#, ##, ###...) được lưu làm "section" cho các block phía sau.
    """
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    blocks: list[ContentBlock] = []
    current_section: str | None = None

    # Tách code fences ra trước, giữ vị trí bằng placeholder để xử lý phần còn lại theo đoạn văn
    code_snippets = []

    def _stash_code(match: re.Match) -> str:
        code_snippets.append(match.group(2))
        return f"\n@@CODE_BLOCK_{len(code_snippets) - 1}@@\n"

    text_without_code = _CODE_FENCE.sub(_stash_code, raw)

    paragraphs = re.split(r"\n\s*\n", text_without_code)
    for para in paragraphs:
        para = para.strip("\n")
        if not para.strip():
            continue

        # Heading -> cập nhật section, không tạo block riêng (trừ khi muốn giữ heading làm context)
        heading_match = re.match(r"^(#{1,6})\s+(.*)", para.strip())
        if heading_match:
            current_section = heading_match.group(2).strip()
            continue

        # Placeholder code block đã tách trước đó
        code_ph = re.match(r"@@CODE_BLOCK_(\d+)@@", para.strip())
        if code_ph:
            idx = int(code_ph.group(1))
            blocks.append(
                ContentBlock(
                    text=code_snippets[idx].strip(),
                    content_type="code",
                    source=file_path.name,
                    section=current_section,
                )
            )
            continue

        content_type = classify_block(para)
        blocks.append(
            ContentBlock(
                text=para.strip(),
                content_type=content_type,
                source=file_path.name,
                section=current_section,
            )
        )

    return blocks


def parse_pdf(file_path: Path) -> list[ContentBlock]:
    """
    Parse 1 file PDF thành danh sách ContentBlock bằng PyMuPDF (fitz).
    - Dùng page.get_text("blocks") để lấy từng khối văn bản kèm vị trí.
    - Dùng page.find_tables() (PyMuPDF >= 1.23) để tách bảng biểu ra riêng.
    - Heading được đoán bằng font-size lớn hơn trung bình trang (heuristic).
    """
    import fitz  # PyMuPDF

    blocks: list[ContentBlock] = []
    doc = fitz.open(file_path)

    try:
        for page_num, page in enumerate(doc, start=1):
            # 1) Tách bảng biểu trước để loại khỏi text block bên dưới
            table_bboxes = []
            try:
                tables = page.find_tables()
                for tbl in tables.tables:
                    table_bboxes.append(tbl.bbox)
                    rows = tbl.extract()
                    table_text = "\n".join(
                        " | ".join(str(cell) if cell is not None else "" for cell in row)
                        for row in rows
                    )
                    if table_text.strip():
                        blocks.append(
                            ContentBlock(
                                text=table_text,
                                content_type="table",
                                source=file_path.name,
                                page=page_num,
                            )
                        )
            except Exception:
                # Một số PDF/scan không hỗ trợ find_tables tốt -> bỏ qua, coi như không có bảng
                pass

            # 2) Lấy các block văn bản còn lại, loại bỏ phần đã thuộc về bảng
            text_dict = page.get_text("dict")
            font_sizes = []
            for blk in text_dict.get("blocks", []):
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span.get("size", 0))
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0

            for blk in text_dict.get("blocks", []):
                bbox = blk.get("bbox")
                if bbox and _bbox_overlaps_any(bbox, table_bboxes):
                    continue  # đã xử lý ở phần table

                block_text_parts = []
                max_span_size = 0
                is_monospace = True
                for line in blk.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    block_text_parts.append(line_text)
                    for span in line.get("spans", []):
                        max_span_size = max(max_span_size, span.get("size", 0))
                        font_name = span.get("font", "").lower()
                        if "mono" not in font_name and "courier" not in font_name and "consolas" not in font_name:
                            is_monospace = False

                block_text = "\n".join(p for p in block_text_parts if p.strip())
                if not block_text.strip():
                    continue

                # Heading: font lớn hơn trung bình đáng kể và ngắn -> coi là section title, không tạo block riêng
                if avg_font_size and max_span_size > avg_font_size * 1.3 and len(block_text) < 120:
                    # Gắn làm section cho các block tiếp theo trong cùng trang (đơn giản hoá:
                    # lưu trực tiếp làm 1 block content_type="text" để không mất thông tin)
                    blocks.append(
                        ContentBlock(
                            text=block_text.strip(),
                            content_type="text",
                            source=file_path.name,
                            page=page_num,
                            section=block_text.strip(),
                        )
                    )
                    continue

                if is_monospace:
                    content_type = "code"
                else:
                    content_type = classify_block(block_text)

                blocks.append(
                    ContentBlock(
                        text=block_text.strip(),
                        content_type=content_type,
                        source=file_path.name,
                        page=page_num,
                    )
                )
    finally:
        doc.close()

    return blocks


def _bbox_overlaps_any(bbox, bboxes, threshold: float = 0.5) -> bool:
    """Kiểm tra bbox có overlap đáng kể với 1 trong các bbox bảng đã tách hay không."""
    x0, y0, x1, y1 = bbox
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if area == 0:
        return False
    for tb in bboxes:
        tx0, ty0, tx1, ty1 = tb
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx1), min(y1, ty1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter / area > threshold:
            return True
    return False


def parse_file(file_path: Path) -> list[ContentBlock]:
    """Điều phối theo phần mở rộng file: .pdf -> parse_pdf, .md/.markdown -> parse_markdown."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        return parse_markdown(file_path)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {suffix}")


def parse_directory(dir_path: Path) -> list[ContentBlock]:
    """Parse toàn bộ file .pdf/.md trong 1 thư mục (không đệ quy thư mục con)."""
    all_blocks: list[ContentBlock] = []
    for file_path in sorted(Path(dir_path).iterdir()):
        if file_path.suffix.lower() in (".pdf", ".md", ".markdown"):
            print(f"[parser] Đang parse: {file_path.name}")
            try:
                all_blocks.extend(parse_file(file_path))
            except Exception as e:
                print(f"[parser] Lỗi khi parse {file_path.name}: {e}")
    return all_blocks
