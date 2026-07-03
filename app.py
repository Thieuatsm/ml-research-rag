import sys
from pathlib import Path
from app.demo import demo


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.generation.llm_client import generate
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import get_retriever


def _format_sources(chunks) -> str:
    if not chunks:
        return "_Không tìm thấy nguồn liên quan._"
    lines = []
    for i, c in enumerate(chunks, start=1):
        page_info = f", trang {c.page}" if c.page else ""
        section_info = f" — {c.section}" if c.section else ""
        lines.append(f"**[{i}]** `{c.source}`{page_info}{section_info} _({c.content_type})_")
    return "\n\n".join(lines)


def answer_question(question: str):
    """Chạy full pipeline retrieve -> generate, trả về câu trả lời + nguồn trích dẫn."""
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", ""

    retriever = get_retriever()
    chunks = retriever.retrieve(question)

    user_prompt = build_user_prompt(question, chunks)
    answer = generate(SYSTEM_PROMPT, user_prompt)

    return answer, _format_sources(chunks)


with gr.Blocks(title="ML Research Assistant (RAG)") as demo:
    gr.Markdown(
        "# ML Research Assistant (RAG)\n"
        "Hỏi đáp trên tập paper/docs ML-AI, dùng hybrid retrieval (BM25 + dense + re-rank). "
        "100% miễn phí — chạy bằng Ollama (local) hoặc Groq (free tier)."
    )
    with gr.Row():
        question_box = gr.Textbox(
            label="Câu hỏi về ML/AI",
            placeholder="Vd: Attention mechanism hoạt động thế nào?",
            lines=2,
        )
    submit_btn = gr.Button("Hỏi", variant="primary")
    answer_box = gr.Textbox(label="Câu trả lời", lines=8)
    sources_box = gr.Markdown(label="Nguồn trích dẫn")

    submit_btn.click(
        fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box]
    )
    question_box.submit(
        fn=answer_question, inputs=question_box, outputs=[answer_box, sources_box]
    )

if __name__ == "__main__":
    demo.launch()