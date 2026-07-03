"""
LLM client: gọi model FREE để sinh câu trả lời cuối cùng từ prompt đã build
ở prompts.py. Không cần Anthropic/OpenAI API key.

- generate_with_ollama(): dùng khi dev local. Cài Ollama (https://ollama.com),
  chạy `ollama pull qwen2.5:3b-instruct` một lần, sau đó gọi qua thư viện
  `ollama` — hoàn toàn free, không giới hạn số request, không cần internet.
- generate_with_groq(): dùng khi deploy demo public lên HF Spaces (free CPU
  không đủ mạnh để chạy Ollama). Groq có free tier khá rộng rãi và tốc độ
  inference rất nhanh. Lấy key miễn phí tại https://console.groq.com/keys
- generate(): điều phối theo GENERATION_BACKEND trong config (mặc định "ollama").
"""
from src.config import (
    GENERATION_BACKEND,
    OLLAMA_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_GENERATION_TOKENS,
)


def generate_with_ollama(system_prompt: str, user_prompt: str) -> str:
    """Gọi Ollama local model. Yêu cầu Ollama server đang chạy (`ollama serve`)."""
    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_predict": MAX_GENERATION_TOKENS},
    )
    return response["message"]["content"]


def generate_with_groq(system_prompt: str, user_prompt: str) -> str:
    """Gọi Groq API (free tier). Yêu cầu GROQ_API_KEY trong .env."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY chưa được cấu hình trong .env. "
            "Lấy key miễn phí tại https://console.groq.com/keys"
        )

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_GENERATION_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def generate(system_prompt: str, user_prompt: str, backend: str | None = None) -> str:
    """Điều phối generation theo backend (mặc định lấy từ config.GENERATION_BACKEND)."""
    backend = backend or GENERATION_BACKEND
    if backend == "ollama":
        return generate_with_ollama(system_prompt, user_prompt)
    elif backend == "groq":
        return generate_with_groq(system_prompt, user_prompt)
    raise ValueError(f"Backend không hợp lệ: {backend} (chỉ hỗ trợ 'ollama' hoặc 'groq')")
