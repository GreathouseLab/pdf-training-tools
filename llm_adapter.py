# Minimal Together adapter for a quick connectivity check
import os
from together import Together  # pip install together
from dotenv import load_dotenv
load_dotenv()  # reads .env so TOGETHER_API_KEY is available

try:
    from together import Together  # pip install together
except Exception as e:
    raise RuntimeError("Install Together SDK: pip install together") from e


def llm_smoke_test(model: str = "OpenAI/gpt-oss-20B") -> None:
    """Ping Together chat API and print a small reply."""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set (put it in .env).")
    client = Together(api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise scientific assistant."},
            {"role": "user", "content": "Reply with exactly: OK"}
        ],
        max_tokens=8,
        temperature=0.0,
    )
    text = (r.choices[0].message.content or "").strip()
    print("[LLM smoke test]:", text)