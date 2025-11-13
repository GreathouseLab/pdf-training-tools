# Minimal Together adapter for a quick connectivity check
import os
from together import Together  # pip install together
from dotenv import load_dotenv

load_dotenv()  # reads .env so TOGETHER_API_KEY is available

def llm_smoke_test(model: str | None = None) -> None:
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    model = model or os.getenv("TOGETHER_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise scientific assistant."},
                {"role":"user","content":"Reply with exactly: OK"}
            ],
            max_tokens=8,
            temperature=0.0,
        )
        print("[LLM smoke test]:", (r.choices[0].message.content or "").strip())
    except Exception as e:
        print(f"[LLM smoke test] Failed for model '{model}': {e}")
        # Helpful: show a few available model ids to pick from
        try:
            models = client.models.list()
            ids = [m.id if hasattr(m, "id") else (m.get("id") if isinstance(m, dict) else str(m)) for m in models.data]
            print("Available models (first 10):")
            for mid in ids[:10]:
                print(" -", mid)
            print("Tip: rerun with --llm_model \"<one of the ids above>\"")
        except Exception as e2:
            print("Also failed to list models:", e2)

# === PTPC / Q&A: Together chat adapter ===
import os, json
from together import Together

def _extract_json_block(text: str) -> str:
    """Grab the JSON object if the model wrapped it with extra text."""
    if not text:
        raise ValueError("Empty model response.")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in model response.")
    return text[start:end+1]

def llm_chat(messages, model: str | None = None, max_tokens: int = 2000, temperature: float = 0.0) -> str:
    """
    Call Together's chat API. Return a STRICT JSON string.
    - messages: [{"role":"system","content":...},{"role":"user","content":...}]
    - model: uses TOGETHER_MODEL env if not provided
    """
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    model = model or os.getenv("TOGETHER_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,   # keep 0.0 for JSON-only outputs
    )
    content = (resp.choices[0].message.content or "").strip()

    # Return clean JSON (or extract it if the model added wrappers)
    try:
        json.loads(content)
        return content
    except Exception:
        cleaned = _extract_json_block(content)
        json.loads(cleaned)  # verify parses
        return cleaned