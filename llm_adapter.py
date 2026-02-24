# ==============================================================================
# NORE Universal LLM Adapter
# ==============================================================================
# Supports both Together AI and OpenAI backends via a unified interface.
# The pipeline auto-detects the backend from the model name, or you can
# force it with --llm_backend.
#
# Together AI models: moonshotai/Kimi-K2-Instruct-0905, meta-llama/*, etc.
# OpenAI models:      gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-5, etc.
#
# Usage:
#   from llm_adapter import llm_chat_universal, detect_backend
#
#   backend = detect_backend("gpt-4.1-mini")  # -> "openai"
#   response = llm_chat_universal(
#       model="gpt-4.1-mini",
#       messages=[{"role": "user", "content": "Hello"}],
#       backend=backend
#   )
# ==============================================================================

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

# Known OpenAI model prefixes
OPENAI_PREFIXES = (
    "gpt-", "o1", "o3", "o4", "o5",
    "chatgpt-", "ft:gpt-",
)

# Known Together AI model patterns (org/model format)
TOGETHER_INDICATORS = ("/",)  # Models with org/ prefix are Together


def detect_backend(model: str) -> str:
    """
    Auto-detect whether a model string targets OpenAI or Together AI.
    
    Rules:
      1. Starts with a known OpenAI prefix -> "openai"
      2. Contains '/' (org/model format like moonshotai/Kimi-K2) -> "together"
      3. Fallback: check which API key is available
      4. Default: "together" (backward compatible)
    
    Returns:
        "openai" or "together"
    """
    model_lower = model.lower().strip()

    # Rule 1: OpenAI prefix match
    for prefix in OPENAI_PREFIXES:
        if model_lower.startswith(prefix):
            return "openai"

    # Rule 2: org/model pattern -> Together
    if "/" in model:
        return "together"

    # Rule 3: fallback to whichever key is present
    if os.getenv("OPENAI_API_KEY") and not os.getenv("TOGETHER_API_KEY"):
        return "openai"

    return "together"


# ---------------------------------------------------------------------------
# Together AI client (singleton)
# ---------------------------------------------------------------------------

_together_client = None


def _get_together_client():
    global _together_client
    if _together_client is None:
        from together import Together
        _together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    return _together_client


def llm_chat_together(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> str:
    """
    Call Together AI's chat completion endpoint.
    
    Returns the assistant's response text (stripped).
    """
    client = _get_together_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# OpenAI client (singleton)
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file or export it:\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def llm_chat_openai(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> str:
    """
    Call OpenAI's chat completion endpoint.
    
    Accepts the same arguments as llm_chat_together() for drop-in compatibility.
    Returns the assistant's response text (stripped).
    """
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Universal dispatcher
# ---------------------------------------------------------------------------

_llm_call_counter = 0


def llm_chat_universal(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    backend: Optional[str] = None,
) -> str:
    """
    Universal LLM chat function that dispatches to the correct backend.
    
    This is the function you inject into the pipeline in place of
    llm_chat_together(). It has the same signature plus an optional
    `backend` parameter.
    
    Args:
        model:       Model identifier (e.g. "gpt-4.1-mini" or "moonshotai/Kimi-K2-Instruct-0905")
        messages:    List of {"role": ..., "content": ...} dicts
        temperature: Sampling temperature
        max_tokens:  Max tokens in response
        top_p:       Nucleus sampling parameter
        backend:     Force "openai" or "together". If None, auto-detected from model name.
    
    Returns:
        Assistant's response text (stripped).
    """
    global _llm_call_counter
    _llm_call_counter += 1
    call_num = _llm_call_counter

    if backend is None:
        backend = detect_backend(model)

    logging.info(f"[LLM call #{call_num}] {backend}:{model} ...")

    if backend == "openai":
        content = llm_chat_openai(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif backend == "together":
        content = llm_chat_together(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'openai' or 'together'.")

    logging.info(f"[LLM call #{call_num}] Got response ({len(content)} chars)")
    return content


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def llm_smoke_test(model: str | None = None, backend: str | None = None) -> None:
    """Quick connectivity check for either backend."""
    model = model or os.getenv("TOGETHER_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    backend = backend or detect_backend(model)

    print(f"[Smoke test] Backend: {backend}, Model: {model}")
    try:
        reply = llm_chat_universal(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise scientific assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            max_tokens=8,
            temperature=0.0,
            backend=backend,
        )
        print(f"[Smoke test] Response: {reply}")
    except Exception as e:
        print(f"[Smoke test] FAILED: {e}")


# ---------------------------------------------------------------------------
# Legacy compat -- the old llm_chat() used by PTPC triage
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Grab the JSON object if the model wrapped it with extra text."""
    if not text:
        raise ValueError("Empty model response.")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in model response.")
    return text[start : end + 1]


def llm_chat(
    messages,
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> str:
    """
    Legacy wrapper for PTPC triage. Returns strict JSON string.
    Uses Together backend by default (backward compatible).
    """
    model = model or os.getenv("TOGETHER_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    content = llm_chat_universal(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        backend="together",
    )
    try:
        json.loads(content)
        return content
    except Exception:
        cleaned = _extract_json_block(content)
        json.loads(cleaned)  # verify parses
        return cleaned


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NORE LLM Adapter -- smoke test & backend info")
    ap.add_argument("--model", default=None, help="Model to test")
    ap.add_argument("--backend", choices=["openai", "together"], default=None,
                    help="Force backend (default: auto-detect)")
    args = ap.parse_args()

    # Show environment status
    print("=" * 60)
    print("NORE LLM Adapter -- Environment Check")
    print("=" * 60)
    print(f"  TOGETHER_API_KEY: {'set' if os.getenv('TOGETHER_API_KEY') else 'not set'}")
    print(f"  OPENAI_API_KEY:   {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print(f"  TOGETHER_MODEL:   {os.getenv('TOGETHER_MODEL', '(not set)')}")
    print()

    # Auto-detect tests
    test_models = [
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-5",
        "moonshotai/Kimi-K2-Instruct-0905", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ]
    print("Backend auto-detection:")
    for m in test_models:
        print(f"  {m:55s} -> {detect_backend(m)}")
    print()

    # Run smoke test
    llm_smoke_test(model=args.model, backend=args.backend)
