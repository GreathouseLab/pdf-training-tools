import argparse
import csv
from pathlib import Path
from typing import List
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

SYSTEM_PROMPT = """You are a careful scientific assistant.
Given a passage from a paper, propose high-quality question/answer pairs that test understanding.
Prefer precise, factual Q/A over trivia. Keep answers concise but specific."""

USER_PROMPT_TMPL = """Create {k} high-quality Q/A pairs from this passage.\n
Return as JSON list of objects with 'question' and 'answer'.\n
Passage:\n---\n{passage}\n---
"""

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(text)

def chunk_text(txt: str, max_len: int = 2000, overlap: int = 200) -> List[str]:
    # Simple whitespace chunker
    words = txt.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_len]
        chunks.append(" ".join(chunk))
        i += max_len - overlap
        if i <= 0:  # safety
            break
    return chunks

def ask_model(client: OpenAI, model: str, prompt: str) -> List[dict]:
    r = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        max_output_tokens=800
    )
    txt = r.output_text.strip()
    # Try to locate JSON in the text
    import json, re
    match = re.search(r'\[.*\]', txt, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        out = []
        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a:
                out.append({"question": q, "answer": a})
        return out
    except Exception:
        return []

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to a PDF")
    ap.add_argument("--out", default="qa.csv", help="Output CSV path")
    ap.add_argument("--per-chunk", type=int, default=2, help="Q/A per chunk")
    ap.add_argument("--model", default="gpt-5-mini")
    args = ap.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY in environment or .env")

    client = OpenAI(api_key=key)

    text = read_pdf_text(Path(args.pdf))
    chunks = chunk_text(text, max_len=1800, overlap=200)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "chunk_id", "question", "answer"])
        for idx, ch in enumerate(tqdm(chunks, desc="Processing chunks")):
            prompt = USER_PROMPT_TMPL.format(k=args.per_chunk, passage=ch)
            pairs = ask_model(client, args.model, prompt)
            for qa in pairs:
                w.writerow([Path(args.pdf).name, idx, qa["question"], qa["answer"]])

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
