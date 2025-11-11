#reads through a PDF, extracts text, chunks it, and for each chunk,
#asks an LLM to generate Q/A pairs, then embeds those using DeepSeek R1
#and stores them in Pinecone.

# python -m venv .venv
# source .venv/bin/activate
# pip install pymupdf openai python-dotenv tqdm pinecone matplotlib pydantic deepseek together
# python -m venv pymupdf-venv
# . pymupdf-venv/bin/activate
# python -m pip install --upgrade pip
# pip install --upgrade pymupdf
# pip install -r requirements.txt 
# export OPENAI_API_KEY=...
# export DEEPSEEK_API_KEY=...
# export PINECONE_API_KEY=......
# export PINECONE_ENV=...us-west1-gcp
# export MODEL=gpt-5-mini
# export PINECONE_INDEX=paper-qa
# python mupdf_trainer.py ./pdfs --k 1 --chunk_size 800 --chunk_overlap 50
# (where ./pdfs is a folder of PDFs to process)
# togerher model = OpenAI/gpt-oss-20B or meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

#function process_pdf_folder(folder_path):
 #   initialize total_pages = 0
  #  for each PDF in folder:
   #     extract pages from PDF
    #    for each page:
     #       extract text
      #      embed text using DeepSeek R1
       #     store embedding in Pinecone
        #    total_pages += 1
    #return total_pages
#-----------IMPORTS----------------
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
import time
import json
import re
import hashlib

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from typing import Tuple 
from typing import List
import pymupdf
import fitz

from together import Together 

import qa_pipeline_skeleton as qps   
import prompts_qa as prompts        

# === PTPC: imports ===
from onc_nutri_triage_prompt import build_messages, validate_triage_json
from llm_adapter import llm_chat


#-----------SETUP----------------# this needs a lot of work on the prompts 
load_dotenv()  # Load environment variables from .env file
import logging
logging.basicConfig(level=logging.INFO) # set logging level to INFO
import sys
logging.getLogger("openai").setLevel(logging.WARNING) # set openai logging to WARNING
logging.getLogger("pinecone").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are a careful scientific assistant.
 Given a passage from a paper, propose high-quality question/answer pairs that test understanding.
Prefer precise, factual Q/A over trivia. Keep answers concise but specific."""
USER_PROMPT_TMPL = """Create {k} high-quality Q/A pairs from this passage.\n    
Return as JSON list of objects with 'question' and 'answer'.\n
Passage:\n---\n{passage}\n---"""

#-----------LLM ADAPTER----------------
qps.llm_chat = llm_chat_together

def llm_chat_together(model: str, messages: list[dict],
                      temperature: float = 0.2, max_tokens: int = 1024, top_p: float = 0.9) -> str:
    """Unifies the LLM call signature used by the skeleton."""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p
    )
    return (resp.choices[0].message.content or "").strip()


#------------Start of main function----------------
def main():
    parser = argparse.ArgumentParser(description="Process PDFs to generate and store Q/A pairs.")
    parser.add_argument("pdf_folder", type=str, help="Path to folder containing PDF files.")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "gpt-5-mini"), help="LLM model to use.")
    parser.add_argument("--k", type=int, default=3, help="Number of Q/A pairs to generate per chunk.")
    parser.add_argument("--chunk_size", type=int, default=800, help="Max tokens per text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Token overlap between chunks.")
    parser.add_argument("--embed_model", type=str, default="deepseek-r1", help="Embedding model to use.")
    parser.add_argument("--pinecone_index", type=str, default=os.getenv("PINECONE_INDEX", "paper-qa"), help="Pinecone index name.")
    parser.add_argument("--pinecone_namespace", type=str, default="paper-qa-namespace", help="Pinecone namespace.")
    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    if not pdf_folder.is_dir():
        logging.error(f"Provided path {pdf_folder} is not a directory.")
        sys.exit(1)

    generate_qa_from_pdf(args.pdf, args.model, args.out)

    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        logging.error("Pinecone API key or environment not set in .env")
        sys.exit(1)
    
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if args.pinecone_index not in pinecone.list_indexes():
        logging.info(f"Creating Pinecone index: {args.pinecone_index}")
        pinecone.create_index(args.pinecone_index, dimension=1024)  # assuming DeepSeek R1 dim=1024
        time.sleep(10)  # wait for index to be ready
    pclient = PineconeClient()
    index = pclient.Index(args.pinecone_index)

#-----------EXTRACTION FUNCTION----------------
def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """
    Extract plain text from a PDF using PyMuPDF and return (text, num_pages).
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"No such file: {pdf_path}")

    doc = pymupdf.open(pdf_path)      
    page_texts: List[str] = []
    for page in doc:                  
        page_texts.append(page.get_text("text") or "")
    raw_text = "\n\f\n".join(page_texts)  # \f = form feed as a page break
    num_pages = doc.page_count
    logging.info(f"Extracted {num_pages} pages from {pdf_path}")

    # sanity check
    if 1:
        print(f"[extract_text_from_pdf] Extracted {num_pages} pages, preview: {raw_text[:300]}")
 
    return raw_text, num_pages

#-----------LLM Paper Triage Function (PTPC)----------------
#class PTPCConfig:
    MIN_USEFULNESS = 4
    MIN_EVIDENCE_SPANS = 1  # raise to 2 once stable

#def ptpc_gate(ptpc: dict) -> str:
    usefulness = int(ptpc.get("usefulness_score", 0))
    spans = ptpc.get("evidence_spans") or []
    if usefulness >= PTPCConfig.MIN_USEFULNESS and len(spans) >= PTPCConfig.MIN_EVIDENCE_SPANS:
        return "GREEN"
    if usefulness >= 2:
        return "YELLOW"
    return "RED"

#def run_ptpc_on_text(paper_text: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = build_messages(paper_text)
    raw = llm_chat(messages)                     # ← uses adapter from llm_adapter.py
    try:
        ptpc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"PTPC JSON parse failed: {e}\nRaw head: {raw[:400]}")
    errors = validate_triage_json(ptpc)
    if errors:
        raise ValueError("PTPC validation failed:\n - " + "\n - ".join(errors))
    (out_dir / "ptpc.json").write_text(json.dumps(ptpc, indent=2), encoding="utf-8")
    return ptpc

#def process_pdf(pdf_path: str, out_root: Path) -> None:
    text, page_count = extract_text_from_pdf(pdf_path)

    # PTPC gate right after text extraction
    doc_id = Path(pdf_path).stem
    out_dir = out_root / doc_id
    try:
        ptpc = run_ptpc_on_text(text, out_dir)
    except Exception as e:
        (out_dir / "status.txt").write_text(f"PTPC ERROR: {e}", encoding="utf-8")
        return

    gate = ptpc_gate(ptpc)
    (out_dir / "status.txt").write_text(f"PTPC={gate}", encoding="utf-8")
    if gate != "GREEN":
        return  # stop here for YELLOW/RED

    # If GREEN, continue with your existing Q&A generation…
    # qna = generate_qna_from_text(text, ptpc)  # pass ptpc into your Q&A builder
    # (out_dir / "qna.json").write_text(json.dumps(qna, indent=2), encoding="utf-8")


#-----------CLEANING FUNCTION----------------may need to add more cleaning here...
def clean_text(text: str) -> str: 
    text = text.replace("\t", " ")
    text = " ".join(text.split()) 
    if 1:
        print(f"[clean_text] Cleaned text preview: {text[:300]}")
    return text.strip()

def extract_and_clean(pdf_path: str) -> Tuple[str, int]:
    """
    Glue function that ensures the parameter `text` to clean_text(...)
    is exactly the extracted string from the PDF.
    """
    raw_text, num_pages = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)  
    if 1:
        print(f"[extract_and_clean] Cleaned text preview: {cleaned_text[:300]}") 
    return cleaned_text, num_pages

#-----------CHUNKING FUNCTION----------------
def chunk_text_words(
    text: str,
    max_words: int = 800,
    overlap_words: int = 50
) -> List[str]:
    """
    Split `text` into overlapping windows measured in WORDS,
    preserving word boundaries.
    """
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if not (0 <= overlap_words < max_words):
        raise ValueError("Require 0 <= overlap_words < max_words")

    words = text.split()
    step = max_words - overlap_words
    chunks: List[str] = []

    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += step
    if 1:
        print(f"[chunk_text_words] Created {len(chunks)} chunks, preview of first chunk: {chunks[0][:300]}")
    return chunks

#-----------Process and SAVE CHUNKS TO TXT----------------
def save_extracted_text(pdf_path: str, out_txt: str) -> None:
    text, _ = extract_text_from_pdf(pdf_path)  # string with \f page breaks
    with open(out_txt, "w", encoding="utf-8") as out:
        out.write(text)

def process_chunk_all(chunk: str, model: str, k_free_qas: int = 3) -> dict:
    # 1) Run the relevance→augment→MCQ→score→reasoning pipeline
    pipe_out = qps.process_chunk(chunk_text=chunk, model=model, prompts=prompts,
                                 thresholds={"relevance": 6, "mcq_min": 7})

    # 2) ALSO run your existing free-form Q/A on the augmented text
    free_qas = []
    if pipe_out.get("accepted") and pipe_out.get("augmented"):
        free_qas = generate_qas_from_chunk_together(
            chunk=pipe_out["augmented"],
            k=k_free_qas,
            model=model,
            max_tokens=800,
            temperature=0.1,
        )
    pipe_out["free_qas"] = free_qas
    return pipe_out


#---------LLM Q/A GENERATION FUNCTION----------------
def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _extract_json(text: str) -> list[dict]:
    """
    Be tolerant to models that wrap JSON in prose or ```json fences.
    Returns a Python list[dict] or raises ValueError.
    """
    if text is None:
        raise ValueError("empty model response")
    t = text.strip()

    # strip code fences if present
    fence = re.compile(r"^```(?:json)?\s*(.*?)```$", re.S | re.I)
    m = fence.match(t)
    if m:
        t = m.group(1).strip()

    # try direct JSON first
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback: find the first JSON array [...] anywhere in the string
    m = re.search(r"\[[\s\S]*?\]", t)  # non-greedy, matches across newlines
    if m:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return data

    raise ValueError("could not parse JSON from model output")

def build_messages(passage: str, k: int) -> list[dict]:
    SYSTEM = (
        "You are a careful scientific assistant. "
        "Return STRICT JSON only. No explanations."
    )
    USER = (
        "Create {k} high-quality question/answer pairs from the passage.\n"
        "Return a JSON array of objects with keys: 'question', 'answer'.\n"
        "Rules:\n"
        " - keep answers concise but specific\n"
        " - DO NOT use any metadata like authors, pages, dates, institutions from the passage\n"
        "Passage:\n---\n{passage}\n---"
    ).format(k=k, passage=passage)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

def generate_qas_from_chunk_together(
    chunk: str,
    k: int,
    model: str,
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> list[dict]:
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(chunk, k),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        stop=None,
    )
    txt = (resp.choices[0].message.content or "").strip()
    data = _extract_json(txt)

    # light validation
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            out.append({"question": q, "answer": a})
    if not out:
        raise ValueError("parsed JSON but no valid Q/A items")
    return out

#-----------FULL PIPELINE FUNCTION----------------
def generate_qa_from_pdf(pdf_path: str, model: str, out_path: str, k_free_qas: int = 3):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text_words(text, chunk_size_words=800, overlap_words=120)

    results = []
    for i, ch in enumerate(chunks):
        r = process_chunk_all(ch, model=model, k_free_qas=k_free_qas)
        r["chunk_id"] = i
        r["chunk_hash"] = _hash(ch)
        results.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

#-----------MAIN PROCESSING LOOP----------------
if __name__ == "__main__":
    import argparse, sys
    import csv
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Dry-run: extract -> clean -> chunk (no LLM).")
    ap.add_argument("path", help="PDF file OR a folder containing PDFs")
    ap.add_argument("--max_words", type=int, default=800)
    ap.add_argument("--overlap_words", type=int, default=50)
    ap.add_argument("--out_dir", default="chunks_csv", help="Folder for per-PDF CSVs")
    ap.add_argument("--save_format", choices=["txt","csv"], default="csv")
    ap.add_argument("--llm_smoke_test", action="store_true", help="Ping Together Chat API and exit")
  
    ap.add_argument("--gen_qa", action="store_true", help="Call LLM on each chunk and write Q/A JSONL")
    ap.add_argument("--qa_k", type=int, default=1, help="Q/A pairs per chunk")
    ap.add_argument("--llm_model", default=os.getenv("TOGETHER_MODEL"), help="Together model id for generation")
    ap.add_argument("--rpm", type=int, default=30, help="Rate limit (requests per minute)")
    ap.add_argument("--qa_out", default="qa_jsonl", help="Output folder for Q/A JSONL files")
  
    args = ap.parse_args()

 # ---------- INSERTED BLOCK (early exit for llm smoke test) ----------
    if args.llm_smoke_test:
        from llm_adapter import llm_smoke_test  # requires llm_adapter.py next to this file
        llm_smoke_test()
        sys.exit(0)
    # ----------------------------------------------------------------
    p = Path(args.path)
    files = []
    if p.is_file() and p.suffix.lower() == ".pdf":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.pdf"))
    else:
        sys.exit(f"Not a PDF file or folder: {p}")

    if not files:
        sys.exit(f"No PDFs found in {p}")

    for pdf in files:
        print(f"\n=== {pdf.name} ===")
        cleaned, n_pages = extract_and_clean(str(pdf))
        chunks = chunk_text_words(cleaned, max_words=args.max_words, overlap_words=args.overlap_words)
        print(f"Pages: {n_pages} | Chunks: {len(chunks)}")
        if chunks:
            print(f"First chunk preview:\n{chunks[0][:300]}\n")

        out_root = Path(args.out_dir)            # e.g., ./chunks_csv
        out_root.mkdir(parents=True, exist_ok=True)
        csv_path = out_root / f"{pdf.stem}_chunks.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "chunk_id", "text"])   # header
            for j, ch in enumerate(chunks):
                w.writerow([pdf.name, j, ch])

        print(f"Wrote CSV: {csv_path.resolve()}  (rows: {len(chunks)})")
#--------Q/A generation per PDF----------------#
if args.gen_qa:
    out_dir = Path(args.qa_out); out_dir.mkdir(parents=True, exist_ok=True)
    jl_path = out_dir / f"{pdf.stem}_qa.jsonl"
    rate_delay = 60.0 / max(args.rpm, 1)

    wrote = 0
    with open(jl_path, "w", encoding="utf-8") as f:
        for j, ch in enumerate(chunks):
            try:
                # run the combined pipeline for this chunk (both structured with MCQs/scoring; and free-form Q/A)
                res = process_chunk_all(       #res = results from process_chunk_all
                    chunk=ch,
                    model=args.llm_model,
                    k_free_qas=args.qa_k,      # how many of your original Q/As you still want
                )

                # 1) structured reasoning from qa_pipeline_skeleton
                structured = res.get("structured") or res  # depending how you stored it
                if structured and structured.get("accepted"):
                    # MCQs
                    for idx, mcq in enumerate(structured.get("mcqs", [])):
                        rec = {
                            "type": "mcq",
                            "file": pdf.name,
                            "chunk_id": j,
                            "qa_id": f"mcq-{j}-{idx}",
                            "passage_hash": _hash(ch),
                            "question": mcq["question"],
                            "options": mcq["options"],
                            "correct_index": mcq["correct_index"],
                            "difficulty": mcq.get("difficulty"),
                            "critique": mcq.get("critique"),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        wrote += 1

                    # reasoning item
                    if structured.get("reasoning"):
                        ritem = structured["reasoning"]
                        rec = {
                            "type": "reasoning",
                            "file": pdf.name,
                            "chunk_id": j,
                            "qa_id": f"reason-{j}",
                            "passage_hash": _hash(ch),
                            "question": ritem["question"],
                            "answer_key": ritem["answer_key"],
                            "rubric": ritem["rubric"],
                            "difficulty": ritem.get("difficulty"),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        wrote += 1

                # 2) free-form Q/As from original generator
                for i, qa in enumerate(res.get("free_qas", [])):
                    rec = {
                        "type": "freeform",
                        "file": pdf.name,
                        "chunk_id": j,
                        "qa_id": f"ff-{j}-{i}",
                        "passage_hash": _hash(ch),
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1

            except Exception as e:
                print(f"[gen_qa] {pdf.name} chunk {j}: {e}")

            time.sleep(rate_delay)

    print(f"Wrote Q/A JSONL: {jl_path.resolve()}  (rows: {wrote})")