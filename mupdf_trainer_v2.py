#Updated version of mupdf_trainer.py to reflect recent changes
#reads through a PDF, extracts text, chunks it, and for each chunk,
#asks an LLM to generate Q/A pairs, then embeds those using DeepSeek R1
#and stores them in Pinecone.
#this command runs the environment setup and the script:cd /Users/leigh_greathouse/Documents/My_Code/Python_code
#bash setup_and_run.sh

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
# python mupdf_trainer_v2.py ./pdfs --k 1 --chunk_size 800 --chunk_overlap 50
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
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import os
import argparse
import time
import json
import re
import hashlib
import sys
import csv
import logging

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import pymupdf

from together import Together 

# Import the v2 skeleton
import qa_pipeline_skeleton_v2 as qps   
# Prompts file is unchanged
import prompts_qa as prompts        

# === PTPC: imports (unchanged) === #not using curretly
from onc_nutri_triage_prompt import build_messages, validate_triage_json
# from llm_adapter import llm_chat # This 'llm_chat' is for PTPC, separate from our pipeline
from llm_adapter import llm_chat,llm_smoke_test

#-----------SETUP----------------
load_dotenv()  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING) 
logging.getLogger("pinecone").setLevel(logging.WARNING)


#-----------LLM ADAPTER----------------
def llm_chat_together(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 0.9
) -> str:
    """Unifies the LLM call signature used by the skeleton."""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p
    )
    return (resp.choices[0].message.content or "").strip()


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


#-----------CLEANING FUNCTION----------------
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

#-----------HELPER FUNCTIONS (Hashing, JSON Parsing)----------------
def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _extract_json_generic(text: str) -> Any:
    """
    Be tolerant to models that wrap JSON in prose or ```json fences.
    Finds the first valid JSON object {} or array [] in the text.
    This is the robust parser we will inject into the skeleton.
    """
    if text is None:
        raise ValueError("empty model response")
    t = text.strip()

    # strip code fences if present
    fence = re.compile(r"^```(?:json)?\s*(.*?)```$", re.S | re.I)
    m = fence.match(t)
    if m:
        t = m.group(1).strip()

    # try direct JSON first (accepts both objects and arrays)
    try:
        data = json.loads(t)
        if isinstance(data, (dict, list)):
            return data
    except Exception:
        pass

    # fallback: find the first JSON object {...} anywhere in the string
    m = re.search(r"\{[\s\S]*?\}", t)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # fallback: find the first JSON array [...] anywhere in the string
    m = re.search(r"\[[\s\S]*?\]", t)  # non-greedy, matches across newlines
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    raise ValueError("could not parse JSON (object or array) from model output")

#-----------FREE-FORM Q/A GENERATOR (Original)----------------
def build_messages_freeform(passage: str, k: int) -> list[dict]:
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
    
    # This function calls the LLM adapter *directly*
    txt = llm_chat_together(
        model=model,
        messages=build_messages_freeform(chunk, k),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # It uses the list-specific JSON parser
    data = _extract_json_generic(txt)

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

#-----------SUMMARY REPORT FUNCTION----------------
def write_summary_report(jl_path: Path, model: str, total_chunks: int, wrote: int,
                         mcq_count: int, reasoning_count: int, freeform_count: int,
                         elapsed_time: float = 0.0):
    """Create a human-readable summary for model comparison."""
    summary_path = jl_path.parent / f"{jl_path.stem}_SUMMARY.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"Q/A Generation Summary\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {jl_path.stem.replace('_qa', '')}\n")
        f.write(f"\nChunks Processed: {total_chunks}\n")
        f.write(f"Total Q/A Items: {wrote}\n")
        f.write(f"\nBreakdown by Type:\n")
        f.write(f"  - MCQs: {mcq_count}\n")
        f.write(f"  - Reasoning: {reasoning_count}\n")
        f.write(f"  - Free-form: {freeform_count}\n")
        f.write(f"\nAverage Q/A per Chunk: {(wrote/max(total_chunks,1)):.2f}\n")
        f.write(f"Acceptance Rate: {((mcq_count + reasoning_count)/max(total_chunks,1)*100):.1f}%\n")

        if elapsed_time > 0:
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60
            f.write(f"\nCompute Time: {hours:02d}:{minutes:02d}:{seconds:05.2f}\n")
            f.write(f"Time per Chunk: {(elapsed_time/max(total_chunks,1)):.2f} seconds\n")
            f.write(f"Time per Q/A: {(elapsed_time/max(wrote,1)):.2f} seconds\n")

        f.write(f"=" * 60 + "\n")

    logging.info(f"Wrote summary: {summary_path}")


def generate_overall_summary(qa_dir: str = "qa_jsonl") -> None:
    """
    Scan all *_qa.jsonl files in the specified directory and generate
    an overall summary report with statistics and compute times.
    """
    qa_path = Path(qa_dir)
    if not qa_path.exists():
        logging.error(f"Directory not found: {qa_path}")
        return

    jsonl_files = sorted(qa_path.glob("*_qa.jsonl"))
    if not jsonl_files:
        logging.error(f"No *_qa.jsonl files found in {qa_path}")
        return

    logging.info(f"Found {len(jsonl_files)} Q/A JSONL files to analyze")

    # Collect stats for each file
    file_stats = []
    total_questions = 0
    total_mcqs = 0
    total_reasoning = 0
    total_freeform = 0

    for jl_file in jsonl_files:
        stats = {
            "filename": jl_file.name,
            "pdf_name": jl_file.stem.replace("_qa", ""),
            "total": 0,
            "mcq": 0,
            "reasoning": 0,
            "freeform": 0,
            "models": set(),
            "timestamps": []
        }

        # Read the JSONL file
        with open(jl_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Handle both standard JSONL and multi-line JSON with --- separators
        if '---' in content:
            json_blocks = content.split('---')
        else:
            json_blocks = content.strip().split('\n')

        for block in json_blocks:
            block = block.strip()
            if not block:
                continue
            try:
                qa = json.loads(block)
                if not isinstance(qa, dict):
                    continue

                stats["total"] += 1
                qa_type = qa.get("type", "")

                if qa_type == "mcq":
                    stats["mcq"] += 1
                elif qa_type == "reasoning":
                    stats["reasoning"] += 1
                elif qa_type == "freeform":
                    stats["freeform"] += 1

                if "model" in qa:
                    stats["models"].add(qa["model"])
                if "timestamp" in qa:
                    stats["timestamps"].append(qa["timestamp"])

            except json.JSONDecodeError:
                continue

        # Calculate time span if timestamps available
        if stats["timestamps"]:
            try:
                timestamps_sorted = sorted(stats["timestamps"])
                start_time = time.strptime(timestamps_sorted[0], '%Y-%m-%d %H:%M:%S')
                end_time = time.strptime(timestamps_sorted[-1], '%Y-%m-%d %H:%M:%S')
                duration = time.mktime(end_time) - time.mktime(start_time)
                stats["duration"] = duration
            except:
                stats["duration"] = None
        else:
            stats["duration"] = None

        file_stats.append(stats)
        total_questions += stats["total"]
        total_mcqs += stats["mcq"]
        total_reasoning += stats["reasoning"]
        total_freeform += stats["freeform"]

    # Write overall summary report
    summary_path = qa_path / "OVERALL_SUMMARY.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL Q/A GENERATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {qa_path.resolve()}\n")
        f.write(f"\nTotal Files Analyzed: {len(jsonl_files)}\n")
        f.write(f"Total Questions Generated: {total_questions}\n")
        f.write(f"\nOverall Breakdown:\n")
        f.write(f"  - MCQs: {total_mcqs} ({(total_mcqs/max(total_questions,1)*100):.1f}%)\n")
        f.write(f"  - Reasoning: {total_reasoning} ({(total_reasoning/max(total_questions,1)*100):.1f}%)\n")
        f.write(f"  - Free-form: {total_freeform} ({(total_freeform/max(total_questions,1)*100):.1f}%)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-FILE STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        for stats in file_stats:
            f.write(f"File: {stats['filename']}\n")
            f.write(f"PDF: {stats['pdf_name']}\n")
            f.write(f"Total Questions: {stats['total']}\n")
            f.write(f"  - MCQs: {stats['mcq']}\n")
            f.write(f"  - Reasoning: {stats['reasoning']}\n")
            f.write(f"  - Free-form: {stats['freeform']}\n")

            if stats['models']:
                f.write(f"Models Used: {', '.join(sorted(stats['models']))}\n")

            if stats['duration'] is not None:
                hours = int(stats['duration'] // 3600)
                minutes = int((stats['duration'] % 3600) // 60)
                seconds = stats['duration'] % 60
                f.write(f"Compute Time: {hours:02d}:{minutes:02d}:{seconds:05.2f}\n")
                f.write(f"Time per Question: {(stats['duration']/max(stats['total'],1)):.2f} seconds\n")
            else:
                f.write(f"Compute Time: Not available (no timestamps or single timestamp)\n")

            f.write("-" * 80 + "\n\n")

        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Average Questions per File: {(total_questions/len(file_stats)):.1f}\n")

        files_with_time = [s for s in file_stats if s['duration'] is not None]
        if files_with_time:
            avg_duration = sum(s['duration'] for s in files_with_time) / len(files_with_time)
            total_compute = sum(s['duration'] for s in files_with_time)

            h = int(avg_duration // 3600)
            m = int((avg_duration % 3600) // 60)
            s = avg_duration % 60
            f.write(f"Average Compute Time per File: {h:02d}:{m:02d}:{s:05.2f}\n")

            h_total = int(total_compute // 3600)
            m_total = int((total_compute % 3600) // 60)
            s_total = total_compute % 60
            f.write(f"Total Compute Time: {h_total:02d}:{m_total:02d}:{s_total:05.2f}\n")

            avg_time_per_q = total_compute / sum(s['total'] for s in files_with_time)
            f.write(f"Average Time per Question: {avg_time_per_q:.2f} seconds\n")

        f.write("=" * 80 + "\n")

    logging.info(f"Wrote overall summary: {summary_path}")
    print(f"\n✓ Overall summary written to: {summary_path}")
    print(f"  Total files analyzed: {len(jsonl_files)}")
    print(f"  Total questions: {total_questions}")
   

#-----------COMBINED PIPELINE FUNCTION (Orchestrator)----------------
def process_chunk_all(chunk: str, model: str, k_free_qas: int = 3) -> dict:
    """
    Runs the full 5-stage pipeline AND the free-form QA generator.
    """
    
    # 1) Run the relevance→augment→MCQ→score→reasoning pipeline
    # We "inject" our real LLM function and parser here.
    pipe_out = qps.process_chunk(
        chunk_text=chunk,
        model=model,
        prompts=prompts,
        thresholds={"relevance": 6, "mcq_min": 7},
        llm_fn=llm_chat_together,
        json_parser_fn=_extract_json_generic
    )

    # 2) ALSO run your existing free-form Q/A on the augmented text
    free_qas = []
    if pipe_out.get("accepted") and pipe_out.get("augmented"):
        try:
            free_qas = generate_qas_from_chunk_together(
                chunk=pipe_out["augmented"],
                k=k_free_qas,
                model=model,
                max_tokens=800,
                temperature=0.1,
            )
        except Exception as e:
            logging.warning(f"Free-form Q/A generation failed: {e}")
            
    pipe_out["free_qas"] = free_qas
    return pipe_out


#-----------MASTER SUMMARY FUNCTION----------------
def write_master_summary(qa_dir: Path, run_stats: list[dict], run_start_time: float):
    """
    Write a master summary file combining statistics from all PDFs in this run.
    """
    if not run_stats:
        return

    run_elapsed = time.time() - run_start_time
    total_pdfs = len(run_stats)
    total_chunks = sum(s['chunks'] for s in run_stats)
    total_questions = sum(s['questions'] for s in run_stats)
    total_mcqs = sum(s['mcq'] for s in run_stats)
    total_reasoning = sum(s['reasoning'] for s in run_stats)
    total_freeform = sum(s['freeform'] for s in run_stats)
    total_time = sum(s['elapsed'] for s in run_stats)

    summary_path = qa_dir / f"MASTER_SUMMARY_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MASTER Q/A GENERATION SUMMARY - BATCH RUN\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {qa_dir.resolve()}\n")
        f.write(f"\n{'='*80}\n")
        f.write("OVERALL STATISTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total PDFs Processed: {total_pdfs}\n")
        f.write(f"Total Chunks Processed: {total_chunks}\n")
        f.write(f"Total Questions Generated: {total_questions}\n")
        f.write(f"\nQuestion Type Breakdown:\n")
        f.write(f"  - MCQs: {total_mcqs} ({(total_mcqs/max(total_questions,1)*100):.1f}%)\n")
        f.write(f"  - Reasoning: {total_reasoning} ({(total_reasoning/max(total_questions,1)*100):.1f}%)\n")
        f.write(f"  - Free-form: {total_freeform} ({(total_freeform/max(total_questions,1)*100):.1f}%)\n")

        # Timing statistics
        run_hours = int(run_elapsed // 3600)
        run_minutes = int((run_elapsed % 3600) // 60)
        run_seconds = run_elapsed % 60
        f.write(f"\n{'='*80}\n")
        f.write("TIMING STATISTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Run Time: {run_hours:02d}:{run_minutes:02d}:{run_seconds:05.2f}\n")
        f.write(f"Total Processing Time: {int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{total_time%60:05.2f}\n")
        f.write(f"Average Time per PDF: {(total_time/max(total_pdfs,1)):.2f} seconds\n")
        f.write(f"Average Time per Chunk: {(total_time/max(total_chunks,1)):.2f} seconds\n")
        f.write(f"Average Time per Question: {(total_time/max(total_questions,1)):.2f} seconds\n")

        # Per-PDF breakdown
        f.write(f"\n{'='*80}\n")
        f.write("PER-PDF BREAKDOWN\n")
        f.write(f"{'='*80}\n\n")

        for i, stats in enumerate(run_stats, 1):
            f.write(f"[{i}/{total_pdfs}] {stats['pdf_name']}\n")
            f.write(f"  Chunks: {stats['chunks']}\n")
            f.write(f"  Total Questions: {stats['questions']}\n")
            f.write(f"    - MCQs: {stats['mcq']}\n")
            f.write(f"    - Reasoning: {stats['reasoning']}\n")
            f.write(f"    - Free-form: {stats['freeform']}\n")

            hours = int(stats['elapsed'] // 3600)
            minutes = int((stats['elapsed'] % 3600) // 60)
            seconds = stats['elapsed'] % 60
            f.write(f"  Processing Time: {hours:02d}:{minutes:02d}:{seconds:05.2f}\n")
            f.write(f"  Avg Time/Question: {(stats['elapsed']/max(stats['questions'],1)):.2f} sec\n")
            f.write(f"  Output File: {stats['output_file']}\n")
            f.write(f"{'-'*80}\n")

        # Summary averages
        f.write(f"\n{'='*80}\n")
        f.write("AVERAGES\n")
        f.write(f"{'='*80}\n")
        f.write(f"Avg Chunks per PDF: {(total_chunks/max(total_pdfs,1)):.1f}\n")
        f.write(f"Avg Questions per PDF: {(total_questions/max(total_pdfs,1)):.1f}\n")
        f.write(f"Avg Questions per Chunk: {(total_questions/max(total_chunks,1)):.2f}\n")

        if run_stats[0].get('model'):
            f.write(f"\nModel Used: {run_stats[0]['model']}\n")

        f.write(f"{'='*80}\n")

    logging.info(f"Wrote master summary: {summary_path}")
    print(f"\n{'='*80}")
    print(f"✓ Master summary written to: {summary_path.name}")
    print(f"  Total PDFs: {total_pdfs} | Total Questions: {total_questions}")
    print(f"  Total Time: {run_hours:02d}:{run_minutes:02d}:{run_seconds:05.2f}")
    print(f"{'='*80}\n")


#-----------MAIN PIPELINE FUNCTION (Moved from __main__)----------------
def run_pipeline(args: argparse.Namespace):
    """
    This function contains the main application logic.
    """
    p = Path(args.path)
    files = []
    if p.is_file() and p.suffix.lower() == ".pdf":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.pdf"))
    else:
        logging.error(f"Not a PDF file or folder: {p}")
        sys.exit(1)

    if not files:
        logging.error(f"No PDFs found in {p}")
        sys.exit(1)

    # Track statistics for master summary
    run_stats = []
    run_start_time = time.time()

    for pdf in files:
        logging.info(f"\n=== Processing: {pdf.name} ===")
        try:
            cleaned, n_pages = extract_and_clean(str(pdf))
        except Exception as e:
            logging.error(f"Failed to extract/clean {pdf.name}: {e}")
            continue # Skip to next PDF

        chunks = chunk_text_words(cleaned, max_words=args.max_words, overlap_words=args.overlap_words)
        logging.info(f"Pages: {n_pages} | Chunks: {len(chunks)}")
        if chunks:
            logging.info(f"First chunk preview:\n{chunks[0][:300]}\n")

        # --- CSV Chunk Output (Unchanged) ---
        if args.save_format == "csv":
            out_root = Path(args.out_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            csv_path = out_root / f"{pdf.stem}_chunks.csv"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "chunk_id", "text"])   # header
                for j, ch in enumerate(chunks):
                    w.writerow([pdf.name, j, ch])
            logging.info(f"Wrote CSV: {csv_path.resolve()}  (rows: {len(chunks)})")
        
        # --- TXT Chunk Output (Unchanged) ---
        if args.save_format == "txt":
             # This seems to be missing from the original logic, but I'm adding
             # based on the --save_format flag.
            out_root = Path(args.out_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            txt_path = out_root / f"{pdf.stem}_chunks.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for j, ch in enumerate(chunks):
                    f.write(f"--- CHUNK {j} ---\n{ch}\n\n")
            logging.info(f"Wrote TXT: {txt_path.resolve()} (chunks: {len(chunks)})")


        #--------Q/A generation per PDF----------------#
        if args.gen_qa:
            if not args.llm_model:
                logging.error("LLM model must be specified with --llm_model or TOGETHER_MODEL env var")
                sys.exit(1)

            out_dir = Path(args.qa_out); out_dir.mkdir(parents=True, exist_ok=True)
            jl_path = out_dir / f"{pdf.stem}_qa.jsonl"
            rate_delay = 60.0 / max(args.rpm, 1)

            wrote = 0
            mcq_count = 0
            reasoning_count = 0
            freeform_count = 0
            logging.info(f"Generating Q/A for {pdf.name} (chunks: {len(chunks)})...")

            # Start timing
            start_time = time.time()

            with open(jl_path, "w", encoding="utf-8") as f:
                for j, ch in enumerate(chunks):
                    logging.info(f"Processing chunk {j+1}/{len(chunks)}...")
                    try:
                        # run the combined pipeline for this chunk
                        res = process_chunk_all(
                            chunk=ch,
                            model=args.llm_model,
                            k_free_qas=args.qa_k,
                        )

                        # Check for pipeline processing errors
                        if res.get("error"):
                            logging.warning(f"Pipeline error on chunk {j}: {res['error']}")
                            continue
                        
                        # 1) structured reasoning from qa_pipeline_skeleton
                        if res.get("accepted"):
                            structured = res # The 'res' is the whole dict
                            
                            # MCQs (mutliple choice questions)
                            for idx, mcq in enumerate(structured.get("mcqs", [])):
                                rec = {
                                    "type": "mcq",
                                    "model": args.llm_model,
                                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
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
                                if args.pretty_json:
                                    f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n---\n")
                                else:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                wrote += 1
                                mcq_count += 1

                            # reasoning item
                            if structured.get("reasoning"):
                                ritem = structured["reasoning"]
                                rec = {
                                    "type": "reasoning",
                                    "model": args.llm_model,
                                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                    "file": pdf.name,
                                    "chunk_id": j,
                                    "qa_id": f"reason-{j}",
                                    "passage_hash": _hash(ch),
                                    "question": ritem["question"],
                                    "answer_key": ritem["answer_key"],
                                    "rubric": ritem["rubric"],
                                    "difficulty": ritem.get("difficulty"),
                                }
                                if args.pretty_json:
                                    f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n---\n")
                                else:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                wrote += 1
                                reasoning_count += 1

                        # 2) free-form Q/As from original generator
                        for i, qa in enumerate(res.get("free_qas", [])):
                            rec = {
                                "type": "freeform",
                                "model": args.llm_model,
                                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "file": pdf.name,
                                "chunk_id": j,
                                "qa_id": f"ff-{j}-{i}",
                                "passage_hash": _hash(ch),
                                "question": qa["question"],
                                "answer": qa["answer"],
                            }
                            if args.pretty_json:
                                f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n---\n")
                            else:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            wrote += 1
                            freeform_count += 1

                    except Exception as e:
                        logging.error(f"[gen_qa] Unhandled exception on {pdf.name} chunk {j}: {e}")

                    time.sleep(rate_delay)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60

            logging.info(f"Wrote Q/A JSONL: {jl_path.resolve()}  (rows: {wrote})")
            logging.info(f"Compute time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")

            # Write summary report for easy model comparison
            write_summary_report(jl_path, args.llm_model, len(chunks), wrote,
                               mcq_count, reasoning_count, freeform_count, elapsed_time)

            # Collect statistics for master summary
            run_stats.append({
                'pdf_name': pdf.name,
                'chunks': len(chunks),
                'questions': wrote,
                'mcq': mcq_count,
                'reasoning': reasoning_count,
                'freeform': freeform_count,
                'elapsed': elapsed_time,
                'output_file': jl_path.name,
                'model': args.llm_model
            })

    # Write master summary if Q/A generation was performed
    if args.gen_qa and run_stats:
        out_dir = Path(args.qa_out)
        write_master_summary(out_dir, run_stats, run_start_time)

#-----------MAIN ENTRY POINT----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract, chunk, and generate Q/A from PDFs.")
    ap.add_argument("path", nargs='?', help="PDF file OR a folder containing PDFs")
    ap.add_argument("--max_words", type=int, default=800)
    ap.add_argument("--overlap_words", type=int, default=50)
    ap.add_argument("--out_dir", default="chunks_csv", help="Folder for per-PDF CSVs/TXTs")
    ap.add_argument("--save_format", choices=["txt","csv"], default="csv")
    ap.add_argument("--llm_smoke_test", action="store_true", help="Ping Together Chat API and exit")

    ap.add_argument("--gen_qa", action="store_true", help="Call LLM on each chunk and write Q/A JSONL")
    ap.add_argument("--qa_k", type=int, default=1, help="Free-form Q/A pairs per chunk")
    ap.add_argument("--llm_model", default=os.getenv("TOGETHER_MODEL"), help="Together model id for generation")
    ap.add_argument("--rpm", type=int, default=30, help="Rate limit (requests per minute)")
    ap.add_argument("--qa_out", default="qa_jsonl", help="Output folder for Q/A JSONL files")
    ap.add_argument("--pretty_json", action="store_true", help="Output pretty-printed JSON for human readability")

    ap.add_argument("--summarize", action="store_true", help="Generate overall summary of all *_qa.jsonl files and exit")
    ap.add_argument("--summary_dir", default="qa_jsonl", help="Directory containing *_qa.jsonl files for summary")

    args = ap.parse_args()

    # ---------- Early exit for overall summary ----------
    if args.summarize:
        logging.info(f"Generating overall summary from {args.summary_dir}...")
        generate_overall_summary(args.summary_dir)
        sys.exit(0)

    # ---------- Early exit for llm smoke test ----------
    if args.llm_smoke_test:
        logging.info("Running LLM smoke test...")
        # Note: This 'llm_smoke_test' is from the separate 'llm_adapter.py'
        # used by your PTPC system.
        llm_smoke_test()
        logging.info("Smoke test complete.")
        sys.exit(0)

    # Ensure path is provided for normal operation
    if not args.path:
        ap.error("the following arguments are required: path (unless using --summarize or --llm_smoke_test)")

    # Call the main pipeline function with the parsed args
    run_pipeline(args)