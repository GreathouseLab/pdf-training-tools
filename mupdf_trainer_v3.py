#==============================================================================
# NORE Q/A Generation Pipeline v3
#==============================================================================
# Updated version with integrated verification system
# 
# Key changes from v2:
# - Integrated verification_qa.py for Q/A quality assurance
# - Added verification loop with up to 2 regeneration attempts
# - Tracks verification statistics (pass/fail/review/discard)
# - Outputs verification metadata with each Q/A pair
#
# Pipeline: PDF â†’ Extract â†’ Chunk â†’ Generate Q/A â†’ Verify â†’ Output
#
# Usage:
## Basic usage (verification disabled)
# python mupdf_trainer_v3.py ./pdfs --gen_qa --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
# With verification enabled
# python mupdf_trainer_v3.py ./pdfs --gen_qa --enable-verification --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --pretty_json
#==============================================================================
# SETUP INSTRUCTIONS
#==============================================================================
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
# python mupdf_trainer_v3.py ./pdfs --gen_qa --llm_model "moonshotai/Kimi-K2-Instruct-0905" --qa_k 1 --rpm 20 --pretty_json
# (where ./pdfs is a folder of PDFs to process)
# togerher model = OpenAI/gpt-oss-20B or meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

#-----------IMPORTS----------------
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
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

# Universal LLM adapter — supports Together AI + OpenAI backends (2026-02-11)
from llm_adapter import llm_chat_universal, detect_backend, llm_smoke_test

# Import the v2 skeleton
import qa_pipeline_skeleton_v2 as qps
# Prompts file
import prompts_qa as prompts
# NEW: Verification module
import verification_qa as verify        

# === PTPC: imports (unchanged) === #not using currently
try:
    from onc_nutri_triage_prompt import build_messages, validate_triage_json
except ImportError:
    logging.warning("onc_nutri_triage_prompt not found - PTPC triage features unavailable")
    build_messages = None
    validate_triage_json = None

#-----------SETUP----------------
load_dotenv()  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING) 
logging.getLogger("pinecone").setLevel(logging.WARNING)


#-----------LLM ADAPTER----------------
# Uses universal backend from llm_adapter.py (2026-02-11)
# Backend is auto-detected from model name:
#   "gpt-4.1-mini"                       -> openai
#   "moonshotai/Kimi-K2-Instruct-0905"   -> together
# Or forced via --llm_backend CLI flag.

_active_backend = None  # Set in run_pipeline() from args

def llm_chat_together(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 0.9
) -> str:
    """Drop-in wrapper that routes through the universal adapter.
    Keeps the same function name so skeleton injection still works."""
    return llm_chat_universal(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        backend=_active_backend,
    )


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
def build_messages_freeform(passage: str, k: int, seen_answers: list[str] | None = None) -> list[dict]:
    """Build freeform Q/A messages using centralized prompts.
    UPDATED 2026-02-11: Added seen_answers injection to reduce within-PDF duplicates."""
    seen_block = prompts.build_seen_answers_block(seen_answers or [])
    user_content = prompts.FREEFORM_USER.format(
        k=k,
        passage=passage,
        no_source_meta=prompts.NO_SOURCE_META,
        strict_json=prompts.STRICT_JSON_INSTRUCTIONS,
        seen_answers_block=seen_block,
    )
    return [
        {"role": "system", "content": prompts.FREEFORM_SYSTEM},
        {"role": "user", "content": user_content},
    ]

def generate_qas_from_chunk_together(
    chunk: str,
    k: int,
    model: str,
    max_tokens: int = 800,
    temperature: float = 0.1,
    seen_answers: list[str] | None = None,
) -> list[dict]:
    
    # This function calls the LLM adapter *directly*
    txt = llm_chat_together(
        model=model,
        messages=build_messages_freeform(chunk, k, seen_answers=seen_answers),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # It uses the list-specific JSON parser
    data = _extract_json_generic(txt)

    # light validation + answer length enforcement (Nick grading guidance 2026-02-10)
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            word_count = len(a.split())
            if word_count > 3:
                logging.warning(f"Freeform answer too long ({word_count} words), skipping: '{a[:60]}...'")
                continue  # Skip answers that are too long for exact-match grading
            out.append({"question": q, "answer": a})

    return out


#-----------VERIFICATION FUNCTIONS (NEW)----------------

def verify_mcq(
    source_chunk: str,
    mcq: Dict[str, Any],
    model: str,
    attempt_number: int = 1
) -> Dict[str, Any]:
    """
    Verify a single MCQ against the source chunk.
    
    Returns verification result with status, confidence, and recommendation.
    """
    # Stage 1: Fast heuristic pre-check
    pre_check = verify.pre_verify_mcq(
        question=mcq.get("question", ""),
        options=mcq.get("options", []),
        correct_answer=mcq.get("options", [""])[mcq.get("correct_index", 0)] if mcq.get("options") else ""
    )
    
    if not pre_check["passes_heuristics"]:
        return {
            "verification_status": "fail",
            "confidence_score": 0.3,
            "pre_check_failed": True,
            "pre_check_issues": pre_check["issues"],
            "safety_terms_found": pre_check.get("safety_terms", {}),
            "recommendation": "regenerate" if attempt_number < verify.MAX_REGENERATION_ATTEMPTS else "discard",
            "attempt_number": attempt_number
        }
    
    # Stage 2: LLM semantic verification
    try:
        messages = verify.build_mcq_verification_messages(
            source_chunk=source_chunk,
            question=mcq.get("question", ""),
            options=mcq.get("options", []),
            correct_index=mcq.get("correct_index", 0)
        )
        
        response_text = llm_chat_together(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        
        llm_result = _extract_json_generic(response_text)
        result = verify.parse_mcq_verification_result(llm_result, attempt_number)
        result["pre_check_passed"] = True
        result["safety_terms_found"] = pre_check.get("safety_terms", {})
        result["requires_enhanced_verification"] = pre_check.get("requires_enhanced_verification", False)
        
        return result
        
    except Exception as e:
        logging.warning(f"MCQ verification LLM call failed: {e}")
        return {
            "verification_status": "flag_for_review",
            "confidence_score": 0.5,
            "error": str(e),
            "recommendation": "accept",  # Accept for human review on error
            "attempt_number": attempt_number
        }


def verify_reasoning(
    source_chunk: str,
    reasoning: Dict[str, Any],
    model: str,
    attempt_number: int = 1
) -> Dict[str, Any]:
    """
    Verify a reasoning question against the source chunk.
    
    Returns verification result with status, confidence, and recommendation.
    """
    # Stage 1: Fast heuristic pre-check
    pre_check = verify.pre_verify_reasoning(
        question=reasoning.get("question", ""),
        answer_key=reasoning.get("answer_key", ""),
        rubric=reasoning.get("rubric", [])
    )
    
    if not pre_check["passes_heuristics"]:
        return {
            "verification_status": "fail",
            "confidence_score": 0.3,
            "pre_check_failed": True,
            "pre_check_issues": pre_check["issues"],
            "safety_terms_found": pre_check.get("safety_terms", {}),
            "recommendation": "regenerate" if attempt_number < verify.MAX_REGENERATION_ATTEMPTS else "discard",
            "attempt_number": attempt_number
        }
    
    # Stage 2: LLM semantic verification
    try:
        messages = verify.build_reasoning_verification_messages(
            source_chunk=source_chunk,
            question=reasoning.get("question", ""),
            answer_key=reasoning.get("answer_key", ""),
            rubric=reasoning.get("rubric", [])
        )
        
        response_text = llm_chat_together(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        
        llm_result = _extract_json_generic(response_text)
        result = verify.parse_reasoning_verification_result(llm_result, attempt_number)
        result["pre_check_passed"] = True
        result["safety_terms_found"] = pre_check.get("safety_terms", {})
        result["requires_enhanced_verification"] = pre_check.get("requires_enhanced_verification", False)
        
        return result
        
    except Exception as e:
        logging.warning(f"Reasoning verification LLM call failed: {e}")
        return {
            "verification_status": "flag_for_review",
            "confidence_score": 0.5,
            "error": str(e),
            "recommendation": "accept",
            "attempt_number": attempt_number
        }


def verify_freeform(
    source_chunk: str,
    qa: Dict[str, Any],
    model: str,
    attempt_number: int = 1
) -> Dict[str, Any]:
    """
    Verify a freeform Q/A pair against the source chunk.
    
    Returns verification result with status, confidence, and recommendation.
    """
    # Stage 1: Fast heuristic pre-check
    pre_check = verify.pre_verify_freeform(
        question=qa.get("question", ""),
        answer=qa.get("answer", "")
    )
    
    if not pre_check["passes_heuristics"]:
        return {
            "verification_status": "fail",
            "confidence_score": 0.3,
            "pre_check_failed": True,
            "pre_check_issues": pre_check["issues"],
            "safety_terms_found": pre_check.get("safety_terms", {}),
            "recommendation": "regenerate" if attempt_number < verify.MAX_REGENERATION_ATTEMPTS else "discard",
            "attempt_number": attempt_number
        }
    
    # Stage 2: LLM semantic verification
    try:
        messages = verify.build_freeform_verification_messages(
            source_chunk=source_chunk,
            question=qa.get("question", ""),
            answer=qa.get("answer", "")
        )
        
        response_text = llm_chat_together(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1200
        )
        
        llm_result = _extract_json_generic(response_text)
        result = verify.parse_freeform_verification_result(llm_result, attempt_number)
        result["pre_check_passed"] = True
        result["safety_terms_found"] = pre_check.get("safety_terms", {})
        result["requires_enhanced_verification"] = pre_check.get("requires_enhanced_verification", False)
        
        return result
        
    except Exception as e:
        logging.warning(f"Freeform verification LLM call failed: {e}")
        return {
            "verification_status": "flag_for_review",
            "confidence_score": 0.5,
            "error": str(e),
            "recommendation": "accept",
            "attempt_number": attempt_number
        }


#-----------SUMMARY REPORT FUNCTION----------------
def write_summary_report(jl_path: Path, model: str, total_chunks: int, wrote: int,
                         mcq_count: int, reasoning_count: int, freeform_count: int,
                         elapsed_time: float = 0.0,
                         verification_stats: Optional[Dict[str, Any]] = None):
    """Create a human-readable summary for model comparison with verification stats."""
    summary_path = jl_path.parent / f"{jl_path.stem}_SUMMARY.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"Q/A Generation Summary (with Verification)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {jl_path.stem.replace('_qa', '')}\n")
        f.write(f"\nChunks Processed: {total_chunks}\n")
        f.write(f"Total Q/A Items Accepted: {wrote}\n")
        f.write(f"\nBreakdown by Type:\n")
        f.write(f"  - MCQs: {mcq_count}\n")
        f.write(f"  - Reasoning: {reasoning_count}\n")
        f.write(f"  - Free-form: {freeform_count}\n")
        f.write(f"\nAverage Q/A per Chunk: {(wrote/max(total_chunks,1)):.2f}\n")
        f.write(f"Acceptance Rate: {((mcq_count + reasoning_count)/max(total_chunks,1)*100):.1f}%\n")

        # NEW: Verification statistics
        if verification_stats:
            f.write(f"\n" + "-" * 60 + "\n")
            f.write(f"VERIFICATION STATISTICS\n")
            f.write(f"-" * 60 + "\n")
            
            total_verified = verification_stats.get("total_verified", 0)
            passed = verification_stats.get("passed", 0)
            flagged = verification_stats.get("flagged_for_review", 0)
            failed = verification_stats.get("failed", 0)
            discarded = verification_stats.get("discarded", 0)
            regenerated = verification_stats.get("regenerated", 0)
            
            f.write(f"Total Items Verified: {total_verified}\n")
            f.write(f"  - Passed (auto-accept): {passed} ({(passed/max(total_verified,1)*100):.1f}%)\n")
            f.write(f"  - Flagged for Review: {flagged} ({(flagged/max(total_verified,1)*100):.1f}%)\n")
            f.write(f"  - Failed: {failed} ({(failed/max(total_verified,1)*100):.1f}%)\n")
            f.write(f"  - Discarded: {discarded} ({(discarded/max(total_verified,1)*100):.1f}%)\n")
            f.write(f"  - Regeneration Attempts: {regenerated}\n")
            
            # Safety stats
            safety_flagged = verification_stats.get("safety_flagged", 0)
            if safety_flagged > 0:
                f.write(f"\nSafety-Critical Content Flagged: {safety_flagged}\n")
            
            # Average confidence
            avg_confidence = verification_stats.get("avg_confidence", 0)
            if avg_confidence > 0:
                f.write(f"Average Confidence Score: {avg_confidence:.3f}\n")

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
    print(f"\nâœ“ Overall summary written to: {summary_path}")
    print(f"  Total files analyzed: {len(jsonl_files)}")
    print(f"  Total questions: {total_questions}")
   

#-----------COMBINED PIPELINE FUNCTION (Orchestrator)----------------
def process_chunk_all(
    chunk: str, 
    model: str, 
    k_free_qas: int = 3,
    enable_verification: bool = True,
    seen_answers: list[str] | None = None,
) -> dict:
    """
    Runs the full 5-stage pipeline AND the free-form QA generator.
    Now includes verification with regeneration loop.
    
    Args:
        chunk: Source text chunk
        model: LLM model identifier
        k_free_qas: Number of freeform Q/A pairs to generate
        enable_verification: Whether to run verification (default True)
        seen_answers: List of answers already generated from prior chunks (dedup)
    
    Returns:
        dict with mcqs, reasoning, free_qas, and verification_stats
    """
    
    verification_stats = {
        "total_verified": 0,
        "passed": 0,
        "flagged_for_review": 0,
        "failed": 0,
        "discarded": 0,
        "regenerated": 0,
        "safety_flagged": 0,
        "confidence_scores": []
    }
    
    # Store original chunk for verification
    source_chunk = chunk
    
    # 1) Run the relevance->augment->MCQ->score->reasoning pipeline
    pipe_out = qps.process_chunk(
        chunk_text=chunk,
        model=model,
        prompts=prompts,
        thresholds={"relevance": 6, "mcq_min": 7},
        llm_fn=llm_chat_together,
        json_parser_fn=_extract_json_generic
    )
    
    # 3) Verify reasoning MCQ with regeneration loop (uses MCQ verification â€” 2026-02-10)
    verified_reasoning = None
    verified_mcqs = []
    if enable_verification and pipe_out.get("accepted") and pipe_out.get("reasoning"):
        attempt = 1
        current_pipe = pipe_out

        while attempt <= verify.MAX_REGENERATION_ATTEMPTS + 1:
            needs_regen = False

            # --- Verify MCQs from current pipeline run ---
            for mcq in current_pipe.get("mcqs", []):
                v_result = verify_mcq(source_chunk, mcq, model, attempt)
                verification_stats["total_verified"] += 1
                verification_stats["confidence_scores"].append(v_result.get("confidence_score", 0))

                if v_result.get("safety_terms_found"):
                    verification_stats["safety_flagged"] += 1

                recommendation = v_result.get("computed_recommendation", v_result.get("recommendation", "accept"))

                if recommendation == "accept":
                    status = v_result.get("computed_status", v_result.get("verification_status", "pass"))
                    if status == "pass":
                        verification_stats["passed"] += 1
                    else:
                        verification_stats["flagged_for_review"] += 1
                    mcq["verification"] = v_result
                    verified_mcqs.append(mcq)
                elif recommendation == "regenerate" and attempt <= verify.MAX_REGENERATION_ATTEMPTS:
                    verification_stats["failed"] += 1
                    needs_regen = True
                    logging.info(f"MCQ failed verification (attempt {attempt}), will regenerate...")
                else:
                    verification_stats["discarded"] += 1
                    logging.info(f"MCQ discarded after {attempt} attempts")

            # --- Verify reasoning MCQ from current pipeline run (MCQ format since 2026-02-10) ---
            current_reasoning = current_pipe.get("reasoning")
            if current_reasoning and verified_reasoning is None:
                v_result = verify_mcq(source_chunk, current_reasoning, model, attempt)
                verification_stats["total_verified"] += 1
                verification_stats["confidence_scores"].append(v_result.get("confidence_score", 0))

                if v_result.get("safety_terms_found"):
                    verification_stats["safety_flagged"] += 1

                recommendation = v_result.get("computed_recommendation", v_result.get("recommendation", "accept"))

                if recommendation == "accept":
                    status = v_result.get("computed_status", v_result.get("verification_status", "pass"))
                    if status == "pass":
                        verification_stats["passed"] += 1
                    else:
                        verification_stats["flagged_for_review"] += 1
                    current_reasoning["verification"] = v_result
                    verified_reasoning = current_reasoning
                elif recommendation == "regenerate" and attempt <= verify.MAX_REGENERATION_ATTEMPTS:
                    verification_stats["failed"] += 1
                    needs_regen = True
                    logging.info(f"Reasoning failed verification (attempt {attempt}), will regenerate...")
                else:
                    verification_stats["discarded"] += 1
                    logging.info(f"Reasoning discarded after {attempt} attempts")

            # --- Re-run pipeline if any items need regeneration ---
            if needs_regen and attempt <= verify.MAX_REGENERATION_ATTEMPTS:
                verification_stats["regenerated"] += 1
                logging.info(f"Re-running pipeline for regeneration (attempt {attempt + 1})...")
                try:
                    current_pipe = qps.process_chunk(
                        chunk_text=chunk,
                        model=model,
                        prompts=prompts,
                        thresholds={"relevance": 6, "mcq_min": 7},
                        llm_fn=llm_chat_together,
                        json_parser_fn=_extract_json_generic
                    )
                except Exception as e:
                    logging.warning(f"Pipeline regeneration failed: {e}")
                    break
                attempt += 1
            else:
                break
    else:
        # Verification disabled - accept all MCQs and reasoning
        verified_mcqs = pipe_out.get("mcqs", [])
        verified_reasoning = pipe_out.get("reasoning")

    pipe_out["mcqs"] = verified_mcqs
    pipe_out["reasoning"] = verified_reasoning

    # 3) Generate and verify free-form Q/A with regeneration
    free_qas = []
    if pipe_out.get("accepted") and pipe_out.get("augmented"):
        try:
            raw_qas = generate_qas_from_chunk_together(
                chunk=pipe_out["augmented"],
                k=k_free_qas,
                model=model,
                max_tokens=800,
                temperature=0.1,
                seen_answers=seen_answers,
            )

            if enable_verification:
                for qa in raw_qas:
                    attempt = 1
                    current_qa = qa

                    while attempt <= verify.MAX_REGENERATION_ATTEMPTS + 1:
                        v_result = verify_freeform(source_chunk, current_qa, model, attempt)
                        verification_stats["total_verified"] += 1
                        verification_stats["confidence_scores"].append(v_result.get("confidence_score", 0))

                        if v_result.get("safety_terms_found"):
                            verification_stats["safety_flagged"] += 1

                        recommendation = v_result.get("computed_recommendation", v_result.get("recommendation", "accept"))

                        if recommendation == "accept":
                            status = v_result.get("computed_status", v_result.get("verification_status", "pass"))
                            if status == "pass":
                                verification_stats["passed"] += 1
                            else:
                                verification_stats["flagged_for_review"] += 1

                            current_qa["verification"] = v_result
                            free_qas.append(current_qa)
                            break

                        elif recommendation == "regenerate" and attempt <= verify.MAX_REGENERATION_ATTEMPTS:
                            verification_stats["regenerated"] += 1
                            verification_stats["failed"] += 1
                            logging.info(f"Freeform Q/A failed verification (attempt {attempt}), regenerating...")
                            # Actually regenerate a new freeform Q/A
                            try:
                                new_qas = generate_qas_from_chunk_together(
                                    chunk=pipe_out["augmented"],
                                    k=1,
                                    model=model,
                                    max_tokens=800,
                                    temperature=0.3,  # Slightly higher for variation
                                    seen_answers=seen_answers,
                                )
                                if new_qas:
                                    current_qa = new_qas[0]
                            except Exception as e:
                                logging.warning(f"Freeform regeneration failed: {e}")
                            attempt += 1

                        else:  # discard
                            verification_stats["discarded"] += 1
                            logging.info(f"Freeform Q/A discarded after {attempt} attempts")
                            break
            else:
                free_qas = raw_qas

        except Exception as e:
            logging.warning(f"Free-form Q/A generation failed: {e}")
    
    # NOTE (2026-02-11): MCQ fallback removed — skeleton v2 is freeform-only
    # (qps.gen_mcq / qps.score_mcq no longer exist in the simplified skeleton)
    fallback_mcqs = []
    if pipe_out.get("accepted") and len(free_qas) == 0:
        logging.warning("Freeform yielded 0 valid short-answer Q/As for this chunk (no fallback available)")
    
    pipe_out["free_qas"] = free_qas
    pipe_out["fallback_mcqs"] = fallback_mcqs
    
    # Calculate average confidence
    if verification_stats["confidence_scores"]:
        verification_stats["avg_confidence"] = sum(verification_stats["confidence_scores"]) / len(verification_stats["confidence_scores"])
    else:
        verification_stats["avg_confidence"] = 0.0
    
    # Remove raw scores list to keep output clean
    del verification_stats["confidence_scores"]
    
    pipe_out["verification_stats"] = verification_stats
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
    print(f"âœ“ Master summary written to: {summary_path.name}")
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

    # Set LLM backend (2026-02-11)
    global _active_backend
    if args.llm_backend:
        _active_backend = args.llm_backend
    elif args.llm_model:
        _active_backend = detect_backend(args.llm_model)
    else:
        _active_backend = "together"
    logging.info(f"LLM backend: {_active_backend} (model: {args.llm_model})")

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
                logging.error("LLM model must be specified with --llm_model or TOGETHER_MODEL/OPENAI_MODEL env var")
                sys.exit(1)

            out_dir = Path(args.qa_out); out_dir.mkdir(parents=True, exist_ok=True)
            jl_path = out_dir / f"{pdf.stem}_qa.jsonl"
            rate_delay = 60.0 / max(args.rpm, 1)

            wrote = 0
            mcq_count = 0
            reasoning_count = 0
            freeform_count = 0
            
            # NEW: Aggregate verification statistics
            agg_verification_stats = {
                "total_verified": 0,
                "passed": 0,
                "flagged_for_review": 0,
                "failed": 0,
                "discarded": 0,
                "regenerated": 0,
                "safety_flagged": 0,
                "confidence_scores": []
            }
            
            logging.info(f"Generating Q/A for {pdf.name} (chunks: {len(chunks)})...")
            if args.enable_verification:
                logging.info("Verification ENABLED")
            else:
                logging.info("Verification DISABLED")

            # Start timing
            start_time = time.time()

            # Seen-answers buffer for within-PDF dedup (2026-02-11)
            seen_answers_for_pdf = [] if not args.no_seen_answers else None

            with open(jl_path, "w", encoding="utf-8") as f:
                for j, ch in enumerate(chunks):
                    logging.info(f"Processing chunk {j+1}/{len(chunks)}...")
                    try:
                        # run the combined pipeline for this chunk
                        res = process_chunk_all(
                            chunk=ch,
                            model=args.llm_model,
                            k_free_qas=args.qa_k,
                            enable_verification=args.enable_verification,
                            seen_answers=seen_answers_for_pdf,
                        )

                        # Check for pipeline processing errors
                        if res.get("error"):
                            logging.warning(f"Pipeline error on chunk {j}: {res['error']}")
                            continue
                        
                        # Aggregate verification stats from this chunk
                        if res.get("verification_stats"):
                            vs = res["verification_stats"]
                            agg_verification_stats["total_verified"] += vs.get("total_verified", 0)
                            agg_verification_stats["passed"] += vs.get("passed", 0)
                            agg_verification_stats["flagged_for_review"] += vs.get("flagged_for_review", 0)
                            agg_verification_stats["failed"] += vs.get("failed", 0)
                            agg_verification_stats["discarded"] += vs.get("discarded", 0)
                            agg_verification_stats["regenerated"] += vs.get("regenerated", 0)
                            agg_verification_stats["safety_flagged"] += vs.get("safety_flagged", 0)
                            if vs.get("avg_confidence", 0) > 0:
                                agg_verification_stats["confidence_scores"].append(vs["avg_confidence"])
                        
                        # 1) structured reasoning from qa_pipeline_skeleton
                        if res.get("accepted"):
                            structured = res # The 'res' is the whole dict
                            
                            # MCQs (multiple choice questions)
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
                                # NEW: Include verification data if present
                                if mcq.get("verification"):
                                    rec["verification"] = {
                                        "status": mcq["verification"].get("computed_status", mcq["verification"].get("verification_status")),
                                        "confidence": mcq["verification"].get("confidence_score"),
                                        "safety_flagged": bool(mcq["verification"].get("safety_terms_found"))
                                    }
                                if args.pretty_json:
                                    f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n---\n")
                                else:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                wrote += 1
                                mcq_count += 1

                            # reasoning item (now MCQ format â€” 2026-02-10)
                            if structured.get("reasoning"):
                                ritem = structured["reasoning"]
                                rec = {
                                    "type": "reasoning_mcq",
                                    "model": args.llm_model,
                                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                    "file": pdf.name,
                                    "chunk_id": j,
                                    "qa_id": f"reason-{j}",
                                    "passage_hash": _hash(ch),
                                    "question": ritem["question"],
                                    "options": ritem["options"],
                                    "correct_index": ritem["correct_index"],
                                    "reasoning_type": ritem.get("reasoning_type", "mechanistic"),
                                    "difficulty": ritem.get("difficulty"),
                                }
                                # NEW: Include verification data if present
                                if ritem.get("verification"):
                                    rec["verification"] = {
                                        "status": ritem["verification"].get("computed_status", ritem["verification"].get("verification_status")),
                                        "confidence": ritem["verification"].get("confidence_score"),
                                        "safety_flagged": bool(ritem["verification"].get("safety_terms_found"))
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
                            # NEW: Include verification data if present
                            if qa.get("verification"):
                                rec["verification"] = {
                                    "status": qa["verification"].get("computed_status", qa["verification"].get("verification_status")),
                                    "confidence": qa["verification"].get("confidence_score"),
                                    "safety_flagged": bool(qa["verification"].get("safety_terms_found"))
                                }
                            if args.pretty_json:
                                f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n---\n")
                            else:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            wrote += 1
                            freeform_count += 1

                            # Accumulate answer for seen-answers dedup (2026-02-11)
                            if seen_answers_for_pdf is not None:
                                seen_answers_for_pdf.append(qa["answer"])

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

            # Calculate aggregate verification confidence
            if agg_verification_stats["confidence_scores"]:
                agg_verification_stats["avg_confidence"] = sum(agg_verification_stats["confidence_scores"]) / len(agg_verification_stats["confidence_scores"])
            else:
                agg_verification_stats["avg_confidence"] = 0.0
            del agg_verification_stats["confidence_scores"]  # Clean up
            
            # Log verification summary
            if args.enable_verification:
                logging.info(f"Verification: {agg_verification_stats['passed']} passed, "
                           f"{agg_verification_stats['flagged_for_review']} flagged, "
                           f"{agg_verification_stats['discarded']} discarded")

            # Write summary report for easy model comparison
            write_summary_report(jl_path, args.llm_model, len(chunks), wrote,
                               mcq_count, reasoning_count, freeform_count, elapsed_time,
                               verification_stats=agg_verification_stats if args.enable_verification else None)

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
                'model': args.llm_model,
                'verification_stats': agg_verification_stats if args.enable_verification else None
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
    ap.add_argument("--llm_model", default=os.getenv("TOGETHER_MODEL") or os.getenv("OPENAI_MODEL"), help="LLM model id (auto-detects backend from name)")
    ap.add_argument("--rpm", type=int, default=30, help="Rate limit (requests per minute)")
    ap.add_argument("--qa_out", default="qa_jsonl", help="Output folder for Q/A JSONL files")
    ap.add_argument("--pretty_json", action="store_true", help="Output pretty-printed JSON for human readability")
    ap.add_argument("--enable-verification", action="store_true", dest="enable_verification",
                    help="Enable Q/A verification with quality checks and regeneration (default: disabled)")
    ap.add_argument("--llm_backend", choices=["openai", "together"], default=None,
                    help="Force LLM backend (default: auto-detect from model name)")
    ap.add_argument("--no-seen-answers", action="store_true", dest="no_seen_answers",
                    help="Disable seen-answers dedup injection (not recommended)")

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