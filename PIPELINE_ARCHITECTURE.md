# NORE Q/A Generation Pipeline — Architecture

> **Version:** v3 (freeform-only)
> **Last updated:** 2026-02-24
> **Author:** Dr. K. Leigh Greathouse
> **Purpose:** Generate short-answer Q/A training data from biomedical PDFs for RLHF grading at Argonne National Lab (Aurora supercomputer)

---

## Overview

The pipeline extracts text from biomedical PDFs, chunks it, generates freeform Q/A pairs via LLM, optionally verifies quality, deduplicates, and compiles to CSV for downstream RLHF training.

```
PDF files
   │
   ▼
┌──────────────────────────┐
│  mupdf_trainer_v3.py     │  Orchestrator (1,364 lines)
│  ├─ Extract text (PyMuPDF)│
│  ├─ Clean & chunk        │
│  ├─ Gate (relevance)     │──▶ qa_pipeline_skeleton_v2.py
│  ├─ Augment chunk        │──▶ qa_pipeline_skeleton_v2.py
│  ├─ Generate freeform Q/A│──▶ prompts_qa.py
│  ├─ Verify (optional)    │──▶ verification_qa.py
│  └─ Write JSONL output   │
└──────────┬───────────────┘
           │  per-PDF *_qa.jsonl files
           ▼
┌──────────────────────────┐
│  duplicate_triage.py     │  Phase 1: Excel triage workbook
│  (human-in-the-loop)     │  Phase 2: Apply decisions → clean JSONL
└──────────┬───────────────┘
           │  qa_clean.jsonl
           ▼
┌──────────────────────────┐
│  compile_qa.py           │  Final CSV export for RLHF grading
└──────────────────────────┘
```

---

## File Inventory

| File | Lines | Role |
|------|------:|------|
| `mupdf_trainer_v3.py` | 1,364 | Main orchestrator — PDF extraction, chunking, Q/A generation, verification loop, JSONL output |
| `llm_adapter.py` | 326 | Universal LLM backend — supports Together AI + OpenAI with auto-detection |
| `prompts_qa.py` | 349 | All prompt templates (relevance, augment, freeform, seen-answers dedup) |
| `qa_pipeline_skeleton_v2.py` | 109 | Freeform-only skeleton — relevance gate + chunk augmentation |
| `verification_qa.py` | 981 | Heuristic pre-checks + LLM semantic verification with confidence scoring |
| `duplicate_triage.py` | 811 | Two-phase dedup: semantic similarity → Excel workbook → human review → clean JSONL |
| `compile_qa.py` | 264 | Final CSV export from deduplicated JSONL |
| **Total** | **4,204** | |

---

## Pipeline Stages (Detail)

### Stage 1: PDF Extraction & Chunking
**File:** `mupdf_trainer_v3.py` (lines 112–179)

```
PDF → PyMuPDF → raw text → clean_text() → chunk_text_words()
```

- Extracts plain text from each PDF page using PyMuPDF
- Cleans whitespace (tabs, multiple spaces)
- Splits into overlapping word-based chunks (default: 800 words, 50-word overlap)

### Stage 2: Relevance Gate
**File:** `qa_pipeline_skeleton_v2.py` → `gate_chunk()`

The skeleton receives injected `llm_fn` and `json_parser_fn` from the trainer (dependency injection pattern). It calls the LLM to score chunk relevance on a 1-10 scale.

- Score < 6 → rejected (junk: references, copyright, metadata)
- Content type in `{references, metadata, copyright}` → rejected
- LLM call: 1 per chunk

### Stage 3: Chunk Augmentation
**File:** `qa_pipeline_skeleton_v2.py` → `augment_chunk()`

Enriches accepted chunks with contextual detail for better Q/A generation.
- LLM call: 1 per accepted chunk

### Stage 4: Freeform Q/A Generation
**File:** `mupdf_trainer_v3.py` → `generate_qas_from_chunk_together()` (lines 248–281)

Generates short-answer Q/A pairs from the augmented chunk.

- Uses `prompts_qa.FREEFORM_USER` template with 5 placeholders: `{k}`, `{passage}`, `{no_source_meta}`, `{strict_json}`, `{seen_answers_block}`
- **Seen-answers dedup** (2026-02-11): Injects prior chunk answers via `build_seen_answers_block()` to reduce within-PDF duplicates
- Answers constrained to 1-3 words for exact-match RLHF grading (longer answers are skipped)
- LLM call: 1 per accepted chunk

### Stage 5: Verification (Optional)
**File:** `verification_qa.py` + trainer wrapper functions (lines 414–473)

Two-stage verification when `--enable-verification` is set:

1. **Heuristic pre-check** (`pre_verify_freeform`): Forbidden reference patterns, safety term detection, structural checks (answer length, question mark presence)
2. **LLM semantic verification** (`build_freeform_verification_messages` → LLM → `parse_freeform_verification_result`): Factual accuracy, self-containment, clinical safety, educational value

Confidence scoring:
- >= 0.9 → auto-pass
- 0.7–0.9 → flag for review
- < 0.7 → fail → regenerate (up to 2 attempts) or discard

### Stage 6: JSONL Output
**File:** `mupdf_trainer_v3.py` → `run_pipeline()` (lines 1139–1269)

Each accepted Q/A pair is written as a JSON record:
```json
{
  "type": "freeform",
  "model": "gpt-4.1-mini",
  "timestamp": "2026-02-24 10:30:00",
  "file": "paper_name.pdf",
  "chunk_id": 5,
  "qa_id": "ff-5-0",
  "passage_hash": "a1b2c3d4e5",
  "question": "What enzyme catalyzes the reaction?",
  "answer": "IDO1",
  "verification": {
    "status": "pass",
    "confidence": 0.95,
    "safety_flagged": false
  }
}
```

Supports two output formats:
- Standard JSONL (one record per line)
- Pretty JSON with `---` separators (`--pretty_json`)

### Stage 7: Duplicate Triage (Post-processing)
**File:** `duplicate_triage.py`

Human-in-the-loop deduplication using sentence-transformers (`all-mpnet-base-v2`):

- **Phase 1 (`--triage`):** Embeds all questions + answers, finds similar pairs via `paraphrase_mining`, classifies as `true_duplicate` / `near_duplicate` / `answer_conflict`, flags safety-critical conflicts, outputs Excel workbook with dropdown decisions
- **Phase 2 (`--apply`):** Reads human decisions (keep/remove/merge/review_later) from workbook, produces clean JSONL

### Stage 8: CSV Compilation (Final Export)
**File:** `compile_qa.py`

Reads deduplicated JSONL → outputs CSV with columns: `qa_id`, `question`, `answer`, `file`, `model`, `confidence`

---

## LLM Backend Architecture

**File:** `llm_adapter.py`

```
mupdf_trainer_v3.py
   │
   │  llm_chat_together() wrapper
   │  (sets _active_backend global)
   │
   ▼
llm_adapter.py
   │
   ├─ detect_backend(model)
   │    ├─ "gpt-*", "o1*", "o3*" ... → "openai"
   │    └─ "org/model" (contains /) → "together"
   │
   └─ llm_chat_universal(model, messages, temperature, max_tokens, top_p, backend)
        ├─ backend="openai"  → OpenAI() client → chat.completions.create()
        └─ backend="together" → Together() client → chat.completions.create()
```

Singleton clients: instantiated once per session to avoid repeated auth.

**Dependency injection chain:**
```
skeleton calls:  llm_fn(model, messages, 0.2, 1024)         [4 positional args]
trainer wrapper: llm_chat_together(model, messages, temp, max_tokens, top_p=0.9)  [5 params, 5th defaults]
adapter:         llm_chat_universal(model, messages, temp, max_tokens, top_p, backend)  [6 params, 5th+6th default]
```

---

## Data Flow Summary

```
PDFs (101 papers)
  │
  ▼
qa_jsonl/                    ← per-PDF JSONL + summary files (205 files)
  │
  ▼
duplicate_triage.py --triage
  │
  ▼
triage_workbook.xlsx         ← human reviews in Excel
  │
  ▼
duplicate_triage.py --apply
  │
  ▼
qa_clean.jsonl               ← deduplicated records
  │
  ▼
compile_qa.py
  │
  ▼
qa_output.csv                ← final dataset for RLHF training (Aurora)
```

---

## CLI Reference

### Main Pipeline
```bash
# Basic generation (verification disabled)
python mupdf_trainer_v3.py ./pdfs --gen_qa \
  --llm_model gpt-4.1-mini \
  --qa_k 1 --rpm 30 --pretty_json

# With verification enabled
python mupdf_trainer_v3.py ./pdfs --gen_qa \
  --llm_model gpt-4.1-mini \
  --enable-verification --qa_k 1 --rpm 30

# Force Together AI backend
python mupdf_trainer_v3.py ./pdfs --gen_qa \
  --llm_model "moonshotai/Kimi-K2-Instruct-0905" \
  --llm_backend together

# Smoke test LLM connectivity
python mupdf_trainer_v3.py --llm_smoke_test

# Generate overall summary of existing JSONL files
python mupdf_trainer_v3.py --summarize --summary_dir qa_jsonl
```

### Deduplication
```bash
# Phase 1: Generate triage workbook
python duplicate_triage.py --triage --dir qa_jsonl --out triage_workbook.xlsx

# Phase 2: Apply human decisions
python duplicate_triage.py --apply --workbook triage_workbook.xlsx --dir qa_jsonl

# Quick stats
python duplicate_triage.py --stats-only --dir qa_jsonl
```

### CSV Compilation
```bash
python compile_qa.py --input-dir qa_jsonl --output qa_output.csv
```

---

## Key Design Decisions

### Why freeform-only? (v3 vs v2)
The v2 pipeline generated MCQs, reasoning questions, AND freeform Q/A. 
- MCQ answers were 7-15 words — too long for exact-match RLHF grading
- Reasoning was converted to MCQ format but was redundant
- Freeform already produces clean 1-3 word answers gradable by exact string match
- Simplifying to freeform-only cut skeleton complexity from 250+ lines to 109 lines

### Why human-in-the-loop dedup?
Previous automated dedup (Union-Find clustering in `duplicate_detector.py`) destroyed 85% of data through transitive chaining. The current `duplicate_triage.py` uses direct pairwise comparison only, with all removal decisions requiring human confirmation in Excel.

### Why seen-answers injection?
Within a single PDF, the same factual question can be generated from overlapping chunks. The `build_seen_answers_block()` function injects prior chunk answers into the prompt to explicitly ask the LLM to avoid repeating them.

---

## Environment

```bash
# Required packages
pip install pymupdf openai together python-dotenv tqdm sentence-transformers pandas openpyxl

# Environment variables (.env)
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
TOGETHER_MODEL=moonshotai/Kimi-K2-Instruct-0905   # optional default
OPENAI_MODEL=gpt-4.1-mini                          # optional default
```

---

## Dead Code / Technical Debt

The following code exists in `mupdf_trainer_v3.py` but is unreachable since the skeleton was simplified to freeform-only:

| Lines | Description | Risk |
|-------|-------------|------|
| 286–347 | `verify_mcq()` function | None (never called) |
| 350–411 | `verify_reasoning()` function | None (never called) |
| 757–841 | MCQ/reasoning verification loop | None (gated by `pipe_out.get("reasoning")` which is always None) |
| 1175–1202 | MCQ JSONL output block | None (iterates empty list) |
| 1205–1233 | Reasoning JSONL output block | None (checks None) |

These can be safely removed to reduce the trainer from ~1,364 to ~1,100 lines.
