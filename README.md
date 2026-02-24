# NORE Q/A Generation Pipeline

A pipeline for generating short-answer biomedical Q/A training data from research PDFs, designed for RLHF fine-tuning on Argonne National Lab's Aurora supercomputer.

**NORE** (Nutrition-Oncology Research Engine) produces exact-match Q/A pairs (1-3 word answers) from clinical nutrition and oncology literature for training domain-specific language models.

## What It Does

```
PDFs  -->  Extract & Chunk  -->  LLM generates Q/A  -->  Verify  -->  Deduplicate  -->  CSV
```

1. **Extract** text from biomedical PDFs using PyMuPDF
2. **Chunk** into overlapping word windows (800 words, 50-word overlap)
3. **Gate** chunks by relevance (filters out references, copyright, metadata)
4. **Augment** accepted chunks with contextual enrichment
5. **Generate** freeform Q/A pairs via LLM (OpenAI or Together AI)
6. **Verify** quality through heuristic checks + LLM semantic verification (optional)
7. **Deduplicate** via human-in-the-loop Excel triage with semantic similarity
8. **Compile** to CSV for downstream RLHF training

## Pipeline Files

| File | Description |
|------|-------------|
| `mupdf_trainer_v3.py` | Main orchestrator: PDF extraction, chunking, Q/A generation, verification loop |
| `llm_adapter.py` | Universal LLM backend supporting Together AI and OpenAI with auto-detection |
| `prompts_qa.py` | Prompt templates for relevance gating, augmentation, and freeform Q/A generation |
| `qa_pipeline_skeleton_v2.py` | Lightweight skeleton: relevance gate + chunk augmentation (dependency injection) |
| `verification_qa.py` | Two-stage quality verification: heuristic pre-checks + LLM semantic validation |
| `duplicate_triage.py` | Human-in-the-loop deduplication: semantic similarity to Excel workbook for review |
| `compile_qa.py` | Final CSV export from deduplicated JSONL |

Supporting files: `llm_smoke_test.py`, `qa_analyzer.py`, `paper_qa.py`, `setup_env.py`, `run_pipeline.sh`

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers pandas openpyxl   # for dedup/triage
```

### 2. Environment Variables

Create a `.env` file (see `.env.template`):

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
```

### 3. Run the Pipeline

```bash
# Generate Q/A from PDFs (OpenAI backend, auto-detected from model name)
python mupdf_trainer_v3.py ./pdfs --gen_qa --llm_model gpt-4.1-mini --qa_k 1 --rpm 30

# Generate Q/A (Together AI backend)
python mupdf_trainer_v3.py ./pdfs --gen_qa --llm_model "moonshotai/Kimi-K2-Instruct-0905"

# With verification enabled
python mupdf_trainer_v3.py ./pdfs --gen_qa --llm_model gpt-4.1-mini --enable-verification

# Test LLM connectivity
python mupdf_trainer_v3.py --llm_smoke_test
```

### 4. Deduplicate

```bash
# Phase 1: Generate triage workbook for human review
python duplicate_triage.py --triage --dir qa_jsonl --out triage_workbook.xlsx

# Human reviews triage_workbook.xlsx in Excel (fill in decisions column)

# Phase 2: Apply human decisions
python duplicate_triage.py --apply --workbook triage_workbook.xlsx --dir qa_jsonl
```

### 5. Compile Final Dataset

```bash
python compile_qa.py --input-dir qa_jsonl --output qa_output.csv
```

## Output Format

Each Q/A record is a JSON object in JSONL format:

```json
{
  "type": "freeform",
  "model": "gpt-4.1-mini",
  "timestamp": "2026-02-24 10:30:00",
  "file": "paper_name.pdf",
  "chunk_id": 5,
  "qa_id": "ff-5-0",
  "passage_hash": "a1b2c3d4e5",
  "question": "What enzyme catalyzes tryptophan degradation in the kynurenine pathway?",
  "answer": "IDO1"
}
```

Answers are constrained to 1-3 words for exact-match RLHF grading.

## LLM Backend

The pipeline supports two LLM providers via `llm_adapter.py`:

| Backend | Model Examples | Detection |
|---------|---------------|-----------|
| **OpenAI** | `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `o1`, `o3` | Prefix match (`gpt-`, `o1`, `o3`, ...) |
| **Together AI** | `moonshotai/Kimi-K2-Instruct-0905`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | Contains `/` (org/model format) |

Backend is auto-detected from the model name, or forced with `--llm_backend openai|together`.

## Verification System

When `--enable-verification` is set, each Q/A pair goes through:

1. **Heuristic pre-check** (no LLM call): forbidden reference patterns, answer length, safety term detection
2. **LLM semantic verification**: factual accuracy vs source chunk, self-containment, clinical safety
3. **Confidence scoring**: >= 0.9 auto-pass, 0.7-0.9 flag for review, < 0.7 regenerate (up to 2 attempts)

## Key Design Decisions

- **Freeform-only** (v3): MCQ and reasoning question types were removed because their answers (7-15 words) were too long for exact-match RLHF grading. Freeform produces clean 1-3 word answers.
- **Human-in-the-loop dedup**: Automated Union-Find clustering destroyed 85% of data through transitive chaining. The current system uses direct pairwise similarity with human confirmation in Excel.
- **Seen-answers injection**: Reduces within-PDF duplicate questions by injecting prior chunk answers as exclusions into the freeform prompt.
- **Dependency injection**: The skeleton receives `llm_fn` and `json_parser_fn` as arguments, keeping it backend-agnostic.

## Project Structure

```
.
├── mupdf_trainer_v3.py          # Main pipeline
├── llm_adapter.py               # LLM backend (OpenAI + Together AI)
├── prompts_qa.py                # Prompt templates
├── qa_pipeline_skeleton_v2.py   # Relevance gate + augmentation
├── verification_qa.py           # Q/A quality verification
├── duplicate_triage.py          # Human-in-the-loop dedup
├── compile_qa.py                # CSV export
├── requirements.txt             # Python dependencies
├── .env.template                # Environment variable template
├── PIPELINE_ARCHITECTURE.md     # Detailed architecture documentation
├── pdfs/                        # Input PDFs
├── qa_jsonl/                    # Generated Q/A output (per-PDF JSONL)
└── chunks_csv/                  # Intermediate chunk CSVs
```

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Current v3 freeform-only pipeline |
| `v2-comprehensive-archive` | Previous v2 pipeline with MCQ + reasoning + freeform generation |

## Requirements

- Python 3.10+
- PyMuPDF >= 1.24.0
- OpenAI SDK >= 1.40.0 and/or Together AI SDK >= 1.2.0
- sentence-transformers (for deduplication)
- openpyxl, pandas, tqdm

## Context

This pipeline is part of the **NORE** (Nutrition-Oncology Research Engine) project at [Baylor University Greathouse Lab](https://github.com/GreathouseLab), developed in collaboration with Argonne National Lab. The generated Q/A datasets are used for RLHF fine-tuning of domain-specific language models on the Aurora supercomputer for clinical nutrition and oncology applications.
