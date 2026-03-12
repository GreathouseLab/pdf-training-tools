# NORE Q/A Generation Pipeline

A two-stage pipeline for harvesting biomedical research papers and generating short-answer Q/A training data, designed for RLHF fine-tuning on Argonne National Lab's Aurora supercomputer.

**NORE** (Nutrition-Oncology Research Engine) produces exact-match Q/A pairs (1-3 word answers) from clinical nutrition and oncology literature for training domain-specific language models.

## What It Does

```
PubMed / Europe PMC
    │
    ▼
┌─ Harvest ──────────────────────────┐
│  Discover → Screen → Locate →      │
│  Extract (Firecrawl default)        │
└────────┬───────────────────────────┘
         │  .firecrawl.md + .meta.json
         ▼
┌─ Train ────────────────────────────┐
│  Text Route → Chunk → Gate →       │
│  Generate Q/A → Verify → Output    │
└────────┬───────────────────────────┘
         │  JSONL
         ▼
  Deduplicate → Compile CSV → RLHF Training (Aurora)
```

### Stage 1: Harvest (`nore_paper_harvester.py`)

1. **Discover** candidate papers via PubMed E-utilities + Europe PMC using topic-specific MeSH queries
2. **Screen** abstracts through an LLM relevance gate (score >= 6/10 to pass)
3. **Locate** open-access PDF URLs via PMC OA, Unpaywall, or Europe PMC
4. **Extract** full text via Firecrawl server-side PDF parsing (default), or download PDFs locally (`--download-pdfs` for RLHF traceability)

### Stage 2: Train (`mupdf_trainer_v3.py`)

5. **Route** text source — auto-detects `.firecrawl.md` or falls back to local PDF extraction via PyMuPDF
6. **Chunk** into overlapping word windows (600 words, 100-word overlap)
7. **Gate** chunks by relevance (filters out references, copyright, metadata)
8. **Augment** accepted chunks with contextual enrichment
9. **Generate** freeform Q/A pairs via LLM (OpenAI or Together AI)
10. **Verify** quality through heuristic checks + LLM semantic verification (optional)
11. **Deduplicate** via human-in-the-loop Excel triage with semantic similarity
12. **Compile** to CSV for downstream RLHF training

## Pipeline Files

| File | Description |
|------|-------------|
| `nore_paper_harvester.py` | Stage 1: discover, screen, and extract papers via Firecrawl or PDF download |
| `mupdf_trainer_v3.py` | Stage 2: text routing, chunking, Q/A generation, verification loop |
| `llm_adapter.py` | Universal LLM backend supporting Together AI and OpenAI with auto-detection |
| `prompts_qa.py` | Prompt templates for relevance gating, augmentation, and freeform Q/A generation |
| `qa_pipeline_skeleton_v2.py` | Lightweight skeleton: relevance gate + chunk augmentation (dependency injection) |
| `verification_qa.py` | Two-stage quality verification: heuristic pre-checks + LLM semantic validation |
| `duplicate_triage.py` | Human-in-the-loop deduplication: semantic similarity to Excel workbook for review |
| `compile_qa.py` | Final CSV export from deduplicated JSONL |
| `firecrawl_extract.py` | Standalone batch tool: retroactively extract text from existing PDFs via Firecrawl |
| `mine_plos_genetics.py` | Reference: Nick's Firecrawl example for PLoS Genetics |

Supporting files: `llm_smoke_test.py`, `qa_analyzer.py`, `paper_qa.py`, `setup_env.py`, `run_pipeline.sh`

## Quick Start

### 1. Setup

```bash
python -m venv pymupdf-venv
source pymupdf-venv/bin/activate
pip install -r requirements.txt
pip install firecrawl                                # Firecrawl SDK (default extraction)
pip install sentence-transformers pandas openpyxl     # for dedup/triage stage
```

### 2. Environment Variables

Create a `.env` file:

```
FIRECRAWL_API_KEY=fc-...        # Required for default Firecrawl mode
NCBI_API_KEY=...                 # Recommended for faster PubMed rate limits
OPENAI_API_KEY=sk-...            # Required for screening + Q/A generation
TOGETHER_API_KEY=...             # Optional alternative LLM backend
```

### 3. Harvest Papers

```bash
# Default: Firecrawl text extraction (no PDFs downloaded)
python nore_paper_harvester.py --email you@university.edu

# Single topic with limit
python nore_paper_harvester.py --email you@university.edu --topic drug_nutrient --max-per-topic 100

# Download PDFs locally instead (for RLHF traceability)
python nore_paper_harvester.py --email you@university.edu --download-pdfs

# Resume interrupted run
python nore_paper_harvester.py --email you@university.edu --resume
```

### 4. Generate Q/A Training Data

```bash
# Process ALL topics at once (recurses into subdirectories)
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# Process a single topic
python mupdf_trainer_v3.py ./harvested_papers/drug_nutrient --gen_qa \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# Force Firecrawl text only (error if .firecrawl.md missing)
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
    --text-source firecrawl --llm_model gpt-4.1-mini --qa_k 1 --rpm 20

# Force local PDF extraction (original v3 behavior)
python mupdf_trainer_v3.py ./pdfs --gen_qa \
    --text-source mupdf --llm_model gpt-4.1-mini --qa_k 1 --rpm 20

# With verification enabled
python mupdf_trainer_v3.py ./harvested_papers --gen_qa --enable-verification \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20

# Test LLM connectivity
python mupdf_trainer_v3.py --llm_smoke_test
```

### 5. Deduplicate

```bash
# Phase 1: Generate triage workbook for human review
python duplicate_triage.py --triage --dir qa_jsonl --out triage_workbook.xlsx

# Human reviews triage_workbook.xlsx in Excel (fill in decisions column)

# Phase 2: Apply human decisions
python duplicate_triage.py --apply --workbook triage_workbook.xlsx --dir qa_jsonl
```

### 6. Compile Final Dataset

```bash
python compile_qa.py --input-dir qa_jsonl --output qa_output.csv
```

## Text Source Routing

The trainer's `--text-source` flag controls how text is extracted:

| Flag | Behavior |
|------|----------|
| `auto` (default) | Use `.firecrawl.md` if it exists, fall back to mupdf |
| `firecrawl` | Require `.firecrawl.md`, error if missing |
| `mupdf` | Always use local PDF extraction via PyMuPDF |

The master summary report tracks which source was used per file:
```
Text Source Breakdown:
  - Firecrawl: 142
  - mupdf (local PDF): 3
```

## Topic Taxonomy

The harvester searches 6 priority areas in nutrition oncology:

| Topic | Description |
|-------|-------------|
| `drug_nutrient` | Drug-Nutrient Interactions in Oncology |
| `cachexia_sarcopenia` | Cancer Cachexia & Sarcopenia |
| `immunotherapy_nutrition` | Immunotherapy & Nutrition |
| `cancer_malnutrition` | Cancer-Related Malnutrition |
| `dietary_patterns` | Dietary Patterns & Cancer Outcomes |
| `microbiome_diet_cancer` | Microbiome, Diet & Cancer |

Each topic defines 6-8 PubMed MeSH queries and 2-3 Europe PMC free-text queries.

## Output Format

Each Q/A record is a JSON object in JSONL format:

```json
{
  "type": "freeform",
  "model": "gpt-4.1-mini",
  "timestamp": "2026-03-12 10:54:37",
  "file": "Curcumin_interactions_a1b2c3d4.pdf",
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

- **Firecrawl as default** (v4): Server-side PDF extraction produces cleaner text than local PyMuPDF (no column blending, header/footer artifacts). PDF download preserved as `--download-pdfs` for RLHF traceability.
- **Freeform-only** (v3): MCQ and reasoning question types were removed because their answers (7-15 words) were too long for exact-match RLHF grading. Freeform produces clean 1-3 word answers.
- **Human-in-the-loop dedup**: Automated Union-Find clustering destroyed 85% of data through transitive chaining. The current system uses direct pairwise similarity with human confirmation in Excel.
- **Seen-answers injection**: Reduces within-PDF duplicate questions by injecting prior chunk answers as exclusions into the freeform prompt.
- **Dependency injection**: The skeleton receives `llm_fn` and `json_parser_fn` as arguments, keeping it backend-agnostic.

## Project Structure

```
.
├── nore_paper_harvester.py      # Stage 1: Paper harvesting
├── mupdf_trainer_v3.py          # Stage 2: Q/A generation pipeline
├── llm_adapter.py               # LLM backend (OpenAI + Together AI)
├── prompts_qa.py                # Prompt templates
├── qa_pipeline_skeleton_v2.py   # Relevance gate + augmentation
├── verification_qa.py           # Q/A quality verification
├── duplicate_triage.py          # Human-in-the-loop dedup
├── compile_qa.py                # CSV export
├── firecrawl_extract.py         # Standalone Firecrawl batch extraction
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not committed)
├── PIPELINE_ARCHITECTURE.md     # Detailed architecture documentation
├── harvested_papers/            # Harvested data (per-topic subdirectories)
│   ├── drug_nutrient/
│   ├── cachexia_sarcopenia/
│   ├── immunotherapy_nutrition/
│   ├── cancer_malnutrition/
│   ├── dietary_patterns/
│   └── microbiome_diet_cancer/
├── qa_jsonl/                    # Generated Q/A output (per-PDF JSONL)
└── chunks_csv/                  # Intermediate chunk CSVs
```

## Branches

| Branch | Description |
|--------|-------------|
| `main` | v3 freeform-only pipeline |
| `feature/firecrawl-integration` | v4 with Firecrawl harvesting + text source routing |
| `v2-comprehensive-archive` | Previous v2 pipeline with MCQ + reasoning + freeform generation |

## Requirements

- Python 3.10+
- PyMuPDF >= 1.24.0
- Firecrawl SDK (`pip install firecrawl`)
- OpenAI SDK >= 1.40.0 and/or Together AI SDK >= 1.2.0
- sentence-transformers (for deduplication)
- openpyxl, pandas, tqdm

## Context

This pipeline is part of the **NORE** (Nutrition-Oncology Research Engine) project at [Baylor University Greathouse Lab](https://github.com/GreathouseLab), developed in collaboration with Argonne National Lab. The generated Q/A datasets are used for RLHF fine-tuning of domain-specific language models on the Aurora supercomputer for clinical nutrition and oncology applications.
