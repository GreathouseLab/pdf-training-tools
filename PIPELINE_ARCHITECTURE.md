# NORE Q/A Generation Pipeline — Architecture

> **Version:** v4 (Firecrawl integration)
> **Last updated:** 2026-03-11
> **Author:** Dr. K. Leigh Greathouse
> **Purpose:** Generate short-answer Q/A training data from biomedical papers for RLHF grading at Argonne National Lab (Aurora supercomputer)

---

## Overview

The NORE (Nutrition Oncology Research Engine) pipeline is a two-stage system: **Harvest** papers, then **Train** Q/A pairs.

**v4 key change:** Firecrawl is now the default text extraction method. Papers are extracted server-side as markdown — no local PDF download required. PDF download is available as an opt-in (`--download-pdfs`) for RLHF traceability.

```
  PubMed / Europe PMC
         │
         ▼
┌─────────────────────────────┐
│  nore_paper_harvester.py    │   Stage 1: Harvest
│  ├─ Discover (MeSH queries) │
│  ├─ Screen (LLM gate)       │
│  ├─ Locate (OA PDF URL)     │
│  └─ Extract text             │
│     ├─ Firecrawl (default)   │ → .firecrawl.md + .meta.json
│     └─ PDF download (opt-in) │ → .pdf + .meta.json
└────────┬────────────────────┘
         │
         ▼
  harvested_papers/
  ├── drug_nutrient/
  ├── cachexia_sarcopenia/
  ├── immunotherapy_nutrition/
  ├── cancer_malnutrition/
  ├── dietary_patterns/
  └── microbiome_diet_cancer/
         │
         ▼
┌─────────────────────────────┐
│  mupdf_trainer_v3.py        │   Stage 2: Train
│  ├─ Text source routing      │   get_text() → auto|firecrawl|mupdf
│  ├─ Chunk text               │
│  ├─ Relevance gate           │
│  ├─ Augment chunk            │
│  ├─ Generate freeform Q/A    │
│  ├─ Verify (optional)        │
│  └─ Write JSONL output       │
└────────┬────────────────────┘
         │  per-PDF *_qa.jsonl files
         ▼
┌─────────────────────────────┐
│  duplicate_triage.py        │   Phase 1: Excel triage workbook
│  (human-in-the-loop)        │   Phase 2: Apply decisions → clean JSONL
└────────┬────────────────────┘
         │  qa_clean.jsonl
         ▼
┌─────────────────────────────┐
│  compile_qa.py              │   Final CSV export for RLHF grading
└─────────────────────────────┘
```

---

## Stage 1: Paper Harvester (`nore_paper_harvester.py`)

### Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Discover** | Query PubMed E-utilities + Europe PMC with topic-specific MeSH terms |
| 2 | **Screen** | LLM relevance gate scores abstracts (default: gpt-4.1-mini, threshold ≥ 6) |
| 3 | **Locate** | Find open-access PDF URL via PMC OA, Unpaywall, or Europe PMC |
| 4 | **Extract** | Acquire paper text via Firecrawl (default) or PDF download |

### Text Extraction Modes

#### Default: Firecrawl (no local PDFs)
- Sends the PDF URL to [Firecrawl](https://firecrawl.dev) for server-side extraction
- Returns clean markdown; saved as `<stem>.firecrawl.md`
- Metadata saved as `<stem>.firecrawl.meta.json`
- No local PDF download — faster, less storage
- Requires: `pip install firecrawl` + `FIRECRAWL_API_KEY` env var

#### Opt-in: PDF Download (`--download-pdfs`)
- Downloads the PDF binary to disk as `<stem>.pdf`
- Metadata saved as `<stem>.meta.json`
- Needed for RLHF traceability — keep original PDFs for auditing trained model outputs back to source text

### Firecrawl SDK Settings

Nick's recommended configuration for academic paper extraction:

```python
FIRECRAWL_SETTINGS = {
    "only_main_content": False,    # Full paper including methods, supplementary
    "max_age": 172800000,          # 48-hour cache (milliseconds)
    "parsers": ["pdf"],            # Force PDF-specific parser
    "formats": ["markdown"],       # Output as markdown
}
```

### Topic Taxonomy (6 priority areas)

| Key | Label | Priority |
|-----|-------|----------|
| `drug_nutrient` | Drug-Nutrient Interactions in Oncology | 1 |
| `cachexia_sarcopenia` | Cancer Cachexia & Sarcopenia | 2 |
| `immunotherapy_nutrition` | Immunotherapy & Nutrition | 3 |
| `cancer_malnutrition` | Cancer-Related Malnutrition | 4 |
| `dietary_patterns` | Dietary Patterns & Cancer Outcomes | 5 |
| `microbiome_diet_cancer` | Microbiome, Diet & Cancer | 6 |

Each topic defines 6-8 PubMed MeSH queries and 2-3 Europe PMC free-text queries.

### Output Structure

```
harvested_papers/
├── drug_nutrient/
│   ├── Curcumin_interactions_a1b2c3d4.firecrawl.md
│   ├── Curcumin_interactions_a1b2c3d4.firecrawl.meta.json
│   ├── Vitamin_D_and_chemo_e5f6g7h8.firecrawl.md
│   ├── Vitamin_D_and_chemo_e5f6g7h8.firecrawl.meta.json
│   └── .progress.json
├── cachexia_sarcopenia/
│   └── ...
└── .progress.json                    # Resume tracking
```

The `.firecrawl.md` files include an HTML comment header with DOI, PMID, title, source URL, and extraction timestamp.

---

## Stage 2: Q/A Trainer (`mupdf_trainer_v3.py`)

### Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Text Source** | Route to Firecrawl markdown or local mupdf PDF extraction |
| 2 | **Chunk** | Split text into word-bounded chunks (default: 600 words, 100 overlap) |
| 3 | **Relevance Gate** | LLM scores chunk relevance (score < 6 → rejected) |
| 4 | **Augment** | LLM enriches accepted chunks with contextual detail |
| 5 | **Generate Q/A** | LLM generates freeform short-answer Q/A pairs per chunk |
| 6 | **Verify** | Optional verification loop with up to 2 regeneration attempts |
| 7 | **Output** | Write JSONL training pairs + master summary report |

### Text Source Routing (`--text-source`)

The `get_text()` function selects extraction method and returns a 3-tuple: `(text, pages, source_used)`.

| Flag | Behavior |
|------|----------|
| `auto` (default) | Use `.firecrawl.md` if it exists alongside the PDF path; otherwise fall back to mupdf |
| `firecrawl` | Require `.firecrawl.md`; error if missing |
| `mupdf` | Always use local PDF extraction via PyMuPDF (original v3 behavior) |

**File mapping convention:**
```
Harvester creates:  stem.firecrawl.md
Trainer discovers:  stem.firecrawl.md → creates pseudo path stem.pdf
get_text() routes:  stem.pdf → looks for stem.firecrawl.md → strips markdown → returns text
```

The `_strip_markdown()` function converts Firecrawl markdown to plain text by removing HTML comments, images, links, headers, bold/italic markers, tables, code fences, blockquotes, and list markers while preserving content.

### Recursive Directory Scanning

The trainer uses `rglob` to recurse into all topic subdirectories automatically:

```bash
# Process ALL topics at once
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# Process a single topic folder
python mupdf_trainer_v3.py ./harvested_papers/drug_nutrient --gen_qa \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json
```

### Observability

**Per-file logging:**
```
=== Processing: Curcumin_interactions_a1b2c3d4.pdf ===
  Text source: firecrawl | 12 pages | 34,521 chars
  Pages: 12 | Chunks: 8
```

**Master summary report** (`MASTER_SUMMARY.txt`) includes:
- Text source breakdown: `Text Sources: 142 firecrawl | 3 mupdf`
- Per-PDF text source in the file-by-file breakdown
- Total PDFs, questions, chunks, and timing

---

## Supporting Tools

### `firecrawl_extract.py` (standalone utility)
Batch-extracts text from already-downloaded PDFs using Firecrawl. Useful for retroactively creating `.firecrawl.md` files from existing PDF collections without re-running the harvester.

### `mine_plos_genetics.py` (Nick's reference)
Reference implementation showing Firecrawl REST API approach for PLoS Genetics. Queries PLoS API → sends article URL to Firecrawl `/scrape` → saves `.txt` with attribution. This was the model for the harvester's Firecrawl integration.

### `trainer_firecrawl_patch.py` (applied — reference only)
Patch notes describing the three changes applied to `mupdf_trainer_v3.py`:
1. `extract_from_firecrawl_md()` — read `.firecrawl.md` files
2. `get_text()` — router function for text source selection
3. `--text-source` CLI argument

---

## File Inventory

| File | Role |
|------|------|
| `nore_paper_harvester.py` | Stage 1 — discover, screen, extract/download papers |
| `mupdf_trainer_v3.py` | Stage 2 — text routing, chunking, Q/A generation, verification, JSONL output |
| `llm_adapter.py` | Universal LLM backend — Together AI + OpenAI with auto-detection |
| `prompts_qa.py` | All prompt templates (relevance, augment, freeform, seen-answers dedup) |
| `qa_pipeline_skeleton_v2.py` | Freeform-only skeleton — relevance gate + chunk augmentation |
| `verification_qa.py` | Heuristic pre-checks + LLM semantic verification with confidence scoring |
| `duplicate_triage.py` | Two-phase dedup: similarity detection → Excel workbook → human review → clean JSONL |
| `compile_qa.py` | Final CSV export from deduplicated JSONL |
| `firecrawl_extract.py` | Standalone batch Firecrawl extraction for existing PDFs |
| `mine_plos_genetics.py` | Nick's PLoS Genetics reference (read-only) |
| `trainer_firecrawl_patch.py` | Patch notes (already applied to trainer) |

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

**Dependency injection chain:**
```
skeleton calls:  llm_fn(model, messages, 0.2, 1024)
trainer wrapper: llm_chat_together(model, messages, temp, max_tokens, top_p=0.9)
adapter:         llm_chat_universal(model, messages, temp, max_tokens, top_p, backend)
```

---

## Data Flow Summary

```
PubMed + Europe PMC
  │
  ▼
nore_paper_harvester.py
  │  (Firecrawl default / PDF opt-in)
  ▼
harvested_papers/              ← .firecrawl.md or .pdf per topic folder
  │
  ▼
mupdf_trainer_v3.py            ← auto-detects text source
  │
  ▼
qa_output/                     ← per-PDF JSONL + MASTER_SUMMARY.txt
  │
  ▼
duplicate_triage.py --triage
  │
  ▼
triage_workbook.xlsx           ← human reviews in Excel
  │
  ▼
duplicate_triage.py --apply
  │
  ▼
qa_clean.jsonl                 ← deduplicated records
  │
  ▼
compile_qa.py
  │
  ▼
qa_output.csv                  ← final dataset for RLHF training (Aurora)
```

---

## CLI Reference

### Harvester
```bash
# Default: Firecrawl text extraction (no PDFs)
python nore_paper_harvester.py --email you@university.edu

# Single topic with limit
python nore_paper_harvester.py --email you@university.edu --topic drug_nutrient --max-per-topic 100

# Download PDFs locally (for RLHF traceability)
python nore_paper_harvester.py --email you@university.edu --download-pdfs

# Resume interrupted run
python nore_paper_harvester.py --email you@university.edu --resume

# Skip LLM screening
python nore_paper_harvester.py --email you@university.edu --no-screen
```

### Trainer
```bash
# Process all topics (recurses into subdirectories)
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
  --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# Force Firecrawl text only (error if .firecrawl.md missing)
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
  --text-source firecrawl --llm_model gpt-4.1-mini --qa_k 1 --rpm 20

# Force local PDF extraction (original v3 behavior)
python mupdf_trainer_v3.py ./pdfs --gen_qa \
  --text-source mupdf --llm_model gpt-4.1-mini --qa_k 1 --rpm 20

# With verification enabled
python mupdf_trainer_v3.py ./harvested_papers --gen_qa --enable-verification \
  --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# Smoke test LLM connectivity
python mupdf_trainer_v3.py --llm_smoke_test

# Summarize existing JSONL files
python mupdf_trainer_v3.py --summarize --summary_dir qa_output
```

### Deduplication
```bash
# Phase 1: Generate triage workbook
python duplicate_triage.py --triage --dir qa_output --out triage_workbook.xlsx

# Phase 2: Apply human decisions
python duplicate_triage.py --apply --workbook triage_workbook.xlsx --dir qa_output

# Quick stats
python duplicate_triage.py --stats-only --dir qa_output
```

### CSV Compilation
```bash
python compile_qa.py --input-dir qa_output --output qa_output.csv
```

---

## Key Design Decisions

### Why Firecrawl as default? (v4)
- **No local PDF dependency:** Server-side extraction eliminates PyMuPDF layout issues (columns, tables, headers/footers blending into text)
- **Better text quality:** Firecrawl's PDF parser produces structured markdown with clean section separation
- **Faster pipeline:** No download + extraction step; single API call returns text
- **PDF opt-in preserved:** `--download-pdfs` keeps original files for RLHF traceability per Nick's recommendation

### Why freeform-only? (v3)
- MCQ answers were 7-15 words — too long for exact-match RLHF grading
- Freeform produces clean 1-3 word answers gradable by exact string match
- Simplifying to freeform-only cut skeleton from 250+ to 109 lines

### Why human-in-the-loop dedup?
Previous automated dedup (Union-Find clustering) destroyed 85% of data through transitive chaining. Current `duplicate_triage.py` uses direct pairwise comparison only, with all removal decisions requiring human confirmation in Excel.

### Why seen-answers injection?
Within a single PDF, the same factual question can be generated from overlapping chunks. `build_seen_answers_block()` injects prior chunk answers into the prompt to reduce within-PDF duplicates.

---

## Environment

```bash
# Activate virtual environment
source pymupdf-venv/bin/activate

# Install dependencies
pip install pymupdf openai together python-dotenv tqdm firecrawl
pip install sentence-transformers pandas openpyxl  # for dedup stage

# Environment variables (.env)
FIRECRAWL_API_KEY=fc-...          # Required for default Firecrawl mode
NCBI_API_KEY=...                   # Recommended for faster PubMed rate limits
OPENAI_API_KEY=sk-...              # Required for screening + Q/A generation
TOGETHER_API_KEY=...               # Optional alternative LLM backend
GEMINI_API_KEY=...                 # Optional alternative LLM backend
```

---

## Quick Start

```bash
# 1. Activate environment
source pymupdf-venv/bin/activate

# 2. Install Firecrawl SDK
pip install firecrawl

# 3. Ensure .env has FIRECRAWL_API_KEY, NCBI_API_KEY, OPENAI_API_KEY

# 4. Harvest papers (Firecrawl mode — default)
python nore_paper_harvester.py --email you@university.edu

# 5. Generate Q/A training pairs (processes all topic folders)
python mupdf_trainer_v3.py ./harvested_papers --gen_qa \
    --llm_model gpt-4.1-mini --qa_k 1 --rpm 20 --pretty_json

# 6. Deduplicate
python duplicate_triage.py --triage --dir qa_output --out triage_workbook.xlsx
# ... review in Excel ...
python duplicate_triage.py --apply --workbook triage_workbook.xlsx --dir qa_output

# 7. Export final CSV
python compile_qa.py --input-dir qa_output --output qa_output.csv
```
