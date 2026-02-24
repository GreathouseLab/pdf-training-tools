# Git Operations Log — Pipeline v3 Migration

> **Date:** 2026-02-24
> **Repo:** `GreathouseLab/pdf-training-tools`
> **Remote:** `git@github.com:GreathouseLab/pdf-training-tools.git`

---

## Summary

Migrated `main` from the v2 comprehensive pipeline (MCQ + reasoning + freeform) to the v3 freeform-only pipeline. The old v2 codebase was preserved in an archive branch before any changes were made to `main`.

**Net change:** 18 files changed, +3,484 lines / -3,887 lines

---

## Branch Structure (After Migration)

```
main                        ← v3 freeform-only pipeline (active)
v2-comprehensive-archive    ← v2 MCQ+reasoning+freeform pipeline (frozen snapshot)
fix/add-pymupdf-dependency  ← earlier hotfix (already merged)
```

---

## Commit History (Full)

```
bf9521e  2026-02-24  Pipeline v3: freeform-only with dual LLM backend, verification, and triage
1aa476b  2025-12-17  Update README with project descriptions
9086dc2  2025-12-17  Updated duplicate_detector and mupdf_trainer_v2 added comprehensive summary report
87b5b0e  2025-12-03  Update quiz game and mupdf trainer
2e61268  2025-11-26  update to prompts_qa.py to reduce hallucinations
a6da96a  2025-11-25  new files created for learning python and modified pdf training files with new prompts
6cb35ec  2025-11-13  feat: add model comparison features (metadata, pretty JSON, summary reports)
040beac  2025-11-13  feat: add setup_env.py for reproducible environment bootstrapping
dd21690  2025-11-11  Merge pull request #2 from GreathouseLab/fix/add-pymupdf-dependency
85ac329  2025-11-11  Fix missing pymupdf dependency in requirements.txt
8b6a896  2025-11-11  Initial commit: PDF training and QA pipeline tools
```

---

## Step-by-Step Operations Performed

### Step 1: Verify Repository State

```bash
git status
```

**Output:** On branch `main`, 1 commit behind `origin/main`. Multiple unstaged changes (deletions of old files, modifications to adapter/prompts/skeleton) and untracked new files (v3 trainer, verification, triage, compile).

```bash
git branch -a
```

**Output:** Two local branches (`main`, `fix/add-pymupdf-dependency`), two remote branches.

```bash
git log --oneline -10
```

**Output:** Confirmed commit history starting from initial commit `8b6a896` through latest `9086dc2`.

### Step 2: Fast-Forward Main to Match Remote

Main was 1 commit behind `origin/main` (a README update pushed via GitHub). Direct `git pull` failed because of unstaged changes, so we stashed first.

```bash
git stash
git pull --ff-only origin main
git stash pop
```

**Result:** Fast-forwarded from `9086dc2` to `1aa476b` (README.md added). Stash restored cleanly with no conflicts.

### Step 3: Create Archive Branch

Created a branch from the current `main` to preserve the complete v2 codebase before making any changes.

```bash
git branch v2-comprehensive-archive main
```

**Result:** Branch `v2-comprehensive-archive` created at commit `1aa476b`. This branch contains:
- `mupdf_trainer.py` (v1 trainer)
- `mupdf_trainer_v2.py` (v2 trainer with MCQ + reasoning + freeform)
- `qa_pipeline_skeleton.py` (v1 skeleton with MCQ generation/scoring)
- `qa_pipeline_skeleton_v2.py` (v2 skeleton, pre-simplification)
- `duplicate_detector.py` (automated Union-Find dedup)
- `llm_adapter.py` (Together AI only)
- `prompts_qa.py` (without seen-answers)
- `microbiome_genesis_framework.py` (HARMONY project)
- `onc_nutri_triage_prompt.py` (PTPC triage)
- `quiz_game.py`, `quiz_game_solution.py` (Python learning)
- `LEARNING_PROJECT_GUIDE.md`, `QUICKSTART.md` (documentation)
- `pdf_trainer.py` (early prototype)

### Step 4: Stage New v3 Pipeline Files

```bash
git add mupdf_trainer_v3.py llm_adapter.py prompts_qa.py qa_pipeline_skeleton_v2.py \
        verification_qa.py compile_qa.py duplicate_triage.py PIPELINE_ARCHITECTURE.md
```

**Files staged:**
| File | Status | Description |
|------|--------|-------------|
| `mupdf_trainer_v3.py` | **New** | Main orchestrator (1,364 lines) |
| `llm_adapter.py` | **Modified** | Dual-backend: Together AI + OpenAI |
| `prompts_qa.py` | **Modified** | Added seen-answers dedup, restored MCQ/reasoning templates |
| `qa_pipeline_skeleton_v2.py` | **Modified** | Simplified to gate + augment only (109 lines) |
| `verification_qa.py` | **New** | Heuristic + LLM quality verification (981 lines) |
| `compile_qa.py` | **New** | CSV export from deduplicated JSONL (264 lines) |
| `duplicate_triage.py` | **New** | Human-in-the-loop Excel triage (811 lines) |
| `PIPELINE_ARCHITECTURE.md` | **New** | Full architecture documentation |

### Step 5: Remove Old Files

```bash
git rm mupdf_trainer.py mupdf_trainer_v2.py qa_pipeline_skeleton.py \
       duplicate_detector.py quiz_game.py quiz_game_solution.py \
       LEARNING_PROJECT_GUIDE.md QUICKSTART.md \
       microbiome_genesis_framework.py onc_nutri_triage_prompt.py pdf_trainer.py
```

**Files removed from main (preserved in `v2-comprehensive-archive`):**
| File | Reason for Removal |
|------|--------------------|
| `mupdf_trainer.py` | Superseded by `mupdf_trainer_v3.py` |
| `mupdf_trainer_v2.py` | Superseded by `mupdf_trainer_v3.py` |
| `qa_pipeline_skeleton.py` | Superseded by simplified `qa_pipeline_skeleton_v2.py` |
| `duplicate_detector.py` | Replaced by `duplicate_triage.py` (human-in-the-loop) |
| `pdf_trainer.py` | Early prototype, no longer used |
| `microbiome_genesis_framework.py` | HARMONY project, separate from NORE pipeline |
| `onc_nutri_triage_prompt.py` | PTPC triage, separate from NORE pipeline |
| `quiz_game.py` | Python learning exercise |
| `quiz_game_solution.py` | Python learning exercise |
| `LEARNING_PROJECT_GUIDE.md` | Learning documentation |
| `QUICKSTART.md` | Outdated setup guide |

### Step 6: Commit

```bash
git commit -m "Pipeline v3: freeform-only with dual LLM backend, verification, and triage

Simplified pipeline from MCQ+reasoning+freeform to freeform-only per
Argonne grading requirements (1-3 word exact-match answers for RLHF).

New files:
- mupdf_trainer_v3.py: Main orchestrator with seen-answers dedup
- verification_qa.py: Heuristic + LLM quality verification with regeneration
- duplicate_triage.py: Human-in-the-loop Excel triage (replaces auto-dedup)
- compile_qa.py: Final CSV export from deduplicated JSONL
- PIPELINE_ARCHITECTURE.md: Full architecture documentation

Updated:
- llm_adapter.py: Dual-backend support (Together AI + OpenAI) with auto-detection
- prompts_qa.py: Added seen-answers dedup injection, restored MCQ/reasoning templates
- qa_pipeline_skeleton_v2.py: Simplified to gate + augment only (109 lines)

Removed (archived in v2-comprehensive-archive branch):
- mupdf_trainer.py, mupdf_trainer_v2.py, qa_pipeline_skeleton.py
- duplicate_detector.py (replaced by duplicate_triage.py)
- microbiome_genesis_framework.py, onc_nutri_triage_prompt.py, pdf_trainer.py
- quiz_game.py, quiz_game_solution.py, LEARNING_PROJECT_GUIDE.md, QUICKSTART.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Result:** Commit `bf9521e` created on `main`.

### Step 7: Push to Remote

```bash
git push origin v2-comprehensive-archive
git push origin main
```

**Result:**
- New branch `v2-comprehensive-archive` pushed to `origin`
- `main` updated from `1aa476b` to `bf9521e`

---

## Verification

```bash
git log --oneline -3
# bf9521e Pipeline v3: freeform-only with dual LLM backend, verification, and triage
# 1aa476b Update README with project descriptions
# 9086dc2 Updated duplicate_detector and mupdf_trainer_v2 ...

git branch -a
# * main
#   v2-comprehensive-archive
#   fix/add-pymupdf-dependency
#   remotes/origin/main
#   remotes/origin/v2-comprehensive-archive
#   remotes/origin/fix/add-pymupdf-dependency

git status
# On branch main — up to date with origin/main
# Untracked: .Rhistory, PDFsv2/, qa_clean.jsonl, triage_workbook.xlsx
```

---

## How to Recover Old Code

```bash
# View old v2 files
git show v2-comprehensive-archive:mupdf_trainer_v2.py

# Restore a specific old file to working directory
git checkout v2-comprehensive-archive -- mupdf_trainer_v2.py

# Switch entirely to old pipeline
git checkout v2-comprehensive-archive

# Compare old vs new
git diff v2-comprehensive-archive main -- llm_adapter.py
```
