"""
Compile Q/A — Merge all deduplicated freeform JSONL into a single CSV

Takes the output of the deduplication pipeline and produces one clean CSV
file ready to send to Nick (Argonne) for RLHF exact-match grading.

Workflow position:
  1. mupdf_trainer_v3.py --gen_qa        → qa_jsonl/*.jsonl (raw)
  2. duplicate_detector.py --auto-remove → qa_deduplicated/*.jsonl (clean)
  3. compile_qa.py                       → final_qa_dataset.csv (this script)

Output columns:
  id         — unique identifier (e.g. ff-0-0)
  question   — the question text
  answer     — 1-2 word exact-match answer
  source_pdf — which PDF this came from
  model      — which LLM generated it
  confidence — verification confidence (if available)

Usage:
  python compile_qa.py --dir qa_deduplicated --out final_qa_dataset.csv
  python compile_qa.py --dir qa_deduplicated --verified-only
  python compile_qa.py --dir qa_deduplicated --min-confidence 0.8

Dependencies:
  pip install pandas

Part of the NORE Q/A Generation Pipeline (final export stage)
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DIR = "qa_deduplicated"
DEFAULT_OUT = "final_qa_dataset.csv"


def parse_jsonl(file_path: Path) -> List[Dict]:
    """Parse JSONL file (standard or '---'-separated pretty format)."""
    records = []
    try:
        content = file_path.read_text(encoding='utf-8')
        blocks = content.split('---') if '---' in content else content.splitlines()

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            try:
                data = json.loads(block)
                if isinstance(data, dict) and 'question' in data and 'answer' in data:
                    records.append(data)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logging.error(f"Error reading {file_path.name}: {e}")

    return records


def compile_dataset(
    input_dir: str,
    output_csv: str = DEFAULT_OUT,
    verified_only: bool = False,
    min_confidence: float = 0.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load all JSONL files, flatten to clean rows, write single CSV.

    Args:
        input_dir: Directory with *_qa.jsonl files (typically qa_deduplicated/)
        output_csv: Output CSV path
        verified_only: If True, skip records without verification data (default: include all)
        min_confidence: Minimum verification confidence to include (0.0 = all)
        verbose: Print progress

    Returns:
        DataFrame of the compiled dataset
    """
    root = Path(input_dir)
    if not root.exists():
        logging.error(f"Directory not found: {input_dir}")
        return pd.DataFrame()

    files = sorted(root.glob("*_qa.jsonl"))
    if not files:
        logging.error(f"No *_qa.jsonl files found in {input_dir}")
        return pd.DataFrame()

    if verbose:
        print(f"Scanning {root} ...")
        print(f"Found {len(files)} JSONL files")

    # --- Load and flatten ---
    rows = []
    skipped_no_verify = 0
    skipped_low_conf = 0
    skipped_long_answer = 0

    for fp in files:
        records = parse_jsonl(fp)
        for rec in records:
            # Extract confidence
            conf = None
            v = rec.get('verification', {})
            if isinstance(v, dict):
                conf = v.get('confidence')

            # Filter: unverified (only when --verified-only is set)
            if verified_only and conf is None:
                skipped_no_verify += 1
                continue

            # Filter: low confidence
            if conf is not None and conf < min_confidence:
                skipped_low_conf += 1
                continue

            # Filter: answer too long (safety net — should have been caught by pipeline)
            answer = rec.get('answer', '').strip()
            if len(answer.split()) > 3:
                skipped_long_answer += 1
                logging.warning(f"Answer too long, skipping: '{answer[:50]}...'")
                continue

            rows.append({
                'id': rec.get('qa_id', ''),
                'question': rec.get('question', '').strip(),
                'answer': answer,
                'source_pdf': rec.get('file', fp.stem.replace('_qa', '') + '.pdf'),
                'model': rec.get('model', ''),
                'confidence': round(conf, 3) if conf is not None else '',
            })

    if not rows:
        logging.error("No records passed filters.")
        return pd.DataFrame()

    # --- Build DataFrame ---
    df = pd.DataFrame(rows)

    # Sort: by source PDF, then by ID for clean ordering
    df = df.sort_values(by=['source_pdf', 'id']).reset_index(drop=True)

    # --- Write CSV ---
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # --- Summary ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPILED Q/A DATASET")
        print(f"{'='*60}")
        print(f"Total records:  {len(df)}")
        print(f"Source PDFs:    {df['source_pdf'].nunique()}")
        if df['model'].nunique() > 0:
            print(f"Models used:    {', '.join(df['model'].unique())}")

        # Per-PDF breakdown
        print(f"\nPer source PDF:")
        for pdf, group in df.groupby('source_pdf'):
            print(f"  {pdf}: {len(group)} questions")

        # Answer length distribution
        df['_alen'] = df['answer'].apply(lambda a: len(a.split()))
        print(f"\nAnswer length distribution:")
        for wc in sorted(df['_alen'].unique()):
            count = len(df[df['_alen'] == wc])
            print(f"  {wc} word(s): {count} ({count/len(df)*100:.1f}%)")
        df.drop(columns=['_alen'], inplace=True)

        # Confidence stats
        conf_vals = pd.to_numeric(df['confidence'], errors='coerce').dropna()
        if len(conf_vals) > 0:
            print(f"\nVerification confidence:")
            print(f"  Mean:   {conf_vals.mean():.3f}")
            print(f"  Median: {conf_vals.median():.3f}")
            print(f"  Min:    {conf_vals.min():.3f}")
            print(f"  Max:    {conf_vals.max():.3f}")

        if skipped_no_verify > 0:
            print(f"\nSkipped (no verification): {skipped_no_verify}")
        if skipped_low_conf > 0:
            print(f"Skipped (low confidence):  {skipped_low_conf}")
        if skipped_long_answer > 0:
            print(f"Skipped (answer too long): {skipped_long_answer}")

        print(f"\nOutput: {out_path.resolve()}")
        print(f"{'='*60}")

        # Preview
        print(f"\nPreview (first 5 rows):")
        print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile deduplicated freeform Q/A into a single CSV for RLHF grading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output columns:
  id, question, answer, source_pdf, model, confidence

Examples:
  python compile_qa.py --dir qa_deduplicated
  python compile_qa.py --dir qa_deduplicated --verified-only
  python compile_qa.py --dir qa_deduplicated --min-confidence 0.8 --out high_conf_qa.csv

Typical workflow:
  1. python mupdf_trainer_v3.py ./pdfs --gen_qa --enable-verification
  2. python duplicate_detector.py --dir qa_jsonl --auto-remove
  3. python compile_qa.py --dir qa_deduplicated --out final_qa_dataset.csv

Part of the NORE Q/A Generation Pipeline (final export stage).
        """
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help=f"Directory containing *_qa.jsonl files (default: {DEFAULT_DIR})"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only include records with verification data (default: include all records)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum verification confidence to include (default: 0.0 = all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    compile_dataset(
        input_dir=args.dir,
        output_csv=args.out,
        verified_only=args.verified_only,
        min_confidence=args.min_confidence,
        verbose=not args.quiet
    )
