#!/usr/bin/env python3
"""
NORE Firecrawl Text Extractor
==============================
Standalone script that reads .meta.json sidecar files from the harvester,
sends PDF URLs to Firecrawl for text extraction, and saves markdown output
alongside each PDF as .firecrawl.md files.

Architecture:
  harvested_papers/
    topic_folder/
      paper_abc123.pdf          ← downloaded by harvester
      paper_abc123.meta.json    ← metadata with source_pdf_url
      paper_abc123.firecrawl.md ← OUTPUT: Firecrawl markdown text

Usage:
  # First install the Firecrawl SDK
  pip install firecrawl

  # Set the API key (or edit FIRECRAWL_API_KEY below)
  export FIRECRAWL_API_KEY=your_key_here

  # Run extraction on harvested papers
  python firecrawl_extract.py ./harvested_papers

  # Resume after interruption
  python firecrawl_extract.py ./harvested_papers --resume

  # Process a single topic folder
  python firecrawl_extract.py ./harvested_papers/drug_nutrient

  # Verbose output with custom batch delay
  python firecrawl_extract.py ./harvested_papers --delay 1.5 -v

After extraction, run the trainer with Firecrawl text:
  python mupdf_trainer_v3.py ./harvested_papers/drug_nutrient --gen_qa --text-source auto

Dependencies:
  pip install firecrawl
"""

import os
import sys
import json
import time
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Firecrawl SDK — Nick's recommended approach
# ─────────────────────────────────────────────────────────────
try:
    from firecrawl import Firecrawl
except ImportError:
    print("ERROR: Firecrawl SDK not installed.")
    print("Run: pip install firecrawl")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# API Key — set via environment variable or edit directly here
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")

# Nick's recommended SDK settings
FIRECRAWL_SETTINGS = {
    "only_main_content": False,    # Full paper including methods, supplementary
    "max_age": 172800000,          # 48 hours cache (milliseconds)
    "parsers": ["pdf"],            # Force PDF-specific parser
    "formats": ["markdown"],       # Output as markdown
}

# Rate limiting
DEFAULT_DELAY = 1.0  # seconds between requests
TIMEOUT = 120        # seconds per request


# ─────────────────────────────────────────────────────────────
# Progress Tracking (resume-capable)
# ─────────────────────────────────────────────────────────────

class ExtractionProgress:
    """Track which PDFs have been extracted, enabling resume."""

    def __init__(self, base_dir: Path):
        self.progress_file = base_dir / ".firecrawl_progress.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "extracted": [],
            "failed": [],
            "stats": {
                "total_attempted": 0,
                "total_extracted": 0,
                "total_failed": 0,
                "total_skipped_no_url": 0,
                "total_chars_extracted": 0,
                "started_at": None,
                "last_updated": None,
            }
        }

    def save(self):
        self.data["stats"]["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_extracted(self, pdf_stem: str) -> bool:
        return pdf_stem in self.data["extracted"]

    def mark_extracted(self, pdf_stem: str, char_count: int):
        if pdf_stem not in self.data["extracted"]:
            self.data["extracted"].append(pdf_stem)
        self.data["stats"]["total_extracted"] += 1
        self.data["stats"]["total_attempted"] += 1
        self.data["stats"]["total_chars_extracted"] += char_count
        self.save()

    def mark_failed(self, pdf_stem: str, reason: str):
        self.data["failed"].append({"pdf": pdf_stem, "reason": reason,
                                     "timestamp": datetime.now().isoformat()})
        self.data["stats"]["total_failed"] += 1
        self.data["stats"]["total_attempted"] += 1
        self.save()

    def mark_skipped(self):
        self.data["stats"]["total_skipped_no_url"] += 1

    def set_start(self):
        if not self.data["stats"]["started_at"]:
            self.data["stats"]["started_at"] = datetime.now().isoformat()


# ─────────────────────────────────────────────────────────────
# Core Extraction
# ─────────────────────────────────────────────────────────────

def read_meta_json(meta_path: Path) -> dict:
    """Read a .meta.json sidecar file and return its contents."""
    with open(meta_path) as f:
        return json.load(f)


def extract_with_firecrawl(app: Firecrawl, url: str, log: logging.Logger) -> str:
    """
    Call Firecrawl to extract text from a PDF URL.

    Uses Nick's recommended settings:
      - parsers=["pdf"]       → forces PDF parser (not HTML scraper)
      - only_main_content=False → full paper text
      - max_age=172800000     → 48hr cache
      - formats=["markdown"]  → markdown output

    Returns: markdown text string, or raises on failure.
    """
    log.debug(f"  Calling Firecrawl: {url[:80]}...")

    data = app.scrape(
        url,
        only_main_content=FIRECRAWL_SETTINGS["only_main_content"],
        max_age=FIRECRAWL_SETTINGS["max_age"],
        parsers=FIRECRAWL_SETTINGS["parsers"],
        formats=FIRECRAWL_SETTINGS["formats"],
    )

    # The SDK returns a dict with 'markdown' key (or 'data' depending on version)
    # Handle both SDK response formats
    markdown = None

    if isinstance(data, dict):
        # SDK v1+ returns data directly
        markdown = data.get("markdown")
        if not markdown:
            # Some versions nest under 'data'
            nested = data.get("data", {})
            if isinstance(nested, dict):
                markdown = nested.get("markdown")

    if not markdown:
        raise ValueError(f"No markdown content returned from Firecrawl for {url[:60]}")

    return markdown


def save_firecrawl_output(md_path: Path, markdown: str, meta: dict):
    """
    Save Firecrawl markdown with a metadata header.
    The header helps the trainer identify the source and extraction method.
    """
    header = (
        f"<!-- FIRECRAWL EXTRACTION -->\n"
        f"<!-- doi: {meta.get('doi', 'N/A')} -->\n"
        f"<!-- pmid: {meta.get('pmid', 'N/A')} -->\n"
        f"<!-- title: {meta.get('title', 'N/A')[:100]} -->\n"
        f"<!-- extracted: {datetime.now().isoformat()} -->\n"
        f"<!-- source_url: {meta.get('source_pdf_url', 'N/A')} -->\n\n"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + markdown)


def strip_markdown_to_text(markdown: str) -> str:
    """
    Convert markdown to plain text for the chunking pipeline.
    Preserves content but removes formatting syntax.

    This is used by mupdf_trainer_v3.py when --text-source is auto or firecrawl.
    """
    text = markdown

    # Remove HTML comments (our metadata header)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Convert links to just text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove headers (keep the text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Convert markdown tables to space-separated text
    # Remove table border rows (|---|---|)
    text = re.sub(r'^\|[-:\s|]+\|$', '', text, flags=re.MULTILINE)
    # Convert table cells: | cell1 | cell2 | → cell1  cell2
    text = re.sub(r'\|', '  ', text)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

    # Remove code fences
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove blockquote markers
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # Remove list markers (but keep content)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# File Discovery
# ─────────────────────────────────────────────────────────────

def find_meta_files(base_path: Path) -> list:
    """
    Find all .meta.json files in the directory tree.
    Works with both flat directories and topic subdirectories.

    Returns: List of (meta_path, pdf_path) tuples.
    """
    pairs = []

    for meta_path in sorted(base_path.rglob("*.meta.json")):
        # The PDF should be the same name without .meta.json
        pdf_path = meta_path.with_suffix("").with_suffix(".pdf")
        # Handle case where meta is .meta.json (stem includes .meta)
        stem = meta_path.stem  # e.g., "paper_abc123.meta"
        if stem.endswith(".meta"):
            stem = stem[:-5]  # Remove .meta suffix
            pdf_path = meta_path.parent / f"{stem}.pdf"

        pairs.append((meta_path, pdf_path, stem))

    return pairs


# ─────────────────────────────────────────────────────────────
# Main Extraction Loop
# ─────────────────────────────────────────────────────────────

def run_extraction(args: argparse.Namespace):
    """Main extraction pipeline."""

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("firecrawl_extract")

    # Validate API key
    api_key = FIRECRAWL_API_KEY
    if not api_key:
        log.error("No Firecrawl API key found.")
        log.error("Set FIRECRAWL_API_KEY environment variable or edit the script.")
        sys.exit(1)

    # Initialize Firecrawl SDK
    app = Firecrawl(api_key=api_key)
    log.info("Firecrawl SDK initialized")

    # Find all .meta.json files
    base_path = Path(args.path)
    if not base_path.exists():
        log.error(f"Path does not exist: {base_path}")
        sys.exit(1)

    meta_pairs = find_meta_files(base_path)
    log.info(f"Found {len(meta_pairs)} PDF metadata files in {base_path}")

    if not meta_pairs:
        log.warning("No .meta.json files found. Run the harvester first.")
        sys.exit(0)

    # Progress tracking
    progress = ExtractionProgress(base_path)
    progress.set_start()

    # Count what we need to do
    to_process = []
    already_done = 0
    no_url = 0

    for meta_path, pdf_path, stem in meta_pairs:
        # Skip if already extracted (resume mode)
        if args.resume and progress.is_extracted(stem):
            already_done += 1
            continue

        # Skip if .firecrawl.md already exists (even without --resume)
        md_path = meta_path.parent / f"{stem}.firecrawl.md"
        if md_path.exists() and not args.force:
            already_done += 1
            continue

        # Read metadata to check for URL
        try:
            meta = read_meta_json(meta_path)
        except Exception as e:
            log.warning(f"  Failed to read {meta_path.name}: {e}")
            continue

        url = meta.get("source_pdf_url")
        if not url:
            log.debug(f"  No source_pdf_url in {meta_path.name}, skipping")
            no_url += 1
            progress.mark_skipped()
            continue

        to_process.append((meta_path, pdf_path, stem, meta, url))

    log.info(f"  Already extracted: {already_done}")
    log.info(f"  No URL available:  {no_url}")
    log.info(f"  To extract:        {len(to_process)}")

    if not to_process:
        log.info("Nothing to extract. Done.")
        return

    # Process each paper
    log.info(f"\nStarting extraction of {len(to_process)} papers...")
    log.info(f"Delay between requests: {args.delay}s")
    print()

    start_time = time.time()
    extracted_count = 0
    failed_count = 0

    for i, (meta_path, pdf_path, stem, meta, url) in enumerate(to_process, 1):
        title = meta.get("title", "Unknown")[:60]
        topic = meta.get("topic", "unknown")
        log.info(f"[{i}/{len(to_process)}] {title}...")
        log.debug(f"  URL: {url}")
        log.debug(f"  Topic: {topic}")

        md_path = meta_path.parent / f"{stem}.firecrawl.md"

        try:
            # Call Firecrawl
            markdown = extract_with_firecrawl(app, url, log)

            # Validate we got meaningful content
            if len(markdown.strip()) < 200:
                log.warning(f"  Firecrawl returned very short content ({len(markdown)} chars), may be incomplete")

            # Save markdown output
            save_firecrawl_output(md_path, markdown, meta)

            char_count = len(markdown)
            word_count = len(markdown.split())
            progress.mark_extracted(stem, char_count)
            extracted_count += 1

            log.info(f"  ✓ Saved: {md_path.name} ({word_count:,} words, {char_count:,} chars)")

        except Exception as e:
            error_msg = str(e)[:200]
            log.error(f"  ✗ Failed: {error_msg}")
            progress.mark_failed(stem, error_msg)
            failed_count += 1

        # Rate limiting
        if i < len(to_process):
            time.sleep(args.delay)

    # Final summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print()
    log.info("=" * 60)
    log.info("FIRECRAWL EXTRACTION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Extracted:    {extracted_count}")
    log.info(f"  Failed:       {failed_count}")
    log.info(f"  Skipped:      {no_url} (no URL)")
    log.info(f"  Already done: {already_done}")
    log.info(f"  Time:         {minutes}m {seconds:.0f}s")

    if extracted_count > 0:
        avg_time = elapsed / extracted_count
        total_chars = progress.data["stats"]["total_chars_extracted"]
        log.info(f"  Avg time/paper: {avg_time:.1f}s")
        log.info(f"  Total chars:    {total_chars:,}")

    log.info("=" * 60)

    # Print next steps
    if extracted_count > 0:
        print()
        log.info("NEXT STEP: Run Q/A generation with Firecrawl text:")
        log.info(f"  python mupdf_trainer_v3.py {base_path} --gen_qa --text-source auto \\")
        log.info(f"      --llm_model <your-model> --pretty_json")
        print()

    progress.save()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract text from harvested PDFs using Firecrawl API",
        epilog="Run after nore_paper_harvester.py, before mupdf_trainer_v3.py"
    )

    parser.add_argument(
        "path",
        help="Directory containing harvested PDFs with .meta.json sidecars"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip papers already extracted (checks progress file)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if .firecrawl.md already exists"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between API requests (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
