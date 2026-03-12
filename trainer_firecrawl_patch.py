"""
NORE Trainer Patch — Firecrawl Text Source Integration
=======================================================
Apply these changes to mupdf_trainer_v3.py to enable --text-source flag.

Three changes:
  1. Add new function: extract_from_firecrawl_md()
  2. Add new function: get_text()  (router)
  3. Modify run_pipeline() to use get_text() instead of extract_and_clean()
  4. Add --text-source argument to argparse

"""

# ══════════════════════════════════════════════════════════════
# CHANGE 1: Add these two functions AFTER extract_and_clean()
#           (after line ~163 in current file)
# ══════════════════════════════════════════════════════════════

# --- PASTE AFTER extract_and_clean() ---

def extract_from_firecrawl_md(pdf_path: str) -> Tuple[str, int]:
    """
    Read pre-extracted Firecrawl markdown for a PDF and convert to plain text.
    Expects a .firecrawl.md file alongside the PDF (created by firecrawl_extract.py).

    Returns (cleaned_text, estimated_pages) — same signature as extract_and_clean().
    """
    pdf_p = Path(pdf_path)
    md_path = pdf_p.with_suffix(".firecrawl.md")

    if not md_path.exists():
        raise FileNotFoundError(f"No Firecrawl text found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        markdown = f.read()

    # Strip markdown formatting to plain text
    text = _strip_markdown(markdown)

    # Estimate page count from text length (~3000 chars per page for academic papers)
    estimated_pages = max(1, len(text) // 3000)

    cleaned = clean_text(text)
    logging.info(f"Loaded Firecrawl text for {pdf_p.name}: {len(cleaned)} chars (~{estimated_pages} pages)")

    return cleaned, estimated_pages


def _strip_markdown(markdown: str) -> str:
    """
    Convert Firecrawl markdown to plain text for the chunking pipeline.
    Preserves content but removes formatting syntax.
    """
    text = markdown

    # Remove HTML comments (metadata header from firecrawl_extract.py)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Convert links to just their text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove header markers (keep the text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Convert markdown tables to space-separated text
    text = re.sub(r'^\|[-:\s|]+\|$', '', text, flags=re.MULTILINE)  # Remove border rows
    text = re.sub(r'\|', '  ', text)  # Convert cell separators

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

    # Remove code fences
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove blockquote markers
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # Remove list markers (keep content)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def get_text(pdf_path: str, text_source: str = "auto") -> Tuple[str, int]:
    """
    Router function that selects extraction method based on --text-source flag.

    Args:
        pdf_path: Path to the PDF file
        text_source: One of 'auto', 'firecrawl', 'mupdf'

    Returns: (cleaned_text, num_pages)

    Behavior:
        'auto'      → use .firecrawl.md if it exists, else fall back to mupdf
        'firecrawl' → require .firecrawl.md, error if missing
        'mupdf'     → always use local PDF extraction (original behavior)
    """
    pdf_p = Path(pdf_path)
    md_path = pdf_p.with_suffix(".firecrawl.md")
    has_firecrawl = md_path.exists()

    if text_source == "firecrawl":
        if not has_firecrawl:
            raise FileNotFoundError(
                f"--text-source=firecrawl but no .firecrawl.md for {pdf_p.name}. "
                f"Run firecrawl_extract.py first."
            )
        return extract_from_firecrawl_md(pdf_path)

    elif text_source == "auto":
        if has_firecrawl:
            logging.info(f"  Using Firecrawl text for {pdf_p.name}")
            return extract_from_firecrawl_md(pdf_path)
        else:
            logging.info(f"  Using mupdf extraction for {pdf_p.name} (no .firecrawl.md)")
            return extract_and_clean(pdf_path)

    else:  # "mupdf"
        return extract_and_clean(pdf_path)


# ══════════════════════════════════════════════════════════════
# CHANGE 2: In the argparse section (around line 1320-1360),
#           add this argument
# ══════════════════════════════════════════════════════════════

# --- PASTE with the other parser.add_argument() calls ---

"""
    parser.add_argument(
        "--text-source",
        choices=["auto", "firecrawl", "mupdf"],
        default="auto",
        dest="text_source",
        help="Text extraction method: "
             "'auto' = prefer .firecrawl.md, fall back to mupdf (default); "
             "'firecrawl' = require .firecrawl.md (error if missing); "
             "'mupdf' = always use local PDF extraction"
    )
"""


# ══════════════════════════════════════════════════════════════
# CHANGE 3: In run_pipeline(), replace the extract_and_clean call
#           (around line 1080)
# ══════════════════════════════════════════════════════════════

# FIND this block (around line 1077-1083):
"""
    for pdf in files:
        logging.info(f"\\n=== Processing: {pdf.name} ===")
        try:
            cleaned, n_pages = extract_and_clean(str(pdf))
        except Exception as e:
            logging.error(f"Failed to extract/clean {pdf.name}: {e}")
            continue # Skip to next PDF
"""

# REPLACE WITH:
"""
    for pdf in files:
        logging.info(f"\\n=== Processing: {pdf.name} ===")
        try:
            cleaned, n_pages = get_text(str(pdf), text_source=args.text_source)
        except Exception as e:
            logging.error(f"Failed to extract/clean {pdf.name}: {e}")
            continue # Skip to next PDF
"""

# That's it! The rest of the pipeline (chunking, relevance gate,
# Q/A generation, verification) works exactly the same regardless
# of whether the text came from Firecrawl or mupdf.
