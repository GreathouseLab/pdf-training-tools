#!/usr/bin/env python3
"""
Mine PLoS Genetics papers from the last 3 years.
Extracts full text using Firecrawl and saves as individual .txt files with attribution.
"""

import os
import re
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
import argparse
import sys


# Configuration
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"
PLOS_API_URL = "https://api.plos.org/search"
OUTPUT_DIR = Path("plos_genetics_papers")
RATE_LIMIT_DELAY = 1.0  # seconds between Firecrawl requests
PLOS_RATE_LIMIT_DELAY = 4.0  # seconds between PLoS API requests (limit: 100/5min = 20/min)


def get_plos_genetics_articles(start_date: str, end_date: str, rows_per_page: int = 100, max_articles: int = None):
    """
    Query PLoS API for PLoS Genetics articles within date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        rows_per_page: Number of results per API call
        max_articles: Maximum number of articles to fetch (None for all)

    Yields:
        dict: Article metadata including DOI, title, authors, etc.
    """
    # PLoS Genetics journal key - filter to only research articles
    journal_query = 'journal:"PLoS Genetics"'
    date_query = f'publication_date:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]'
    # Filter out sub-documents by requiring doc_type:full
    query = f"{journal_query} AND {date_query} AND doc_type:full"

    start = 0
    total_fetched = 0

    while True:
        params = {
            "q": query,
            "fl": "id,title,author,publication_date,abstract,article_type",
            "wt": "json",
            "rows": rows_per_page,
            "start": start,
        }

        print(f"Fetching articles {start} to {start + rows_per_page}...")
        response = requests.get(PLOS_API_URL, params=params)
        response.raise_for_status()

        data = response.json()
        docs = data["response"]["docs"]
        num_found = data["response"]["numFound"]

        if not docs:
            break

        for doc in docs:
            if max_articles and total_fetched >= max_articles:
                return
            # Skip sub-documents (like /title, /abstract fragments)
            doi = doc.get("id", "")
            if "/" in doi.split("journal.pgen.")[-1]:
                continue
            yield doc
            total_fetched += 1

        print(f"  Found {num_found} total articles, fetched {total_fetched} so far")

        start += rows_per_page
        if start >= num_found:
            break

        time.sleep(PLOS_RATE_LIMIT_DELAY)  # Be nice to PLoS API


def doi_to_url(doi: str) -> str:
    """Convert PLoS DOI to article URL."""
    # PLoS DOIs look like: 10.1371/journal.pgen.1234567
    return f"https://journals.plos.org/plosgenetics/article?id={doi}"


def extract_with_firecrawl(url: str) -> dict:
    """
    Extract article content using Firecrawl scrape endpoint.

    Args:
        url: Article URL to scrape

    Returns:
        dict: Extracted content with markdown and metadata
    """
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }

    response = requests.post(
        f"{FIRECRAWL_BASE_URL}/scrape",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    return response.json()


def sanitize_filename(title: str, doi: str) -> str:
    """Create a safe filename from article title and DOI."""
    # Extract the numeric part of the DOI for uniqueness (handle potential slashes)
    doi_suffix = doi.split(".")[-1].split("/")[0] if doi else "unknown"

    # Clean title - remove all non-alphanumeric except spaces and hyphens
    clean_title = re.sub(r'[^\w\s-]', '', title[:80])
    clean_title = re.sub(r'\s+', '_', clean_title.strip())

    # Ensure no empty filename
    if not clean_title:
        clean_title = "untitled"

    return f"{clean_title}_{doi_suffix}.txt"


def format_attribution(article: dict) -> str:
    """Format article metadata as attribution header."""
    doi = article.get("id", "Unknown DOI")
    title = article.get("title", "Unknown Title")
    authors = article.get("author", [])
    pub_date = article.get("publication_date", "Unknown Date")
    article_type = article.get("article_type", "Unknown Type")

    # Format authors
    if isinstance(authors, list):
        if len(authors) > 5:
            author_str = ", ".join(authors[:5]) + f", et al. ({len(authors)} authors)"
        else:
            author_str = ", ".join(authors)
    else:
        author_str = str(authors)

    # Format date
    if pub_date and "T" in pub_date:
        pub_date = pub_date.split("T")[0]

    attribution = f"""# Attribution
DOI: {doi}
Title: {title}
Authors: {author_str}
Publication Date: {pub_date}
Journal: PLoS Genetics
Article Type: {article_type}
URL: {doi_to_url(doi)}
Extracted: {datetime.now().isoformat()}

# Full Text
"""
    return attribution


def save_article(article: dict, content: str, output_dir: Path) -> Path:
    """Save article with attribution to text file."""
    doi = article.get("id", "unknown")
    title = article.get("title", "Untitled")

    filename = sanitize_filename(title, doi)
    filepath = output_dir / filename

    attribution = format_attribution(article)
    full_content = attribution + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return filepath


def load_progress(output_dir: Path) -> set:
    """Load set of already processed DOIs from progress file."""
    progress_file = output_dir / ".progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return set(json.load(f))
    return set()


def save_progress(output_dir: Path, processed_dois: set):
    """Save set of processed DOIs to progress file."""
    progress_file = output_dir / ".progress.json"
    with open(progress_file, "w") as f:
        json.dump(list(processed_dois), f)


def main():
    parser = argparse.ArgumentParser(description="Mine PLoS Genetics papers")
    parser.add_argument("--start-date", default=None,
                        help="Start date (YYYY-MM-DD). Default: 3 years ago")
    parser.add_argument("--end-date", default=None,
                        help="End date (YYYY-MM-DD). Default: today")
    parser.add_argument("--output-dir", default="plos_genetics_papers",
                        help="Output directory for text files")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Maximum number of articles to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skip already processed")
    args = parser.parse_args()

    # Set date range
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")

    print(f"Mining PLoS Genetics articles from {start_date} to {end_date}")

    # Check API key
    if not FIRECRAWL_API_KEY:
        print("ERROR: FIRECRAWL_API_KEY environment variable not set")
        print("Run: export FIRECRAWL_API_KEY=your_key_here")
        return 1

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress if resuming
    processed_dois = load_progress(output_dir) if args.resume else set()
    if processed_dois:
        print(f"Resuming: {len(processed_dois)} articles already processed")

    # Statistics
    stats = {
        "total_found": 0,
        "processed": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Fetch and process articles
    try:
        for article in get_plos_genetics_articles(start_date, end_date, max_articles=args.max_articles):
            stats["total_found"] += 1
            doi = article.get("id", "")
            title = article.get("title", "Untitled")[:60]

            # Skip if already processed
            if doi in processed_dois:
                print(f"  Skipping (already processed): {title}...")
                stats["skipped"] += 1
                continue

            print(f"\nProcessing: {title}...")

            try:
                # Get article URL
                url = doi_to_url(doi)

                # Extract with Firecrawl
                result = extract_with_firecrawl(url)

                if result.get("success") and result.get("data", {}).get("markdown"):
                    content = result["data"]["markdown"]
                    filepath = save_article(article, content, output_dir)
                    print(f"  Saved: {filepath.name}")
                    stats["processed"] += 1
                    processed_dois.add(doi)
                else:
                    print(f"  Warning: No content extracted for {doi}")
                    stats["errors"] += 1

                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)

            except requests.exceptions.RequestException as e:
                print(f"  Error extracting {doi}: {e}")
                stats["errors"] += 1
                continue

            # Save progress periodically
            if stats["processed"] % 10 == 0:
                save_progress(output_dir, processed_dois)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")

    finally:
        # Save final progress
        save_progress(output_dir, processed_dois)

        # Print summary
        print("\n" + "="*50)
        print("Mining Complete!")
        print("="*50)
        print(f"Total articles found: {stats['total_found']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Skipped (already done): {stats['skipped']}")
        print(f"Errors: {stats['errors']}")
        print(f"Output directory: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    exit(main())
