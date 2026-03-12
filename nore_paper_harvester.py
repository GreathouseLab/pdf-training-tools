#!/usr/bin/env python3
"""
NORE Multi-Source Paper Harvester
=================================
Discovers, pre-screens, and downloads open-access nutrition oncology PDFs
from multiple sources (PMC, Unpaywall, Europe PMC) for Q/A pair generation.

Architecture:
  1. DISCOVER  — Search PMC E-utilities / Europe PMC by topic-specific MeSH queries
  2. SCREEN    — Pull abstracts, run LLM relevance gate (reuses NORE RELEVANCE prompt)
  3. LOCATE    — Find open-access PDF via PMC OA, Unpaywall, or publisher
  4. DOWNLOAD  — Save PDF with attribution metadata sidecar
  5. TRACK     — Progress file enables resume after interruption

Output feeds directly into: mupdf_trainer_v3.py --gen_qa

Usage:
  # Harvest all topics — DEFAULT uses Firecrawl for text extraction (no PDFs downloaded)
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu

  # Single topic with limit
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu --topic drug_nutrient --max-per-topic 100

  # Download PDFs locally (for RLHF traceability) instead of Firecrawl
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu --download-pdfs

  # Resume interrupted run
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu --resume

  # Skip LLM screening (download everything that matches MeSH query)
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu --no-screen

  # Use specific LLM backend for screening
  python nore_paper_harvester.py --email Leigh_Greathouse@baylor.edu --screen-model gpt-4.1-mini
"""

import os
import re
import json
import time
import hashlib
import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Generator
from textwrap import dedent

import requests
from urllib.parse import quote, urlencode

# ──────────────────────────────────────────────
# Firecrawl SDK (optional — only needed in default mode)
# ──────────────────────────────────────────────
_FIRECRAWL_AVAILABLE = False
try:
    from firecrawl import Firecrawl
    _FIRECRAWL_AVAILABLE = True
except ImportError:
    pass  # OK — will error at runtime only if Firecrawl mode is used

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# API endpoints
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ELINK   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
PMC_OA_FTP = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Rate limits
NCBI_DELAY = 0.34          # 3 req/sec without API key (be conservative)
NCBI_DELAY_WITH_KEY = 0.11 # 10 req/sec with API key
UNPAYWALL_DELAY = 0.1      # 100K/day, generous limit
EUROPE_PMC_DELAY = 0.2     # Be nice
PDF_DOWNLOAD_DELAY = 1.0   # Courtesy delay for publisher servers
FIRECRAWL_DELAY = 1.0      # Courtesy delay for Firecrawl API

# Firecrawl settings (Nick's recommended SDK settings)
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_SETTINGS = {
    "only_main_content": False,    # Full paper including methods, supplementary
    "max_age": 172800000,          # 48 hours cache (milliseconds)
    "parsers": ["pdf"],            # Force PDF-specific parser
    "formats": ["markdown"],       # Output as markdown
}

# Defaults
DEFAULT_OUTPUT_DIR = Path("harvested_papers")
DEFAULT_YEARS_BACK = 5
DEFAULT_MAX_PER_TOPIC = 500
MIN_RELEVANCE_SCORE = 6    # Abstracts scoring below this are skipped

# ──────────────────────────────────────────────
# Topic Definitions — MeSH queries for each priority area
# ──────────────────────────────────────────────

TOPIC_QUERIES = {
    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 1: Drug-Nutrient Interactions in Oncology
    #  MeSH qualifiers: /adverse effects, /pharmacokinetics, /drug effects, /diet therapy
    # ═══════════════════════════════════════════════════════════════
    "drug_nutrient": {
        "label": "Drug-Nutrient Interactions in Oncology",
        "priority": 1,
        "queries": [
            # Core MeSH heading — Food-Drug Interactions as Major topic + Cancer
            '"Food-Drug Interactions"[Majr] AND "Neoplasms"[MeSH]',
            # Herb-drug interactions with cancer drugs — /adverse effects targets harm
            '"Herb-Drug Interactions"[MeSH] AND "Antineoplastic Agents/adverse effects"[MeSH]',
            # Cancer drug pharmacokinetics affected by diet/food
            '"Antineoplastic Agents/pharmacokinetics"[MeSH] AND ("Diet"[MeSH] OR "Food"[MeSH])',
            # Dietary supplements interacting with cancer therapy
            '"Dietary Supplements/adverse effects"[MeSH] AND "Neoplasms/drug therapy"[MeSH]',
            # CYP enzymes affected by dietary compounds in cancer
            '"Cytochrome P-450 Enzyme System/drug effects"[MeSH] AND ("Diet"[MeSH] OR "Dietary Supplements"[MeSH]) AND "Neoplasms"[MeSH]',
            # Drug interactions with nutritional treatment of cancer
            '"Drug Interactions"[MeSH] AND "Neoplasms/diet therapy"[MeSH]',
            # Known problematic supplements — /adverse effects qualifier
            '("Curcumin/adverse effects"[MeSH] OR "Tea/adverse effects"[MeSH] OR "Grapefruit/adverse effects"[MeSH] OR "Hypericum/adverse effects"[MeSH]) AND "Neoplasms/drug therapy"[MeSH]',
            # Nutritional status affecting drug metabolism in cancer
            '"Nutritional Status"[MeSH] AND "Antineoplastic Agents/pharmacokinetics"[MeSH]',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"food-drug interaction" OR TITLE_ABS:"drug-nutrient interaction") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"oncology") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"herb-drug interaction") AND (TITLE_ABS:"chemotherapy" OR TITLE_ABS:"antineoplastic") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"dietary supplement" AND TITLE_ABS:"adverse" AND TITLE_ABS:"cancer treatment") AND (OPEN_ACCESS:y)',
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 2: Cancer Cachexia & Sarcopenia
    #  MeSH qualifiers: /diet therapy, /therapy, /complications, /prevention & control
    # ═══════════════════════════════════════════════════════════════
    "cachexia_sarcopenia": {
        "label": "Cancer Cachexia & Sarcopenia",
        "priority": 2,
        "queries": [
            # Cachexia diet therapy — /diet therapy IS the nutritional treatment qualifier
            '"Cachexia/diet therapy"[MeSH] AND "Neoplasms/complications"[MeSH]',
            # Cachexia general therapy + nutritional support
            '"Cachexia/therapy"[MeSH] AND "Neoplasms/complications"[MeSH] AND ("Nutrition Therapy"[MeSH] OR "Nutritional Support"[MeSH])',
            # Sarcopenia + nutritional status/body composition in cancer
            '"Sarcopenia"[MeSH] AND "Neoplasms"[MeSH] AND ("Nutritional Status"[MeSH] OR "Body Composition"[MeSH])',
            # Skeletal muscle affected by nutrition in cancer — /drug effects qualifier
            '"Muscle, Skeletal/drug effects"[MeSH] AND "Neoplasms/complications"[MeSH] AND ("Dietary Supplements/therapeutic use"[MeSH] OR "Nutrition Therapy"[MeSH])',
            # Sarcopenic obesity + cancer + diet
            '"Sarcopenia"[MeSH] AND "Obesity"[MeSH] AND "Neoplasms"[MeSH] AND ("Diet"[MeSH] OR "Nutritional Status"[MeSH])',
            # Weight loss prevention in cancer with nutritional support
            '"Weight Loss/prevention & control"[MeSH] AND "Neoplasms/complications"[MeSH] AND ("Nutrition Therapy"[MeSH] OR "Dietary Supplements/therapeutic use"[MeSH])',
            # Protein supplementation for cancer muscle preservation
            '"Dietary Proteins/therapeutic use"[MeSH] AND ("Cachexia"[MeSH] OR "Sarcopenia"[MeSH]) AND "Neoplasms"[MeSH]',
            # Malnutrition diet therapy in cancer
            '"Malnutrition/diet therapy"[MeSH] AND "Neoplasms/complications"[MeSH]',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"cancer cachexia") AND (TITLE_ABS:"nutrition" OR TITLE_ABS:"dietary intervention") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"sarcopenia") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"oncology") AND (TITLE_ABS:"protein" OR TITLE_ABS:"nutrition") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"muscle wasting" OR TITLE_ABS:"lean body mass") AND (TITLE_ABS:"cancer") AND (TITLE_ABS:"nutritional support") AND (OPEN_ACCESS:y)',
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 3: Immunotherapy & Nutrition (UPDATED)
    #  REMOVED: Microbiome queries (moved to Topic 6)
    #  MeSH qualifiers: /immunology, /adverse effects, /diet therapy
    # ═══════════════════════════════════════════════════════════════
    "immunotherapy_nutrition": {
        "label": "Immunotherapy & Nutrition",
        "priority": 3,
        "queries": [
            # Immunotherapy + nutritional status in cancer
            '"Immunotherapy"[MeSH] AND "Nutritional Status"[MeSH] AND "Neoplasms"[MeSH]',
            # Immune checkpoint inhibitors + diet/nutrition therapy
            '"Immune Checkpoint Inhibitors"[MeSH] AND ("Nutrition Therapy"[MeSH] OR "Diet"[MeSH] OR "Nutritional Status"[MeSH])',
            # Cancer immunology + immunotherapy + body composition/malnutrition
            '"Neoplasms/immunology"[MeSH] AND "Immunotherapy"[MeSH] AND ("Body Composition"[MeSH] OR "Body Mass Index"[MeSH] OR "Malnutrition"[MeSH])',
            # Cachexia/sarcopenia impacting immunotherapy outcomes
            '("Cachexia"[MeSH] OR "Sarcopenia"[MeSH]) AND "Immune Checkpoint Inhibitors"[MeSH] AND "Neoplasms"[MeSH]',
            # Dietary supplements during immunotherapy — /therapeutic use + /adverse effects
            '"Dietary Supplements/therapeutic use"[MeSH] AND "Immunotherapy/adverse effects"[MeSH] AND "Neoplasms"[MeSH]',
            # Cancer diet therapy + immune system
            '"Neoplasms/diet therapy"[MeSH] AND ("Immunity"[MeSH] OR "Immune System"[MeSH])',
            # Obesity/BMI effect on immunotherapy response
            '("Obesity"[MeSH] OR "Body Mass Index"[MeSH]) AND "Immune Checkpoint Inhibitors"[MeSH] AND "Treatment Outcome"[MeSH]',
            # Nutritional intervention during immunotherapy adverse events (colitis, etc.)
            '"Immunotherapy/adverse effects"[MeSH] AND ("Nutrition Therapy"[MeSH] OR "Nutritional Support"[MeSH]) AND "Neoplasms"[MeSH]',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"immunotherapy" OR TITLE_ABS:"checkpoint inhibitor") AND (TITLE_ABS:"nutrition" OR TITLE_ABS:"diet") AND (TITLE_ABS:"cancer") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"immune checkpoint") AND (TITLE_ABS:"nutritional status" OR TITLE_ABS:"body composition" OR TITLE_ABS:"sarcopenia") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"immunotherapy") AND (TITLE_ABS:"obesity" OR TITLE_ABS:"BMI") AND (TITLE_ABS:"cancer outcome" OR TITLE_ABS:"treatment response") AND (OPEN_ACCESS:y)',
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 4: Cancer Malnutrition — Screening, Assessment & Intervention (NEW)
    #  The foundational clinical topic — affects 20-70% of cancer patients.
    #  MeSH qualifiers: /diagnosis, /diet therapy, /therapy, /prevention & control
    # ═══════════════════════════════════════════════════════════════
    "cancer_malnutrition": {
        "label": "Cancer Malnutrition: Screening, Assessment & Intervention",
        "priority": 2,
        "queries": [
            # Malnutrition + cancer — direct nutritional treatment
            '"Malnutrition/diet therapy"[MeSH] AND "Neoplasms"[MeSH]',
            # Malnutrition diagnosis/screening in cancer — PG-SGA, NRS-2002, MUST, GLIM
            '"Malnutrition/diagnosis"[MeSH] AND "Neoplasms"[MeSH]',
            # Nutrition assessment in cancer (broad — captures all screening tools)
            '"Nutrition Assessment"[MeSH] AND "Neoplasms"[MeSH]',
            # Nutritional support interventions in cancer — ONS, enteral, parenteral
            '"Nutritional Support"[MeSH] AND "Neoplasms/diet therapy"[MeSH]',
            # Enteral nutrition in cancer — tube feeding, ONS during treatment
            '"Enteral Nutrition"[MeSH] AND "Neoplasms"[MeSH]',
            # Parenteral nutrition in cancer — when enteral is not possible
            '"Parenteral Nutrition"[MeSH] AND "Neoplasms"[MeSH]',
            # Perioperative nutrition — pre/post-surgical optimization in cancer
            '("Preoperative Care"[MeSH] OR "Postoperative Care"[MeSH]) AND "Nutritional Status"[MeSH] AND "Neoplasms"[MeSH]',
            # Malnutrition impact on treatment tolerance and outcomes
            '"Malnutrition"[MeSH] AND "Neoplasms/therapy"[MeSH] AND ("Treatment Outcome"[MeSH] OR "Patient Outcome Assessment"[MeSH])',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"malnutrition" OR TITLE_ABS:"nutritional screening") AND (TITLE_ABS:"cancer") AND (TITLE_ABS:"assessment" OR TITLE_ABS:"PG-SGA" OR TITLE_ABS:"GLIM") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"nutritional support" OR TITLE_ABS:"oral nutritional supplement") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"oncology") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"perioperative nutrition" OR TITLE_ABS:"prehabilitation") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"surgery") AND (TITLE_ABS:"nutritional") AND (OPEN_ACCESS:y)',
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 5: Dietary Patterns in Cancer (NEW — replaces Mediterranean-only)
    #  Covers: Mediterranean, ketogenic, fasting/TRE/FMD, plant-based,
    #          anti-inflammatory, dietary inflammatory index
    #  MeSH qualifiers: /prevention & control, /diet therapy, /mortality,
    #                    /therapeutic use, /adverse effects
    # ═══════════════════════════════════════════════════════════════
    "dietary_patterns": {
        "label": "Dietary Patterns in Cancer",
        "priority": 4,
        "queries": [
            # Mediterranean diet + cancer (retained from old topic)
            '"Diet, Mediterranean"[Majr] AND "Neoplasms"[MeSH]',
            # Ketogenic diet + cancer — metabolic targeting, Warburg effect
            '"Diet, Ketogenic"[MeSH] AND "Neoplasms"[MeSH]',
            # Ketogenic diet during cancer treatment — adjuvant role
            '"Diet, Ketogenic"[MeSH] AND "Neoplasms/diet therapy"[MeSH]',
            # Intermittent fasting + cancer — chemo sensitization, metabolic switching
            '"Intermittent Fasting"[MeSH] AND "Neoplasms"[MeSH]',
            # Fasting + cancer treatment — fasting-mimicking diet, short-term fasting
            '"Fasting"[MeSH] AND "Neoplasms/diet therapy"[MeSH]',
            # Caloric restriction + cancer — prevention and treatment
            '"Caloric Restriction"[MeSH] AND "Neoplasms"[MeSH]',
            # Anti-inflammatory dietary pattern + cancer outcomes
            '"Inflammation"[MeSH] AND "Diet"[MeSH] AND "Neoplasms/diet therapy"[MeSH]',
            # Plant-based / vegetarian diet + cancer
            '("Diet, Vegetarian"[MeSH] OR "Diet, Vegan"[MeSH]) AND "Neoplasms"[MeSH]',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"ketogenic diet") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"glioblastoma" OR TITLE_ABS:"tumor") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"intermittent fasting" OR TITLE_ABS:"time-restricted eating" OR TITLE_ABS:"fasting-mimicking") AND (TITLE_ABS:"cancer") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"Mediterranean diet" OR TITLE_ABS:"anti-inflammatory diet" OR TITLE_ABS:"dietary inflammatory index") AND (TITLE_ABS:"cancer") AND (OPEN_ACCESS:y)',
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    #  TOPIC 6: Microbiome, Diet & Cancer Outcomes (NEW — split from immunotherapy)
    #  The microbiome-diet connection extends beyond immunotherapy to chemo
    #  response, radiation enteritis, GVHD, and surgical recovery.
    #  MeSH qualifiers: /drug effects, /therapeutic use, /diet therapy
    # ═══════════════════════════════════════════════════════════════
    "microbiome_diet_cancer": {
        "label": "Microbiome, Diet & Cancer Outcomes",
        "priority": 5,
        "queries": [
            # Gut microbiome + diet + cancer (broadest capture)
            '"Gastrointestinal Microbiome"[MeSH] AND "Diet"[MeSH] AND "Neoplasms"[MeSH]',
            # Microbiome modulation through diet affecting cancer immune response
            # (moved from immunotherapy topic)
            '"Gastrointestinal Microbiome/drug effects"[MeSH] AND ("Diet"[MeSH] OR "Probiotics/therapeutic use"[MeSH]) AND "Neoplasms"[MeSH]',
            # Gut microbiome + immunotherapy + diet (moved from immunotherapy topic)
            '"Gastrointestinal Microbiome"[MeSH] AND "Immunotherapy"[MeSH] AND ("Diet"[MeSH] OR "Dietary Fiber"[MeSH])',
            # Probiotics/prebiotics during cancer treatment
            '("Probiotics/therapeutic use"[MeSH] OR "Prebiotics/therapeutic use"[MeSH]) AND "Neoplasms"[MeSH]',
            # Dietary fiber + microbiome + cancer (fiber as prebiotic modulator)
            '"Dietary Fiber"[MeSH] AND "Gastrointestinal Microbiome"[MeSH] AND "Neoplasms"[MeSH]',
            # Microbiome and chemotherapy response/toxicity
            '"Gastrointestinal Microbiome"[MeSH] AND "Antineoplastic Agents/adverse effects"[MeSH]',
            # Microbiome and radiation therapy — enteritis, gut damage recovery
            '"Gastrointestinal Microbiome"[MeSH] AND ("Radiation Injuries"[MeSH] OR "Radiotherapy"[MeSH]) AND ("Diet"[MeSH] OR "Probiotics"[MeSH])',
            # Dysbiosis + cancer + nutritional intervention
            '"Dysbiosis"[MeSH] AND "Neoplasms"[MeSH] AND ("Diet"[MeSH] OR "Nutrition Therapy"[MeSH])',
        ],
        "europe_pmc_queries": [
            '(TITLE_ABS:"gut microbiome" OR TITLE_ABS:"gut microbiota") AND (TITLE_ABS:"diet" OR TITLE_ABS:"fiber" OR TITLE_ABS:"prebiotic") AND (TITLE_ABS:"cancer") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"probiotics" OR TITLE_ABS:"synbiotics") AND (TITLE_ABS:"cancer" OR TITLE_ABS:"chemotherapy") AND (TITLE_ABS:"nutrition" OR TITLE_ABS:"diet") AND (OPEN_ACCESS:y)',
            '(TITLE_ABS:"microbiome") AND (TITLE_ABS:"immunotherapy" OR TITLE_ABS:"checkpoint inhibitor") AND (TITLE_ABS:"diet" OR TITLE_ABS:"fiber") AND (OPEN_ACCESS:y)',
        ]
    },
}


# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────

def setup_logging(output_dir: Path, verbose: bool = False):
    """Configure logging to both console and file."""
    log_file = output_dir / "harvest.log"
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("harvester")


# ──────────────────────────────────────────────
# Progress Tracking
# ──────────────────────────────────────────────

class ProgressTracker:
    """Tracks harvested DOIs/PMIDs to enable resume and deduplication across topics."""

    def __init__(self, output_dir: Path):
        self.progress_file = output_dir / ".harvest_progress.json"
        self.data = self._load()
        # Build lookup sets from loaded lists for O(1) membership checks
        self._doi_set = set(self.data["processed_dois"])
        self._pmid_set = set(self.data["processed_pmids"])
        self._screened_set = set(self.data["screened_out"])
        self._error_set = set(self.data["download_errors"])

    def _load(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {
            "processed_dois": [],
            "processed_pmids": [],
            "screened_out": [],     # Abstracts that failed relevance gate
            "download_errors": [],
            "stats_by_topic": {},
            "last_updated": None,
        }

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_processed(self, doi: str = None, pmid: str = None) -> bool:
        if doi and doi in self._doi_set:
            return True
        if pmid and pmid in self._pmid_set:
            return True
        return False

    def is_screened_out(self, doi: str = None, pmid: str = None) -> bool:
        if doi and doi in self._screened_set:
            return True
        if pmid and pmid in self._screened_set:
            return True
        return False

    def mark_processed(self, doi: str = None, pmid: str = None, topic: str = "unknown"):
        if doi and doi not in self._doi_set:
            self._doi_set.add(doi)
            self.data["processed_dois"].append(doi)
        if pmid and pmid not in self._pmid_set:
            self._pmid_set.add(pmid)
            self.data["processed_pmids"].append(pmid)
        # Update topic stats
        if topic not in self.data["stats_by_topic"]:
            self.data["stats_by_topic"][topic] = {"downloaded": 0, "screened_out": 0, "errors": 0}
        self.data["stats_by_topic"][topic]["downloaded"] += 1

    def mark_screened_out(self, identifier: str, topic: str = "unknown",
                          doi: str = None, pmid: str = None):
        if doi and doi not in self._screened_set:
            self._screened_set.add(doi)
            self.data["screened_out"].append(doi)
        if pmid and pmid not in self._screened_set:
            self._screened_set.add(pmid)
            self.data["screened_out"].append(pmid)
        if not doi and not pmid and identifier not in self._screened_set:
            self._screened_set.add(identifier)
            self.data["screened_out"].append(identifier)
        if topic not in self.data["stats_by_topic"]:
            self.data["stats_by_topic"][topic] = {"downloaded": 0, "screened_out": 0, "errors": 0}
        self.data["stats_by_topic"][topic]["screened_out"] += 1

    def mark_error(self, identifier: str, topic: str = "unknown"):
        if identifier not in self._error_set:
            self._error_set.add(identifier)
            self.data["download_errors"].append(identifier)
        if topic not in self.data["stats_by_topic"]:
            self.data["stats_by_topic"][topic] = {"downloaded": 0, "screened_out": 0, "errors": 0}
        self.data["stats_by_topic"][topic]["errors"] += 1

    @property
    def total_downloaded(self) -> int:
        return len(self._doi_set | self._pmid_set)

    def get_summary(self) -> str:
        lines = ["=" * 60, "HARVEST SUMMARY", "=" * 60]
        lines.append(f"Total PDFs downloaded: {self.total_downloaded}")
        lines.append(f"Total screened out:    {len(self.data['screened_out'])}")
        lines.append(f"Total errors:          {len(self.data['download_errors'])}")
        lines.append("")
        for topic, stats in self.data["stats_by_topic"].items():
            label = TOPIC_QUERIES.get(topic, {}).get("label", topic)
            lines.append(f"  {label}:")
            lines.append(f"    Downloaded:   {stats['downloaded']}")
            lines.append(f"    Screened out: {stats['screened_out']}")
            lines.append(f"    Errors:       {stats['errors']}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Source 1: PubMed Central via NCBI E-utilities
# ──────────────────────────────────────────────

def search_pmc(query: str, email: str, max_results: int = 500,
               api_key: str = None, min_date: str = None, max_date: str = None,
               logger=None) -> list[dict]:
    """
    Search PubMed for articles matching query. Returns list of PMIDs with metadata.

    Uses E-utilities esearch -> efetch pipeline.
    Returns: List of dicts with keys: pmid, doi, title, authors, abstract, pub_date, journal
    """
    log = logger or logging.getLogger("harvester")
    delay = NCBI_DELAY_WITH_KEY if api_key else NCBI_DELAY

    # Step 1: esearch — get PMIDs
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 10000),  # NCBI max per request
        "retmode": "json",
        "sort": "relevance",
        "email": email,
    }
    if api_key:
        esearch_params["api_key"] = api_key
    if min_date:
        esearch_params["mindate"] = min_date
        esearch_params["datetype"] = "pdat"
    if max_date:
        esearch_params["maxdate"] = max_date
        esearch_params["datetype"] = "pdat"

    log.info(f"  Searching PubMed: {query[:80]}...")
    time.sleep(delay)
    resp = requests.get(NCBI_ESEARCH, params=esearch_params, timeout=30)
    resp.raise_for_status()
    search_data = resp.json()

    pmids = search_data.get("esearchresult", {}).get("idlist", [])
    total_count = int(search_data.get("esearchresult", {}).get("count", 0))
    log.info(f"  Found {total_count} total results, retrieving {len(pmids)} PMIDs")

    if not pmids:
        return []

    # Step 2: efetch — get metadata for all PMIDs (batch by 200)
    articles = []
    for batch_start in range(0, len(pmids), 200):
        batch = pmids[batch_start:batch_start + 200]
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
            "email": email,
        }
        if api_key:
            efetch_params["api_key"] = api_key

        time.sleep(delay)
        resp = requests.get(NCBI_EFETCH, params=efetch_params, timeout=60)
        resp.raise_for_status()

        # Parse XML response (lightweight — no lxml dependency needed)
        articles.extend(_parse_pubmed_xml(resp.text))
        log.debug(f"  Fetched metadata for {len(articles)} articles so far")

    return articles


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """
    Parse PubMed efetch XML into article dicts.
    Uses regex-based extraction to avoid lxml dependency.
    """
    articles = []

    # Split into individual articles
    article_blocks = re.findall(
        r'<PubmedArticle>(.*?)</PubmedArticle>',
        xml_text, re.DOTALL
    )

    for block in article_blocks:
        article = {}

        # PMID
        pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
        article["pmid"] = pmid_match.group(1) if pmid_match else None

        # Title
        title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', block, re.DOTALL)
        article["title"] = _strip_xml_tags(title_match.group(1)) if title_match else "Untitled"

        # Abstract
        abstract_parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', block, re.DOTALL)
        article["abstract"] = " ".join(_strip_xml_tags(p) for p in abstract_parts) if abstract_parts else ""

        # Authors — parse each <Author> block individually to avoid
        # catastrophic regex backtracking on corporate/consortium authors
        authors = []
        for author_block in re.findall(r'<Author[^>]*>(.*?)</Author>', block, re.DOTALL):
            ln_match = re.search(r'<LastName>(.*?)</LastName>', author_block)
            fn_match = re.search(r'<ForeName>(.*?)</ForeName>', author_block)
            if ln_match and fn_match:
                authors.append(f"{fn_match.group(1)} {ln_match.group(1)}")
            elif ln_match:
                authors.append(ln_match.group(1))
        article["authors"] = authors

        # DOI
        doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', block)
        article["doi"] = doi_match.group(1).strip() if doi_match else None

        # PMC ID — normalize to always have PMC prefix
        pmc_match = re.search(r'<ArticleId IdType="pmc">(.*?)</ArticleId>', block)
        pmcid = pmc_match.group(1).strip() if pmc_match else None
        if pmcid and not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        article["pmcid"] = pmcid

        # Publication date
        year_match = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', block, re.DOTALL)
        month_match = re.search(r'<PubDate>.*?<Month>(\w+)</Month>', block, re.DOTALL)
        if year_match:
            year = year_match.group(1)
            month = month_match.group(1) if month_match else "01"
            article["pub_date"] = f"{year}-{month}"
        else:
            article["pub_date"] = "Unknown"

        # Journal
        journal_match = re.search(r'<Title>(.*?)</Title>', block)
        article["journal"] = _strip_xml_tags(journal_match.group(1)) if journal_match else "Unknown"

        # Article type hints
        pub_type_matches = re.findall(r'<PublicationType[^>]*>(.*?)</PublicationType>', block)
        article["pub_types"] = [_strip_xml_tags(pt) for pt in pub_type_matches]

        if article["pmid"]:
            articles.append(article)

    return articles


def _strip_xml_tags(text: str) -> str:
    """Remove XML tags from text."""
    return re.sub(r'<[^>]+>', '', text).strip()


# ──────────────────────────────────────────────
# Source 2: Europe PMC (supplements PMC with European OA content)
# ──────────────────────────────────────────────

def search_europe_pmc(query: str, max_results: int = 200, logger=None) -> list[dict]:
    """Search Europe PMC for open-access articles."""
    log = logger or logging.getLogger("harvester")
    articles = []
    page_size = min(max_results, 100)
    cursor = "*"

    while len(articles) < max_results:
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
            "resultType": "core",
        }

        time.sleep(EUROPE_PMC_DELAY)
        log.debug(f"  Europe PMC query: {query[:60]}... (cursor: {cursor[:20]})")
        resp = requests.get(EUROPE_PMC_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        for r in results:
            articles.append({
                "pmid": r.get("pmid"),
                "doi": r.get("doi"),
                "pmcid": r.get("pmcid"),
                "title": r.get("title", "Untitled"),
                "abstract": r.get("abstractText", ""),
                "authors": [
                    f"{a.get('firstName', '')} {a.get('lastName', '')}"
                    for a in r.get("authorList", {}).get("author", [])
                ],
                "pub_date": r.get("firstPublicationDate", "Unknown"),
                "journal": r.get("journalTitle", "Unknown"),
                "pub_types": [],
                "source": "europe_pmc",
            })

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    log.info(f"  Europe PMC returned {len(articles)} results")
    return articles[:max_results]


# ──────────────────────────────────────────────
# PDF Locator: Tries multiple sources to find OA PDF
# ──────────────────────────────────────────────

def find_pdf_url(doi: str = None, pmcid: str = None, email: str = "",
                 logger=None) -> Optional[str]:
    """
    Try to find an open-access PDF URL using multiple strategies:
    1. PMC OA service (if PMCID available) - direct PDF from NIH
    2. Unpaywall API (if DOI available) - finds legal OA copies
    3. Europe PMC full-text PDF link
    4. Direct PMC PDF URL construction (last resort)

    Returns PDF URL or None.
    """
    log = logger or logging.getLogger("harvester")

    # Normalize PMCID format — some sources return bare numbers
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    # Strategy 1: PMC OA Service (best - direct NIH-hosted PDF)
    if pmcid:
        try:
            time.sleep(NCBI_DELAY)
            resp = requests.get(PMC_OA_FTP, params={"id": pmcid, "format": "pdf"}, timeout=15)
            if resp.status_code == 200:
                # Prefer HTTPS links
                link_match = re.search(r'href="(https?://[^"]+\.pdf[^"]*)"', resp.text)
                if link_match:
                    log.debug(f"  Found PDF via PMC OA (HTTPS): {pmcid}")
                    return link_match.group(1)
                # FIX #1: Convert FTP to HTTPS (requests can't do FTP)
                ftp_match = re.search(r'href="(ftp://ftp\.ncbi\.nlm\.nih\.gov/[^"]+)"', resp.text)
                if ftp_match:
                    https_url = ftp_match.group(1).replace(
                        "ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov"
                    )
                    log.debug(f"  Found PDF via PMC OA (FTP->HTTPS): {pmcid}")
                    return https_url
        except Exception as e:
            log.debug(f"  PMC OA lookup failed for {pmcid}: {e}")

    # Strategy 2: Unpaywall
    # FIX #2: Scan ALL oa_locations for url_for_pdf, not just best_oa
    if doi and email:
        try:
            time.sleep(UNPAYWALL_DELAY)
            encoded_doi = quote(doi, safe='')
            url = f"{UNPAYWALL_API}/{encoded_doi}?email={email}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()

                # First try best_oa_location.url_for_pdf
                best_oa = data.get("best_oa_location") or {}
                pdf_url = best_oa.get("url_for_pdf")
                if pdf_url:
                    log.debug(f"  Found PDF via Unpaywall (best_oa.url_for_pdf): {doi}")
                    return pdf_url

                # Scan ALL OA locations for any url_for_pdf
                for loc in data.get("oa_locations", []):
                    pdf_url = loc.get("url_for_pdf")
                    if pdf_url:
                        log.debug(f"  Found PDF via Unpaywall (oa_locations.url_for_pdf): {doi}")
                        return pdf_url

                # Last resort: best_oa url (may be landing page)
                fallback_url = best_oa.get("url")
                if fallback_url:
                    log.debug(f"  Trying Unpaywall fallback URL (may be landing page): {doi}")
                    return fallback_url
        except Exception as e:
            log.debug(f"  Unpaywall lookup failed for {doi}: {e}")

    # FIX #3: New Strategy 3 - Europe PMC full-text PDF
    if pmcid:
        epmc_pdf = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        try:
            time.sleep(0.5)
            head_resp = requests.head(epmc_pdf, timeout=10, allow_redirects=True)
            ct = head_resp.headers.get("Content-Type", "")
            if head_resp.status_code == 200 and "pdf" in ct.lower():
                log.debug(f"  Found PDF via Europe PMC: {pmcid}")
                return epmc_pdf
            else:
                log.debug(f"  Europe PMC PDF not available for {pmcid} (status={head_resp.status_code}, type={ct})")
        except Exception as e:
            log.debug(f"  Europe PMC lookup failed for {pmcid}: {e}")

    # Strategy 4: Direct PMC PDF URL construction (last resort)
    if pmcid:
        clean_pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        direct_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{clean_pmcid}/pdf/"
        log.debug(f"  Trying direct PMC PDF URL (last resort): {direct_url}")
        return direct_url

    return None


# ──────────────────────────────────────────────
# Abstract Pre-Screening via LLM Relevance Gate
# ──────────────────────────────────────────────

def screen_abstract(abstract: str, title: str, topic_label: str,
                    model: str = "gpt-4.1-mini", logger=None) -> tuple[int, str]:
    """
    Run abstract through NORE relevance gate to determine if paper is worth downloading.

    Returns: (relevance_score, reasoning)
    """
    log = logger or logging.getLogger("harvester")

    if not abstract or len(abstract.strip()) < 50:
        log.debug(f"  Abstract too short for screening, auto-pass")
        return (7, "Abstract too short to screen; passing to avoid false negative")

    # Import the universal LLM adapter (same one mupdf_trainer uses)
    try:
        from llm_adapter import llm_chat_universal
    except ImportError:
        log.warning("  llm_adapter.py not found — skipping LLM screening, auto-pass")
        return (7, "LLM adapter not available; auto-pass")

    system_prompt = dedent(f"""\
    You are a content relevance screener for the NORE (Nutrition Oncology Reasoning Engine) project.
    You evaluate whether a paper abstract is relevant to the topic: {topic_label}.

    The paper must contain information useful for training an AI system that generates
    personalized nutrition plans for cancer patients. Focus on clinical applicability.

    Rate relevance 1-10:
    - 8-10: Highly relevant — directly addresses nutrition interventions, clinical outcomes,
            mechanisms, or guidelines in oncology context for this topic
    - 5-7:  Moderately relevant — related but tangential (e.g., general cancer biology
            without nutrition focus, or nutrition without cancer context)
    - 1-4:  Not relevant — wrong domain, animal-only study with no clinical translation,
            purely genetic/molecular without nutritional implications

    Return ONLY valid JSON: {{"relevance_score": <int>, "reasoning": "<one sentence>"}}
    """)

    user_prompt = f"TITLE: {title}\n\nABSTRACT:\n{abstract[:3000]}"

    try:
        response = llm_chat_universal(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        # Parse JSON response
        text = response.strip()
        # Remove markdown fences if present
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        data = json.loads(text)
        score = int(data.get("relevance_score", 5))
        reasoning = data.get("reasoning", "No reasoning provided")

        return (score, reasoning)

    except Exception as e:
        log.warning(f"  LLM screening failed: {e} — auto-pass with score 6")
        return (6, f"Screening error: {e}")


# ──────────────────────────────────────────────
# PDF Download & Save
# ──────────────────────────────────────────────

def download_pdf(url: str, filepath: Path, logger=None) -> bool:
    """Download PDF from URL to filepath. Returns True on success."""
    log = logger or logging.getLogger("harvester")

    try:
        time.sleep(PDF_DOWNLOAD_DELAY)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*",
        }
        with requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True) as resp:
            # Check if we actually got a PDF
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200:
                log.debug(f"  Download failed: HTTP {resp.status_code} from {url[:80]}")
                return False

            # Use a single iterator to avoid fragile multi-generator streaming
            chunks_iter = resp.iter_content(chunk_size=8192)

            first_bytes = b""
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                # Could be HTML error page — check first bytes
                first_bytes = next(chunks_iter, b"")
                if not first_bytes.startswith(b"%PDF"):
                    log.debug(f"  Not a PDF (content-type: {content_type})")
                    return False

            # Save the PDF
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                # If we already read first bytes in content-type check above
                if first_bytes:
                    f.write(first_bytes)
                for chunk in chunks_iter:
                    f.write(chunk)

            # Verify file size (PDFs should be > 10KB)
            size = filepath.stat().st_size
            if size < 10240:
                log.debug(f"  Downloaded file too small ({size} bytes), likely error page")
                filepath.unlink(missing_ok=True)
                return False

            log.info(f"  ✓ Saved: {filepath.name} ({size // 1024} KB)")
            return True

    except Exception as e:
        log.debug(f"  Download error: {e}")
        filepath.unlink(missing_ok=True)
        return False


def save_metadata_sidecar(article: dict, filepath: Path, topic: str,
                          relevance_score: int = None, relevance_reasoning: str = None,
                          pdf_url: str = None):
    """Save article metadata as JSON sidecar file alongside PDF."""
    sidecar_path = filepath.with_suffix(".meta.json")
    meta = {
        "doi": article.get("doi"),
        "pmid": article.get("pmid"),
        "pmcid": article.get("pmcid"),
        "title": article.get("title"),
        "authors": article.get("authors", [])[:10],
        "journal": article.get("journal"),
        "pub_date": article.get("pub_date"),
        "pub_types": article.get("pub_types", []),
        "topic": topic,
        "topic_label": TOPIC_QUERIES.get(topic, {}).get("label", topic),
        "relevance_score": relevance_score,
        "relevance_reasoning": relevance_reasoning,
        "harvested_at": datetime.now().isoformat(),
        "source_pdf_url": pdf_url,
    }
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2)


def extract_with_firecrawl(app, url: str, logger=None) -> str:
    """
    Extract text from a PDF URL using Firecrawl SDK (Nick's approach).
    Sends the URL to Firecrawl which parses the PDF server-side — no local download needed.

    Returns: markdown text string, or raises on failure.
    """
    log = logger or logging.getLogger("harvester")
    log.debug(f"  Calling Firecrawl: {url[:80]}...")

    data = app.scrape(
        url,
        only_main_content=FIRECRAWL_SETTINGS["only_main_content"],
        max_age=FIRECRAWL_SETTINGS["max_age"],
        parsers=FIRECRAWL_SETTINGS["parsers"],
        formats=FIRECRAWL_SETTINGS["formats"],
    )

    # Handle SDK response: v4+ returns a Pydantic Document with .markdown attribute
    markdown = None
    if hasattr(data, "markdown"):
        markdown = data.markdown
    elif isinstance(data, dict):
        markdown = data.get("markdown")
        if not markdown:
            nested = data.get("data", {})
            if isinstance(nested, dict):
                markdown = nested.get("markdown")

    if not markdown:
        raise ValueError(f"No markdown content returned from Firecrawl for {url[:60]}")

    return markdown


def save_firecrawl_output(md_path: Path, markdown: str, article: dict,
                          topic: str, relevance_score=None,
                          relevance_reasoning=None, pdf_url=None):
    """
    Save Firecrawl markdown with metadata header + JSON sidecar.
    The .firecrawl.md file is consumed directly by mupdf_trainer_v3.py (--text-source auto).
    """
    header = (
        f"<!-- FIRECRAWL EXTRACTION -->\n"
        f"<!-- doi: {article.get('doi', 'N/A')} -->\n"
        f"<!-- pmid: {article.get('pmid', 'N/A')} -->\n"
        f"<!-- title: {article.get('title', 'N/A')[:100]} -->\n"
        f"<!-- extracted: {datetime.now().isoformat()} -->\n"
        f"<!-- source_url: {pdf_url or 'N/A'} -->\n\n"
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + markdown)

    # Also save .meta.json sidecar (same format as PDF mode for compatibility)
    sidecar_path = md_path.with_suffix(".meta.json")
    meta = {
        "doi": article.get("doi"),
        "pmid": article.get("pmid"),
        "pmcid": article.get("pmcid"),
        "title": article.get("title"),
        "authors": article.get("authors", [])[:10],
        "journal": article.get("journal"),
        "pub_date": article.get("pub_date"),
        "pub_types": article.get("pub_types", []),
        "topic": topic,
        "topic_label": TOPIC_QUERIES.get(topic, {}).get("label", topic),
        "relevance_score": relevance_score,
        "relevance_reasoning": relevance_reasoning,
        "harvested_at": datetime.now().isoformat(),
        "source_pdf_url": pdf_url,
        "extraction_method": "firecrawl",
    }
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2)


def sanitize_filename(title: str, identifier: str, extension: str = ".pdf") -> str:
    """Create safe filename from title and DOI/PMID."""
    clean = re.sub(r'[^\w\s-]', '', title[:80])
    clean = re.sub(r'\s+', '_', clean.strip())
    if not clean:
        clean = "untitled"

    # Use hash of identifier for uniqueness
    id_hash = hashlib.md5(identifier.encode()).hexdigest()[:8]
    return f"{clean}_{id_hash}{extension}"


# ──────────────────────────────────────────────
# Main Harvest Pipeline
# ──────────────────────────────────────────────

def harvest_topic(topic_key: str, email: str, output_dir: Path,
                  tracker: ProgressTracker, max_papers: int = 500,
                  screen: bool = True, screen_model: str = "gpt-4.1-mini",
                  min_date: str = None, max_date: str = None,
                  api_key: str = None, logger=None,
                  download_pdfs: bool = False, firecrawl_app=None):
    """
    Full harvest pipeline for a single topic:
    1. Search PMC + Europe PMC
    2. Deduplicate candidates
    3. Screen abstracts (optional)
    4. Extract text via Firecrawl (default) OR download PDFs (--download-pdfs)
    """
    log = logger or logging.getLogger("harvester")
    topic = TOPIC_QUERIES[topic_key]
    topic_dir = output_dir / topic_key

    log.info(f"\n{'='*60}")
    log.info(f"TOPIC: {topic['label']} (priority {topic['priority']})")
    log.info(f"{'='*60}")

    # ── Phase 1: Discover candidates ──
    log.info("Phase 1: Discovering candidate papers...")
    all_candidates = {}  # keyed by DOI or PMID to deduplicate

    # Search PubMed via E-utilities
    for i, query in enumerate(topic["queries"]):
        log.info(f"  PubMed query {i+1}/{len(topic['queries'])}")
        try:
            results = search_pmc(
                query=query, email=email, max_results=200,
                api_key=api_key, min_date=min_date, max_date=max_date,
                logger=log,
            )
            for article in results:
                key = article.get("doi") or article.get("pmid") or ""
                if key and key not in all_candidates:
                    all_candidates[key] = article
        except Exception as e:
            log.warning(f"  PubMed search failed: {e}")

    # Search Europe PMC
    for i, query in enumerate(topic.get("europe_pmc_queries", [])):
        log.info(f"  Europe PMC query {i+1}/{len(topic.get('europe_pmc_queries', []))}")
        try:
            results = search_europe_pmc(query=query, max_results=100, logger=log)
            for article in results:
                key = article.get("doi") or article.get("pmid") or ""
                if key and key not in all_candidates:
                    all_candidates[key] = article
        except Exception as e:
            log.warning(f"  Europe PMC search failed: {e}")

    log.info(f"  Total unique candidates: {len(all_candidates)}")

    # ── Phase 2: Filter already-processed ──
    candidates = []
    for key, article in all_candidates.items():
        doi = article.get("doi")
        pmid = article.get("pmid")
        if tracker.is_processed(doi=doi, pmid=pmid):
            continue
        if tracker.is_screened_out(doi=doi, pmid=pmid):
            continue
        candidates.append(article)

    log.info(f"  After filtering already-processed: {len(candidates)} remaining")

    # ── Phase 3: Screen abstracts ──
    screened = []
    if screen:
        log.info("Phase 2: Screening abstracts via LLM relevance gate...")
        for i, article in enumerate(candidates):
            if len(screened) >= max_papers:
                break

            title = article.get("title", "Untitled")
            abstract = article.get("abstract", "")
            identifier = article.get("doi") or article.get("pmid") or str(i)

            score, reasoning = screen_abstract(
                abstract=abstract, title=title,
                topic_label=topic["label"], model=screen_model, logger=log,
            )

            if score >= MIN_RELEVANCE_SCORE:
                article["_relevance_score"] = score
                article["_relevance_reasoning"] = reasoning
                screened.append(article)
                log.debug(f"  ✓ [{score}/10] {title[:60]}...")
            else:
                tracker.mark_screened_out(
                    identifier, topic=topic_key,
                    doi=article.get("doi"), pmid=article.get("pmid"),
                )
                log.debug(f"  ✗ [{score}/10] {title[:60]}... — {reasoning}")

            # Save progress periodically
            if (i + 1) % 50 == 0:
                tracker.save()
                log.info(f"  Screened {i+1}/{len(candidates)}: {len(screened)} passed")

        log.info(f"  Screening complete: {len(screened)}/{len(candidates)} passed (score ≥ {MIN_RELEVANCE_SCORE})")
    else:
        screened = candidates[:max_papers]
        for article in screened:
            article["_relevance_score"] = None
            article["_relevance_reasoning"] = "Screening disabled"
        log.info(f"  Screening disabled — using all {len(screened)} candidates")

    # ── Phase 4: Acquire paper text ──
    if download_pdfs:
        # ── Mode A: Download PDFs locally (for RLHF traceability) ──
        log.info(f"Phase 3: Downloading PDFs for {len(screened)} papers...")
    else:
        # ── Mode B: Extract text via Firecrawl (default — no local PDFs) ──
        log.info(f"Phase 3: Extracting text via Firecrawl for {len(screened)} papers...")

    acquired = 0

    for i, article in enumerate(screened):
        doi = article.get("doi")
        pmid = article.get("pmid")
        pmcid = article.get("pmcid")
        title = article.get("title", "Untitled")
        identifier = doi or pmid or f"unknown_{i}"

        # Find PDF URL (needed for both modes)
        pdf_url = find_pdf_url(doi=doi, pmcid=pmcid, email=email, logger=log)

        if not pdf_url:
            log.debug(f"  No OA PDF found for: {title[:60]}...")
            tracker.mark_error(identifier, topic=topic_key)
            continue

        if download_pdfs:
            # ── Mode A: Download PDF ──
            filename = sanitize_filename(title, identifier, extension=".pdf")
            filepath = topic_dir / filename

            if filepath.exists():
                log.debug(f"  Already exists: {filename}")
                tracker.mark_processed(doi=doi, pmid=pmid, topic=topic_key)
                acquired += 1
                continue

            success = download_pdf(pdf_url, filepath, logger=log)

            # If download failed and we have a PMCID, try Europe PMC as fallback
            if not success and pmcid:
                epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                if epmc_url != pdf_url:
                    log.debug(f"  Retrying via Europe PMC fallback: {pmcid}")
                    success = download_pdf(epmc_url, filepath, logger=log)

            if success:
                save_metadata_sidecar(
                    article, filepath, topic=topic_key,
                    relevance_score=article.get("_relevance_score"),
                    relevance_reasoning=article.get("_relevance_reasoning"),
                    pdf_url=pdf_url,
                )
                tracker.mark_processed(doi=doi, pmid=pmid, topic=topic_key)
                acquired += 1
            else:
                tracker.mark_error(identifier, topic=topic_key)

        else:
            # ── Mode B: Firecrawl extraction (default) ──
            filename = sanitize_filename(title, identifier, extension=".firecrawl.md")
            md_path = topic_dir / filename

            if md_path.exists():
                log.debug(f"  Already exists: {filename}")
                tracker.mark_processed(doi=doi, pmid=pmid, topic=topic_key)
                acquired += 1
                continue

            try:
                markdown = extract_with_firecrawl(firecrawl_app, pdf_url, logger=log)

                if len(markdown.strip()) < 200:
                    log.warning(f"  Firecrawl returned very short content ({len(markdown)} chars)")

                save_firecrawl_output(
                    md_path, markdown, article, topic=topic_key,
                    relevance_score=article.get("_relevance_score"),
                    relevance_reasoning=article.get("_relevance_reasoning"),
                    pdf_url=pdf_url,
                )

                word_count = len(markdown.split())
                log.info(f"  Saved: {md_path.name} ({word_count:,} words)")
                tracker.mark_processed(doi=doi, pmid=pmid, topic=topic_key)
                acquired += 1

            except Exception as e:
                log.error(f"  Firecrawl extraction failed for {identifier}: {str(e)[:200]}")
                tracker.mark_error(identifier, topic=topic_key)

            time.sleep(FIRECRAWL_DELAY)

        # Save progress periodically
        if (i + 1) % 10 == 0:
            tracker.save()
            mode_label = "downloaded" if download_pdfs else "extracted"
            log.info(f"  Progress: {acquired} {mode_label}, {i+1}/{len(screened)} attempted")

    tracker.save()
    mode_label = "PDFs downloaded" if download_pdfs else "texts extracted via Firecrawl"
    log.info(f"  Topic complete: {acquired} {mode_label}")
    return acquired


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NORE Multi-Source Paper Harvester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
        Examples:
          # Harvest all topics (default: Firecrawl text extraction, no PDFs)
          python nore_paper_harvester.py --email you@example.com

          # Download PDFs locally (for RLHF traceability)
          python nore_paper_harvester.py --email you@example.com --download-pdfs

          # Single topic
          python nore_paper_harvester.py --email you@example.com --topic drug_nutrient

          # Resume interrupted run
          python nore_paper_harvester.py --email you@example.com --resume

          # No LLM screening (faster, more papers, less targeted)
          python nore_paper_harvester.py --email you@example.com --no-screen
        """)
    )

    global MIN_RELEVANCE_SCORE

    parser.add_argument("--email", required=True,
                        help="Email for NCBI/Unpaywall API (required, not stored)")
    parser.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY"),
                        help="NCBI API key for faster rate limits (env: NCBI_API_KEY)")
    parser.add_argument("--topic", choices=list(TOPIC_QUERIES.keys()),
                        help="Harvest single topic (default: all)")
    parser.add_argument("--max-per-topic", type=int, default=DEFAULT_MAX_PER_TOPIC,
                        help=f"Max PDFs per topic (default: {DEFAULT_MAX_PER_TOPIC})")
    parser.add_argument("--years-back", type=int, default=DEFAULT_YEARS_BACK,
                        help=f"How many years back to search (default: {DEFAULT_YEARS_BACK})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--no-screen", action="store_true",
                        help="Skip LLM abstract screening (download everything)")
    parser.add_argument("--screen-model", default="gpt-4.1-mini",
                        help="LLM model for abstract screening (default: gpt-4.1-mini)")
    parser.add_argument("--min-relevance", type=int, default=MIN_RELEVANCE_SCORE,
                        help=f"Minimum relevance score to download (default: {MIN_RELEVANCE_SCORE})")
    parser.add_argument("--download-pdfs", action="store_true",
                        help="Download PDFs locally instead of using Firecrawl text extraction "
                             "(useful for RLHF traceability). Default: use Firecrawl.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir, verbose=args.verbose)

    # Override minimum relevance if specified
    MIN_RELEVANCE_SCORE = args.min_relevance

    # Date range
    max_date = datetime.now().strftime("%Y/%m/%d")
    min_date = (datetime.now() - timedelta(days=args.years_back * 365)).strftime("%Y/%m/%d")

    # Initialize Firecrawl if using default mode
    firecrawl_app = None
    if not args.download_pdfs:
        if not _FIRECRAWL_AVAILABLE:
            log.error("Firecrawl SDK not installed. Run: pip install firecrawl")
            log.error("Or use --download-pdfs to download PDFs locally instead.")
            return 1
        if not FIRECRAWL_API_KEY:
            log.error("FIRECRAWL_API_KEY environment variable not set.")
            log.error("Run: export FIRECRAWL_API_KEY=your_key_here")
            log.error("Or use --download-pdfs to download PDFs locally instead.")
            return 1
        firecrawl_app = Firecrawl(api_key=FIRECRAWL_API_KEY)
        log.info("Firecrawl SDK initialized")

    mode_str = "PDF download" if args.download_pdfs else "Firecrawl text extraction"
    log.info(f"NORE Paper Harvester starting")
    log.info(f"Mode: {mode_str}")
    log.info(f"Date range: {min_date} to {max_date}")
    log.info(f"Output: {output_dir.absolute()}")
    log.info(f"Screening: {'disabled' if args.no_screen else f'enabled (model: {args.screen_model}, min score: {MIN_RELEVANCE_SCORE})'}")

    # Progress tracker
    tracker = ProgressTracker(output_dir)
    if args.resume:
        log.info(f"Resuming: {tracker.total_downloaded} papers already processed")

    # Determine topics to harvest
    topics = [args.topic] if args.topic else sorted(
        TOPIC_QUERIES.keys(),
        key=lambda t: TOPIC_QUERIES[t]["priority"]
    )

    # Run harvest
    total = 0
    try:
        for topic_key in topics:
            count = harvest_topic(
                topic_key=topic_key,
                email=args.email,
                output_dir=output_dir,
                tracker=tracker,
                max_papers=args.max_per_topic,
                screen=not args.no_screen,
                screen_model=args.screen_model,
                min_date=min_date,
                max_date=max_date,
                api_key=args.ncbi_api_key,
                logger=log,
                download_pdfs=args.download_pdfs,
                firecrawl_app=firecrawl_app,
            )
            total += count

    except KeyboardInterrupt:
        log.info("\nInterrupted — saving progress...")

    finally:
        tracker.save()
        summary = tracker.get_summary()
        log.info(f"\n{summary}")

        # Also save summary as file
        with open(output_dir / "harvest_summary.txt", "w") as f:
            f.write(summary)

    return 0


if __name__ == "__main__":
    exit(main())
