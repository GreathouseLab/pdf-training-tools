#reads through a PDF, extracts text, chunks it, and for each chunk,
#asks an LLM to generate Q/A pairs, then embeds those using DeepSeek R1
#and stores them in Pinecone.

#function process_pdf_folder(folder_path):
 #   initialize total_pages = 0
  #  for each PDF in folder:
   #     extract pages from PDF
    #    for each page:
     #       extract text
      #      embed text using DeepSeek R1
       #     store embedding in Pinecone
        #    total_pages += 1
    #return total_pages

from pathlib import Path
from typing import List
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pinecone
import argparse
import time
import sqlite3
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import json
import re
from pydantic import BaseModel
from deepseek import DeepSeek
from deepseek import DeepSeekR1
from pinecone import PineconeClient
from typing import Tuple 
from typing import List

#-----------SETUP----------------
load_dotenv()  # Load environment variables from .env file
import logging
logging.basicConfig(level=logging.INFO) # set logging level to INFO
import sys
logging.getLogger("openai").setLevel(logging.WARNING) # set openai logging to WARNING
logging.getLogger("pinecone").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are a careful scientific assistant.
 Given a passage from a paper, propose high-quality question/answer pairs that test understanding.
Prefer precise, factual Q/A over trivia. Keep answers concise but specific."""
USER_PROMPT_TMPL = """Create {k} high-quality Q/A pairs from this passage.\n    
Return as JSON list of objects with 'question' and 'answer'.\n
Passage:\n---\n{passage}\n---"""
#------------Start of main function----------------
def main():
    parser = argparse.ArgumentParser(description="Process PDFs to generate and store Q/A pairs.")
    parser.add_argument("pdf_folder", type=str, help="Path to folder containing PDF files.")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "gpt-5-mini"), help="LLM model to use.")
    parser.add_argument("--k", type=int, default=3, help="Number of Q/A pairs to generate per chunk.")
    parser.add_argument("--chunk_size", type=int, default=800, help="Max tokens per text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Token overlap between chunks.")
    parser.add_argument("--embed_model", type=str, default="deepseek-r1", help="Embedding model to use.")
    parser.add_argument("--pinecone_index", type=str, default=os.getenv("PINECONE_INDEX", "paper-qa"), help="Pinecone index name.")
    parser.add_argument("--pinecone_namespace", type=str, default="paper-qa-namespace", help="Pinecone namespace.")
    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    if not pdf_folder.is_dir():
        logging.error(f"Provided path {pdf_folder} is not a directory.")
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ds = DeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY"))
    embedder = DeepSeekR1(ds)
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        logging.error("Pinecone API key or environment not set in .env")
        sys.exit(1)
    
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if args.pinecone_index not in pinecone.list_indexes():
        logging.info(f"Creating Pinecone index: {args.pinecone_index}")
        pinecone.create_index(args.pinecone_index, dimension=1024)  # assuming DeepSeek R1 dim=1024
        time.sleep(10)  # wait for index to be ready
    pclient = PineconeClient()
    index = pclient.Index(args.pinecone_index)
#-----------EXTRACTION FUNCTION----------------
def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """
    Return a tuple: (raw_text, num_pages).
    Postcondition: len(raw_text) >=0 and num_pages == len(reader.page)
    """
    reader = PdfReader(pdf_path)
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    raw_text = "\n".join(page_texts)
    num_pages = len(reader.pages)
# Sanity check: Ensure extraction matches actual number of pages
    if len(page_texts) != num_pages:
        logging.warning(f"Sanity check failed: collected {len(page_texts)} pages, but PdfReader reports {num_pages} pages for {pdf_path}")
    else:
        logging.info(f"Sanity check passed: Extracted {num_pages} pages from {pdf_path}")
    return raw_text, num_pages
#If 1:
    #print (…)
#-----------CLEANING FUNCTION----------------
def clean_text(text: str) -> str: 
    text = text.replace("\t", " ")
    text = " ".join(text.split()) 
    return text.strip()

def extract_and_clean(pdf_path: str) -> Tuple[str, int]:
    """
    Glue function that ensures the parameter `text` to clean_text(...)
    is exactly the extracted string from the PDF.
    """
    raw_text, num_pages = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)   
    return cleaned_text, num_pages
#-----------CHUNKING FUNCTION----------------
def chunk_text_words(
    text: str,
    max_words: int = 800,
    overlap_words: int = 50
) -> List[str]:
    """
    Split `text` into overlapping windows measured in WORDS,
    preserving word boundaries.
    """
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if not (0 <= overlap_words < max_words):
        raise ValueError("Require 0 <= overlap_words < max_words")

    words = text.split()
    step = max_words - overlap_words
    chunks: List[str] = []

    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += step

    return chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaned, n_pages = extract_and_clean("paper.pdf")
    print(f"Pages: {n_pages}")
    print(cleaned[:500])  # preview