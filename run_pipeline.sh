#!/bin/bash
# Quick run script for mupdf_trainer_v2.py
# Assumes virtual environment is already set up

set -e

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Run setup_and_run.sh first."
    exit 1
fi

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: No .env file found. Make sure environment variables are set."
fi

# Default values
PDF_PATH="${1:-./my_pdfs}"
MODEL="${2:-meta-llama/Llama-3.3-70B-Instruct-Turbo}"
QA_K="${3:-1}"
RPM="${4:-20}"

echo "=== Running PDF Training Pipeline ==="
echo "PDF Path: $PDF_PATH"
echo "Model: $MODEL"
echo "Q/A pairs per chunk: $QA_K"
echo "Rate limit: $RPM requests/min"
echo ""

python mupdf_trainer_v2.py "$PDF_PATH" \
    --gen_qa \
    --llm_model "$MODEL" \
    --qa_k "$QA_K" \
    --rpm "$RPM" \
    --pretty_json

echo ""
echo "=== Pipeline complete! ==="
echo "Check the qa_jsonl folder for output files and summaries."
