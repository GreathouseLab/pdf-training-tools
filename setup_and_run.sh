#!/bin/bash
# Setup script for mupdf_trainer_v2.py
# Usage: bash setup_and_run.sh

set -e  # Exit on error

echo "=== Setting up virtual environment ==="
python -m venv .venv
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Environment variables ==="
echo "Please set the following in your .env file or export them:"
echo "  OPENAI_API_KEY=your_key_here"
echo "  DEEPSEEK_API_KEY=your_key_here"
echo "  PINECONE_API_KEY=your_key_here"
echo "  PINECONE_ENV=us-west1-gcp"
echo "  TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo"
echo "  TOGETHER_API_KEY=your_key_here"
echo "  PINECONE_INDEX=paper-qa"
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "✓ Found .env file, loading environment variables..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠ No .env file found. Create one with the variables above."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the pipeline, use:"
echo "  python mupdf_trainer_v2.py ./my_pdfs --gen_qa --llm_model \"meta-llama/Llama-3.3-70B-Instruct-Turbo\" --qa_k 1 --rpm 20"
