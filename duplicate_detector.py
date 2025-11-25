#!/usr/bin/env python3
"""
Duplicate Question Detector - Phase 2
Finds similar/duplicate questions across multiple Q/A files

Learning objectives:
- String similarity algorithms
- Working with multiple files
- Data structures (sets, dictionaries)
- Text normalization
"""

import json
from pathlib import Path
from difflib import SequenceMatcher


def normalize_question(question: str) -> str:
    """
    Normalize a question for comparison.

    TODO: Implement this function
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    - Return cleaned string

    Hints:
    - Use .lower(), .strip()
    - Consider string.punctuation
    """
    # YOUR CODE HERE
    pass


def calculate_similarity(q1: str, q2: str) -> float:
    """
    Calculate similarity between two questions (0.0 to 1.0).

    TODO: Implement this function using SequenceMatcher
    - Normalize both questions first
    - Use SequenceMatcher.ratio()
    - Return similarity score

    Hint: SequenceMatcher(None, str1, str2).ratio()
    """
    # YOUR CODE HERE
    pass


def load_all_questions(folder: str) -> list[dict]:
    """
    Load all Q/A pairs from all JSONL files in a folder.

    TODO: Implement this function
    - Use Path.glob("*.jsonl") to find all JSONL files
    - Load questions from each file
    - Add "source_file" to each question dict
    - Return combined list
    """
    questions = []
    # YOUR CODE HERE
    return questions


def find_duplicates(questions: list[dict], threshold: float = 0.85) -> list[tuple]:
    """
    Find duplicate or very similar questions.

    TODO: Implement this function
    - Compare each question with every other question
    - If similarity > threshold, record as duplicate
    - Return list of (question1, question2, similarity_score) tuples
    - Avoid comparing a question with itself!

    Challenge: Make this efficient (hint: only compare i with j where j > i)
    """
    duplicates = []
    # YOUR CODE HERE
    return duplicates


def print_duplicates(duplicates: list[tuple]):
    """Pretty print duplicate questions."""
    if not duplicates:
        print("No duplicates found!")
        return

    print(f"\nFound {len(duplicates)} potential duplicates:\n")
    for i, (q1, q2, score) in enumerate(duplicates, 1):
        print(f"--- Duplicate {i} (similarity: {score:.1%}) ---")
        print(f"Q1: {q1['question']}")
        print(f"    (from {q1['source_file']})")
        print(f"Q2: {q2['question']}")
        print(f"    (from {q2['source_file']})")
        print()


def main():
    """Main duplicate detection."""
    print("=" * 60)
    print("Duplicate Question Detector")
    print("=" * 60)

    # TODO: Get folder path from user
    folder = input("Enter folder with JSONL files (default: qa_jsonl): ").strip()
    if not folder:
        folder = "qa_jsonl"

    # TODO: Get similarity threshold
    threshold = float(input("Similarity threshold (0.0-1.0, default 0.85): ") or "0.85")

    print(f"\nLoading questions from {folder}...")
    questions = load_all_questions(folder)
    print(f"Loaded {len(questions)} questions")

    print("Finding duplicates...")
    duplicates = find_duplicates(questions, threshold)

    print_duplicates(duplicates)

    # TODO: Optional - save duplicates to file
    # save_path = "duplicates_report.txt"
    # with open(save_path, "w") as f:
    #     for q1, q2, score in duplicates:
    #         f.write(f"Similarity: {score:.1%}\n")
    #         f.write(f"Q1: {q1['question']}\n")
    #         f.write(f"Q2: {q2['question']}\n\n")


if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Phase 2):
# 1. Use Levenshtein distance instead of SequenceMatcher
# 2. Group duplicates into clusters (transitivity: if A~B and B~C, then A~B~C)
# 3. Add statistics: avg similarity, most duplicated question
# 4. Create a "deduplicated" JSONL output file
