#!/usr/bin/env python3
"""
Interactive Quiz Game - Phase 1
Loads Q/A pairs from JSONL and quizzes the user

Learning objectives:
- File I/O with JSONL
- Working with dictionaries
- User input handling
- Basic scoring
"""

import json
import random
from pathlib import Path


def load_questions(jsonl_file: str) -> list[dict]:
    """
    Load Q/A pairs from JSONL file.

    TODO: Implement this function
    - Open the file
    - Read each line
    - Parse JSON
    - Filter for question types you want (freeform, mcq, reasoning)
    - Return a list of question dictionaries

    Hints:
    - Use json.loads() to parse each line
    - Each line is a separate JSON object
    """
    questions = []
    # YOUR CODE HERE
    return questions


def ask_question(qa: dict, question_num: int, total: int) -> bool:
    """
    Ask a single question and check the answer.

    TODO: Implement this function
    - Display the question number and total
    - Show the question
    - Get user's answer
    - Compare with correct answer (case-insensitive)
    - Return True if correct, False if wrong

    For MCQs: Display options and accept A/B/C/D
    For other types: Accept free-form text
    """
    # YOUR CODE HERE
    pass


def calculate_score(correct: int, total: int) -> tuple[int, str]:
    """
    Calculate percentage and grade.

    TODO: Implement this function
    - Calculate percentage
    - Assign letter grade (A: 90+, B: 80-89, C: 70-79, D: 60-69, F: <60)
    - Return (percentage, grade)
    """
    # YOUR CODE HERE
    pass


def main():
    """Main quiz game loop."""
    print("=" * 60)
    print("PDF Q/A Quiz Game")
    print("=" * 60)

    # TODO: Get JSONL file from user or use default
    jsonl_file = input("Enter path to Q/A file (or press Enter for default): ").strip()
    if not jsonl_file:
        jsonl_file = "qa_jsonl/example_qa.jsonl"

    # TODO: Load questions
    questions = load_questions(jsonl_file)

    if not questions:
        print("No questions found!")
        return

    # TODO: Ask how many questions to quiz
    num_questions = int(input(f"How many questions? (1-{len(questions)}): "))
    num_questions = min(num_questions, len(questions))

    # TODO: Randomly select questions
    selected = random.sample(questions, num_questions)

    # TODO: Quiz loop
    correct = 0
    for i, qa in enumerate(selected, 1):
        if ask_question(qa, i, num_questions):
            correct += 1
            print("✓ Correct!\n")
        else:
            print("✗ Incorrect\n")

    # TODO: Calculate and display final score
    percentage, grade = calculate_score(correct, num_questions)
    print("=" * 60)
    print(f"Final Score: {correct}/{num_questions} ({percentage}%)")
    print(f"Grade: {grade}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Phase 1):
# 1. Add a timer for each question
# 2. Show the correct answer after wrong answers
# 3. Add difficulty filtering (only show "easy" questions)
# 4. Save quiz results to a file
