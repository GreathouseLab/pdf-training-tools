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


def load_questions(jsonl_file: str) -> list[dict]:
    """
    Load Q/A pairs from JSONL file.

    Opens the file, reads each line, parses JSON, and filters for
    question types (freeform, mcq, reasoning).
    Returns a list of question dictionaries.

    Supports both standard JSONL (one JSON per line) and
    multi-line JSON separated by '---'.
    """
    questions = []
    with open(jsonl_file, 'r') as f:
        content = f.read()

    # Check if file uses '---' separator (multi-line JSON format)
    if '---' in content:
        # Split by separator and parse each block
        json_blocks = content.split('---')
        for block_num, block in enumerate(json_blocks, 1):
            block = block.strip()
            if not block:
                continue
            try:
                qa = json.loads(block)
                # Skip if not a dictionary
                if not isinstance(qa, dict):
                    continue
                # Filter for desired question types
                if qa.get('type') in {'freeform', 'mcq', 'reasoning'}:
                    questions.append(qa)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON in block {block_num}: {e}")
                continue
    else:
        # Standard JSONL: one JSON object per line
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                qa = json.loads(line)
                # Skip if not a dictionary
                if not isinstance(qa, dict):
                    continue
                # Filter for desired question types
                if qa.get('type') in {'freeform', 'mcq', 'reasoning'}:
                    questions.append(qa)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

    return questions


def ask_question(qa: dict, question_num: int, total: int) -> bool:
    """
    Ask a single question and check the answer.

    Displays question number and total, shows the question, gets user's answer,
    and compares with correct answer (case-insensitive).

    For MCQs: Display options and accept A/B/C/D
    For other types: Accept free-form text

    Returns True if correct, False if wrong.
    """
    # show progress
    print(f"\nQuestion {question_num}/{total}:")
    print(qa["question"])

    # multiple-choice question
    if qa.get("type") == "mcq":
        options = qa["options"]
        # print options with labels A, B, C, ...
        for idx, option in enumerate(options):
            label = chr(ord("A") + idx)
            print(f"  {label}. {option}")

        user_answer = input("Your answer (letter): ").strip().upper()

        correct_index = qa["correct_index"]
        correct_label = chr(ord("A") + correct_index)

        return user_answer == correct_label

    # freeform / reasoning / other text answers
    else:
        user_answer = input("Your answer: ").strip()
        correct_answer = qa.get("answer_key") or qa.get("answer", "").strip()
        return user_answer.lower() == correct_answer.lower()


def calculate_score(correct: int, total: int) -> tuple[int, str]:
    """
    Calculate percentage and grade.

    Calculates percentage and assigns letter grade:
    - A: 90+
    - B: 80-89
    - C: 70-79
    - D: 60-69
    - F: <60

    Returns (percentage, grade).
    """
    percentage = round((correct / total) * 100)
    if percentage >= 90:
        grade = "A"
    elif percentage >= 80:
        grade = "B"
    elif percentage >= 70:
        grade = "C"
    elif percentage >= 60:
        grade = "D"
    else:
        grade = "F"
    return percentage, grade


def main():
    """Main quiz game loop."""
    print("=" * 60)
    print("PDF Q/A Quiz Game")
    print("=" * 60)

    # Get JSONL file from user or use default
    jsonl_file = input("Enter path to Q/A file (or press Enter for default): ").strip()
    if not jsonl_file:
        jsonl_file = "qa_jsonl/example_qa.jsonl"

    # Load questions
    questions = load_questions(jsonl_file)

    if not questions:
        print("No questions found!")
        return

    # Ask how many questions to quiz
    num_questions = int(input(f"How many questions? (1-{len(questions)}): "))
    num_questions = min(num_questions, len(questions))

    # Randomly select questions
    selected = random.sample(questions, num_questions)

    # Quiz loop
    correct = 0
    for i, qa in enumerate(selected, 1):
        if ask_question(qa, i, num_questions):
            correct += 1
            print("✓ Correct!\n")
        else:
            print("✗ Incorrect\n")

    # Calculate and display final score
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
