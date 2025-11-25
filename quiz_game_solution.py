#!/usr/bin/env python3
"""
Interactive Quiz Game - SOLUTION (Phase 1)

This is a complete solution to help you if you get stuck.
Try to implement it yourself first!
"""

import json
import random
from pathlib import Path


def load_questions(jsonl_file: str) -> list[dict]:
    """Load Q/A pairs from JSONL file."""
    questions = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa = json.loads(line)
                    questions.append(qa)
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found!")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

    return questions


def ask_question(qa: dict, question_num: int, total: int) -> bool:
    """Ask a single question and check the answer."""
    print(f"\n{'='*60}")
    print(f"Question {question_num}/{total}")
    print(f"{'='*60}")
    print(f"Type: {qa.get('type', 'unknown').upper()}")
    if 'difficulty' in qa:
        print(f"Difficulty: {qa['difficulty']}")
    print()
    print(qa['question'])
    print()

    # Handle MCQ questions
    if qa['type'] == 'mcq':
        options = qa.get('options', [])
        for i, option in enumerate(options):
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            print(f"  {letter}) {option}")

        user_answer = input("\nYour answer (A/B/C/D): ").strip().upper()

        # Validate input
        while user_answer not in ['A', 'B', 'C', 'D']:
            user_answer = input("Please enter A, B, C, or D: ").strip().upper()

        correct_index = qa['correct_index']
        correct_letter = chr(65 + correct_index)

        is_correct = (user_answer == correct_letter)

        if not is_correct:
            print(f"\nThe correct answer was: {correct_letter}) {options[correct_index]}")

        return is_correct

    # Handle reasoning/freeform questions
    else:
        user_answer = input("\nYour answer: ").strip()

        # Get the correct answer
        correct_answer = qa.get('answer') or qa.get('answer_key', '')

        # Simple check: see if key terms are present
        # This is basic - you could make it smarter!
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()

        # Check if answer is similar (contains key words)
        is_correct = user_lower in correct_lower or correct_lower in user_lower

        if not is_correct:
            print(f"\nExpected answer: {correct_answer}")

        return is_correct


def calculate_score(correct: int, total: int) -> tuple[int, str]:
    """Calculate percentage and letter grade."""
    if total == 0:
        return 0, "N/A"

    percentage = round((correct / total) * 100)

    # Assign letter grade
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
    print(" " * 20 + "PDF Q/A QUIZ GAME")
    print("=" * 60)
    print()

    # Get JSONL file from user
    jsonl_file = input("Enter path to Q/A file (or press Enter for default): ").strip()
    if not jsonl_file:
        # Default to any JSONL file in qa_jsonl folder
        folder = Path("qa_jsonl")
        if folder.exists():
            jsonl_files = list(folder.glob("*.jsonl"))
            if jsonl_files:
                jsonl_file = str(jsonl_files[0])
                print(f"Using: {jsonl_file}")
            else:
                print("No JSONL files found in qa_jsonl folder!")
                return
        else:
            print("qa_jsonl folder not found!")
            return

    # Load questions
    print("\nLoading questions...")
    questions = load_questions(jsonl_file)

    if not questions:
        print("No questions found or error loading file!")
        return

    print(f"Loaded {len(questions)} questions")

    # Ask how many questions
    while True:
        try:
            num_str = input(f"\nHow many questions? (1-{len(questions)}, Enter for all): ").strip()
            if not num_str:
                num_questions = len(questions)
            else:
                num_questions = int(num_str)

            if 1 <= num_questions <= len(questions):
                break
            else:
                print(f"Please enter a number between 1 and {len(questions)}")
        except ValueError:
            print("Please enter a valid number")

    # Randomly select questions
    selected = random.sample(questions, num_questions)

    # Quiz loop
    correct = 0
    for i, qa in enumerate(selected, 1):
        if ask_question(qa, i, num_questions):
            correct += 1
            print("✓ CORRECT!")
        else:
            print("✗ INCORRECT")

        # Pause between questions
        if i < num_questions:
            input("\nPress Enter for next question...")

    # Calculate and display final score
    percentage, grade = calculate_score(correct, num_questions)

    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL RESULTS")
    print("=" * 60)
    print(f"\nScore: {correct}/{num_questions}")
    print(f"Percentage: {percentage}%")
    print(f"Grade: {grade}")
    print("\n" + "=" * 60)

    # Give feedback
    if percentage >= 90:
        print("Excellent work! 🌟")
    elif percentage >= 70:
        print("Good job! 👍")
    elif percentage >= 50:
        print("Not bad, but room for improvement!")
    else:
        print("Keep studying!")


if __name__ == "__main__":
    main()
