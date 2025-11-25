#!/usr/bin/env python3
"""
Q/A Quality Analyzer - Phase 3 (Object-Oriented Programming)
Analyzes quality metrics across your Q/A dataset

Learning objectives:
- Object-oriented programming (classes)
- Data aggregation and statistics
- Working with multiple data sources
- Creating reports
"""

from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict
import json
import statistics


class QADataset:
    """
    Represents a collection of Q/A pairs with analysis methods.

    TODO: Implement this class
    This teaches you Object-Oriented Programming!
    """

    def __init__(self, name: str):
        """
        Initialize the dataset.

        TODO: Set up instance variables
        - self.name = name
        - self.questions = []  # list of question dicts
        - self.stats = {}  # will hold computed statistics
        """
        # YOUR CODE HERE
        pass

    def load_from_file(self, jsonl_path: str):
        """
        Load Q/A pairs from a JSONL file.

        TODO: Implement this method
        - Open and read the JSONL file
        - Parse each line as JSON
        - Append to self.questions
        """
        # YOUR CODE HERE
        pass

    def load_from_folder(self, folder_path: str):
        """
        Load all JSONL files from a folder.

        TODO: Implement this method
        - Use Path.glob("*.jsonl")
        - Call load_from_file() for each file
        """
        # YOUR CODE HERE
        pass

    def count_by_type(self) -> Dict[str, int]:
        """
        Count questions by type (mcq, reasoning, freeform).

        TODO: Implement this method
        - Use Counter or a dict
        - Iterate through self.questions
        - Count each type
        - Return the counts
        """
        # YOUR CODE HERE
        pass

    def count_by_model(self) -> Dict[str, int]:
        """
        Count questions by which model generated them.

        TODO: Implement this method
        - Similar to count_by_type()
        - Group by "model" field
        """
        # YOUR CODE HERE
        pass

    def count_by_difficulty(self) -> Dict[str, int]:
        """
        Count questions by difficulty level.

        TODO: Implement this method
        - Group by "difficulty" field
        - Handle cases where difficulty might be missing
        """
        # YOUR CODE HERE
        pass

    def analyze_question_lengths(self) -> Dict[str, float]:
        """
        Compute statistics about question lengths.

        TODO: Implement this method
        - Calculate lengths of all questions (in words or characters)
        - Compute: min, max, mean, median
        - Return as dict

        Hint: Use statistics.mean(), statistics.median()
        """
        # YOUR CODE HERE
        pass

    def generate_report(self) -> str:
        """
        Generate a text report with all statistics.

        TODO: Implement this method
        - Call all the analysis methods
        - Format results nicely
        - Return as string

        This is great practice for string formatting!
        """
        report = f"Dataset: {self.name}\n"
        report += "=" * 60 + "\n"

        # Add total count
        report += f"Total questions: {len(self.questions)}\n\n"

        # Add type breakdown
        types = self.count_by_type()
        report += "By Type:\n"
        for qtype, count in types.items():
            report += f"  {qtype}: {count}\n"

        # TODO: Add more sections
        # - By model
        # - By difficulty
        # - Question length stats

        return report

    def save_report(self, output_path: str):
        """
        Save the report to a file.

        TODO: Implement this method
        - Call generate_report()
        - Write to file
        """
        # YOUR CODE HERE
        pass


class ModelComparison:
    """
    Compare Q/A generation quality across different models.

    TODO: Implement this class (Advanced challenge!)
    This teaches you how to work with multiple datasets!
    """

    def __init__(self):
        """Initialize with empty datasets dict."""
        self.datasets = {}  # model_name -> QADataset

    def add_model(self, model_name: str, dataset: QADataset):
        """Add a dataset for a specific model."""
        self.datasets[model_name] = dataset

    def compare_metrics(self) -> Dict[str, Dict]:
        """
        Compare key metrics across all models.

        TODO: Implement this method
        - For each model's dataset
        - Collect metrics (total count, types, difficulties)
        - Return nested dict of results

        Example output:
        {
            "meta-llama/Llama-3.3": {"total": 50, "mcq_count": 30, ...},
            "mistral/Mistral-7B": {"total": 45, "mcq_count": 25, ...}
        }
        """
        # YOUR CODE HERE
        pass


def main():
    """Main analysis program."""
    print("=" * 60)
    print("Q/A Quality Analyzer")
    print("=" * 60)

    # TODO: Create a QADataset
    dataset = QADataset("My Q/A Collection")

    # TODO: Load data
    folder = input("Enter folder with JSONL files (default: qa_jsonl): ").strip()
    if not folder:
        folder = "qa_jsonl"

    dataset.load_from_folder(folder)

    # TODO: Generate and print report
    report = dataset.generate_report()
    print(report)

    # TODO: Ask if user wants to save report
    save = input("\nSave report to file? (y/n): ").lower()
    if save == 'y':
        output_path = input("Output filename (default: qa_report.txt): ").strip()
        if not output_path:
            output_path = "qa_report.txt"
        dataset.save_report(output_path)
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()


# BONUS CHALLENGES (Phase 3):
# 1. Add visualization with matplotlib (bar charts of type distribution)
# 2. Implement the ModelComparison class fully
# 3. Add a method to find the "best" questions (longest, most complex)
# 4. Export statistics to CSV for analysis in Excel/Google Sheets
