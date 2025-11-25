# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Generate Some Test Data

First, create some Q/A pairs to work with:

```bash
cd /Users/leigh_greathouse/Documents/My_Code/Python_code

# Make sure you have a PDF
# If you don't have one, download a sample from the internet

# Generate Q/A pairs (this will take a few minutes)
python mupdf_trainer_v2.py ./your_pdf.pdf --gen_qa --llm_model "meta-llama/Llama-3.3-70B-Instruct-Turbo" --qa_k 1 --max_words 400
```

This will create:
- `qa_jsonl/your_pdf_qa.jsonl` - Your Q/A data
- `qa_jsonl/your_pdf_qa_SUMMARY.txt` - Summary stats

### Step 2: Choose Your First Project

Pick ONE to start with:

#### **Option A: Quiz Game** (Easiest - Start Here!)
```bash
python quiz_game.py
```

**TODO List for Quiz Game:**
1. [ ] Implement `load_questions()` - read the JSONL file
2. [ ] Implement `ask_question()` - display question and get answer
3. [ ] Implement `calculate_score()` - compute percentage and grade
4. [ ] Test with your generated data
5. [ ] Add 1 bonus feature (timer, show answer, etc.)

**Stuck?** Look at `quiz_game_solution.py` for hints!

#### **Option B: Duplicate Detector** (Medium)
```bash
python duplicate_detector.py
```

**TODO List:**
1. [ ] Implement `normalize_question()` - clean text
2. [ ] Implement `calculate_similarity()` - compare questions
3. [ ] Implement `find_duplicates()` - find similar questions
4. [ ] Test and optimize

#### **Option C: Quality Analyzer** (Harder - OOP Practice)
```bash
python qa_analyzer.py
```

**TODO List:**
1. [ ] Implement `__init__()` method
2. [ ] Implement `load_from_file()` method
3. [ ] Implement counting methods
4. [ ] Implement statistics methods
5. [ ] Test and generate report

---

## 📖 Learning Path

```
Week 1: Quiz Game
├─ Day 1-2: Implement basic functions
├─ Day 3: Add MCQ support
├─ Day 4: Polish and add 1 bonus feature
└─ Day 5: Test with real data

Week 2: Duplicate Detector
├─ Day 1-2: Text normalization and similarity
├─ Day 3: Find duplicates algorithm
└─ Day 4-5: Optimize and test

Week 3: Quality Analyzer
├─ Day 1-2: Learn OOP concepts
├─ Day 3-4: Implement QADataset class
└─ Day 5: Generate reports
```

---

## 🆘 If You Get Stuck

### Common Issues

**1. "FileNotFoundError"**
```
Make sure your JSONL file path is correct:
- Use absolute path: /full/path/to/file.jsonl
- Or relative from current directory: qa_jsonl/file.jsonl
```

**2. "JSONDecodeError"**
```
Your JSONL file might be corrupted. Check:
- Each line should be valid JSON
- No empty lines at the end
- Use a JSON validator online
```

**3. "KeyError: 'question'"**
```
The data structure might be different than expected.
Add debug prints:
    print(f"Keys in qa: {qa.keys()}")
    print(f"Full qa dict: {qa}")
```

**4. "I don't know how to start!"**
```
1. Read the function signature (what it takes, what it returns)
2. Write the simplest version first (ignore edge cases)
3. Test it
4. Improve it
5. Repeat
```

### Debugging Checklist

- [ ] Print variable values to see what you have
- [ ] Test with just 1-2 questions first
- [ ] Read error messages carefully (they tell you the line!)
- [ ] Check the solution file for hints
- [ ] Google the error message

---

## ✅ Daily Coding Routine

**10 minutes before coding:**
1. Read the function docstring
2. Understand what input/output is expected
3. Think about the steps needed

**While coding:**
1. Write a little bit
2. Test it immediately
3. Fix errors
4. Repeat

**After coding:**
1. Test with real data
2. Clean up your code
3. Add comments
4. Commit to git!

---

## 🎯 Success Criteria

You know you're done with a phase when:

✅ **Phase 1 (Quiz):**
- Can load questions from file
- Can quiz user with 10 questions
- Shows correct score at end
- Works with both MCQ and free-form questions

✅ **Phase 2 (Duplicates):**
- Can find similar questions
- Threshold setting works
- Shows meaningful results
- Handles multiple files

✅ **Phase 3 (Analyzer):**
- QADataset class works
- Can generate a complete report
- Statistics are accurate
- Can save to file

---

## 📚 Additional Resources

### Python Basics Review
- [Official Python Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python Tutorials](https://realpython.com/)
- [Python for Everybody](https://www.py4e.com/)

### Specific Topics
- **File I/O**: [Python Files](https://www.w3schools.com/python/python_file_handling.asp)
- **JSON**: [Working with JSON](https://realpython.com/python-json/)
- **OOP**: [Python Classes](https://realpython.com/python3-object-oriented-programming/)
- **String Methods**: [Python Strings](https://www.w3schools.com/python/python_strings_methods.asp)

### When You're Ready for More
- [Project Euler](https://projecteuler.net/) - Math/programming challenges
- [Advent of Code](https://adventofcode.com/) - Annual coding puzzles
- [LeetCode](https://leetcode.com/) - Interview prep

---

## 🎉 Celebrate Your Progress!

After each milestone:
1. Commit your code to git
2. Write a short note about what you learned
3. Take a break!
4. Show someone your work

Remember: **Every expert was once a beginner!**

---

Good luck! Start with `quiz_game.py` and work your way through. You've got this! 💪
