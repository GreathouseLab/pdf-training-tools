# Learning Project: Q/A Pipeline Enhancement Tools

**Goal**: Build tools to analyze and interact with your PDF Q/A pipeline output while learning intermediate Python concepts.

**Prerequisites**: CS50 Python (completed ✅)

**Estimated Time**: 10-15 hours total across all phases

---

## 📚 Learning Path Overview

```
Phase 1: Interactive Quiz Game (3-4 hrs)
    ↓
Phase 2: Duplicate Detector (3-4 hrs)
    ↓
Phase 3: Quality Analyzer with OOP (4-6 hrs)
    ↓
Bonus: Advanced Features
```

---

## 🎯 Phase 1: Interactive Quiz Game

### **File**: `quiz_game.py`

### **What You'll Learn**
- Reading and parsing JSONL files
- Working with dictionaries and lists
- User input/output
- Random selection
- Basic scoring logic

### **Step-by-Step Implementation Guide**

#### **Step 1.1: Load Questions Function**
```python
def load_questions(jsonl_file: str) -> list[dict]:
    questions = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                qa = json.loads(line)
                questions.append(qa)
    return questions
```

**What to learn here:**
- `with open()` - context managers
- `.strip()` - remove whitespace
- `json.loads()` - parse JSON from string

#### **Step 1.2: Ask Question Function**
```python
def ask_question(qa: dict, question_num: int, total: int) -> bool:
    print(f"\n--- Question {question_num}/{total} ---")
    print(qa['question'])

    # Handle different question types
    if qa['type'] == 'mcq':
        # Show options A, B, C, D
        for i, option in enumerate(qa['options']):
            print(f"  {chr(65+i)}) {option}")  # 65 = ASCII for 'A'

        user_answer = input("\nYour answer (A/B/C/D): ").upper()
        correct_index = qa['correct_index']
        correct_letter = chr(65 + correct_index)

        return user_answer == correct_letter
    else:
        # Free-form answer
        user_answer = input("\nYour answer: ")
        correct_answer = qa.get('answer') or qa.get('answer_key')

        # Simple comparison (you can make this smarter!)
        return user_answer.lower().strip() in correct_answer.lower()
```

**What to learn here:**
- `chr()` - ASCII to character conversion
- Dictionary `.get()` method
- String comparison and normalization

#### **Step 1.3: Calculate Score**
```python
def calculate_score(correct: int, total: int) -> tuple[int, str]:
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
```

### **Testing Your Phase 1**
1. First, generate some Q/A pairs:
   ```bash
   python mupdf_trainer_v2.py ./test_pdf.pdf --gen_qa --llm_model "meta-llama/Llama-3.3-70B-Instruct-Turbo" --qa_k 1
   ```

2. Run your quiz:
   ```bash
   python quiz_game.py
   ```

### **Challenges for Phase 1**
- [ ] Add a timer for each question
- [ ] Show the correct answer after wrong answers
- [ ] Filter by difficulty level
- [ ] Save quiz results with timestamp

---

## 🔍 Phase 2: Duplicate Question Detector

### **File**: `duplicate_detector.py`

### **What You'll Learn**
- String similarity algorithms
- Working with multiple files
- Efficient comparisons (avoiding O(n²))
- Text normalization techniques

### **Key Concepts**

#### **String Similarity with SequenceMatcher**
```python
from difflib import SequenceMatcher

# Example
s1 = "What is photosynthesis?"
s2 = "What is photosynthesis"  # Missing question mark
similarity = SequenceMatcher(None, s1, s2).ratio()
print(similarity)  # Output: 0.96 (96% similar)
```

#### **Normalizing Text**
```python
import string

def normalize_question(question: str) -> str:
    # Convert to lowercase
    q = question.lower()

    # Remove punctuation
    q = q.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    q = ' '.join(q.split())

    return q
```

### **Implementation Strategy**

1. **Load all questions** from folder
2. **For each pair of questions**:
   - Normalize both
   - Calculate similarity
   - If > threshold, record as duplicate
3. **Display results**

### **Optimization Challenge**
Instead of comparing all N questions with all N questions (N² comparisons),
only compare each question with questions that come after it:

```python
for i in range(len(questions)):
    for j in range(i + 1, len(questions)):  # Only j > i
        # Compare questions[i] with questions[j]
```

This cuts comparisons in half!

---

## 🏗️ Phase 3: Quality Analyzer (OOP)

### **File**: `qa_analyzer.py`

### **What You'll Learn**
- **Object-Oriented Programming** (classes, methods)
- Data aggregation with `Counter` and `defaultdict`
- Statistical analysis with `statistics` module
- Generating formatted reports

### **OOP Concepts Review**

#### **What is a Class?**
A class is a blueprint for creating objects that bundle data and functionality.

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # instance variable
        self.age = age

    def bark(self):  # method
        print(f"{self.name} says woof!")

# Creating objects (instances)
my_dog = Dog("Buddy", 3)
my_dog.bark()  # Output: Buddy says woof!
```

#### **Why Use Classes for QADataset?**
- **Encapsulation**: Keep data (questions) and operations (analysis) together
- **Reusability**: Create multiple datasets easily
- **Organization**: Cleaner code structure

### **Implementation Guide**

#### **Step 3.1: Basic Class Structure**
```python
class QADataset:
    def __init__(self, name: str):
        self.name = name
        self.questions = []
        self.stats = {}
```

#### **Step 3.2: Loading Data**
```python
def load_from_file(self, jsonl_path: str):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                self.questions.append(json.loads(line))
    print(f"Loaded {len(self.questions)} questions from {jsonl_path}")
```

#### **Step 3.3: Analysis Methods**
```python
from collections import Counter

def count_by_type(self) -> Dict[str, int]:
    types = [q['type'] for q in self.questions]
    return dict(Counter(types))

def count_by_model(self) -> Dict[str, int]:
    models = [q.get('model', 'unknown') for q in self.questions]
    return dict(Counter(models))
```

#### **Step 3.4: Statistics**
```python
import statistics

def analyze_question_lengths(self) -> Dict[str, float]:
    lengths = [len(q['question'].split()) for q in self.questions]

    return {
        'min': min(lengths),
        'max': max(lengths),
        'mean': statistics.mean(lengths),
        'median': statistics.median(lengths)
    }
```

---

## 🎓 Learning Resources

### **Python Concepts Covered**

| Concept | Phase | Resource |
|---------|-------|----------|
| File I/O | 1, 2, 3 | [Python Docs: Files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files) |
| JSON parsing | 1, 2, 3 | [JSON module](https://docs.python.org/3/library/json.html) |
| String methods | 1, 2 | [String methods](https://docs.python.org/3/library/stdtypes.html#string-methods) |
| List comprehensions | 3 | `[x for x in items if condition]` |
| Classes | 3 | [OOP Tutorial](https://docs.python.org/3/tutorial/classes.html) |
| Counter | 3 | [collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter) |

### **Debugging Tips**

1. **Add print statements**:
   ```python
   print(f"DEBUG: questions loaded: {len(questions)}")
   print(f"DEBUG: first question: {questions[0]}")
   ```

2. **Use `type()` to check types**:
   ```python
   print(type(my_variable))
   ```

3. **Test with small data first**:
   - Use just 1-2 questions before processing hundreds

4. **Check for KeyErrors**:
   ```python
   # Instead of:
   answer = qa['answer']  # Crashes if key doesn't exist

   # Use:
   answer = qa.get('answer', 'No answer')  # Safe
   ```

---

## 🚀 Bonus Advanced Features

Once you complete all 3 phases, try these challenges:

### **1. Web Dashboard** (Flask)
Create a web interface to visualize your Q/A statistics.

### **2. Answer Validator**
Use the LLM to grade user answers (not just exact match).

### **3. Spaced Repetition System**
Track which questions users get wrong and show them again later.

### **4. Export to Anki**
Convert your Q/A pairs to Anki flashcard format.

---

## 📝 Project Milestones Checklist

### Phase 1: Quiz Game
- [ ] Load questions from JSONL
- [ ] Display questions to user
- [ ] Check answers
- [ ] Calculate and display score
- [ ] Handle MCQ and free-form questions
- [ ] Add at least 1 bonus feature

### Phase 2: Duplicate Detector
- [ ] Load questions from multiple files
- [ ] Normalize text
- [ ] Calculate similarity
- [ ] Find and display duplicates
- [ ] Optimize to avoid unnecessary comparisons
- [ ] Add at least 1 bonus feature

### Phase 3: Quality Analyzer
- [ ] Create QADataset class
- [ ] Implement all counting methods
- [ ] Add statistical analysis
- [ ] Generate formatted reports
- [ ] Save reports to file
- [ ] Add at least 1 bonus feature

---

## 💡 Getting Help

1. **Read error messages carefully** - they tell you the line number and problem
2. **Google the error** - add "python" to your search
3. **Use Python documentation** - official docs are excellent
4. **Break problems into smaller pieces** - solve one function at a time
5. **Test frequently** - don't write 100 lines before testing!

---

## 🎉 Completion

When you finish all phases, you will have:
- ✅ Built 3 complete Python programs
- ✅ Learned file I/O, JSON, OOP, algorithms
- ✅ Created useful tools for your pipeline
- ✅ Something impressive for your portfolio!

**Next Steps**: Show me your completed code and I'll review it, suggest improvements, and recommend your next learning project!

---

*Good luck and happy coding! Remember: the best way to learn is by doing. Don't be afraid to make mistakes - they're how we learn! 🚀*
