"""
Verification module for Q/A quality assurance in the NORE pipeline.

This module provides:
1. Verification prompts for MCQs and Reasoning questions
2. Safety watchlists for clinical nutrition content
3. Forbidden reference patterns
4. Verification functions with confidence scoring

Design Decisions (from context_updated.json):
- Inline verification at generation time
- 2 regeneration attempts before discard
- Confidence thresholds: >=0.9 auto-pass, 0.7-0.9 flag for review, <0.7 auto-fail
- Separate verification logic for MCQs vs Reasoning questions
- 1% target false positive rate
"""

from textwrap import dedent
import re
from typing import Dict, List, Any, Tuple
from enum import Enum

# Import shared helpers from prompts_qa
from prompts_qa import STRICT_JSON_INSTRUCTIONS, NO_SOURCE_META, BIOMED_SAFETY


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class VerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    FLAG_FOR_REVIEW = "flag_for_review"


class VerificationRecommendation(Enum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    DISCARD = "discard"


# Confidence thresholds
CONFIDENCE_AUTO_PASS = 0.9
CONFIDENCE_REVIEW_MIN = 0.7
MAX_REGENERATION_ATTEMPTS = 2


# ============================================================================
# SAFETY WATCHLIST
# ============================================================================
# Sources: Project PDFs + ESPEN/ASCO guidelines + Standard oncology references

SAFETY_WATCHLIST = {
    # ----- Chemotherapy and Targeted Therapy Drugs -----
    # These require careful attention to drug-nutrient interactions
    "chemotherapy_drugs": [
        # Antimetabolites (folate pathway - critical interactions)
        "methotrexate", "pemetrexed", "capecitabine", "5-fluorouracil", "5-fu",
        "fluorouracil", "gemcitabine", "cytarabine",
        # Platinum compounds
        "cisplatin", "carboplatin", "oxaliplatin",
        # Taxanes
        "paclitaxel", "docetaxel", "cabazitaxel",
        # Anthracyclines
        "doxorubicin", "epirubicin", "daunorubicin",
        # Alkylating agents
        "cyclophosphamide", "ifosfamide", "temozolomide",
        # Vinca alkaloids
        "vincristine", "vinblastine", "vinorelbine",
        # Topoisomerase inhibitors
        "irinotecan", "topotecan", "etoposide",
        # Targeted therapies
        "imatinib", "erlotinib", "gefitinib", "sunitinib", "sorafenib",
        "bevacizumab", "trastuzumab", "cetuximab", "rituximab",
        # Immunotherapy
        "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab",
        # Hormone therapies
        "tamoxifen", "letrozole", "anastrozole", "enzalutamide",
    ],
    
    # ----- High-Risk Nutrients (toxicity or interaction potential) -----
    "high_risk_nutrients": [
        # Vitamins with toxicity risk or drug interactions
        "vitamin k", "vitamin a", "vitamin d", "vitamin e", "vitamin b12",
        "folate", "folic acid", "folinic acid", "leucovorin",
        "niacin", "vitamin b6", "pyridoxine",
        # Minerals with narrow therapeutic window
        "potassium", "magnesium", "calcium", "iron", "zinc", "selenium",
        "copper", "phosphorus", "sodium",
        # Fatty acids
        "omega-3", "omega-6", "fish oil", "epa", "dha",
        # Other supplements with interaction potential
        "coenzyme q10", "coq10", "st john's wort", "ginkgo",
        "ginseng", "turmeric", "curcumin", "green tea extract",
    ],
    
    # ----- Vulnerable Patient Populations -----
    "vulnerable_populations": [
        # Metabolic states
        "cachexia", "cancer cachexia", "sarcopenia", "sarcopenic obesity",
        "malnutrition", "severe malnutrition", "refeeding syndrome",
        "anorexia", "cancer anorexia",
        # Organ function impairment
        "renal impairment", "renal failure", "kidney disease", "ckd",
        "hepatic impairment", "liver failure", "cirrhosis",
        "cardiac cachexia", "heart failure",
        # Immune status
        "neutropenia", "neutropenic", "immunocompromised", "immunosuppressed",
        # GI complications
        "mucositis", "oral mucositis", "dysphagia", "esophagitis",
        "enteritis", "colitis", "diarrhea", "constipation",
        "nausea", "vomiting", "anorexia",
        "short bowel syndrome", "malabsorption",
        # Metabolic disorders
        "diabetes", "hyperglycemia", "hypoglycemia",
        "inborn errors of metabolism", "iem", "phenylketonuria", "pku",
    ],
    
    # ----- Critical Clinical Conditions -----
    "critical_conditions": [
        # Electrolyte imbalances
        "hyperkalemia", "hypokalemia", "hypernatremia", "hyponatremia",
        "hypercalcemia", "hypocalcemia", "hypomagnesemia",
        "tumor lysis syndrome",
        # Nutritional emergencies
        "refeeding syndrome", "wernicke encephalopathy",
        # Treatment side effects
        "hand-foot syndrome", "peripheral neuropathy",
        "cardiotoxicity", "nephrotoxicity", "hepatotoxicity",
    ],
    
    # ----- Specific Food-Drug Interactions -----
    "food_drug_interactions": [
        # Grapefruit interactions
        "grapefruit", "grapefruit juice",
        # Vitamin K foods (warfarin interaction)
        "leafy greens", "green leafy vegetables", "kale", "spinach",
        # Tyramine-containing foods (MAOIs)
        "aged cheese", "fermented foods",
        # High-potassium foods
        "banana", "orange juice", "potatoes",
        # Calcium-rich foods (absorption interactions)
        "dairy", "milk", "calcium supplements",
    ],
}

# Flatten for easy lookup
ALL_SAFETY_TERMS = []
for category, terms in SAFETY_WATCHLIST.items():
    ALL_SAFETY_TERMS.extend(terms)
ALL_SAFETY_TERMS = list(set(ALL_SAFETY_TERMS))  # Remove duplicates


# ============================================================================
# FORBIDDEN REFERENCE PATTERNS
# ============================================================================
# Questions and answers must be self-contained without referencing source material

FORBIDDEN_REFERENCE_PATTERNS = [
    # Direct text references
    r"\bthe text\b",
    r"\bthe passage\b",
    r"\bthe document\b",
    r"\bthe paper\b",
    r"\bthe study\b",
    r"\bthe article\b",
    r"\bthis paper\b",
    r"\bthis study\b",
    r"\bthis article\b",
    
    # Author references
    r"\bthe authors?\b",
    r"\bthe researchers?\b",
    r"\bthe investigators?\b",
    r"\baccording to the text\b",
    r"\baccording to the passage\b",
    r"\baccording to the study\b",
    r"\bthe author states\b",
    r"\bas mentioned\b",
    r"\bas described\b",
    r"\bas stated\b",
    r"\bas noted\b",
    r"\bas shown\b",
    r"\bas indicated\b",
    r"\bas reported\b",
    
    # Section/figure/table references
    r"\bappendix\b",
    r"\bfigure\s*\d*\b",
    r"\btable\s*\d*\b",
    r"\bsection\s*\d*\b",
    r"\bchapter\s*\d*\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bpreviously mentioned\b",
    r"\bfollowing\b",
    
    # Meta-references to results
    r"\bthese results\b",
    r"\bthis result\b",
    r"\bthe results\b",
    r"\bthe findings\b",
    r"\bthe thesis\b",
    r"\bthe model\b",  # when referring to a model described in text
    
    # External content references
    r"\bsource materials?\b",
    r"\bexternal content\b",
]

# Compile patterns for efficiency
FORBIDDEN_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_REFERENCE_PATTERNS]


# ============================================================================
# VERIFICATION PROMPTS - MCQ
# ============================================================================

VERIFY_MCQ_SYSTEM = dedent("""\
You are an expert assessment validator specializing in biomedical education for clinical nutrition 
and oncology. Your task is to verify that multiple-choice questions meet quality standards for 
training healthcare professionals.

You evaluate questions on factual accuracy, educational validity, safety considerations, and 
self-containment. You are rigorous but fair, understanding that high-quality training data 
requires strict standards.
""")

VERIFY_MCQ_USER = dedent("""\
Verify the following multiple-choice question for quality and accuracy.

SOURCE CHUNK (the original text the question was derived from):
{source_chunk}

MCQ TO VERIFY:
Question: {question}
Options: {options}
Stated Correct Answer Index: {correct_index}

VERIFICATION CRITERIA:

1. FACTUAL ACCURACY (Critical)
   - Is the correct answer actually supported by the source chunk?
   - Are the facts in the question accurate according to the source?
   - Are there any hallucinated details not present in the source?

2. SINGLE CORRECT ANSWER (Critical)
   - Is there exactly ONE unambiguously correct answer?
   - Could any other option also be defended as correct?
   - Is the stated correct index actually the best answer?

3. DISTRACTOR QUALITY
   - Are incorrect options plausible but clearly wrong?
   - Are distractors based on common misconceptions (not obviously absurd)?
   - Do distractors avoid being "trick" answers or overly similar to correct answer?

4. SELF-CONTAINMENT (Critical)
   - Does the question make sense without access to the source text?
   - Does it avoid references like "the study," "the authors," "as mentioned," etc.?
   - Would this read like a general knowledge question on the topic?

5. CLINICAL SAFETY
   - If the question involves medications, nutrients, or patient populations, is it safe?
   - Are there any missing safety caveats that could mislead learners?
   - Does the correct answer avoid potentially harmful recommendations?

6. EDUCATIONAL VALUE
   - Does this question test understanding rather than trivial recall?
   - Is the difficulty appropriate for graduate-level healthcare professionals?
   - Would answering this question correctly indicate clinical competency?

AUTOMATIC FAILURE CONDITIONS (any one = fail):
- Correct answer is not supported by source chunk
- Multiple options could be correct
- Contains forbidden reference phrases (the text, the study, as mentioned, etc.)
- Contains hallucinated clinical information not in source
- Could lead to patient harm if learned incorrectly

{strict_json}

Output JSON:
{{
  "verification_status": "<pass | fail | flag_for_review>",
  "confidence_score": <float 0.0-1.0>,
  "factual_accuracy": {{
    "score": <1-10>,
    "issues": ["<list any factual problems>"]
  }},
  "single_correct_answer": {{
    "verified": <true | false>,
    "issues": ["<list any ambiguity problems>"]
  }},
  "distractor_quality": {{
    "score": <1-10>,
    "issues": ["<list any distractor problems>"]
  }},
  "self_containment": {{
    "verified": <true | false>,
    "forbidden_phrases_found": ["<list any found>"]
  }},
  "clinical_safety": {{
    "safe": <true | false>,
    "concerns": ["<list any safety concerns>"]
  }},
  "educational_value": {{
    "score": <1-10>,
    "comments": "<brief assessment>"
  }},
  "failure_reasons": ["<list of critical issues if status is fail>"],
  "recommendation": "<accept | regenerate | discard>",
  "improvement_suggestions": ["<specific suggestions if regenerate>"]
}}
""")


# ============================================================================
# VERIFICATION PROMPTS - REASONING QUESTIONS
# ============================================================================

VERIFY_REASONING_SYSTEM = dedent("""\
You are an expert assessment validator specializing in biomedical education for clinical nutrition 
and oncology. Your task is to verify that free-response reasoning questions and their answer keys 
meet quality standards for training healthcare professionals.

You evaluate questions for depth of reasoning required, factual accuracy of answers, rubric 
alignment, and absence of hallucinated content. You are rigorous but fair.
""")

VERIFY_REASONING_USER = dedent("""\
Verify the following reasoning question, answer key, and rubric for quality and accuracy.

SOURCE CHUNK (the original text the question was derived from):
{source_chunk}

REASONING QUESTION TO VERIFY:
Question: {question}
Answer Key: {answer_key}
Rubric: {rubric}

VERIFICATION CRITERIA:

1. ANSWER ACCURACY (Critical)
   - Is every claim in the answer key supported by the source chunk?
   - Are there any hallucinated mechanisms, pathways, or processes not in the source?
   - Are numerical values, statistics, or effect sizes accurate to the source?

2. QUESTION-ANSWER ALIGNMENT
   - Does the answer actually address what the question asks?
   - Is the answer appropriately comprehensive for the question scope?
   - Would an expert consider this a complete answer?

3. RUBRIC VALIDITY
   - Does each rubric point correspond to content in the answer key?
   - Are rubric criteria specific and measurable?
   - Would following this rubric fairly assess student responses?

4. REASONING DEPTH
   - Does the question require multi-step reasoning (not simple recall)?
   - Does it test understanding of mechanisms, trade-offs, or clinical judgment?
   - Is it appropriately challenging for graduate-level learners?

5. SELF-CONTAINMENT (Critical)
   - Is the question answerable without access to the source text?
   - Does the answer avoid references like "the study found" or "as described"?
   - Would this work as a standalone exam question?

6. CLINICAL SAFETY
   - Does the answer key avoid potentially harmful recommendations?
   - Are appropriate caveats included for medications or interventions?
   - Would learning this answer lead to safe clinical practice?

7. HALLUCINATION CHECK (Critical)
   - Are all biological mechanisms mentioned actually in the source?
   - Are all pathways, processes, or interactions supported by the source?
   - Does the answer stay within the scope of what the source establishes?

AUTOMATIC FAILURE CONDITIONS (any one = fail):
- Answer contains claims not supported by source chunk
- Hallucinated mechanisms or pathways not in source
- Question cannot be answered without source access
- Contains forbidden reference phrases
- Could lead to patient harm if learned incorrectly

{strict_json}

Output JSON:
{{
  "verification_status": "<pass | fail | flag_for_review>",
  "confidence_score": <float 0.0-1.0>,
  "answer_accuracy": {{
    "score": <1-10>,
    "unsupported_claims": ["<list any claims not in source>"],
    "hallucinations_detected": ["<list any hallucinated content>"]
  }},
  "question_answer_alignment": {{
    "score": <1-10>,
    "issues": ["<list any alignment problems>"]
  }},
  "rubric_validity": {{
    "score": <1-10>,
    "issues": ["<list any rubric problems>"]
  }},
  "reasoning_depth": {{
    "score": <1-10>,
    "question_type": "<critical_analysis | mechanistic | limitation | comparative | application>",
    "comments": "<assessment of reasoning requirements>"
  }},
  "self_containment": {{
    "verified": <true | false>,
    "forbidden_phrases_found": ["<list any found>"]
  }},
  "clinical_safety": {{
    "safe": <true | false>,
    "concerns": ["<list any safety concerns>"]
  }},
  "failure_reasons": ["<list of critical issues if status is fail>"],
  "recommendation": "<accept | regenerate | discard>",
  "improvement_suggestions": ["<specific suggestions if regenerate>"]
}}
""")


# ============================================================================
# VERIFICATION PROMPTS - FREEFORM Q/A (Simple Question-Answer Pairs)
# ============================================================================

VERIFY_FREEFORM_SYSTEM = dedent("""\
You are an expert fact-checker and content validator specializing in biomedical education for 
clinical nutrition and oncology. Your task is to verify that simple question-answer pairs are 
factually accurate and suitable for training healthcare professionals.

You evaluate Q/A pairs for factual accuracy against the source material, self-containment, 
and clinical safety. You are rigorous but efficient.
""")

VERIFY_FREEFORM_USER = dedent("""\
Verify the following question-answer pair for accuracy and quality.

SOURCE CHUNK (the original text the Q/A was derived from):
{source_chunk}

Q/A PAIR TO VERIFY:
Question: {question}
Answer: {answer}

VERIFICATION CRITERIA:

1. FACTUAL ACCURACY (Critical)
   - Is the answer supported by the source chunk?
   - Are all facts, numbers, and claims accurate to the source?
   - Are there any hallucinated or fabricated details not in the source?

2. ANSWER COMPLETENESS
   - Does the answer adequately address the question?
   - Is the answer appropriately concise yet specific?
   - Would a learner gain correct understanding from this answer?

3. SELF-CONTAINMENT (Critical)
   - Does the Q/A make sense without access to the source text?
   - Does it avoid references like "the text states," "according to the passage," etc.?
   - Would this work as a standalone flashcard or study item?

4. CLINICAL SAFETY
   - If the answer involves clinical recommendations, is it safe?
   - Are there any potentially harmful omissions or simplifications?
   - Would learning this answer support safe clinical practice?

5. EDUCATIONAL VALUE
   - Does this Q/A test meaningful knowledge (not trivial facts)?
   - Is it relevant to clinical nutrition or oncology practice?
   - Would a dietitian or oncologist benefit from knowing this?

AUTOMATIC FAILURE CONDITIONS (any one = fail):
- Answer contains claims not supported by source chunk
- Answer contains hallucinated/fabricated information
- Contains forbidden reference phrases to source material
- Could lead to patient harm if learned incorrectly
- Answer is factually incorrect

{strict_json}

Output JSON:
{{
  "verification_status": "<pass | fail | flag_for_review>",
  "confidence_score": <float 0.0-1.0>,
  "factual_accuracy": {{
    "score": <1-10>,
    "supported_by_source": <true | false>,
    "unsupported_claims": ["<list any claims not in source>"],
    "hallucinations_detected": ["<list any fabricated content>"]
  }},
  "answer_completeness": {{
    "score": <1-10>,
    "issues": ["<list any completeness problems>"]
  }},
  "self_containment": {{
    "verified": <true | false>,
    "forbidden_phrases_found": ["<list any found>"]
  }},
  "clinical_safety": {{
    "safe": <true | false>,
    "concerns": ["<list any safety concerns>"]
  }},
  "educational_value": {{
    "score": <1-10>,
    "comments": "<brief assessment>"
  }},
  "failure_reasons": ["<list of critical issues if status is fail>"],
  "recommendation": "<accept | regenerate | discard>",
  "improvement_suggestions": ["<specific suggestions if regenerate>"]
}}
""")


# ============================================================================
# VERIFICATION HELPER FUNCTIONS
# ============================================================================

def check_forbidden_references(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains any forbidden reference patterns.
    
    Returns:
        Tuple of (has_forbidden, list_of_matches)
    """
    found_patterns = []
    for pattern in FORBIDDEN_PATTERNS_COMPILED:
        matches = pattern.findall(text)
        if matches:
            found_patterns.extend(matches)
    
    return len(found_patterns) > 0, list(set(found_patterns))


def check_safety_terms(text: str) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Check if text contains safety-relevant terms that require enhanced verification.
    
    Returns:
        Tuple of (has_safety_terms, dict_of_category_to_terms_found)
    """
    text_lower = text.lower()
    found_terms = {}
    
    for category, terms in SAFETY_WATCHLIST.items():
        category_matches = []
        for term in terms:
            if term.lower() in text_lower:
                category_matches.append(term)
        if category_matches:
            found_terms[category] = category_matches
    
    return len(found_terms) > 0, found_terms


def calculate_confidence_from_scores(scores: Dict[str, int], critical_checks: Dict[str, bool]) -> float:
    """
    Calculate overall confidence score from individual verification scores.
    
    Args:
        scores: Dict of score_name -> score (1-10)
        critical_checks: Dict of check_name -> passed (True/False)
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # If any critical check fails, cap confidence at 0.5
    if not all(critical_checks.values()):
        max_confidence = 0.5
    else:
        max_confidence = 1.0
    
    # Average the scores and normalize to 0-1
    if scores:
        avg_score = sum(scores.values()) / len(scores)
        normalized_score = avg_score / 10.0
    else:
        normalized_score = 0.5
    
    return min(normalized_score, max_confidence)


def determine_recommendation(
    confidence: float,
    has_safety_concerns: bool,
    attempt_number: int
) -> VerificationRecommendation:
    """
    Determine the recommendation based on confidence score and context.
    
    Args:
        confidence: Confidence score 0.0-1.0
        has_safety_concerns: Whether safety issues were detected
        attempt_number: Current regeneration attempt (1-based)
    
    Returns:
        VerificationRecommendation enum value
    """
    # Safety concerns always require human review or discard
    if has_safety_concerns and confidence < CONFIDENCE_AUTO_PASS:
        if attempt_number >= MAX_REGENERATION_ATTEMPTS:
            return VerificationRecommendation.DISCARD
        return VerificationRecommendation.REGENERATE
    
    # High confidence = accept
    if confidence >= CONFIDENCE_AUTO_PASS:
        return VerificationRecommendation.ACCEPT
    
    # Medium confidence = flag for review or regenerate
    if confidence >= CONFIDENCE_REVIEW_MIN:
        # On later attempts, accept for human review rather than keep regenerating
        if attempt_number >= MAX_REGENERATION_ATTEMPTS:
            return VerificationRecommendation.ACCEPT  # Will be flagged for review
        return VerificationRecommendation.REGENERATE
    
    # Low confidence = regenerate or discard
    if attempt_number >= MAX_REGENERATION_ATTEMPTS:
        return VerificationRecommendation.DISCARD
    return VerificationRecommendation.REGENERATE


def determine_status(confidence: float) -> VerificationStatus:
    """
    Determine verification status from confidence score.
    """
    if confidence >= CONFIDENCE_AUTO_PASS:
        return VerificationStatus.PASS
    elif confidence >= CONFIDENCE_REVIEW_MIN:
        return VerificationStatus.FLAG_FOR_REVIEW
    else:
        return VerificationStatus.FAIL


# ============================================================================
# MAIN VERIFICATION FUNCTIONS (to be called from pipeline)
# ============================================================================

def build_mcq_verification_messages(
    source_chunk: str,
    question: str,
    options: List[str],
    correct_index: int
) -> List[Dict[str, str]]:
    """
    Build the messages list for MCQ verification LLM call.
    """
    options_str = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
    
    user_content = VERIFY_MCQ_USER.format(
        source_chunk=source_chunk,
        question=question,
        options=options_str,
        correct_index=correct_index,
        strict_json=STRICT_JSON_INSTRUCTIONS
    )
    
    return [
        {"role": "system", "content": VERIFY_MCQ_SYSTEM},
        {"role": "user", "content": user_content}
    ]


def build_reasoning_verification_messages(
    source_chunk: str,
    question: str,
    answer_key: str,
    rubric: List[str]
) -> List[Dict[str, str]]:
    """
    Build the messages list for reasoning question verification LLM call.
    """
    rubric_str = "\n".join([f"  - {item}" for item in rubric])
    
    user_content = VERIFY_REASONING_USER.format(
        source_chunk=source_chunk,
        question=question,
        answer_key=answer_key,
        rubric=rubric_str,
        strict_json=STRICT_JSON_INSTRUCTIONS
    )
    
    return [
        {"role": "system", "content": VERIFY_REASONING_SYSTEM},
        {"role": "user", "content": user_content}
    ]


def build_freeform_verification_messages(
    source_chunk: str,
    question: str,
    answer: str
) -> List[Dict[str, str]]:
    """
    Build the messages list for freeform Q/A verification LLM call.
    """
    user_content = VERIFY_FREEFORM_USER.format(
        source_chunk=source_chunk,
        question=question,
        answer=answer,
        strict_json=STRICT_JSON_INSTRUCTIONS
    )
    
    return [
        {"role": "system", "content": VERIFY_FREEFORM_SYSTEM},
        {"role": "user", "content": user_content}
    ]


def pre_verify_mcq(question: str, options: List[str], correct_answer: str) -> Dict[str, Any]:
    """
    Fast heuristic pre-verification for MCQs (no LLM call).
    Catches obvious failures before expensive LLM verification.
    
    Returns dict with:
        - passes_heuristics: bool
        - issues: list of issues found
        - safety_terms: dict of safety terms found
    """
    issues = []
    
    # Combine all text for checking
    all_text = question + " " + " ".join(options) + " " + correct_answer
    
    # Check forbidden references
    has_forbidden, forbidden_found = check_forbidden_references(all_text)
    if has_forbidden:
        issues.append(f"Forbidden reference patterns found: {forbidden_found}")
    
    # Check safety terms (not a failure, but flags for enhanced verification)
    has_safety, safety_found = check_safety_terms(all_text)
    
    # Basic structural checks
    if len(options) < 3:
        issues.append("Fewer than 3 answer options")
    
    if len(question.strip()) < 20:
        issues.append("Question is too short (< 20 characters)")
    
    # Check for empty options
    empty_options = [i for i, opt in enumerate(options) if len(opt.strip()) < 2]
    if empty_options:
        issues.append(f"Empty or very short options at indices: {empty_options}")
    
    return {
        "passes_heuristics": len(issues) == 0 or (len(issues) == 1 and has_forbidden),
        "issues": issues,
        "safety_terms": safety_found,
        "requires_enhanced_verification": has_safety
    }


def pre_verify_reasoning(question: str, answer_key: str, rubric: List[str]) -> Dict[str, Any]:
    """
    Fast heuristic pre-verification for reasoning questions (no LLM call).
    
    Returns dict with:
        - passes_heuristics: bool
        - issues: list of issues found
        - safety_terms: dict of safety terms found
    """
    issues = []
    
    # Combine all text for checking
    all_text = question + " " + answer_key + " " + " ".join(rubric)
    
    # Check forbidden references
    has_forbidden, forbidden_found = check_forbidden_references(all_text)
    if has_forbidden:
        issues.append(f"Forbidden reference patterns found: {forbidden_found}")
    
    # Check safety terms
    has_safety, safety_found = check_safety_terms(all_text)
    
    # Basic structural checks
    if len(question.strip()) < 30:
        issues.append("Question is too short for reasoning (< 30 characters)")
    
    if len(answer_key.strip()) < 50:
        issues.append("Answer key is too short (< 50 characters)")
    
    if len(rubric) < 2:
        issues.append("Rubric has fewer than 2 criteria")
    
    return {
        "passes_heuristics": len(issues) == 0 or (len(issues) == 1 and has_forbidden),
        "issues": issues,
        "safety_terms": safety_found,
        "requires_enhanced_verification": has_safety
    }


def pre_verify_freeform(question: str, answer: str) -> Dict[str, Any]:
    """
    Fast heuristic pre-verification for freeform Q/A pairs (no LLM call).
    
    Returns dict with:
        - passes_heuristics: bool
        - issues: list of issues found
        - safety_terms: dict of safety terms found
    """
    issues = []
    
    # Combine all text for checking
    all_text = question + " " + answer
    
    # Check forbidden references
    has_forbidden, forbidden_found = check_forbidden_references(all_text)
    if has_forbidden:
        issues.append(f"Forbidden reference patterns found: {forbidden_found}")
    
    # Check safety terms
    has_safety, safety_found = check_safety_terms(all_text)
    
    # Basic structural checks
    if len(question.strip()) < 15:
        issues.append("Question is too short (< 15 characters)")
    
    if len(answer.strip()) < 2:
        issues.append("Answer is too short (< 2 characters)")
    
    # ADDED 2026-02-10: Answer length enforcement for exact-match grading
    answer_word_count = len(answer.strip().split())
    if answer_word_count > 3:
        issues.append(f"Answer too long for exact-match grading ({answer_word_count} words, max 3)")
    
    # Check for question mark in question
    if "?" not in question:
        issues.append("Question does not contain a question mark")
    
    return {
        "passes_heuristics": len(issues) == 0 or (len(issues) == 1 and has_forbidden),
        "issues": issues,
        "safety_terms": safety_found,
        "requires_enhanced_verification": has_safety
    }


# ============================================================================
# VERIFICATION RESULT PARSING
# ============================================================================

def parse_mcq_verification_result(llm_response: Dict[str, Any], attempt_number: int = 1) -> Dict[str, Any]:
    """
    Parse and enrich the LLM verification response for MCQs.
    
    Adds computed fields like final recommendation based on thresholds.
    """
    confidence = llm_response.get("confidence_score", 0.5)
    has_safety_concerns = not llm_response.get("clinical_safety", {}).get("safe", True)
    
    # Override status based on our thresholds
    computed_status = determine_status(confidence)
    computed_recommendation = determine_recommendation(confidence, has_safety_concerns, attempt_number)
    
    # Enrich the response
    result = llm_response.copy()
    result["computed_status"] = computed_status.value
    result["computed_recommendation"] = computed_recommendation.value
    result["attempt_number"] = attempt_number
    result["thresholds_applied"] = {
        "auto_pass": CONFIDENCE_AUTO_PASS,
        "review_min": CONFIDENCE_REVIEW_MIN,
        "max_attempts": MAX_REGENERATION_ATTEMPTS
    }
    
    return result


def parse_reasoning_verification_result(llm_response: Dict[str, Any], attempt_number: int = 1) -> Dict[str, Any]:
    """
    Parse and enrich the LLM verification response for reasoning questions.
    """
    confidence = llm_response.get("confidence_score", 0.5)
    has_safety_concerns = not llm_response.get("clinical_safety", {}).get("safe", True)
    
    # Check for hallucinations - this is critical for reasoning questions
    hallucinations = llm_response.get("answer_accuracy", {}).get("hallucinations_detected", [])
    if hallucinations:
        # Hallucinations severely penalize confidence
        confidence = min(confidence, 0.4)
    
    computed_status = determine_status(confidence)
    computed_recommendation = determine_recommendation(confidence, has_safety_concerns, attempt_number)
    
    result = llm_response.copy()
    result["computed_status"] = computed_status.value
    result["computed_recommendation"] = computed_recommendation.value
    result["attempt_number"] = attempt_number
    result["hallucination_penalty_applied"] = len(hallucinations) > 0
    result["thresholds_applied"] = {
        "auto_pass": CONFIDENCE_AUTO_PASS,
        "review_min": CONFIDENCE_REVIEW_MIN,
        "max_attempts": MAX_REGENERATION_ATTEMPTS
    }
    
    return result


def parse_freeform_verification_result(llm_response: Dict[str, Any], attempt_number: int = 1) -> Dict[str, Any]:
    """
    Parse and enrich the LLM verification response for freeform Q/A pairs.
    """
    confidence = llm_response.get("confidence_score", 0.5)
    has_safety_concerns = not llm_response.get("clinical_safety", {}).get("safe", True)
    
    # Check for hallucinations - important for freeform too
    hallucinations = llm_response.get("factual_accuracy", {}).get("hallucinations_detected", [])
    if hallucinations:
        # Hallucinations penalize confidence
        confidence = min(confidence, 0.4)
    
    # Check if answer is supported by source
    supported = llm_response.get("factual_accuracy", {}).get("supported_by_source", True)
    if not supported:
        confidence = min(confidence, 0.5)
    
    computed_status = determine_status(confidence)
    computed_recommendation = determine_recommendation(confidence, has_safety_concerns, attempt_number)
    
    result = llm_response.copy()
    result["computed_status"] = computed_status.value
    result["computed_recommendation"] = computed_recommendation.value
    result["attempt_number"] = attempt_number
    result["hallucination_penalty_applied"] = len(hallucinations) > 0
    result["unsupported_penalty_applied"] = not supported
    result["thresholds_applied"] = {
        "auto_pass": CONFIDENCE_AUTO_PASS,
        "review_min": CONFIDENCE_REVIEW_MIN,
        "max_attempts": MAX_REGENERATION_ATTEMPTS
    }
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "VerificationStatus",
    "VerificationRecommendation",
    
    # Constants
    "CONFIDENCE_AUTO_PASS",
    "CONFIDENCE_REVIEW_MIN", 
    "MAX_REGENERATION_ATTEMPTS",
    "SAFETY_WATCHLIST",
    "ALL_SAFETY_TERMS",
    "FORBIDDEN_REFERENCE_PATTERNS",
    
    # Prompts - MCQ
    "VERIFY_MCQ_SYSTEM",
    "VERIFY_MCQ_USER",
    # Prompts - Reasoning
    "VERIFY_REASONING_SYSTEM",
    "VERIFY_REASONING_USER",
    # Prompts - Freeform
    "VERIFY_FREEFORM_SYSTEM",
    "VERIFY_FREEFORM_USER",
    
    # Functions - Core
    "check_forbidden_references",
    "check_safety_terms",
    "calculate_confidence_from_scores",
    "determine_recommendation",
    "determine_status",
    # Functions - Message builders
    "build_mcq_verification_messages",
    "build_reasoning_verification_messages",
    "build_freeform_verification_messages",
    # Functions - Pre-verification (heuristic)
    "pre_verify_mcq",
    "pre_verify_reasoning",
    "pre_verify_freeform",
    # Functions - Result parsing
    "parse_mcq_verification_result",
    "parse_reasoning_verification_result",
    "parse_freeform_verification_result",
]
