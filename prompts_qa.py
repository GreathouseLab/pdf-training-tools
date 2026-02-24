
"""
Centralized prompt templates for biomedical Q/A generation.
All prompts require strict JSON or strictly formatted text to make parsing robust.

UPDATED 2026-02-10: Per Nick (Argonne) grading guidance:
- Freeform answers constrained to 1-2 words for exact-match RLHF grading
- Reasoning questions converted to MCQ format (hard stems + multiple choice)
- All outputs must be gradable by exact string match (no semantic scoring)

UPDATED 2026-02-11: Added seen-answers dedup injection to reduce cross-chunk
duplicate questions within a single PDF. Accumulates answers from prior chunks
and injects them as exclusions into the freeform prompt.
"""

from textwrap import dedent

# ---------- 0) Shared helpers ----------

STRICT_JSON_INSTRUCTIONS = dedent("""\
Return ONLY valid JSON. Do not include markdown fences or commentary.
If a field is unknown, use null. Keys are case-sensitive and must match exactly.
""")

NO_SOURCE_META = dedent("""\
Do NOT reference 'the text', 'the passage', 'this paper', 'the authors', figure/table captions,
headings, references, copyright notices, or metadata. Produce self-contained content only.
""")

BIOMED_SAFETY = dedent("""\
Follow biomedical safety: no patient-identifying information; no medical advice; avoid prescriptive treatment.
Frame answers as educational summaries of study content and methods.
""")

# ---------- 1) Chunk relevance gate ----------

RELEVANCE_SYSTEM = dedent("""\
You are an expert content evaluator who determines if text content is relevant to the core 
scientific/technical content of a biomedical paper versus non-relevant material like copyright 
notices, licensing information, references, acknowledgments, or metadata. Your evaluation 
directly impacts whether this content will be used to generate educational questions for 
clinical nutrition and oncology training.
""")

RELEVANCE_USER = dedent("""\
Evaluate the following text chunk and determine if it contains core scientific/technical content 
that would be appropriate for generating educational questions for dietitians and oncologists.

TEXT CHUNK:
{chunk_text}

EVALUATION CRITERIA:

CORE CONTENT (High relevance):
- Scientific concepts, research findings, clinical outcomes
- Methodology, study design, data analysis approaches
- Theories, mechanisms, pathways (e.g., metabolic, inflammatory)
- Experimental results, statistical findings, effect sizes
- Clinical recommendations, nutritional interventions, guidelines
- Patient population characteristics, inclusion/exclusion criteria
- Drug-nutrient interactions, contraindications, safety data

NON-CORE CONTENT (Low relevance):
- Copyright notices, licensing text, publisher boilerplate
- Reference lists, bibliography sections, citation numbers
- Acknowledgments, funding statements, author affiliations
- Publication metadata, DOI, journal information
- Figure/table captions without substantive content
- Page headers/footers, disclaimers, conflict of interest statements

SCORING GUIDE:
- Score 8-10: Rich core content ideal for question generation (methods, results, clinical implications)
- Score 5-7: Mixed content - contains useful material but also non-relevant sections
- Score 1-4: Primarily non-relevant content (references, metadata, copyright, etc.)

{strict_json}

Output JSON with:
{{
  "relevance_score": <integer 1-10>,
  "content_type": "<one of: core_scientific | mixed | references | metadata | copyright | acknowledgments | other>",
  "reasoning": "<one-sentence explanation of why this content is or is not suitable for generating clinical nutrition Q/A>"
}}
""")

# ---------- 2) Augment/summarize chunk ----------

AUGMENT_SYSTEM = dedent("""\
You are a helpful assistant who produces concise bullet summaries of biomedical text and
expands with general domain knowledge that remains faithful to the text.
""")

AUGMENT_USER = dedent("""\
Given the CHUNK below, do TWO things:
1) Summarize as 5-8 crisp bullets.
2) Expand with general domain knowledge that is appropriate for graduate-level readers,
   without introducing claims that contradict the CHUNK.

CHUNK:
{chunk_text}

FORMAT (plain text):
augmented_chunk:
- <bullet 1>
- <bullet 2>
- ...
""")

# ---------- 3) Multiple-choice question generation  ----------

MCQ_SYSTEM = dedent("""\
You are an expert item-writer creating high-quality multiple-choice questions for biomedical topics.
Each question must be self-contained and test understanding (not trivia).
""")
# added line to MCQ prompt below to be grounded in biomedical knowledge
MCQ_USER = dedent("""\
Using the content below, write ONE multiple-choice question with exactly {num_answers} options.
Place the correct answer at position {target_correct_position}. Ensure distractors are plausible but incorrect.
Avoid any reference to the source text. {no_source_meta}

CONTEXT (use to derive content):
{context_text}

REQUIREMENTS:
- Ask about a concept or result that is clearly supported by the CONTEXT.
- One unambiguously correct answer; remaining options are incorrect but plausible.
- Graduate-level difficulty; avoid rote recall.
- Self-contained wording (no 'as mentioned above').
- Keep question stem under 2 sentences.
- Base ONLY the correct answer on facts explicitly stated in the CONTEXT. Do NOT introduce biological processes, mechanisms, or pathways not mentioned in the CONTEXT.
- Distractors should be plausible but clearly contradicted by or absent from the CONTEXT.

CRITICAL - ANTI-HALLUCINATION RULES:
- Use ONLY information explicitly present in the CONTEXT above.
- Do NOT invent biological processes, mechanisms, pathways, or terminology not in the CONTEXT.
- Do NOT draw from general biomedical knowledge beyond what the CONTEXT states.
- If the CONTEXT lacks sufficient detail for a good question, acknowledge this limitation rather than fabricating details.

Return ONLY this JSON:
{{
  "question": "<string>",
  "options": ["<A>", "<B>", "..."],
  "correct_index": {correct_index},
  "tags": ["biomedical", "oncology", "nutrition"],
  "difficulty": "<introductory|intermediate|advanced>"
}}

{strict_json}
""")

# ---------- 4) Multiple-choice scoring/critique ----------

MCQ_SCORE_SYSTEM = dedent("""\
You are a rigorous educator scoring a multiple-choice item for clarity, difficulty, plausibility of distractors,
self-containment, and content relevance. Disqualify any question that references source meta or is ambiguous.
""")

MCQ_SCORE_USER = dedent("""\
Score the MCQ below on 1-10. Also provide a brief critique.

MCQ JSON:
{mcq_json}

SCORING DIMENSIONS (1-10 each; overall is not an average):
- Clarity
- Content relevance to biomedical topic
- Difficulty appropriate for graduates
- Distractors plausible
- Self-contained wording

AUTOMATIC DISQUALIFIERS (overall score must be 1 if any present):
- References 'the text/paper/figure/table/section/appendix' or similar.
- Requires access to the source beyond the MCQ text.
- Ambiguous correct answer or multiple correct answers.

Return ONLY this JSON:
{{
  "score": <integer 1-10>,
  "reasons": "<brief critique>",
  "disqualified": <true|false>
}}

{strict_json}
""")

# ---------- 5) Reasoning/free-response question ----------
# UPDATED 2026-02-10: Converted to MCQ format per Nick's grading guidance.
# Keeps the hard reasoning-level question stem but answers are now multiple choice
# for exact-match grading compatibility with RLHF.

REASON_SYSTEM = dedent("""\
You are an expert researcher who writes challenging, reasoning-heavy multiple-choice questions 
for biomedical topics. Questions must assess deep understanding (mechanisms, limitations, 
alternative explanations, trade-offs) but provide exactly 4 answer options for gradability.
The question stem should require multi-step reasoning, but the correct answer must be one 
unambiguous choice.
""")

REASON_USER = dedent("""\
From the CONTEXT below, produce ONE challenging reasoning-level multiple-choice question.
The question stem should require systematic, multi-step reasoning to answer correctly.
Place the correct answer at position {target_correct_position}.

CONTEXT:
{context_text}

QUESTION STEM REQUIREMENTS (what makes it reasoning-level):
- Requires multi-step logical reasoning, not simple recall
- Tests understanding of mechanisms, trade-offs, or limitations
- Uses sophisticated framing: "Given X and Y, which conclusion follows?", 
  "What is the most likely explanation for...?", "Which factor best accounts for...?"
- Should challenge graduate-level learners

ANSWER OPTIONS REQUIREMENTS (what makes it gradable):
- Exactly 4 options (A, B, C, D)
- One unambiguously correct answer
- Distractors should represent common misconceptions or partial reasoning
- Each option should be a short phrase or single sentence (not a paragraph)

CRITICAL - ANTI-HALLUCINATION RULES:
- Use ONLY information explicitly present in the CONTEXT above.
- Do NOT invent biological processes, mechanisms, pathways, or terminology not in the CONTEXT.
- Do NOT draw from general biomedical knowledge beyond what the CONTEXT states.
- If the CONTEXT lacks sufficient detail for a good question, acknowledge this limitation rather than fabricating details.

Return ONLY this JSON:
{{
  "question": "<challenging reasoning question stem>",
  "options": ["<A>", "<B>", "<C>", "<D>"],
  "correct_index": {correct_index},
  "reasoning_type": "<one of: mechanistic | comparative | limitation_analysis | causal | application>",
  "tags": ["biomedical", "oncology", "reasoning"],
  "difficulty": "advanced"
}}

{strict_json}
""")


# ==========================================================================
# 6) FREEFORM SHORT-ANSWER Q/A (UPDATED 2026-02-11)
# ==========================================================================
# Per Nick (Argonne) grading guidance: Freeform answers MUST be 1-2 words
# for exact-match grading in RLHF. Semantic scoring fails in biomedical 
# domain because similar-sounding terms can have opposite clinical effects.
# Design questions so the answer is a single specific term, value, or name.
#
# NEW: Seen-answers injection reduces within-PDF duplication by telling
# the model which facts have already been extracted from prior chunks.

FREEFORM_SYSTEM = dedent("""\
You are a careful scientific assistant creating short-answer Q/A pairs for biomedical training.
Every answer MUST be exactly 1-2 words -- a single specific term, number, name, or short phrase.
This constraint is critical: answers will be graded by exact string match.
Return STRICT JSON only. No explanations.
""")

FREEFORM_USER = dedent("""\
Create {k} high-quality short-answer question/answer pairs from the passage below.

CRITICAL ANSWER FORMAT RULE:
Every answer MUST be 1-2 words only. Design questions so the correct answer IS a specific:
- Scientific term (e.g., "cachexia", "sarcopenia", "dysbiosis")
- Nutrient or compound name (e.g., "omega-3", "vitamin D", "glutamine") 
- Numeric value or percentage (e.g., "30%", "85%", "1.5 g/kg")
- Diet or intervention name (e.g., "Mediterranean", "ketogenic", "enteral")
- Clinical tool or scale name (e.g., "PG-SGA", "MST", "GLIM")
- Biological process (e.g., "apoptosis", "angiogenesis", "gluconeogenesis")
- Yes/No or True/False for binary factual questions

GOOD EXAMPLES (answer is 1-2 words, exact-match gradable):
Q: "What screening tool did the ASPEN systematic review identify as most practical for outpatient oncology settings?"
A: "MST"

Q: "What percentage of US cancer centers implement validated malnutrition screening?"
A: "35%"

Q: "Which dietary pattern was associated with improved survival in metastatic colorectal cancer patients?"
A: "plant-based"

Q: "What is the minimum recommended protein intake (g/kg/day) for cancer patients per ESPEN guidelines?"
A: "1.0 g/kg"

BAD EXAMPLES (answer too long, NOT exact-match gradable):
Q: "How does fiber affect colorectal cancer outcomes?"
A: "Dietary fiber has been shown to reduce recurrence rates through modulation of the gut microbiome."
(TOO LONG -- this is a sentence, not a term)

Q: "What are the benefits of omega-3 supplementation?"
A: "Omega-3 fatty acids can help maintain body weight and improve inflammatory markers during chemotherapy."
(TOO LONG -- rewrite as: Q: "What effect do omega-3 fatty acids have on body weight during chemotherapy?" A: "weight maintenance")

RULES:
- Answers MUST be 1-2 words (3 words acceptable ONLY for numeric values like "1.5 g/kg")
- Design the question so only one specific term/value is the correct answer
- DO NOT use any metadata like authors, pages, dates, institutions from the passage
- Questions must be self-contained (no "according to the passage" or "in this study")
- If the passage lacks specific terms/values for short answers, generate fewer questions rather than forcing long answers
- {no_source_meta}
{seen_answers_block}
Return a JSON array of objects with keys: 'question', 'answer'.

Passage:
---
{passage}
---

{strict_json}
""")


# ==========================================================================
# 7) SEEN-ANSWERS DEDUP BLOCK (NEW - 2026-02-11)
# ==========================================================================
# This block is injected into FREEFORM_USER when there are previously
# generated answers from earlier chunks in the same PDF. It steers the
# model away from extracting the same headline facts repeatedly.

SEEN_ANSWERS_TEMPLATE = dedent("""\

DUPLICATE AVOIDANCE (CRITICAL):
The following answers have ALREADY been generated from earlier sections of this document.
Do NOT generate any question whose answer matches or closely resembles any of these:
{seen_answers_list}
Instead, find DIFFERENT facts, values, terms, or concepts from this passage.
If every extractable fact has already been covered, return an empty JSON array: []
""")


def build_seen_answers_block(seen_answers: list[str]) -> str:
    """
    Build the seen-answers injection block for the freeform prompt.
    
    Args:
        seen_answers: List of answer strings from prior chunks in this PDF.
                     e.g. ["MST", "85%", "omega-3", "1.0 g/kg"]
    
    Returns:
        Formatted block to inject into FREEFORM_USER, or empty string if
        no prior answers exist.
    """
    if not seen_answers:
        return ""
    
    # Deduplicate and format as a bullet list
    unique_answers = sorted(set(seen_answers))
    formatted_list = "\n".join(f"  - {a}" for a in unique_answers)
    
    return SEEN_ANSWERS_TEMPLATE.format(seen_answers_list=formatted_list)
