
"""
Centralized prompt templates for biomedical Q/A generation.
All prompts require strict JSON or strictly formatted text to make parsing robust.
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
You are a senior content evaluator determining whether a text chunk is relevant to the core
scientific/technical content of a biomedical paper (vs non-relevant material like references, metadata, copyright).
""")

RELEVANCE_USER = dedent("""\
Evaluate the following text CHUNK for scientific/technical relevance to core content suitable for generating Q/A.

CHUNK:
{chunk_text}

EVALUATION CRITERIA:
- High relevance: scientific concepts, research findings, methods, analyses, experimental design, mechanisms, results.
- Low/No relevance: references, licensing/copyright, acknowledgments, footers/headers, figure/table captions only, publisher metadata.
- Mixed: clearly contains both core and non-core material.

{strict_json}

Output JSON with:
{{
  "relevance_score": <integer 1-10>,
  "content_type": "<one of: core_scientific | mixed | references | metadata | copyright | other>",
  "reason": "<one-sentence justification>"
}}
""")

# ---------- 2) Augment/summarize chunk ----------

AUGMENT_SYSTEM = dedent("""\
You are a helpful assistant who produces concise bullet summaries of biomedical text and
expands with general domain knowledge that remains faithful to the text.
""")

AUGMENT_USER = dedent("""\
Given the CHUNK below, do TWO things:
1) Summarize as 5–8 crisp bullets.
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

# ---------- 3) Multiple-choice question generation (MCQ) ----------

MCQ_SYSTEM = dedent("""\
You are an expert item-writer creating high-quality multiple-choice questions for biomedical topics.
Each question must be self-contained and test understanding (not trivia).
""")

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
Score the MCQ below on 1–10. Also provide a brief critique.

MCQ JSON:
{mcq_json}

SCORING DIMENSIONS (1–10 each; overall is not an average):
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

REASON_SYSTEM = dedent("""\
You are an expert researcher who writes challenging, reasoning-heavy free-response questions for biomedical topics.
Questions must assess deep understanding (mechanisms, limitations, alternative explanations, trade-offs).
""")

REASON_USER = dedent("""\
From the CONTEXT below, produce ONE challenging free-response question that requires systematic reasoning.
Also provide a concise expert answer and a short grading rubric (no chain-of-thought; concise bullet justification).

CONTEXT:
{context_text}

Return ONLY this JSON:
{{
  "question": "<string>",
  "answer_key": "<3-6 sentence expert answer>",
  "rubric": ["<bullet criterion 1>", "<bullet 2>", "<bullet 3>"],
  "tags": ["biomedical", "oncology", "reasoning"],
  "difficulty": "<intermediate|advanced>"
}}

{strict_json}
""")
