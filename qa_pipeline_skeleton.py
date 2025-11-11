
"""
Skeleton that shows EXACTLY where to call each prompt and how to parse outputs.
Integrate these functions into your mupdf_trainer.py after chunking.
"""

import json, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Swap this adapter to your actual LLM client (Together, OpenAI, DeepSeek, etc.)
def llm_chat(model: str, messages: List[Dict[str, str]], temperature: float=0.2, max_tokens: int=1024) -> str:
    """
    Return the raw text content from the model. Implement using your preferred SDK.
    messages = [{"role":"system","content":...}, {"role":"user","content":...}]
    """
    raise NotImplementedError("Wire this to your LLM provider.")

# ---------- Messages helper ----------
def msgs(sys: str, user: str) -> List[Dict[str, str]]:
    return [{"role":"system","content": sys}, {"role":"user","content": user}]

# ---------- Stage outputs ----------
@dataclass
class RelevanceResult:
    score: int
    content_type: str
    reason: str

@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_index: int
    tags: List[str]
    difficulty: str

@dataclass
class MCQScore:
    score: int
    reasons: str
    disqualified: bool

@dataclass
class ReasoningItem:
    question: str
    answer_key: str
    rubric: List[str]
    tags: List[str]
    difficulty: str

# ---------- Stage 1: Gate by relevance ----------
def gate_chunk(chunk_text: str, model: str, prompts) -> Optional[RelevanceResult]:
    user = prompts.RELEVANCE_USER.format(chunk_text=chunk_text, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    out = llm_chat(model, msgs(prompts.RELEVANCE_SYSTEM, user))
    data = json.loads(out)
    return RelevanceResult(score=int(data["relevance_score"]),
                           content_type=str(data["content_type"]),
                           reason=str(data["reason"]))

# ---------- Stage 2: Augment chunk ----------
def augment_chunk(chunk_text: str, model: str, prompts) -> str:
    user = prompts.AUGMENT_USER.format(chunk_text=chunk_text)
    return llm_chat(model, msgs(prompts.AUGMENT_SYSTEM, user)).strip()

# ---------- Stage 3: Generate MCQ ----------
def gen_mcq(context_text: str, model: str, prompts, num_answers: int=4, seed: Optional[int]=None) -> MCQ:
    rng = random.Random(seed)
    correct_pos = rng.randint(0, num_answers-1)
    user = prompts.MCQ_USER.format(
        context_text=context_text,
        num_answers=num_answers,
        target_correct_position=correct_pos+1,  # human-readable (1..N)
        correct_index=correct_pos,
        no_source_meta=prompts.NO_SOURCE_META,
        strict_json=prompts.STRICT_JSON_INSTRUCTIONS
    )
    out = llm_chat(model, msgs(prompts.MCQ_SYSTEM, user))
    data = json.loads(out)
    return MCQ(question=data["question"],
               options=data["options"],
               correct_index=int(data["correct_index"]),
               tags=data.get("tags", []),
               difficulty=data.get("difficulty","intermediate"))

# ---------- Stage 4: Score MCQ ----------
def score_mcq(mcq: MCQ, model: str, prompts) -> MCQScore:
    mcq_json = json.dumps(mcq.__dict__, ensure_ascii=False)
    user = prompts.MCQ_SCORE_USER.format(mcq_json=mcq_json, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    out = llm_chat(model, msgs(prompts.MCQ_SCORE_SYSTEM, user))
    data = json.loads(out)
    return MCQScore(score=int(data["score"]), reasons=str(data["reasons"]), disqualified=bool(data["disqualified"]))

# ---------- Stage 5: Reasoning/free-response ----------
def gen_reasoning(context_text: str, model: str, prompts) -> ReasoningItem:
    user = prompts.REASON_USER.format(context_text=context_text, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    out = llm_chat(model, msgs(prompts.REASON_SYSTEM, user))
    data = json.loads(out)
    return ReasoningItem(question=data["question"],
                         answer_key=data["answer_key"],
                         rubric=data["rubric"],
                         tags=data.get("tags",[]),
                         difficulty=data.get("difficulty","advanced"))

# ---------- Orchestration per chunk ----------
def process_chunk(chunk_text: str, model: str, prompts, thresholds: dict) -> Dict[str, Any]:
    """
    Returns a dict with gated flag, augmented text, a list of accepted MCQs, and one reasoning item.
    thresholds = {"relevance": 6, "mcq_min": 7}
    """
    out: Dict[str, Any] = {"accepted": False, "relevance": None, "augmented": None, "mcqs": [], "reasoning": None}

    # 1) Relevance gate
    rel = gate_chunk(chunk_text, model, prompts)
    out["relevance"] = rel.__dict__
    if rel.score < thresholds.get("relevance", 6) or rel.content_type in {"references","metadata","copyright"}:
        return out  # rejected

    # 2) Augment
    augmented = augment_chunk(chunk_text, model, prompts)
    out["augmented"] = augmented

    # 3) Multiple MCQs with scoring loop
    accepted_mcqs: List[Dict[str, Any]] = []
    for _ in range(3):  # generate up to 3 MCQs per chunk
        mcq = gen_mcq(augmented, model, prompts, num_answers=4)
        score = score_mcq(mcq, model, prompts)
        if (not score.disqualified) and (score.score >= thresholds.get("mcq_min", 7)):
            accepted_mcqs.append({**mcq.__dict__, "score": score.score, "critique": score.reasons})

    out["mcqs"] = accepted_mcqs

    # 4) One reasoning item
    out["reasoning"] = gen_reasoning(augmented, model, prompts).__dict__

    out["accepted"] = True
    return out

# ---------- Example usage (wire to your chunk loop) ----------
def run_on_chunks(chunks: List[str], model: str="gpt-5-mini") -> List[Dict[str, Any]]:
    import prompts_qa as prompts
    results = []
    for ch in chunks:
        results.append(process_chunk(ch, model, prompts, thresholds={"relevance":6, "mcq_min":7}))
    return results
