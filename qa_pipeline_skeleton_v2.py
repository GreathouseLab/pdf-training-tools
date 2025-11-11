#updated to qps pipeline skeleton v2.py

"""
Refactored skeleton to use explicit dependency injection.
This file defines the abstract 5-stage pipeline for a single chunk.
It NO LONGER has its own llm_chat function. Instead, the "real"
functions are passed in from the trainer.
"""

import json, random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable

# ---------- Messages helper ----------
def msgs(sys: str, user: str) -> List[Dict[str, str]]:
    return [{"role":"system","content": sys}, {"role":"user","content": user}]

# ---------- Stage outputs (Data Contracts) ----------
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

# Type hint for the LLM function we expect to be passed in
# It takes (model, messages, temp, max_tokens) and returns a string
LLMFunction = Callable[[str, List[Dict[str, str]], float, int], str]
# Type hint for the JSON parser function
JSONParserFunction = Callable[[str], Any]

# ---------- Stage 1: Gate by relevance ----------
def gate_chunk(
    chunk_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction
) -> Optional[RelevanceResult]:
    
    user = prompts.RELEVANCE_USER.format(chunk_text=chunk_text, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    # Call the injected LLM function
    out_str = llm_fn(model, msgs(prompts.RELEVANCE_SYSTEM, user), 0.2, 1024)
    # Call the injected JSON parser
    data = json_parser_fn(out_str)
    
    return RelevanceResult(score=int(data["relevance_score"]),
                           content_type=str(data["content_type"]),
                           reason=str(data["reason"]))

# ---------- Stage 2: Augment chunk ----------
def augment_chunk(
    chunk_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction
) -> str:
    user = prompts.AUGMENT_USER.format(chunk_text=chunk_text)
    # Call the injected LLM function
    return llm_fn(model, msgs(prompts.AUGMENT_SYSTEM, user), 0.2, 1024).strip()

# ---------- Stage 3: Generate MCQ ----------
def gen_mcq(
    context_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction,
    num_answers: int=4,
    seed: Optional[int]=None
) -> MCQ:
    
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
    # Call the injected LLM function
    out_str = llm_fn(model, msgs(prompts.MCQ_SYSTEM, user), 0.2, 1024)
    # Call the injected JSON parser
    data = json_parser_fn(out_str)
    
    return MCQ(question=data["question"],
               options=data["options"],
               correct_index=int(data["correct_index"]),
               tags=data.get("tags", []),
               difficulty=data.get("difficulty","intermediate"))

# ---------- Stage 4: Score MCQ ----------
def score_mcq(
    mcq: MCQ,
    model: str,
    prompts,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction
) -> MCQScore:
    
    mcq_json = json.dumps(mcq.__dict__, ensure_ascii=False)
    user = prompts.MCQ_SCORE_USER.format(mcq_json=mcq_json, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    # Call the injected LLM function
    out_str = llm_fn(model, msgs(prompts.MCQ_SCORE_SYSTEM, user), 0.2, 1024)
    # Call the injected JSON parser
    data = json_parser_fn(out_str)
    
    return MCQScore(score=int(data["score"]), reasons=str(data["reasons"]), disqualified=bool(data["disqualified"]))

# ---------- Stage 5: Reasoning/free-response ----------
def gen_reasoning(
    context_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction
) -> ReasoningItem:
    
    user = prompts.REASON_USER.format(context_text=context_text, strict_json=prompts.STRICT_JSON_INSTRUCTIONS)
    # Call the injected LLM function
    out_str = llm_fn(model, msgs(prompts.REASON_SYSTEM, user), 0.2, 1024)
    # Call the injected JSON parser
    data = json_parser_fn(out_str)
    
    return ReasoningItem(question=data["question"],
                         answer_key=data["answer_key"],
                         rubric=data["rubric"],
                         tags=data.get("tags",[]),
                         difficulty=data.get("difficulty","advanced"))

# ---------- Orchestration per chunk ----------
def process_chunk(
    chunk_text: str,
    model: str,
    prompts,
    thresholds: dict,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction
) -> Dict[str, Any]:
    """
    Returns a dict with gated flag, augmented text, a list of accepted MCQs, and one reasoning item.
    thresholds = {"relevance": 6, "mcq_min": 7}
    """
    out: Dict[str, Any] = {"accepted": False, "relevance": None, "augmented": None, "mcqs": [], "reasoning": None}

    try:
        # 1) Relevance gate
        # Pass the injected functions down to the stage
        rel = gate_chunk(chunk_text, model, prompts, llm_fn, json_parser_fn)
        out["relevance"] = rel.__dict__
        if rel.score < thresholds.get("relevance", 6) or rel.content_type in {"references","metadata","copyright"}:
            return out  # rejected

        # 2) Augment
        augmented = augment_chunk(chunk_text, model, prompts, llm_fn)
        out["augmented"] = augmented

        # 3) Multiple MCQs with scoring loop
        accepted_mcqs: List[Dict[str, Any]] = []
        for _ in range(3):  # generate up to 3 MCQs per chunk
            try:
                mcq = gen_mcq(augmented, model, prompts, llm_fn, json_parser_fn, num_answers=4)
                score = score_mcq(mcq, model, prompts, llm_fn, json_parser_fn)
                if (not score.disqualified) and (score.score >= thresholds.get("mcq_min", 7)):
                    accepted_mcqs.append({**mcq.__dict__, "score": score.score, "critique": score.reasons})
            except Exception as e:
                logging.warning(f"MCQ generation/scoring failed for chunk: {e}")
                pass # Continue to next attempt

        out["mcqs"] = accepted_mcqs

        # 4) One reasoning item
        try:
            out["reasoning"] = gen_reasoning(augmented, model, prompts, llm_fn, json_parser_fn).__dict__
        except Exception as e:
            logging.warning(f"Reasoning generation failed for chunk: {e}")
            out["reasoning"] = None

        out["accepted"] = True
        
    except Exception as e:
        logging.error(f"Error processing chunk: {e}. Chunk text (start): {chunk_text[:100]}...")
        out["accepted"] = False
        # Optionally, store the error in the output dict
        out["error"] = str(e)

    return out