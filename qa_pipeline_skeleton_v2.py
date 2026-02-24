# qa_pipeline_skeleton_v2.py
# SIMPLIFIED 2026-02-10: Freeform-only pipeline
# Removed: MCQ generation, MCQ scoring, reasoning questions
# Rationale: MCQ answers were 7-15 words (ungradable), duplicated within chunks,
# and reasoning was redundant (same format as MCQ after conversion).
# Freeform already produces clean 1-2 word exact-match answers.
#
# This skeleton now does TWO things:
#   1) gate_chunk()  — relevance filter (skip junk chunks)
#   2) augment_chunk() — enrich text for downstream freeform generation
#
# Freeform Q/A generation happens in the trainer (mupdf_trainer_v3.py)
# because it uses the trainer's LLM adapter directly.

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

# ---------- Messages helper ----------
def msgs(sys: str, user: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

# ---------- Data Contracts ----------
@dataclass
class RelevanceResult:
    score: int
    content_type: str
    reason: str

# Type hints for injected functions
LLMFunction = Callable[[str, List[Dict[str, str]], float, int], str]
JSONParserFunction = Callable[[str], Any]

# ---------- Stage 1: Gate by relevance ----------
def gate_chunk(
    chunk_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction,
    json_parser_fn: JSONParserFunction
) -> Optional[RelevanceResult]:
    
    user = prompts.RELEVANCE_USER.format(
        chunk_text=chunk_text,
        strict_json=prompts.STRICT_JSON_INSTRUCTIONS
    )
    out_str = llm_fn(model, msgs(prompts.RELEVANCE_SYSTEM, user), 0.2, 1024)
    data = json_parser_fn(out_str)
    
    return RelevanceResult(
        score=int(data["relevance_score"]),
        content_type=str(data["content_type"]),
        reason=str(data.get("reasoning", data.get("reason", "")))
    )

# ---------- Stage 2: Augment chunk ----------
def augment_chunk(
    chunk_text: str,
    model: str,
    prompts,
    llm_fn: LLMFunction
) -> str:
    user = prompts.AUGMENT_USER.format(chunk_text=chunk_text)
    return llm_fn(model, msgs(prompts.AUGMENT_SYSTEM, user), 0.2, 1024).strip()

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
    Gates and augments a single chunk. Returns dict with:
      - accepted: bool
      - relevance: dict with score, content_type, reason
      - augmented: str (enriched text for freeform generation)
    
    thresholds = {"relevance": 6}
    
    LLM calls: exactly 2 per accepted chunk (gate + augment)
    """
    out: Dict[str, Any] = {
        "accepted": False,
        "relevance": None,
        "augmented": None
    }

    try:
        # 1) Relevance gate
        rel = gate_chunk(chunk_text, model, prompts, llm_fn, json_parser_fn)
        out["relevance"] = rel.__dict__
        if rel.score < thresholds.get("relevance", 6) or \
           rel.content_type in {"references", "metadata", "copyright"}:
            return out  # rejected

        # 2) Augment
        augmented = augment_chunk(chunk_text, model, prompts, llm_fn)
        out["augmented"] = augmented
        out["accepted"] = True
        
    except Exception as e:
        logging.error(f"Error processing chunk: {e}. Chunk text (start): {chunk_text[:100]}...")
        out["accepted"] = False
        out["error"] = str(e)

    return out
