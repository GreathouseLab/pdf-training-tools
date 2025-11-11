
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import json
import argparse
import sys
from textwrap import dedent

# ------------------------------
# Data model (lightweight schema)
# ------------------------------

@dataclass
class PaperMeta:
    title: str = "NA"
    venue: str = "NA"
    year: str = "NA"
    study_type: str = "NA"  # guideline, RCT, meta, mechanistic, observational, case, review
    population: str = "NA"
    cancer_types: str = "NA"
    treatments: str = "NA"
    country_or_setting: str = "NA"


@dataclass
class EvidenceSpan:
    claim_id: str
    quote: str
    location: str  # e.g., "p.12, Results", "Section 3.1", etc.


@dataclass
class TriagePayload:
    paper_meta: PaperMeta
    key_claims: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)  # doses, ranges, cutoffs, interactions
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)  # gaps/limitations/contraindications
    usefulness_score: int = 0  # 1–5

# ------------------------------
# Prompt template
# ------------------------------

SYSTEM_PROMPT = (
    "You are an oncology nutrition and pharmacology analyst. "
    "Be precise, and cite exact spans with page/section locations."
)

USER_PROMPT_TEMPLATE = """\
From the paper below, extract ONLY what is needed to create clinically useful, answerable Q&A for oncology nutrition. 
Return STRICT JSON with keys exactly as follows:
- paper_meta: {title, venue, year, study_type ∈ [guideline,RCT,meta,mechanistic,observational,case,review], population, cancer_types, treatments, country_or_setting}
- key_claims: [up to 10 one-sentence claims clinicians care about]
- constraints: [doses, ranges, cutoffs, contraindications, drug–nutrient interactions]
- evidence_spans: [{{claim_id, quote, location}}]
- uncertainties: [gaps/limitations/contraindications]
- usefulness_score: 1–5 (is this paper good fodder for Q&A?)

If a field is absent in the paper, use "NA".

Paper:
<<<BEGIN PAPER>>>
{paper_text}
<<<END PAPER>>>
"""

STRICT_JSON_INSTRUCTIONS = """\
Rules:
1) Output JSON ONLY. No prose, no markdown.
2) All strings must be double-quoted.
3) Use integers for usefulness_score (1–5). If unclear, pick your best estimate.
4) Each evidence_spans item MUST include claim_id, quote (≤ 40 words), and location.
5) If something is not specified in the paper, set the value to "NA".
"""

def build_messages(paper_text: str) -> List[Dict[str, str]]:
    """Build chat-style messages for common LLM APIs."""
    user_prompt = USER_PROMPT_TEMPLATE.format(paper_text=paper_text.strip())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STRICT_JSON_INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# ------------------------------
# Validation (no external libs)
# ------------------------------

def _is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def validate_triage_json(obj: Any) -> List[str]:
    """
    Validate the structure of the triage JSON.
    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["Root must be a JSON object."]

    # paper_meta
    pm = obj.get("paper_meta")
    if not isinstance(pm, dict):
        errors.append("paper_meta must be an object.")
    else:
        required_pm = [
            "title", "venue", "year", "study_type",
            "population", "cancer_types", "treatments", "country_or_setting"
        ]
        for key in required_pm:
            if key not in pm:
                errors.append(f"paper_meta.{key} missing.")
            else:
                if not isinstance(pm[key], str):
                    errors.append(f"paper_meta.{key} must be a string.")

    # key_claims
    kc = obj.get("key_claims")
    if not isinstance(kc, list):
        errors.append("key_claims must be a list.")
    else:
        if len(kc) > 10:
            errors.append("key_claims must have at most 10 items.")
        for i, item in enumerate(kc):
            if not _is_nonempty_string(item):
                errors.append(f"key_claims[{i}] must be a nonempty string.")

    # constraints
    constraints = obj.get("constraints")
    if not isinstance(constraints, list):
        errors.append("constraints must be a list.")
    else:
        for i, item in enumerate(constraints):
            if not _is_nonempty_string(item):
                errors.append(f"constraints[{i}] must be a nonempty string.")

    # evidence_spans
    evs = obj.get("evidence_spans")
    if not isinstance(evs, list):
        errors.append("evidence_spans must be a list.")
    else:
        for i, item in enumerate(evs):
            if not isinstance(item, dict):
                errors.append(f"evidence_spans[{i}] must be an object.")
                continue
            for key in ["claim_id", "quote", "location"]:
                if key not in item:
                    errors.append(f"evidence_spans[{i}].{key} missing.")
                else:
                    if not _is_nonempty_string(item[key]):
                        errors.append(f"evidence_spans[{i}].{key} must be a nonempty string.")
            # enforce ≤ 40 words in quote (soft check)
            if "quote" in item and isinstance(item["quote"], str):
                if len(item["quote"].split()) > 40:
                    errors.append(f"evidence_spans[{i}].quote should be ≤ 40 words.")

    # uncertainties
    un = obj.get("uncertainties")
    if not isinstance(un, list):
        errors.append("uncertainties must be a list.")
    else:
        for i, item in enumerate(un):
            if not _is_nonempty_string(item):
                errors.append(f"uncertainties[{i}] must be a nonempty string.")

    # usefulness_score
    us = obj.get("usefulness_score")
    if not isinstance(us, int):
        errors.append("usefulness_score must be an integer 1–5.")
    else:
        if not (1 <= us <= 5):
            errors.append("usefulness_score must be between 1 and 5.")

    return errors

# ------------------------------
# Optional: provider call stub
# ------------------------------

def call_model_stub(messages: List[Dict[str, str]]) -> str:
    """
    Stub to demonstrate where you would call your LLM.
    Return a JSON string. Replace this with your provider's SDK.
    """
    mock = {
        "paper_meta": {
            "title": "NA", "venue": "NA", "year": "NA", "study_type": "NA",
            "population": "NA", "cancer_types": "NA", "treatments": "NA", "country_or_setting": "NA"
        },
        "key_claims": [],
        "constraints": [],
        "evidence_spans": [],
        "uncertainties": [],
        "usefulness_score": 3,
    }
    return json.dumps(mock, ensure_ascii=False)

# ------------------------------
# CLI
# ------------------------------

CLI_HELP = dedent("""\
    Oncology-Nutrition Paper Triage Prompt Builder
    ----------------------------------------------
    This tool creates a strict JSON-extraction prompt (system+user) for an LLM,
    to distill clinically useful, answerable Q&A ingredients from a paper.

    Examples:
      # Build prompts for a paper.txt and print them as JSON lines:
      python onc_nutri_triage_prompt.py --paper paper.txt --print-messages

      # Call your model (stubbed here), validate, and write result:
      python onc_nutri_triage_prompt.py --paper paper.txt --run-stub --out triage.json
""")

def cli():
    parser = argparse.ArgumentParser(description="Oncology-Nutrition Triage Prompt Builder", epilog=CLI_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=str, help="Path to a text file containing the paper (abstract or full text). If omitted, reads from stdin.")
    parser.add_argument("--print-messages", action="store_true", help="Print the constructed messages (system+user) as a JSON list.")
    parser.add_argument("--run-stub", action="store_true", help="Run the stub model call and validate output.")
    parser.add_argument("--out", type=str, help="Where to write the model JSON output (when --run-stub).")
    args = parser.parse_args()

    # Read paper text
    if args.paper:
        with open(args.paper, "r", encoding="utf-8") as f:
            paper_text = f.read()
    else:
        paper_text = sys.stdin.read()

    messages = build_messages(paper_text)

    if args.print_messages:
        print(json.dumps(messages, ensure_ascii=False, indent=2))

    if args.run_stub:
        raw = call_model_stub(messages)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Model output is not valid JSON: {e}", file=sys.stderr)
            sys.exit(2)

        errors = validate_triage_json(obj)
        if errors:
            print("[INVALID] The output failed validation with the following issues:", file=sys.stderr)
            for err in errors:
                print(" - " + err, file=sys.stderr)
            sys.exit(3)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            print(f"[OK] Wrote validated triage JSON to: {args.out}")
        else:
            print(json.dumps(obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # Avoid running the CLI when this file is imported programmatically.
    # Only run if actual CLI flags (other than Jupyter's) are present.
    if any(arg for arg in sys.argv[1:] if arg.startswith("--")):
        cli()
    else:
        # Print a short usage hint instead of erroring in notebook contexts.
        print("onc_nutri_triage_prompt.py ready. Run with --help for usage.", file=sys.stderr)
