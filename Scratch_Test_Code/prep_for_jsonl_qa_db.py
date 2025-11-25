# --- helpers ---
def _norm_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    # collapse internal whitespace a bit (simple version)
    return " ".join(s.split())

def _answer_letter(idx):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if isinstance(idx, int) and 0 <= idx < len(letters):
        return letters[idx]
    return None

# --- 1) canonicalize: normalize shape, no opinions ---
def canonicalize(rec):
    typ = rec.get("type", "qa")
    if typ not in ("mcq", "qa"):
        typ = "qa"

    question = _norm_text(rec.get("question"))
    if typ == "mcq":
        options = [ _norm_text(o) for o in (rec.get("options") or []) ]
        answer_index = rec.get("answer_index")
        # turn "1" -> 1 safely
        if answer_index is not None:
            try:
                answer_index = int(answer_index)
            except (ValueError, TypeError):
                answer_index = None
        return {
            "type": "mcq",
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "answer_letter": _answer_letter(answer_index),
            "answer_text": None,   # not used for mcq
        }
    else:
        answer_text = _norm_text(rec.get("answer_text"))
        return {
            "type": "qa",
            "question": question,
            "options": None,       # not used for qa
            "answer_index": None,
            "answer_letter": None,
            "answer_text": answer_text,
        }

# --- 2) validate: apply simple rules & return verdict ---
def validate(canon):
    flags = []
    if not canon.get("question"):
        flags.append("missing-question")

    if canon["type"] == "mcq":
        opts = canon.get("options") or []
        if len(opts) < 3:
            flags.append("options-too-few")
        if canon.get("answer_index") is None:
            flags.append("missing-answer_index")
        elif not (0 <= canon["answer_index"] < len(opts)):
            flags.append("answer_index-out-of-range")
    else:  # qa
        if not canon.get("answer_text"):
            flags.append("missing-answer_text")

    return {"good": len(flags) == 0, "flags": flags}