"""Local quality metrics (plan section 11). Conservative normalisation; no substring matching.

Exact match: normalised string equality (unicode NFKC, whitespace, dashes, apostrophes, case,
numeric formatting). Numeric: exact numeric equality. Answers with several items match when
the gold string equals any item (ambiguous multi-candidate answers are marked separately).
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "norm-v1"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"^(the )", "", s)
    if re.fullmatch(r"[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def is_numeric(s: str) -> bool:
    return bool(re.fullmatch(r"-?[\d,]+(\.\d+)?", (s or "").strip()))


def score_answer(pred: list[str], gold: list[str]) -> dict:
    p = [normalize(x) for x in pred if x]
    g = [normalize(x) for x in gold if x]
    if not g:
        return {"exact": None, "numeric": None, "note": "no gold"}
    out = {"exact": False, "numeric": None, "ambiguous_multi": len(p) > 1, "n_pred": len(p), "n_gold": len(g)}
    if not p:
        out["note"] = "empty prediction"
        return out
    gold_numeric = all(is_numeric(x) for x in gold)
    if gold_numeric:
        try:
            gv = {float(x) for x in g}
            pv = {float(x) for x in p if is_numeric(x)}
            out["numeric"] = bool(pv) and pv == gv
            out["exact"] = out["numeric"]
        except ValueError:
            out["numeric"] = False
        return out
    # non-numeric: set equality is exact; gold contained in a multi-candidate answer is 'contains'
    out["exact"] = set(p) == set(g)
    out["gold_in_pred"] = all(x in p for x in g)
    return out
