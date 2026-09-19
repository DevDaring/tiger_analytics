"""LLM-as-judge (different model family from the answer model), blind to pipeline labels.

Scores correctness, completeness and evidence support against the reference answer and the
cited corpus spans. Judgments are cached by (question, prediction, evidence, rubric, model).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import stable_hash
from ..llm.client import complete_json
from ..settings import CACHE, settings

RUBRIC_VERSION = "judge-rubric-v1"
SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean", "description": "the predicted answer states the same fact as the reference"},
        "completeness": {"type": "number", "description": "0-1: fraction of the requested answer content provided (entity, quantity, scope)"},
        "evidence_support": {"type": "number", "description": "0-1: how well the cited spans support the answer"},
        "justification": {"type": "string"},
    },
}
SYSTEM = """You are a strict evaluator for a question-answering benchmark over a frozen corpus.
Judge ONLY by comparing the prediction with the reference answer and the cited evidence spans.
- correct=true if the prediction's answer denotes the same value/entity as the reference (formatting
  differences such as spacing, accents or concatenated team-member names are acceptable; a different
  number or a different person is wrong). If the prediction lists several candidates and one of them is
  the reference while the others are wrong, correct=false but completeness may be partial.
- completeness in [0,1]: does the answer fully address what was asked (entity, quantity, scope)?
- evidence_support in [0,1]: do the cited spans actually support the stated answer?"""

_cache_path = CACHE / "judge_cache.jsonl"
_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = {}
        if _cache_path.exists():
            for line in open(_cache_path, encoding="utf-8"):
                r = json.loads(line)
                _cache[r["key"]] = r["value"]
    return _cache


def judge(question: str, prediction: list[str], explanation: str, reference: list[str], citations: list[dict]) -> dict:
    ev = "\n".join(f"- [{c.get('title','')}] \"{(c.get('quote') or '')[:300]}\"" for c in citations[:6])
    key = stable_hash([question, prediction, explanation[:500], reference, ev, RUBRIC_VERSION, str(settings.judge_model)])
    cache = _load()
    if key in cache:
        return {**cache[key], "cached": True}
    user = f"Question: {question}\nReference answer: {json.dumps(reference, ensure_ascii=False)}\n\nPrediction answer: {json.dumps(prediction, ensure_ascii=False)}\nPrediction explanation: {explanation[:1200]}\n\nCited evidence spans:\n{ev or '(none)'}"
    try:
        data, u = complete_json("judge", SYSTEM, user, SCHEMA, model=settings.judge_model, max_tokens=400)
        val = {"correct": bool(data.get("correct")), "completeness": float(data.get("completeness", 0)), "evidence_support": float(data.get("evidence_support", 0)), "justification": data.get("justification", ""), "judge_model": str(settings.judge_model), "judge_tokens": u.total, "failed": False}
    except Exception as e:  # noqa: BLE001
        val = {"correct": None, "completeness": None, "evidence_support": None, "justification": f"judge failed: {str(e)[:200]}", "judge_model": str(settings.judge_model), "judge_tokens": 0, "failed": True}
        return val
    cache[key] = val
    with open(_cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": val}, ensure_ascii=False) + "\n")
    return val
