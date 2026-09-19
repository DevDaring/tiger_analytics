"""Offline evaluation: scores per record, citation validity against the frozen corpus,
gold-document recall, LLM judge, and aggregate summaries by pipeline and question type.

Gold answers are read here and only here - after predictions are written.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..settings import CORPUS_PATH, RUNS
from .judge import judge
from .runner import load_questions
from .scoring import NORMALIZATION_VERSION, score_answer


@lru_cache(maxsize=1)
def corpus_text() -> dict[str, str]:
    out = {}
    for line in open(CORPUS_PATH, encoding="utf-8"):
        d = json.loads(line)
        out[d["doc_id"]] = d["text"]
    return out


def citation_validity(citations: list[dict]) -> dict:
    texts = corpus_text()
    n = len(citations)
    valid = 0
    for c in citations:
        t = texts.get(c.get("doc_id"))
        if t is None:
            continue
        cs, ce, q = c.get("char_start"), c.get("char_end"), c.get("quote") or ""
        if cs is not None and ce is not None and 0 <= cs <= ce <= len(t):
            span = t[cs:ce]
            if not q or q.strip()[:80] in span or span.strip()[:80] in q:
                valid += 1
                continue
        if q and q[:120] in t:
            valid += 1
    return {"n": n, "valid": valid, "all_valid": n > 0 and valid == n}


def pct(x, n):
    return round(100.0 * x / n, 1) if n else None


def p(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def evaluate(run_id: str, split: str = "public", use_judge: bool = True) -> dict:
    run_dir = RUNS / run_id
    preds = [json.loads(l) for l in open(run_dir / f"predictions_{split}.jsonl", encoding="utf-8")]
    gold = {q["qid"]: q for q in load_questions(split)} if split != "hidden" else {}
    scored = []
    for r in preds:
        res = r["result"]
        g = gold.get(r["qid"], {})
        s = {"qid": r["qid"], "pipeline": r["pipeline"], "qtype": r.get("qtype") or g.get("qtype"), "status": res["status"], "stop_reason": res["stop_reason"], "answer": res["answer"], "gold": g.get("answer"), "tokens": res["usage"]["total_tokens"], "input_tokens": res["usage"]["input_tokens"], "output_tokens": res["usage"]["output_tokens"], "context_tokens": res["usage"]["context_tokens"], "llm_calls": res["usage"]["llm_calls"], "latency_ms": res["latency_ms"], "n_steps": len(res["trace"]["events"]), "n_retrieval_steps": res["trace"]["n_retrieval_steps"], "n_llm_decisions": res["trace"]["n_llm_decisions"], "strategy_changed": res["trace"]["strategy_changed"], "n_citations": len(res["citations"]), "n_chunks": res["trace"]["n_chunks"], "coverage_complete": res["coverage"]["complete"], "unknown_values": res["coverage"]["unknown_values"], "error": res.get("error")}
        if g.get("answer") is not None:
            s.update({f"em_{k}": v for k, v in score_answer(res["answer"], g["answer"]).items()})
            s["correct"] = bool(s.get("em_exact"))
            gd = set(g.get("gold_doc_ids") or [])
            cited_docs = {c["doc_id"] for c in res["citations"]}
            examined = set(res["coverage"].get("examined_doc_ids") or [])
            s["gold_doc_recall_cited"] = round(len(gd & cited_docs) / len(gd), 3) if gd else None
            s["gold_doc_recall_examined"] = round(len(gd & (examined | cited_docs)) / len(gd), 3) if gd else None
            if use_judge and res["status"] != "error":
                j = judge(r["question"], res["answer"], res.get("explanation", ""), g["answer"], res["citations"])
                s["judge_correct"], s["judge_completeness"], s["judge_support"], s["judge_justification"], s["judge_failed"] = j["correct"], j["completeness"], j["evidence_support"], j["justification"], j["failed"]
        cv = citation_validity(res["citations"])
        s["citations_valid"] = cv["valid"]
        s["citations_all_valid"] = cv["all_valid"]
        scored.append(s)
    with open(run_dir / f"scores_{split}.jsonl", "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = summarize(scored, split)
    summary["run_id"] = run_id
    summary["normalization_version"] = NORMALIZATION_VERSION
    json.dump(summary, open(run_dir / f"summary_{split}.json", "w"), indent=1, ensure_ascii=False)
    return summary


def summarize(scored: list[dict], split: str) -> dict:
    by_p = defaultdict(list)
    for s in scored:
        by_p[s["pipeline"]].append(s)
    has_gold = split != "hidden"

    def agg(rows):
        n = len(rows)
        toks = [r["tokens"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        out = {
            "n": n,
            "answered": sum(1 for r in rows if r["status"] == "answered"),
            "partial": sum(1 for r in rows if r["status"] == "partial"),
            "abstained": sum(1 for r in rows if r["status"] == "abstained"),
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "mean_tokens": round(statistics.mean(toks), 1) if toks else None,
            "median_tokens": statistics.median(toks) if toks else None,
            "total_tokens": sum(toks),
            "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows), 1) if rows else None,
            "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1) if rows else None,
            "mean_context_tokens": round(statistics.mean(r["context_tokens"] for r in rows), 1) if rows else None,
            "mean_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 2) if rows else None,
            "p50_latency_ms": p(lat, 0.5),
            "p95_latency_ms": p(lat, 0.95),
            "mean_steps": round(statistics.mean(r["n_steps"] for r in rows), 2) if rows else None,
            "mean_retrieval_steps": round(statistics.mean(r["n_retrieval_steps"] for r in rows), 2) if rows else None,
            "strategy_change_rate": pct(sum(1 for r in rows if r["strategy_changed"]), n),
            "citations_all_valid_rate": pct(sum(1 for r in rows if r["citations_all_valid"]), n),
            "coverage_complete_rate": pct(sum(1 for r in rows if r["coverage_complete"]), n),
            "stop_reasons": dict(sorted(((k, sum(1 for r in rows if r["stop_reason"] == k)) for k in {r["stop_reason"] for r in rows}), key=lambda x: -x[1])),
        }
        if has_gold:
            out["accuracy_exact"] = pct(sum(1 for r in rows if r.get("correct")), n)
            out["accuracy_exact_answered_only"] = pct(sum(1 for r in rows if r.get("correct")), sum(1 for r in rows if r["status"] in ("answered", "partial")))
            judged = [r for r in rows if r.get("judge_correct") is not None]
            out["judged_n"] = len(judged)
            out["accuracy_judge"] = pct(sum(1 for r in judged if r["judge_correct"]), len(judged))
            out["completeness_judge"] = round(statistics.mean(r["judge_completeness"] for r in judged), 3) if judged else None
            out["evidence_support_judge"] = round(statistics.mean(r["judge_support"] for r in judged), 3) if judged else None
            rec = [r["gold_doc_recall_examined"] for r in rows if r.get("gold_doc_recall_examined") is not None]
            out["gold_doc_recall_examined"] = round(statistics.mean(rec), 3) if rec else None
            rec2 = [r["gold_doc_recall_cited"] for r in rows if r.get("gold_doc_recall_cited") is not None]
            out["gold_doc_recall_cited"] = round(statistics.mean(rec2), 3) if rec2 else None
            correct = sum(1 for r in rows if r.get("correct"))
            out["tokens_per_correct_answer"] = round(sum(toks) / correct, 1) if correct else None
        return out

    summary = {"split": split, "pipelines": {k: agg(v) for k, v in by_p.items()}, "by_qtype": {}}
    qtypes = sorted({s["qtype"] for s in scored if s.get("qtype")})
    for qt in qtypes:
        summary["by_qtype"][qt] = {k: agg([r for r in v if r["qtype"] == qt]) for k, v in by_p.items()}
    # paired differences between graphrag and agentic
    if has_gold and "graphrag" in by_p and "agentic_graphrag" in by_p:
        g = {r["qid"]: r for r in by_p["graphrag"]}
        a = {r["qid"]: r for r in by_p["agentic_graphrag"]}
        common = sorted(set(g) & set(a))
        wins = [q for q in common if a[q].get("correct") and not g[q].get("correct")]
        losses = [q for q in common if g[q].get("correct") and not a[q].get("correct")]
        extra_tokens = sum(a[q]["tokens"] - g[q]["tokens"] for q in common)
        summary["paired_graphrag_vs_agentic"] = {
            "n": len(common), "agent_wins": wins, "agent_losses": losses,
            "quality_lift_pct_points": round(100 * (len(wins) - len(losses)) / len(common), 1) if common else None,
            "mean_token_delta": round(extra_tokens / len(common), 1) if common else None,
            "extra_tokens_per_extra_correct": round(extra_tokens / (len(wins) - len(losses)), 1) if len(wins) > len(losses) else "N/A",
            "bootstrap_95ci_lift": _bootstrap_ci([(1 if a[q].get("correct") else 0) - (1 if g[q].get("correct") else 0) for q in common]),
        }
    summary["failures"] = [{"qid": s["qid"], "pipeline": s["pipeline"], "qtype": s["qtype"], "answer": s["answer"], "gold": s["gold"], "status": s["status"], "stop_reason": s["stop_reason"], "error": s.get("error")} for s in scored if has_gold and not s.get("correct")]
    return summary


def _bootstrap_ci(diffs: list[int], n_boot: int = 2000, seed: int = 7) -> list[float] | None:
    import random

    if not diffs:
        return None
    rnd = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        s = [diffs[rnd.randrange(n)] for _ in range(n)]
        means.append(100 * sum(s) / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 1), round(means[int(0.975 * n_boot)], 1)]
