"""Validate a frozen run and package the submission artifacts.

Fails visibly on missing records, duplicate (qid, pipeline) keys, broken trace references or
missing usage. Hidden outputs are exported raw (answers, tokens, traces) without any accuracy claim.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from podium.settings import HIDDEN_QUESTIONS, MANIFESTS, PUBLIC_QUESTIONS, RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
run_dir = RUNS / a.run
out = Path(a.out) if a.out else run_dir / "submission"
out.mkdir(parents=True, exist_ok=True)
HEADLINE = ["rag", "graphrag", "agentic_graphrag"]
problems = []


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def check(split, questions_path):
    recs = load(run_dir / f"predictions_{split}.jsonl")
    qids = [json.loads(l)["qid"] for l in open(questions_path, encoding="utf-8")]
    keys = [(r["qid"], r["pipeline"]) for r in recs if r["pipeline"] in HEADLINE]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        problems.append(f"{split}: duplicate records {sorted(dupes)[:5]}")
    expected = {(q, p) for q in qids for p in HEADLINE}
    missing = expected - set(keys)
    if missing:
        problems.append(f"{split}: {len(missing)} missing (qid, pipeline) records, e.g. {sorted(missing)[:5]}")
    for r in recs:
        res = r["result"]
        if res["usage"]["llm_calls"] == 0 and res["status"] != "error":
            problems.append(f"{split}: {r['qid']}/{r['pipeline']} has no LLM usage recorded")
        if not res["trace"].get("trace_id"):
            problems.append(f"{split}: {r['qid']}/{r['pipeline']} has no trace id")
        for c in res["citations"]:
            if not c.get("doc_id"):
                problems.append(f"{split}: {r['qid']}/{r['pipeline']} citation without doc_id")
    # export compact + full
    full = [r for r in recs if r["pipeline"] in HEADLINE]
    with open(out / f"predictions_{split}.jsonl", "w", encoding="utf-8") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    compact = []
    for r in full:
        res = r["result"]
        compact.append({
            "qid": r["qid"], "pipeline": r["pipeline"], "question": r["question"], "answer": res["answer"], "explanation": res["explanation"], "status": res["status"], "stop_reason": res["stop_reason"],
            "tokens": {"context_tokens": res["usage"]["context_tokens"], "llm_input_tokens": res["usage"]["input_tokens"], "llm_output_tokens": res["usage"]["output_tokens"], "total_tokens": res["usage"]["total_tokens"], "embedding_tokens": res["usage"]["embedding_tokens"], "llm_calls": res["usage"]["llm_calls"]},
            "latency_ms": res["latency_ms"],
            "citations": [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "title": c["title"], "url": c["url"], "char_start": c["char_start"], "char_end": c["char_end"], "quote": c["quote"]} for c in res["citations"]],
            "trace": {"trace_id": res["trace"]["trace_id"], "n_steps": len(res["trace"]["events"]), "n_retrieval_steps": res["trace"]["n_retrieval_steps"], "n_reasoning_steps": res["trace"]["n_reasoning_steps"], "retrieval_methods": res["trace"]["retrieval_methods"], "specialists_invoked": res["trace"]["specialists_invoked"], "tools_called": res["trace"]["tools_called"], "strategy_changed": res["trace"]["strategy_changed"], "stop_reason": res["trace"]["stop_reason"], "stop_explanation": res["trace"].get("stop_explanation", ""), "n_chunks": res["trace"]["n_chunks"], "n_citations": res["trace"]["n_citations"],
                      "steps": [{"step": e["step"], "specialist": e["specialist"], "tool": e["tool"], "execution_kind": e["execution_kind"], "arguments": e["arguments"], "result": e["result_summary"], "latency_ms": round(e["latency_ms"], 1), "tokens": e["tokens"], "strategy_change": e["strategy_change"]} for e in res["trace"]["events"]]},
            "model_id": r["model_id"], "graph_snapshot_id": r["graph_snapshot_id"], "corpus_hash": r["corpus_hash"], "config_hash": r["config_hash"], "code_commit": r["code_commit"], "recorded_at": r["recorded_at"],
        })
    with open(out / f"predictions_{split}_compact.jsonl", "w", encoding="utf-8") as f:
        for r in compact:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(full)


n_pub = check("public", PUBLIC_QUESTIONS)
n_hid = check("hidden", HIDDEN_QUESTIONS)
for name in ("summary_public.json", "scores_public.jsonl", "config_snapshot.json", "summary_robustness.json", "scores_robustness.jsonl", "predictions_robustness.jsonl"):
    if (run_dir / name).exists():
        shutil.copy(run_dir / name, out / name)
for name in ("ingestion_manifest.json", "graph_verification.json"):
    if (MANIFESTS / name).exists():
        shutil.copy(MANIFESTS / name, out / name)
manifest = {"run_id": a.run, "public_records": n_pub, "hidden_records": n_hid, "pipelines": HEADLINE, "problems": problems, "note": "hidden predictions are raw outputs; their accuracy is unknown to us"}
json.dump(manifest, open(out / "submission_manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
if problems:
    print("EXPORT FAILED", file=sys.stderr)
    sys.exit(1)
print("EXPORT OK ->", out)
