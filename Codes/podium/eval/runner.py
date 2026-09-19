"""Benchmark runner: runs pipelines on question files, writes append-only prediction records,
and is resumable. The inference call receives question text only; qid/qtype/gold are attached
to the envelope afterwards (plan section 10/11).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..contracts import AnswerResult, PredictionRecord, stable_hash
from ..settings import HIDDEN_QUESTIONS, MANIFESTS, PUBLIC_QUESTIONS, ROOT, RUNS, settings

PIPELINES = {
    "rag": lambda q: __import__("podium.pipelines.rag", fromlist=["run"]).run(q),
    "graphrag": lambda q: __import__("podium.pipelines.graphrag", fromlist=["run"]).run(q),
    "agentic_graphrag": lambda q: __import__("podium.agent.orchestrator", fromlist=["run"]).run(q, "full"),
    "fixed_plan": lambda q: __import__("podium.agent.orchestrator", fromlist=["run"]).run(q, "fixed_plan"),
    "agentic_nocov": lambda q: __import__("podium.agent.orchestrator", fromlist=["run"]).run(q, "nocov"),
}
HEADLINE = ["rag", "graphrag", "agentic_graphrag"]


def load_questions(split: str) -> list[dict]:
    if split == "public":
        path = PUBLIC_QUESTIONS
    elif split == "hidden":
        path = HIDDEN_QUESTIONS
    elif split == "robustness":
        path = ROOT / "configs" / "robustness_questions.jsonl"
    else:
        path = Path(split)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "uncommitted"


def snapshot_ids() -> tuple[str, str]:
    m = json.load(open(MANIFESTS / "ingestion_manifest.json"))
    return m["corpus_sha256"], m["graph_snapshot_id"]


def run_one(pipeline: str, question: str) -> AnswerResult:
    return PIPELINES[pipeline](question)


def run_benchmark(run_id: str, split: str, pipelines: list[str] | None = None, qids: list[str] | None = None, concurrency: int = 3, limit: int | None = None) -> Path:
    pipelines = pipelines or HEADLINE
    qs = load_questions(split)
    if qids:
        qs = [q for q in qs if q["qid"] in set(qids)]
    if limit:
        qs = qs[:limit]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"predictions_{split if split in ('public','hidden','robustness') else 'custom'}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["qid"], r["pipeline"]))
    corpus_hash, snap = snapshot_ids()
    config = {"answer_model": str(settings.answer_model), "embed_model": str(settings.embed_model), "rag_top_k": settings.rag_top_k, "context_char_budget": settings.context_char_budget, "agent_limits": {"decisions": settings.agent_max_decisions, "tool_calls": settings.agent_max_tool_calls, "tokens": settings.agent_token_budget, "wall_clock_s": settings.agent_wall_clock_s}}
    cfg_hash = stable_hash(config)
    json.dump({**config, "config_hash": cfg_hash, "corpus_hash": corpus_hash, "graph_snapshot_id": snap, "code_commit": code_commit(), "run_id": run_id, "split": split, "pipelines": pipelines, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(run_dir / "config_snapshot.json", "w"), indent=1)
    jobs = [(q, p) for q in qs for p in pipelines if (q["qid"], p) not in done]
    print(f"run {run_id}/{split}: {len(jobs)} jobs ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    t0 = time.time()

    def work(q, p):
        # Inference boundary: only the question text crosses it.
        res = run_one(p, q["question"])
        rec = PredictionRecord(run_id=run_id, qid=q["qid"], pipeline=p, question=q["question"], qtype=q.get("qtype"), result=res, model_id=str(settings.answer_model), model_parameters={"effort": "low", "answer_model": str(settings.answer_model)}, prompt_hash=stable_hash([p, "prompts-v1"]), corpus_hash=corpus_hash, graph_snapshot_id=snap, config_hash=cfg_hash, code_commit=code_commit(), recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return rec

    n = 0
    with ThreadPoolExecutor(concurrency) as ex, open(out_path, "a", encoding="utf-8") as out:
        futs = {ex.submit(work, q, p): (q, p) for q, p in jobs}
        for f in as_completed(futs):
            q, p = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                rec = PredictionRecord(run_id=run_id, qid=q["qid"], pipeline=p, question=q["question"], qtype=q.get("qtype"), result=AnswerResult(status="error", stop_reason="error", error=str(e)[:300]), model_id=str(settings.answer_model), corpus_hash=corpus_hash, graph_snapshot_id=snap, config_hash=cfg_hash)
            with lock:
                out.write(rec.model_dump_json() + "\n")
                out.flush()
                n += 1
                r = rec.result
                print(f"[{n}/{len(jobs)} {time.time()-t0:.0f}s] {q['qid']} {p:18s} {r.status:9s} {r.stop_reason:20s} tok={r.usage.total_tokens:6d} {r.latency_ms/1000:5.1f}s -> {r.answer[:2]}", flush=True)
    return out_path
