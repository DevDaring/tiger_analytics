"""FastAPI service: live three-way inference, recorded-run replay, artifacts, and the dashboard.

Run: uvicorn podium.api:app --host 127.0.0.1 --port 8120
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .contracts import AnswerResult, PredictionRecord, stable_hash
from .eval.runner import HEADLINE, PIPELINES, load_questions, run_one, snapshot_ids
from .settings import CACHE, CORPUS_PATH, MANIFESTS, ROOT, RUNS, settings

app = FastAPI(title="Podium - Evidence-Driven Agentic GraphRAG", version="1.0")
WEB = ROOT / "web"
_lock = threading.Lock()
_live_dir = RUNS / "live"
_live_dir.mkdir(parents=True, exist_ok=True)
_corpus: dict[str, dict] | None = None


def corpus() -> dict[str, dict]:
    global _corpus
    if _corpus is None:
        c = {}
        for line in open(CORPUS_PATH, encoding="utf-8"):
            d = json.loads(line)
            c[d["doc_id"]] = d
        _corpus = c
    return _corpus


class AskRequest(BaseModel):
    question: str
    pipelines: list[str] = HEADLINE


@app.get("/api/health")
def health():
    out = {"settings": settings.redacted(), "time": time.time()}
    try:
        from .db.tg import get_tg

        tg = get_tg()
        out["tigergraph"] = {"reachable": tg.ping(), "host": tg.host, "graph": tg.graph}
        if (MANIFESTS / "graph_verification.json").exists():
            out["graph_verification"] = json.load(open(MANIFESTS / "graph_verification.json"))
    except Exception as e:  # noqa: BLE001
        out["tigergraph"] = {"reachable": False, "error": str(e)[:200]}
    if (MANIFESTS / "ingestion_manifest.json").exists():
        out["ingestion"] = json.load(open(MANIFESTS / "ingestion_manifest.json"))
    return out


@app.post("/api/ask")
def ask(req: AskRequest):
    pipes = [p for p in req.pipelines if p in PIPELINES]
    if not pipes:
        raise HTTPException(400, "no valid pipeline")
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    corpus_hash, snap = snapshot_ids()
    results: dict[str, AnswerResult] = {}
    with ThreadPoolExecutor(len(pipes)) as ex:
        futs = {p: ex.submit(run_one, p, q) for p in pipes}
        for p, f in futs.items():
            try:
                results[p] = f.result()
            except Exception as e:  # noqa: BLE001
                results[p] = AnswerResult(status="error", stop_reason="error", error=str(e)[:300])
    recs = []
    live_id = "live-" + stable_hash([q, time.time()])[:8]
    for p, r in results.items():
        rec = PredictionRecord(run_id="live", qid=live_id, pipeline=p, question=q, result=r, model_id=str(settings.answer_model), corpus_hash=corpus_hash, graph_snapshot_id=snap, recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        recs.append(rec)
    with _lock, open(_live_dir / "predictions_live.jsonl", "a", encoding="utf-8") as f:
        for rec in recs:
            f.write(rec.model_dump_json() + "\n")
    return {"qid": live_id, "mode": "live", "records": [json.loads(r.model_dump_json()) for r in recs]}


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        info = {"run_id": d.name, "files": sorted(p.name for p in d.iterdir()), "summaries": {}}
        for s in d.glob("summary_*.json"):
            info["summaries"][s.stem.split("_", 1)[1]] = json.load(open(s))
        if (d / "config_snapshot.json").exists():
            info["config"] = json.load(open(d / "config_snapshot.json"))
        out.append(info)
    return out


@app.get("/api/runs/{run_id}/summary")
def run_summary(run_id: str, split: str = "public"):
    p = RUNS / run_id / f"summary_{split}.json"
    if not p.exists():
        raise HTTPException(404, "no summary")
    return json.load(open(p))


@app.get("/api/runs/{run_id}/scores")
def run_scores(run_id: str, split: str = "public"):
    p = RUNS / run_id / f"scores_{split}.jsonl"
    if not p.exists():
        raise HTTPException(404, "no scores")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


@app.get("/api/runs/{run_id}/predictions")
def run_predictions(run_id: str, split: str = "public", qid: str | None = None, pipeline: str | None = None, compact: bool = True):
    p = RUNS / run_id / f"predictions_{split}.jsonl"
    if not p.exists():
        raise HTTPException(404, "no predictions")
    out = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if qid and r["qid"] != qid:
            continue
        if pipeline and r["pipeline"] != pipeline:
            continue
        if compact and not qid:
            res = r["result"]
            r = {"qid": r["qid"], "pipeline": r["pipeline"], "qtype": r.get("qtype"), "question": r["question"], "answer": res["answer"], "status": res["status"], "stop_reason": res["stop_reason"], "tokens": res["usage"]["total_tokens"], "latency_ms": res["latency_ms"], "n_steps": len(res["trace"]["events"]), "strategy_changed": res["trace"]["strategy_changed"]}
        out.append(r)
    return out


@app.get("/api/questions")
def questions(split: str = "public"):
    qs = load_questions(split)
    return [{k: v for k, v in q.items() if k in ("qid", "qtype", "question", "answer")} for q in qs]


@app.get("/api/doc/{doc_id}")
def doc(doc_id: str):
    d = corpus().get(doc_id)
    if not d:
        raise HTTPException(404, "unknown doc")
    return {"doc_id": d["doc_id"], "title": d["title"], "url": d["url"], "text": d["text"]}


@app.get("/api/event/{event_id}/subgraph")
def subgraph(event_id: str):
    from .db.tg import get_tg, results_by_name, vattrs

    try:
        res = results_by_name(get_tg().run_query("event_subgraph", {"ev": {"id": event_id, "type": "Event"}}))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"TigerGraph error: {str(e)[:200]}")
    nodes = [vattrs(v) for v in res.get("event", [])] + [vattrs(v) for v in res.get("neighbours", [])]
    return {"nodes": nodes, "edges": res.get("edges", [])}


@app.get("/api/runs/{run_id}/download/{name}")
def download(run_id: str, name: str):
    p = RUNS / run_id / name
    if not p.exists() or ".." in name:
        raise HTTPException(404)
    return FileResponse(p, filename=name)


@app.get("/api/manifests")
def manifests():
    out = {}
    for p in MANIFESTS.glob("*.json"):
        out[p.stem] = json.load(open(p))
    return out


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
