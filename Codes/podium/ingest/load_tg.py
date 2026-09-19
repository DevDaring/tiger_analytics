"""Create the Podium schema on TigerGraph, load vertices/edges/vectors, install queries.

Idempotent: upserts are safe to re-run; schema creation is skipped when the graph exists.
Usage: python -m podium.ingest.load_tg [--schema] [--load] [--vectors] [--queries] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..db.tg import TG, TigerGraphError, get_tg
from ..ingest.embed import cache_path, load_cache
from ..settings import CACHE, GSQL_DIR, MANIFESTS, settings

BATCH_V = 500
BATCH_VEC = 100
WORKERS = 6


def rows(name: str):
    with open(CACHE / name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _batches(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def create_schema(tg: TG) -> None:
    existing = tg.gsql("ls")
    if f"Graph {tg.graph}(" in existing or f"- Graph {tg.graph}" in existing:
        print(f"graph {tg.graph} already exists; skipping schema creation")
    else:
        schema = (GSQL_DIR / "schema.gsql").read_text()
        out = tg.gsql(schema)
        print(out[-1500:])
        if "Failed" in out or "error" in out.lower() and "successfully" not in out.lower():
            raise TigerGraphError("schema creation reported errors; see output above")
    # vector attribute on Chunk (global schema change job)
    ls = tg.gsql("ls", graph=tg.graph)
    if "embedding" in ls and "VECTOR" in ls.upper():
        print("vector attribute already present")
        return
    job = f"""
CREATE GLOBAL SCHEMA_CHANGE JOB add_chunk_embedding {{
  ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding(DIMENSION={settings.embed_dim}, METRIC="COSINE");
}}
RUN GLOBAL SCHEMA_CHANGE JOB add_chunk_embedding
"""
    print(tg.gsql(job)[-1500:])


def _upsert_batch(tg: TG, vtype: str, batch: list[tuple[str, dict]]) -> int:
    payload = {vtype: {vid: {k: {"value": v} for k, v in attrs.items()} for vid, attrs in batch}}
    for attempt in range(4):
        try:
            r = tg.upsert(vertices=payload)
            return r["results"][0]["accepted_vertices"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def _upsert_edges(tg: TG, spec: dict) -> int:
    for attempt in range(4):
        try:
            r = tg.upsert(edges=spec)
            return r["results"][0]["accepted_edges"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return 0


def load_vertices(tg: TG, vtype: str, items, attrs_fn, batch=BATCH_V) -> int:
    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_upsert_batch, tg, vtype, [(str(r[0]), r[1]) for r in b]) for b in _batches(((x[0], attrs_fn(x[1])) for x in items), batch)]
        for f in futs:
            total += f.result()
    print(f"  {vtype}: {total} vertices in {time.time()-t0:.1f}s", flush=True)
    return total


def load_edges(tg: TG, etype: str, from_type: str, to_type: str, triples, batch=2000) -> int:
    """triples: iterable of (from_id, to_id, attrs)"""
    t0 = time.time()
    total = 0

    def spec_for(b):
        spec: dict = {from_type: {}}
        for f, t, a in b:
            spec[from_type].setdefault(f, {}).setdefault(etype, {}).setdefault(to_type, {})[t] = {k: {"value": v} for k, v in (a or {}).items()}
        return spec

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_upsert_edges, tg, spec_for(b)) for b in _batches(triples, batch)]
        for f in futs:
            total += f.result()
    print(f"  {etype}: {total} edges in {time.time()-t0:.1f}s", flush=True)
    return total


def load_all(tg: TG) -> dict:
    counts = {}
    docs = list(rows("documents.jsonl"))
    counts["Document"] = load_vertices(
        tg, "Document", ((d["doc_id"], d) for d in docs),
        lambda d: {k: d[k] for k in ("title", "url", "wikidata_qid", "infobox_type", "is_olympic_event", "n_chunks", "n_chars", "content_hash")},
    )
    chunks = list(rows("chunks.jsonl"))
    counts["Chunk"] = load_vertices(
        tg, "Chunk", ((c["chunk_id"], c) for c in chunks),
        lambda c: {k: c[k] for k in ("doc_id", "ordinal", "section", "kind", "text", "char_start", "char_end", "tokens", "content_hash")}, batch=200,
    )
    counts["HAS_CHUNK"] = load_edges(tg, "HAS_CHUNK", "Document", "Chunk", ((c["doc_id"], c["chunk_id"], {}) for c in chunks))
    events = list(rows("events.jsonl"))
    ev_keys = [
        "title", "sport", "sport_norm", "season", "edition_year", "edition_id", "event_name", "event_norm", "event_field", "venue_raw", "venue_id",
        "date_raw", "date_norm", "competitors", "has_competitors", "competitors_raw", "competitors_unit", "competitors_status", "nations", "has_nations",
        "nations_raw", "nations_status", "gold_raw", "gold_noc", "silver_raw", "silver_noc", "bronze_raw", "bronze_noc", "bronze2_raw", "bronze2_noc",
        "prev_raw", "next_raw", "prev_year", "next_year", "win_value", "parse_status", "infobox_chunk_id",
    ]
    counts["Event"] = load_vertices(
        tg, "Event", ((e["event_id"], e) for e in events),
        lambda e: {**{k: e[k] for k in ev_keys}, "date_days": ",".join(e["date_days"]), "date_years": ",".join(str(y) for y in e["date_years"])},
    )
    ed = json.load(open(CACHE / "editions.json"))
    counts["Edition"] = load_vertices(tg, "Edition", ((k, v) for k, v in ed["editions"].items()), lambda v: {k: v[k] for k in ("year", "season", "label", "n_events")})
    counts["PREVIOUS_EDITION"] = load_edges(tg, "PREVIOUS_EDITION", "Edition", "Edition", ((a, b, {"source": "corpus_edition_order"}) for a, b in ed["prev_edition"].items() if b))
    sports = json.load(open(CACHE / "sports.json"))
    from ..corpus.parse import sport_norm

    counts["Sport"] = load_vertices(tg, "Sport", ((sport_norm(k), {"name": k, "n_events": n}) for k, n in sports.items()), lambda v: v)
    venues = list(rows("venues.jsonl"))
    counts["Venue"] = load_vertices(tg, "Venue", ((v["venue_id"], v) for v in venues), lambda v: {"name": v["name"], "aliases": " | ".join(v["aliases"]), "n_events": v["n_events"]})
    counts["DESCRIBES_EVENT"] = load_edges(tg, "DESCRIBES_EVENT", "Document", "Event", ((e["doc_id"], e["event_id"], {}) for e in events))
    counts["IN_EDITION"] = load_edges(tg, "IN_EDITION", "Event", "Edition", ((e["event_id"], e["edition_id"], {}) for e in events))
    counts["IN_SPORT"] = load_edges(tg, "IN_SPORT", "Event", "Sport", ((e["event_id"], e["sport_norm"], {}) for e in events))
    counts["AT_VENUE"] = load_edges(tg, "AT_VENUE", "Event", "Venue", ((e["event_id"], e["venue_id"], {}) for e in events if e["venue_id"]))
    counts["PREVIOUS_EVENT"] = load_edges(tg, "PREVIOUS_EVENT", "Event", "Event", ((p["from"], p["to"], {"source": p["source"], "prev_raw": p["prev_raw"]}) for p in rows("prev_links.jsonl")))
    facts = list(rows("facts.jsonl"))
    counts["Fact"] = load_vertices(
        tg, "Fact", ((f["fact_id"], f) for f in facts),
        lambda f: {**{k: f[k] for k in ("subject_id", "predicate", "value_raw", "unit", "status", "note", "method", "chunk_id", "char_start", "char_end")}, "value_num": float(f["value_num"]) if f["value_num"] is not None else -1.0},
    )
    counts["HAS_FACT"] = load_edges(tg, "HAS_FACT", "Event", "Fact", ((f["subject_id"], f["fact_id"], {}) for f in facts))
    counts["SUPPORTED_BY"] = load_edges(tg, "SUPPORTED_BY", "Fact", "Chunk", ((f["fact_id"], f["chunk_id"], {}) for f in facts))
    ents = list(rows("entities.jsonl"))
    counts["Entity"] = load_vertices(
        tg, "Entity", ((e["entity_id"], e) for e in ents),
        lambda e: {"name": e["name"], "name_norm": e["name_norm"], "entity_type": e["type"], "noc": e.get("noc", ""), "doc_id": e.get("doc_id", ""), "members": " | ".join(e.get("members", []) or [])},
    )
    counts["DESCRIBES_ENTITY"] = load_edges(tg, "DESCRIBES_ENTITY", "Document", "Entity", ((e["doc_id"], e["entity_id"], {}) for e in ents if e.get("doc_id")))
    counts["WON_MEDAL"] = load_edges(
        tg, "WON_MEDAL", "Entity", "Event",
        ((e["entity_id"], m["event_id"], {"medal": m["medal"], "noc": m["noc"], "fact_id": m["fact_id"]}) for e in ents for m in e.get("medals", []) or []),
    )
    counts["REPRESENTS"] = load_edges(tg, "REPRESENTS", "Entity", "Entity", ((e["entity_id"], "noc:" + e["noc"], {}) for e in ents if e.get("type") == "medallist" and e.get("noc")))
    return counts


def load_vectors(tg: TG) -> int:
    cache = load_cache(cache_path())
    chunks = list(rows("chunks.jsonl"))
    missing = [c["chunk_id"] for c in chunks if c["content_hash"] not in cache]
    if missing:
        print(f"WARNING: {len(missing)} chunks have no cached embedding; run podium.ingest.embed first")
    items = [(c["chunk_id"], {"embedding": cache[c["content_hash"]]}) for c in chunks if c["content_hash"] in cache]
    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_upsert_batch, tg, "Chunk", b) for b in _batches(items, BATCH_VEC)]
        for i, f in enumerate(futs):
            total += f.result()
            if i % 20 == 0:
                print(f"  vectors {total}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Chunk.embedding: {total} vectors in {time.time()-t0:.1f}s")
    return total


def install_queries(tg: TG) -> str:
    q = (GSQL_DIR / "queries.gsql").read_text()
    out = tg.gsql(q)
    print(out[-3000:])
    return out


def verify(tg: TG) -> dict:
    res = tg.run_query("graph_stats")
    stats = {}
    for r in res:
        stats.update(r)
    manifest = json.load(open(MANIFESTS / "ingestion_manifest.json"))
    checks = {
        "documents": (stats.get("documents"), manifest["documents"]),
        "chunks": (stats.get("chunks"), manifest["chunks"]),
        "events": (stats.get("events"), manifest["events"]),
        "facts": (stats.get("facts"), manifest["facts"]),
        "entities": (stats.get("entities"), manifest["entities"]),
        "venues": (stats.get("venues"), manifest["venues"]),
        "previous_event_edges": (stats.get("previous_event_edges"), manifest["prev_links"]),
    }
    ok = all(a == b for a, b in checks.values())
    report = {"graph": tg.graph, "graph_snapshot_id": manifest["graph_snapshot_id"], "checks": {k: {"graph": a, "manifest": b, "ok": a == b} for k, (a, b) in checks.items()}, "reconciled": ok, "stats": stats}
    json.dump(report, open(MANIFESTS / "graph_verification.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--vectors", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    tg = get_tg()
    print("TigerGraph:", tg.version().get("version"))
    if a.schema or a.all:
        create_schema(tg)
    if a.load or a.all:
        print(json.dumps(load_all(tg), indent=1))
    if a.vectors or a.all:
        load_vectors(tg)
    if a.queries or a.all:
        install_queries(tg)
    if a.verify or a.all:
        verify(tg)


if __name__ == "__main__":
    main()
