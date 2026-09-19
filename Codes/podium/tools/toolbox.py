"""Retrieval and reasoning tools over TigerGraph (plan section 7).

Every tool is bounded, parameterised, read-only and records a TraceEvent with its
execution kind (`graph_query`, `vector_search`, `deterministic`, `llm`). The same toolbox is
used by GraphRAG (fixed plan) and by the agent (adaptive plan) so the comparison is fair.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

from ..contracts import Candidate, Citation, LLMCallUsage, Trace, TraceEvent, UsageSummary
from ..corpus.parse import date_match, event_norm, norm_key, norm_text, sport_norm, split_concatenated_names
from ..db.tg import TG, get_tg, results_by_name, vattrs
from ..llm.client import complete_json, embed_query
from ..settings import CACHE, settings


@lru_cache(maxsize=1)
def doc_index() -> dict[str, dict]:
    """Document metadata (title/url) for citations; derived from the corpus at ingestion."""
    out = {}
    with open(CACHE / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


@lru_cache(maxsize=1)
def sports_index() -> dict[str, str]:
    data = json.load(open(CACHE / "sports.json"))
    return {sport_norm(k): k for k in data}


def resolve_sport(name: str) -> tuple[str, str]:
    """Map a free-text sport to (sport_norm, display). Falls back to best token overlap."""
    if not name:
        return "", ""
    idx = sports_index()
    n = sport_norm(name)
    if n in idx:
        return n, idx[n]
    toks = set(n.split())
    best, score = "", 0.0
    for k in idx:
        kt = set(k.split())
        s = len(toks & kt) / max(1, len(toks | kt))
        if s > score:
            best, score = k, s
    if score >= 0.5:
        return best, idx[best]
    return n, name


class RunContext:
    """Per-question state shared by tools: evidence handles, trace, usage, budgets."""

    def __init__(self, question: str, pipeline: str):
        self.question = question
        self.pipeline = pipeline
        self.trace = Trace(pipeline=pipeline)
        self.usage = UsageSummary()
        self.chunks: dict[str, dict] = {}
        self.events: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}
        self.llm_seen_chunks: set[str] = set()
        self.t0 = time.time()
        self.step = 0
        self.graph_path: list[dict] = []
        self.action_log: list[dict] = []

    def elapsed(self) -> float:
        return time.time() - self.t0

    def add_usage(self, u: LLMCallUsage) -> None:
        self.usage.add(u)

    def add_chunk(self, c: dict) -> dict:
        cid = c.get("chunk_id") or c.get("v_id")
        d = doc_index().get(c.get("doc_id", ""), {})
        rec = {**c, "chunk_id": cid, "title": d.get("title", ""), "url": d.get("url", "")}
        self.chunks[cid] = rec
        return rec

    def add_event(self, e: dict) -> dict:
        eid = e.get("event_id") or e.get("v_id")
        e = {**e, "event_id": eid}
        self.events[eid] = e
        return e

    def add_edge(self, src: str, rel: str, dst: str, label: str = "") -> None:
        edge = {"from": src, "rel": rel, "to": dst, "label": label}
        if edge not in self.graph_path:
            self.graph_path.append(edge)


class Toolbox:
    def __init__(self, ctx: RunContext, tg: TG | None = None):
        self.ctx = ctx
        self.tg = tg or get_tg()

    # ---- tracing --------------------------------------------------------------------
    def _record(self, specialist: str, tool: str, kind: str, args: dict, summary: str, n: int, t0: float, evidence_ids=None, error=None, tokens=0, rationale="", strategy_change=False) -> TraceEvent:
        self.ctx.step += 1
        ev = TraceEvent(
            step=self.ctx.step, specialist=specialist, tool=tool, execution_kind=kind, arguments=args, rationale=rationale,
            result_summary=summary, evidence_ids=list(evidence_ids or [])[:50], n_results=n, latency_ms=(time.time() - t0) * 1000,
            tokens=tokens, error=error, strategy_change=strategy_change,
        )
        self.ctx.trace.record(ev)
        self.ctx.action_log.append({"tool": tool, "args": args, "n": n, "error": error})
        return ev

    # ---- vector retrieval -------------------------------------------------------------
    def search_chunks(self, query: str, k: int | None = None, sport: str = "", year: int = 0, season: str = "", rationale: str = "") -> list[dict]:
        k = k or settings.rag_top_k
        t0 = time.time()
        vec, u = embed_query(query)
        self.ctx.add_usage(u)
        args = {"query": query, "k": k}
        try:
            if sport or year or season:
                sn, _ = resolve_sport(sport)
                args.update({"sport": sn, "year": year, "season": season})
                res = self.tg.run_query("search_chunks_scoped", {"query_vector": vec, "k": k, "sport_norm": sn, "year": year, "season": season})
            else:
                res = self.tg.run_query("search_chunks", {"query_vector": vec, "k": k})
        except Exception as e:  # noqa: BLE001
            self._record("similarity_search", "search_chunks", "vector_search", args, "error", 0, t0, error=str(e)[:300], rationale=rationale)
            raise
        r = results_by_name(res)
        dist = r.get("distances", {}) or {}
        out = []
        for v in r.get("chunks", []):
            c = vattrs(v)
            c["distance"] = dist.get(v["v_id"])
            out.append(self.ctx.add_chunk(c))
        out.sort(key=lambda c: (c["distance"] if c["distance"] is not None else 9))
        self._record("similarity_search", "search_chunks", "vector_search", args, f"{len(out)} chunks", len(out), t0, [c["chunk_id"] for c in out], rationale=rationale)
        return out

    # ---- graph lookups ------------------------------------------------------------------
    def find_events(self, sport: str = "", year: int = 0, season: str = "", event_name: str = "", venue: str = "", title_needle: str = "", rationale: str = "") -> list[dict]:
        t0 = time.time()
        sn, _ = resolve_sport(sport)
        params = {"sport_norm": sn, "year": int(year or 0), "season": season or "", "event_norm": event_norm(event_name) if event_name else "", "venue_id": norm_key(venue) if venue else "", "title_needle": title_needle or ""}
        res = results_by_name(self.tg.run_query("find_events", params))
        events = [self.ctx.add_event(vattrs(v)) for v in res.get("events", [])]
        for e in events:
            self.ctx.add_edge(e["event_id"], "IN_EDITION", e["edition_id"])
            self.ctx.add_edge(e["event_id"], "IN_SPORT", e["sport_norm"])
        self._record("entity_linker", "find_events", "graph_query", params, f"{len(events)} events", len(events), t0, [e["event_id"] for e in events], rationale=rationale)
        return events

    def events_like(self, needle: str, year: int = 0, season: str = "", rationale: str = "") -> list[dict]:
        t0 = time.time()
        params = {"needle": needle, "year": int(year or 0), "season": season or ""}
        res = results_by_name(self.tg.run_query("events_like", params))
        events = [self.ctx.add_event(vattrs(v)) for v in res.get("events", [])]
        self._record("entity_linker", "events_like", "graph_query", params, f"{len(events)} events", len(events), t0, [e["event_id"] for e in events], rationale=rationale)
        return events

    def events_at_venue(self, venue: str, year: int = 0, rationale: str = "", fuzzy: bool = False) -> tuple[list[dict], list[dict]]:
        """Return (events, venues_considered). Exact normalised venue first; token search when fuzzy."""
        t0 = time.time()
        vid = norm_key(venue)
        params = {"venue_id": vid, "year": int(year or 0), "fuzzy": fuzzy}
        res = results_by_name(self.tg.run_query("events_at_venue", {"venue_id": vid, "year": int(year or 0)}))
        events = [self.ctx.add_event(vattrs(v)) for v in res.get("events", [])]
        venues = [{"venue_id": vid, "name": venue, "match": "exact"}] if events else []
        if not events and fuzzy:
            toks = [t for t in re.findall(r"[a-z0-9]+", norm_text(venue)) if len(t) >= 4 and t not in ("olympic", "centre", "center", "arena", "stadium", "hall", "park", "the")]
            toks.sort(key=len, reverse=True)
            seen = {}
            for tok in toks[:3]:
                vres = results_by_name(self.tg.run_query("venues_like", {"needle": tok, "year": 0}))
                for v in vres.get("venues", []):
                    va = vattrs(v)
                    score = sum(1 for t in toks if t in va["venue_id"])
                    if va["venue_id"] not in seen or seen[va["venue_id"]]["score"] < score:
                        seen[va["venue_id"]] = {"venue_id": va["venue_id"], "name": va["name"], "score": score, "match": f"token:{tok}"}
            ranked = sorted(seen.values(), key=lambda x: -x["score"])
            top = ranked[0]["score"] if ranked else 0
            venues = [v for v in ranked if v["score"] == top][:5]
            for v in venues:
                r2 = results_by_name(self.tg.run_query("events_at_venue", {"venue_id": v["venue_id"], "year": int(year or 0)}))
                for x in r2.get("events", []):
                    events.append(self.ctx.add_event(vattrs(x)))
        for e in events:
            self.ctx.add_edge(e["venue_id"], "VENUE_HOSTED", e["event_id"])
        self._record("entity_linker", "events_at_venue", "graph_query", params, f"{len(events)} events at {len(venues)} venue(s)", len(events), t0, [e["event_id"] for e in events], rationale=rationale)
        return events, venues

    def match_date(self, events: list[dict], date: str, rationale: str = "") -> dict:
        """Deterministic date filter. Returns {'best': [...], 'score': int, 'reason': str, 'scored': [...]}."""
        t0 = time.time()
        scored = []
        for e in events:
            s, why = date_match(date, e.get("date_raw") or "")
            scored.append({"event_id": e["event_id"], "title": e["title"], "date_raw": e.get("date_raw"), "score": s, "reason": why})
        best_score = max((s["score"] for s in scored), default=0)
        best = [s for s in scored if s["score"] == best_score and best_score > 0]
        out = {"best": best, "score": best_score, "reason": best[0]["reason"] if best else "no date match", "scored": scored, "ambiguous": len(best) > 1}
        self._record("multi_hop_reasoner", "match_date", "deterministic", {"date": date, "n_events": len(events)}, f"{len(best)} match(es) score={best_score}", len(best), t0, [b["event_id"] for b in best], rationale=rationale)
        return out

    def aggregate_competitors(self, sport: str, year: int, season: str, threshold: int, op: str = ">", rationale: str = "") -> dict:
        t0 = time.time()
        sn, disp = resolve_sport(sport)
        params = {"sport_norm": sn, "year": int(year), "season": season, "threshold": int(threshold), "op": op}
        res = results_by_name(self.tg.run_query("aggregate_competitors", params))
        cands = [self.ctx.add_event(vattrs(v)) for v in res.get("candidates", [])]
        inc, exc, unk = set(res.get("included", [])), set(res.get("excluded", [])), set(res.get("unknown", []))
        manifest = []
        for e in cands:
            st = "included" if e["event_id"] in inc else "excluded" if e["event_id"] in exc else "unknown"
            reason = f"{e.get('competitors_raw') or 'no value'} {op} {threshold}" if st != "unknown" else f"competitors field {e.get('competitors_status')}"
            if e.get("competitors_status") == "suspicious":
                reason += " (implausible infobox value; verify against prose)"
            manifest.append(Candidate(event_id=e["event_id"], title=e["title"], value=e["competitors"] if e.get("has_competitors") else None, value_raw=e.get("competitors_raw"), unit=e.get("competitors_unit"), status=st, reason=reason, source="infobox" if e.get("has_competitors") else "none", chunk_id=e.get("infobox_chunk_id")))
            self.ctx.add_edge(e["event_id"], "IN_SPORT", sn)
        out = {"sport": disp, "sport_norm": sn, "year": year, "season": season, "threshold": threshold, "op": op, "count": res.get("n_included", 0), "n_candidates": res.get("n_candidates", 0), "n_excluded": res.get("n_excluded", 0), "n_unknown": res.get("n_unknown", 0), "unknown_ids": sorted(unk), "manifest": manifest}
        self._record("aggregator", "aggregate_competitors", "graph_query", params, f"count={out['count']} of {out['n_candidates']} candidates ({out['n_unknown']} unknown)", out["n_candidates"], t0, [c.event_id for c in manifest], rationale=rationale)
        return out

    def max_competitors(self, sport: str, year: int, season: str, rationale: str = "") -> dict:
        t0 = time.time()
        sn, disp = resolve_sport(sport)
        params = {"sport_norm": sn, "year": int(year), "season": season}
        res = results_by_name(self.tg.run_query("max_competitors", params))
        cands = [self.ctx.add_event(vattrs(v)) for v in res.get("candidates", [])]
        winners = [self.ctx.add_event(vattrs(v)) for v in res.get("winners", [])]
        wids = {w["event_id"] for w in winners}
        manifest = [
            Candidate(event_id=e["event_id"], title=e["title"], value=e["competitors"] if e.get("has_competitors") else None, value_raw=e.get("competitors_raw"), unit=e.get("competitors_unit"),
                      status="included" if e["event_id"] in wids else ("unknown" if not e.get("has_competitors") else "excluded"),
                      reason=("maximum" if e["event_id"] in wids else ("competitors field " + str(e.get("competitors_status")) if not e.get("has_competitors") else "below maximum")) + (" (implausible infobox value; verify against prose)" if e.get("competitors_status") == "suspicious" else ""),
                      source="infobox" if e.get("has_competitors") else "none", chunk_id=e.get("infobox_chunk_id"))
            for e in cands
        ]
        out = {"sport": disp, "sport_norm": sn, "year": year, "season": season, "max_value": res.get("max_value"), "winners": winners, "n_candidates": res.get("n_candidates", 0), "n_unknown": res.get("n_unknown", 0), "unknown_ids": sorted(res.get("unknown", [])), "manifest": manifest}
        self._record("aggregator", "max_competitors", "graph_query", params, f"max={out['max_value']} winners={len(winners)} of {out['n_candidates']} ({out['n_unknown']} unknown)", out["n_candidates"], t0, [w["event_id"] for w in winners], rationale=rationale)
        return out

    def previous_event(self, event_id: str, rationale: str = "") -> dict:
        t0 = time.time()
        res = results_by_name(self.tg.run_query("previous_event", {"ev": {"id": event_id, "type": "Event"}}))
        prev = [self.ctx.add_event(vattrs(v)) for v in res.get("previous", [])]
        prev_ed = [vattrs(v) for v in res.get("previous_edition", [])]
        sources = res.get("sources", [])
        for p in prev:
            self.ctx.add_edge(event_id, "PREVIOUS_EVENT", p["event_id"], sources[0] if sources else "")
        out = {"event_id": event_id, "previous": prev, "sources": sources, "previous_edition": prev_ed[0] if prev_ed else None}
        self._record("graph_traverser", "previous_event", "graph_query", {"event_id": event_id}, f"{len(prev)} predecessor(s) via {sources or 'none'}", len(prev), t0, [p["event_id"] for p in prev], rationale=rationale)
        return out

    def event_facts(self, event_id: str, rationale: str = "") -> dict:
        t0 = time.time()
        res = results_by_name(self.tg.run_query("event_facts", {"ev": {"id": event_id, "type": "Event"}}))
        ev = [self.ctx.add_event(vattrs(v)) for v in res.get("event", [])]
        facts = [vattrs(v) for v in res.get("facts", [])]
        for f in facts:
            self.ctx.facts[f["fact_id"]] = f
            self.ctx.add_edge(event_id, "HAS_FACT", f["fact_id"], f["predicate"])
        chunks = [self.ctx.add_chunk(vattrs(v)) for v in res.get("chunks", [])]
        meds = [vattrs(v) for v in res.get("medallists", [])]
        out = {"event": ev[0] if ev else None, "facts": facts, "chunks": chunks, "medallists": meds}
        self._record("document_retriever", "event_facts", "graph_query", {"event_id": event_id}, f"{len(facts)} facts, {len(chunks)} source chunk(s)", len(facts), t0, [c["chunk_id"] for c in chunks], rationale=rationale)
        return out

    def fetch_document(self, doc_id: str, rationale: str = "") -> list[dict]:
        t0 = time.time()
        res = results_by_name(self.tg.run_query("doc_chunks", {"doc": {"id": doc_id, "type": "Document"}}))
        chunks = [self.ctx.add_chunk(vattrs(v)) for v in res.get("chunks", [])]
        chunks.sort(key=lambda c: c.get("ordinal", 0))
        self._record("document_retriever", "fetch_document", "graph_query", {"doc_id": doc_id}, f"{len(chunks)} chunks", len(chunks), t0, [c["chunk_id"] for c in chunks], rationale=rationale)
        return chunks

    def resolve_entity(self, name: str, rationale: str = "") -> dict:
        t0 = time.time()
        nk = norm_key(name)
        res = results_by_name(self.tg.run_query("resolve_entity", {"name_norm": nk, "needle": nk if len(nk) >= 5 else ""}))
        ents = [vattrs(v) for v in res.get("entities", [])]
        events = [self.ctx.add_event(vattrs(v)) for v in res.get("events", [])]
        docs = [vattrs(v) for v in res.get("documents", [])]
        medals = res.get("medals", [])
        for m in medals:
            parts = m.split("|")
            if len(parts) == 4:
                self.ctx.add_edge(parts[0], "WON_MEDAL", parts[1], parts[2])
        out = {"entities": ents, "events": events, "documents": docs, "medals": medals}
        self._record("entity_linker", "resolve_entity", "graph_query", {"name": name}, f"{len(ents)} entities, {len(events)} events, {len(docs)} docs", len(ents), t0, [e["entity_id"] for e in ents], rationale=rationale)
        return out

    # ---- LLM specialists ------------------------------------------------------------------
    EXTRACT_SCHEMA = {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "value": {"type": "integer", "description": "the number of competitors/participants in the event overall (0 if not found)"},
            "quote": {"type": "string", "description": "exact verbatim span from the passages that states the value"},
            "chunk_id": {"type": "string"},
            "interpretation": {"type": "string", "description": "which round/scope the number refers to, e.g. 'riders in the qualifying round = all entrants'"},
            "confidence": {"type": "number"},
        },
    }

    def extract_count_from_prose(self, event: dict, predicate: str = "competitors", rationale: str = "") -> dict:
        """Evidence repair: recover a missing infobox count from the document prose, with a verified quote."""
        t0 = time.time()
        chunks = self.fetch_document(event["doc_id"] if "doc_id" in event else event["event_id"], rationale="fetch source document for prose extraction")
        prose = [c for c in chunks if c.get("kind") != "infobox"][:6]
        budget = settings.context_char_budget
        packed, used = [], 0
        for c in prose:
            t = c["text"][: max(0, budget - used)]
            if not t:
                break
            packed.append(f"[chunk {c['chunk_id']}]\n{t}")
            used += len(t)
            self.ctx.llm_seen_chunks.add(c["chunk_id"])
        system = (
            "You extract one numeric fact from Wikipedia-derived passages about an Olympic event. Answer ONLY from the passages. "
            "The quote must be copied verbatim from a passage (it will be checked character-for-character). "
            "Do not sum counts from different rounds; report the total number of entrants/competitors in the event if stated, "
            "or the number in the first/qualifying round when that is the full field."
        )
        user = f"Event: {event.get('title')}\nFact to extract: total number of {predicate} in this event.\n\nPassages:\n" + "\n\n".join(packed)
        ctx_tokens = used // 4
        try:
            data, u = complete_json("extract", system, user, self.EXTRACT_SCHEMA, max_tokens=400, context_tokens=ctx_tokens)
        except Exception as e:  # noqa: BLE001
            self._record("evidence_evaluator", "extract_count_from_prose", "llm", {"event_id": event["event_id"], "predicate": predicate}, "llm error", 0, t0, error=str(e)[:200], rationale=rationale)
            return {"found": False, "error": str(e)[:200]}
        self.ctx.add_usage(u)
        out = {"found": bool(data.get("found")), "value": data.get("value"), "quote": data.get("quote", ""), "chunk_id": data.get("chunk_id", ""), "interpretation": data.get("interpretation", ""), "confidence": data.get("confidence", 0), "verified": False}
        # deterministic verification of the quoted span
        if out["found"] and out["quote"]:
            for c in prose:
                idx = c["text"].find(out["quote"])
                if idx != -1:
                    out.update({"verified": True, "chunk_id": c["chunk_id"], "char_start": c["char_start"] + idx, "char_end": c["char_start"] + idx + len(out["quote"])})
                    break
            if out["verified"] and not re.search(r"\d", out["quote"]) and not re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b", out["quote"].lower()):
                out["verified"] = False
        if out["found"] and not out["verified"]:
            out["found"] = False
            out["reason"] = "quote not found verbatim in source"
        summ = f"value={out.get('value')} verified={out['verified']}" if out["found"] else "not found"
        self._record("evidence_evaluator", "extract_count_from_prose", "llm", {"event_id": event["event_id"], "predicate": predicate}, summ, 1 if out["found"] else 0, t0, [out.get("chunk_id")] if out.get("verified") else [], tokens=u.total, rationale=rationale)
        return out

    # ---- helpers ----------------------------------------------------------------------
    def citation_for_fact(self, event: dict, predicate: str, claim_id: str = "") -> Citation | None:
        """Citation pointing at the exact infobox span of `predicate` for `event`."""
        for f in self.ctx.facts.values():
            if f.get("subject_id") == event["event_id"] and f.get("predicate") == predicate:
                c = self.ctx.chunks.get(f.get("chunk_id"))
                quote = f.get("value_raw", "")
                return Citation(doc_id=event["event_id"], chunk_id=f.get("chunk_id"), title=event.get("title", ""), url=doc_index().get(event["event_id"], {}).get("url", ""), char_start=f.get("char_start"), char_end=f.get("char_end"), quote=quote, claim_ids=[claim_id] if claim_id else [], source="fact")
        # fall back to the infobox chunk without offsets
        cid = event.get("infobox_chunk_id")
        if cid:
            return Citation(doc_id=event["event_id"], chunk_id=cid, title=event.get("title", ""), url=doc_index().get(event["event_id"], {}).get("url", ""), quote=str(event.get(predicate + "_raw", ""))[:200], claim_ids=[claim_id] if claim_id else [], source="graph")
        return None

    def citation_for_chunk(self, chunk: dict, claim_id: str = "", quote: str = "") -> Citation:
        return Citation(doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"], title=chunk.get("title", ""), url=chunk.get("url", ""), char_start=chunk.get("char_start"), char_end=chunk.get("char_end"), quote=(quote or chunk.get("text", ""))[:300], claim_ids=[claim_id] if claim_id else [], source="vector")


def medallist_display(raw: str) -> str:
    parts = split_concatenated_names(raw)
    return " / ".join(parts) if len(parts) > 1 else raw
