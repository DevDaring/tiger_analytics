"""Pipeline B: GraphRAG. Interpret the question once, execute ONE fixed retrieval template for
the intent (graph relations / complete-set aggregate + a supplementary vector search), pack the
evidence and generate an answer. It never replans after inspecting missing evidence.
"""
from __future__ import annotations

from ..contracts import AnswerResult, CoverageReport, Requirement
from ..tools.interpret import interpret
from ..tools.toolbox import RunContext, Toolbox
from .answer import build_citations, clean_answer, finalize, generate_answer


def _gold_result(tb: Toolbox, ev: dict, label: str) -> tuple[dict, list]:
    facts = tb.event_facts(ev["event_id"], rationale=f"fetch medal facts for {label}")
    e = facts["event"] or ev
    g = {"event": e["title"], "gold": e.get("gold_raw"), "gold_noc": e.get("gold_noc"), "silver": e.get("silver_raw"), "bronze": e.get("bronze_raw"), "venue": e.get("venue_raw"), "date": e.get("date_raw")}
    return g, facts["chunks"]


def run(question: str) -> AnswerResult:
    ctx = RunContext(question, "graphrag")
    tb = Toolbox(ctx)
    res = AnswerResult(stop_reason="single_pass")
    try:
        interp, u = interpret(question)
        ctx.add_usage(u)
        tb._record("orchestrator", "interpret", "llm", {}, f"intent={interp['intent']}", 1, ctx.t0, tokens=u.total, rationale="one-shot interpretation; fixed template follows")
        res.interpretation = interp
        intent = interp["intent"]
        chunks: list[dict] = []
        graph_result: dict | None = None
        machine_answer: list[str] = []
        citations = []
        reqs: list[Requirement] = []
        cov = CoverageReport()

        if intent == "count_events" and interp.get("sport") and interp.get("year"):
            agg = tb.aggregate_competitors(interp["sport"], interp["year"], interp.get("season") or "", interp.get("threshold", 0), interp.get("operator") or ">", rationale="fixed template: complete-cohort aggregate")
            graph_result = {k: v for k, v in agg.items() if k != "manifest"} | {"manifest": [c.model_dump() for c in agg["manifest"]]}
            machine_answer = [str(agg["count"])] if agg["n_candidates"] > 0 else []
            cov = CoverageReport(candidate_scope=f"{agg['sport']} events at the {agg['year']} {agg['season']} Olympics (corpus-defined)", candidate_count=agg["n_candidates"], known_values=agg["n_candidates"] - agg["n_unknown"], unknown_values=agg["n_unknown"], candidates=agg["manifest"], complete=agg["n_unknown"] == 0)
            if agg["n_unknown"]:
                cov.bounds = {"lower": agg["count"], "upper": agg["count"] + agg["n_unknown"]}
                cov.notes.append(f"{agg['n_unknown']} candidate(s) have no parseable competitors field; the count assumes they do not qualify (fixed plan cannot repair).")
            reqs = [Requirement(req_id="R1", description="complete cohort enumerated", status="satisfied" if agg["n_candidates"] else "missing"), Requirement(req_id="R2", description="predicate known for every candidate", status="satisfied" if agg["n_unknown"] == 0 else "missing"), Requirement(req_id="R3", description="count computed by graph query", status="satisfied")]
            for c in agg["manifest"]:
                if c.status == "included":
                    cit = tb.citation_for_fact(ctx.events[c.event_id], "competitors", "count")
                    if cit:
                        citations.append(cit)
            if not agg["n_candidates"]:
                chunks = tb.search_chunks(question, k=6, rationale="fixed template: supplementary passages")
        elif intent == "max_events" and interp.get("sport") and interp.get("year"):
            mx = tb.max_competitors(interp["sport"], interp["year"], interp.get("season") or "", rationale="fixed template: complete-cohort maximum")
            graph_result = {k: v for k, v in mx.items() if k not in ("manifest", "winners")} | {"winners": [w["title"] for w in mx["winners"]], "manifest": [c.model_dump() for c in mx["manifest"]]}
            machine_answer = [w["title"] for w in mx["winners"]]
            cov = CoverageReport(candidate_scope=f"{mx['sport']} events at the {mx['year']} {mx['season']} Olympics (corpus-defined)", candidate_count=mx["n_candidates"], known_values=mx["n_candidates"] - mx["n_unknown"], unknown_values=mx["n_unknown"], candidates=mx["manifest"], complete=mx["n_unknown"] == 0)
            if mx["n_unknown"]:
                cov.notes.append(f"{mx['n_unknown']} candidate(s) have no parseable competitors field; the maximum is over known values only.")
            reqs = [Requirement(req_id="R1", description="complete cohort enumerated", status="satisfied" if mx["n_candidates"] else "missing"), Requirement(req_id="R2", description="value known for every candidate", status="satisfied" if mx["n_unknown"] == 0 else "missing"), Requirement(req_id="R3", description="maximum computed by graph query (ties preserved)", status="satisfied" if mx["winners"] else "missing")]
            for w in mx["winners"]:
                cit = tb.citation_for_fact(w, "competitors", "max")
                if cit:
                    citations.append(cit)
            if not mx["n_candidates"]:
                chunks = tb.search_chunks(question, k=6, rationale="fixed template: supplementary passages")
        elif intent == "lookup_field" and interp.get("event_name"):
            events = tb.find_events(interp.get("sport", ""), interp.get("year", 0), interp.get("season") or "", interp.get("event_name", ""), rationale="fixed template: exact event lookup")
            if not events:
                events = tb.events_like(interp.get("event_name", ""), interp.get("year", 0), interp.get("season") or "", rationale="fixed template: title substring lookup")
            fld = interp.get("field") or "nations"
            reqs = [Requirement(req_id="R1", description="event resolved unambiguously", status="satisfied" if len(events) == 1 else "missing", note=f"{len(events)} candidates"), Requirement(req_id="R2", description=f"{fld} value supported by a source span", status="missing")]
            if len(events) == 1:
                facts = tb.event_facts(events[0]["event_id"], rationale="fetch facts for the resolved event")
                e = facts["event"] or events[0]
                raw = e.get(f"{fld}_raw") if f"{fld}_raw" in e else e.get(fld)
                if fld in ("nations", "competitors") and e.get(f"has_{fld}"):
                    machine_answer = [str(e[fld])]
                elif raw:
                    machine_answer = [str(raw)]
                graph_result = {"event": e["title"], fld: raw, "venue": e.get("venue_raw"), "date": e.get("date_raw")}
                chunks = facts["chunks"]
                cit = tb.citation_for_fact(e, fld, "lookup")
                if cit:
                    citations.append(cit)
                    reqs[1].status = "satisfied"
                    reqs[1].evidence_ids = [cit.chunk_id or ""]
            else:
                chunks = tb.search_chunks(question, k=6, rationale="fixed template: supplementary passages")
            cov = CoverageReport(candidate_scope="single event", candidate_count=len(events), known_values=1 if machine_answer else 0)
        elif intent == "event_by_venue_date" and interp.get("venue"):
            events, venues = tb.events_at_venue(interp["venue"], interp.get("year", 0), rationale="fixed template: events at venue (exact normalised name)")
            dm = tb.match_date(events, interp.get("date", ""), rationale="fixed template: date filter") if events else {"best": [], "ambiguous": False, "score": 0}
            best = [ctx.events[b["event_id"]] for b in dm["best"]]
            reqs = [Requirement(req_id="R1", description="venue resolved", status="satisfied" if events else "missing"), Requirement(req_id="R2", description="exactly one event matches venue and date", status="satisfied" if len(best) == 1 else ("conflicting" if len(best) > 1 else "missing"), note=f"{len(best)} match(es)"), Requirement(req_id="R3", description="gold medallist supported by a source span", status="missing")]
            golds = []
            for ev in best[:3]:
                g, ch = _gold_result(tb, ev, ev["title"])
                golds.append(g)
                chunks.extend(ch)
                cit = tb.citation_for_fact(ctx.events[ev["event_id"]], "gold", "gold")
                if cit:
                    citations.append(cit)
                    reqs[2].status = "satisfied"
            if golds:
                graph_result = {"matched_events": golds, "date_match": dm["reason"]}
                machine_answer = [g["gold"] for g in golds if g.get("gold")]
            else:
                chunks = tb.search_chunks(question, k=6, rationale="fixed template: supplementary passages")
            cov = CoverageReport(candidate_scope=f"events at venue '{interp['venue']}'", candidate_count=len(events), known_values=len(best), notes=(["ambiguous: more than one event matches venue and date"] if len(best) > 1 else []))
        elif intent == "previous_edition_winner" and interp.get("event_name"):
            ref = tb.find_events(interp.get("sport", ""), interp.get("year", 0), interp.get("season") or "", interp.get("event_name", ""), rationale="fixed template: resolve the reference-edition event")
            if not ref:
                ref = tb.events_like(interp.get("event_name", ""), interp.get("year", 0), interp.get("season") or "", rationale="fixed template: title substring lookup")
            reqs = [Requirement(req_id="R1", description="reference event resolved", status="satisfied" if len(ref) == 1 else "missing", note=f"{len(ref)} candidates"), Requirement(req_id="R2", description="predecessor event justified (infobox prev or edition order)", status="missing"), Requirement(req_id="R3", description="gold medallist of predecessor supported by a source span", status="missing")]
            if len(ref) == 1:
                pv = tb.previous_event(ref[0]["event_id"], rationale="fixed template: follow PREVIOUS_EVENT edge")
                if pv["previous"]:
                    reqs[1].status = "satisfied"
                    reqs[1].note = f"source={pv['sources']}"
                    g, ch = _gold_result(tb, pv["previous"][0], "predecessor")
                    chunks = ch
                    graph_result = {"reference_event": ref[0]["title"], "predecessor_event": g["event"], "predecessor_source": pv["sources"], "gold": g["gold"], "gold_noc": g["gold_noc"]}
                    machine_answer = [g["gold"]] if g.get("gold") else []
                    cit = tb.citation_for_fact(ctx.events[pv["previous"][0]["event_id"]], "gold", "gold")
                    if cit:
                        citations.append(cit)
                        reqs[2].status = "satisfied"
                else:
                    cov.notes.append("no predecessor document in the corpus for this event")
            if not machine_answer:
                chunks = chunks or tb.search_chunks(question, k=6, rationale="fixed template: supplementary passages")
            cov = CoverageReport(candidate_scope="reference event and its predecessor", candidate_count=len(ref), known_values=1 if machine_answer else 0, notes=cov.notes)
        else:
            chunks = tb.search_chunks(question, k=8, rationale="fixed template: similarity search")
            for name in (interp.get("entities") or [])[:2]:
                ent = tb.resolve_entity(name, rationale="fixed template: entity linking")
                for d in ent["documents"][:1]:
                    chunks.extend(tb.fetch_document(d["doc_id"], rationale="fixed template: fetch linked document")[:3])
            reqs = [Requirement(req_id="R1", description="answer supported by retrieved passage", status="missing")]
            cov = CoverageReport(candidate_scope="retrieved passages", candidate_count=len(chunks))

        data, seen, _ = generate_answer(ctx, tb, question, chunks, graph_result)
        res.evidence_passages = seen
        res.explanation = data.get("explanation", "")
        llm_answer = clean_answer(data.get("answer", []))
        res.answer = machine_answer or llm_answer
        if intent in ("count_events", "max_events") and graph_result is not None and cov.candidate_count == 0:
            res.answer = []  # empty cohort: never fall back to a model guess
        res.interpretation["llm_answer"] = llm_answer
        res.citations = citations or build_citations(ctx, tb, data.get("cited_chunk_ids", []), chunks)
        cov.requirements = reqs
        if intent == "open_question" and data.get("supported") and not data.get("abstain"):
            reqs[0].status = "satisfied"
        res.coverage = cov
        if not res.answer or (data.get("abstain") and not machine_answer):
            res.status = "abstained"
            res.stop_reason = "no_evidence"
            res.explanation = res.explanation or data.get("abstain_reason", "")
        elif any(r.status != "satisfied" for r in reqs):
            res.status = "partial"
            res.stop_reason = "incomplete_scope" if cov.unknown_values else "single_pass"
    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.stop_reason = "error"
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
    return finalize(res, ctx)
