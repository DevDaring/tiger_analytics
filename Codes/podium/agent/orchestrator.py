"""Pipeline C: Agentic GraphRAG. A stateful harness where an orchestrator (LLM) chooses the next
specialist action from the current evidence and the coverage gaps, within explicit budgets.

Modes (ablations, plan section 8):
  full        - adaptive: coverage gaps + LLM decisions + evidence repair
  fixed_plan  - the same planner/tools but no replanning: the primary action per requirement,
                executed once, no repair or fallback strategies
  nocov       - LLM decides without the coverage ledger (no gap/requirement information)
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..contracts import AnswerResult, Candidate, CoverageReport, Requirement, stable_hash
from ..corpus.parse import norm_text
from ..llm.client import complete_json
from ..pipelines.answer import build_citations, clean_answer, finalize, generate_answer
from ..settings import settings
from ..tools.interpret import interpret
from ..tools.toolbox import RunContext, Toolbox
from .coverage import InvestigationState, assess, build_requirements

ACTIONS = ["resolve_event", "resolve_venue_events", "match_date", "aggregate", "superlative", "previous_event", "event_facts", "repair_missing_count", "search_chunks", "fetch_document", "resolve_entity", "disambiguate", "reinterpret", "finish", "abstain"]
FALLBACK_ACTIONS = {"repair_missing_count", "reinterpret", "disambiguate"}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "arguments": {
            "type": "object",
            "properties": {
                "sport": {"type": "string"}, "year": {"type": "integer"}, "season": {"type": "string"}, "event_name": {"type": "string"},
                "mode": {"type": "string", "enum": ["exact", "substring", "semantic", ""]}, "venue": {"type": "string"}, "date": {"type": "string"},
                "fuzzy": {"type": "boolean"}, "threshold": {"type": "integer"}, "op": {"type": "string"}, "event_id": {"type": "string"},
                "doc_id": {"type": "string"}, "query": {"type": "string"}, "k": {"type": "integer"}, "scoped": {"type": "boolean"}, "name": {"type": "string"},
                "hint": {"type": "string"}, "reason": {"type": "string"}, "event_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        "rationale": {"type": "string", "description": "one sentence: why this action, given the gaps"},
    },
}

DECIDE_SYSTEM = """You are the orchestrator of an evidence-driven investigation over a frozen corpus stored in a graph
database (Olympic event pages with infobox facts, chunks with vector embeddings, plus other Wikipedia articles).
You choose ONE next action. Choose based on the question, the evidence gathered so far and the unresolved
evidence requirements. Prefer the cheapest action that closes a gap; when the graph already answered the
question with complete coverage, finish. Do not repeat an action that already produced no new information -
change strategy instead (substring or semantic matching, fuzzy venue search, prose repair, re-interpretation).
If an ambiguity cannot be resolved from the corpus, finish and report it (the answer will list all candidates).
Abstain only when the corpus truly lacks the evidence."""


class Agent:
    def __init__(self, mode: str = "full"):
        assert mode in ("full", "fixed_plan", "nocov")
        self.mode = mode
        self.pipeline = {"full": "agentic_graphrag", "fixed_plan": "fixed_plan", "nocov": "agentic_nocov"}[mode]

    # ------------------------------------------------------------------------------------
    def run(self, question: str) -> AnswerResult:
        ctx = RunContext(question, self.pipeline)
        tb = Toolbox(ctx)
        res = AnswerResult()
        try:
            interp, u = interpret(question)
            ctx.add_usage(u)
            tb._record("orchestrator", "interpret", "llm", {}, f"intent={interp['intent']} sport={interp.get('sport')} year={interp.get('year')}", 1, ctx.t0, tokens=u.total, rationale="construct evidence requirements from the question")
            st = InvestigationState(question=question, interp=interp, requirements=build_requirements(interp))
            res.interpretation = interp
            stop_reason, stop_expl = self._loop(ctx, tb, st)
            self._synthesize(ctx, tb, st, res, stop_reason, stop_expl)
        except Exception as e:  # noqa: BLE001
            res.status = "error"
            res.stop_reason = "error"
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
        return finalize(res, ctx)

    # ------------------------------------------------------------------------------------
    def _loop(self, ctx: RunContext, tb: Toolbox, st: InvestigationState) -> tuple[str, str]:
        decisions = 0
        tool_calls = 0
        no_progress = 0
        primary_done: set[str] = set()
        while True:
            gaps = assess(st)
            missing = [r for r in st.requirements if r.status != "satisfied"]
            if not gaps or all(r.status in ("satisfied", "unresolvable", "conflicting") for r in st.requirements):
                if any(r.status == "conflicting" for r in st.requirements):
                    return "ambiguous_entity", "more than one corpus event satisfies the constraints; all candidates are reported"
                if any(r.status == "unresolvable" for r in st.requirements):
                    return "incomplete_scope", "some candidates could not be resolved from the corpus; bounds are reported"
                return "sufficient_evidence", "every evidence requirement is satisfied"
            if ctx.elapsed() > settings.agent_wall_clock_s:
                return "budget_exhausted", f"wall clock {settings.agent_wall_clock_s}s exceeded"
            if ctx.usage.total_tokens > settings.agent_token_budget:
                return "budget_exhausted", f"token budget {settings.agent_token_budget} exceeded"
            if tool_calls >= settings.agent_max_tool_calls or decisions >= settings.agent_max_decisions:
                return "budget_exhausted", "step budget exhausted"
            if no_progress >= 2:
                return "no_new_evidence", "two consecutive actions produced no new evidence"

            # ---- choose the next action
            if self.mode == "fixed_plan":
                choice = None
                for g in gaps:
                    for s in g["suggested_actions"]:
                        if s["action"] in FALLBACK_ACTIONS or s["arguments"].get("mode") in ("substring", "semantic") or s["arguments"].get("fuzzy"):
                            continue
                        sig = s["action"] + ":" + g["req_id"]
                        if sig not in primary_done:
                            choice = {**s, "rationale": "fixed plan: primary action for " + g["req_id"]}
                            primary_done.add(sig)
                            break
                    if choice:
                        break
                if choice is None:
                    return "single_pass", "fixed plan exhausted without replanning"
            else:
                choice = self._decide(ctx, tb, st, gaps if self.mode == "full" else None)
                decisions += 1
            action, args = choice["action"], choice.get("arguments") or {}
            if action in ("finish", "abstain"):
                if action == "abstain":
                    if tool_calls == 0 and gaps:
                        # never abstain before looking: take the top suggestion instead
                        st.notes.append("abstain before retrieval blocked; executing top suggestion")
                        choice = {**gaps[0]["suggested_actions"][0], "rationale": "guard: retrieve before abstaining"}
                        action, args = choice["action"], choice["arguments"]
                    else:
                        st.notes.append("orchestrator abstained: " + str(args.get("reason", "")))
                        return "no_evidence", str(args.get("reason", "orchestrator abstained"))
            if action == "finish":
                if st.intent == "open_question":
                    st.verified = True  # generation + verification happens in synthesis
                    return "sufficient_evidence", str(args.get("reason", "orchestrator finished"))
                # pre-finish evidence check: cheap deterministic fetches that close a gap run before stopping
                ran = 0
                for g in gaps:
                    for sug in g["suggested_actions"]:
                        if sug["action"] in ("event_facts", "match_date", "previous_event") and ran < 3:
                            sig2 = sug["action"] + ":" + stable_hash({k: v for k, v in sug["arguments"].items() if k != "reason"})
                            if sig2 not in st.tried:
                                st.tried[sig2] = 1
                                self._apply(ctx, tb, st, sug["action"], sug["arguments"], "pre-finish evidence check: " + sug["why"], False)
                                ran += 1
                                tool_calls += 1
                if ran:
                    continue  # re-assess with the new evidence, then let the orchestrator finish
                gaps_now = assess(st)
                missing = [r for r in st.requirements if r.status != "satisfied"]
                if any(r.status == "conflicting" for r in st.requirements):
                    return "ambiguous_entity", "more than one corpus event satisfies the constraints; all candidates are reported"
                return ("sufficient_evidence" if not missing else "incomplete_scope"), str(args.get("reason", "orchestrator finished"))
            sig = action + ":" + stable_hash({k: v for k, v in args.items() if k != "reason"})
            if st.tried.get(sig, 0) >= 1 and self.mode != "fixed_plan":
                # repeated action without new information: block once, then take the top suggestion
                st.notes.append(f"blocked repeated action {action}")
                alt = next((s for g in gaps for s in g["suggested_actions"] if (s["action"] + ":" + stable_hash({k: v for k, v in s['arguments'].items() if k != 'reason'})) not in st.tried), None)
                if alt is None:
                    return "no_new_evidence", "no untried action left for the remaining gaps"
                choice, action, args = alt, alt["action"], alt["arguments"]
                sig = action + ":" + stable_hash({k: v for k, v in args.items() if k != "reason"})
            st.tried[sig] = st.tried.get(sig, 0) + 1
            before = len(missing)
            strategy_change = action in FALLBACK_ACTIONS or args.get("mode") in ("substring", "semantic") or bool(args.get("fuzzy")) or (action == "search_chunks" and st.intent != "open_question")
            progressed = self._apply(ctx, tb, st, action, args, choice.get("rationale", ""), strategy_change)
            tool_calls += 1
            after = len([r for r in assess(st) if True]) if False else len([r for r in st.requirements if r.status != "satisfied"])
            ev = ctx.trace.events[-1]
            ev.coverage_before, ev.coverage_after = before, after
            no_progress = 0 if progressed else no_progress + 1

    # ------------------------------------------------------------------------------------
    def _decide(self, ctx: RunContext, tb: Toolbox, st: InvestigationState, gaps: list[dict] | None) -> dict:
        t0 = time.time()
        summary = self._state_summary(st)
        parts = [f"Question: {st.question}", "Interpretation: " + json.dumps({k: v for k, v in st.interp.items() if v not in ("", 0, [], None)}, ensure_ascii=False), "Evidence so far:\n" + summary]
        if gaps is not None:
            parts.append("Evidence requirements:\n" + "\n".join(f"  {r.req_id} [{r.status}] {r.description}" + (f" ({r.note})" if r.note else "") for r in st.requirements))
            parts.append("Unresolved gaps and candidate actions:\n" + "\n".join(f"  {g['req_id']}: " + "; ".join(f"{s['action']}({json.dumps(s['arguments'], ensure_ascii=False)}) - {s['why']}" for s in g["suggested_actions"]) for g in gaps))
        else:
            parts.append("Available actions: " + ", ".join(ACTIONS))
        hist = [f"  {a['tool']}({json.dumps(a['args'], ensure_ascii=False)[:160]}) -> {a['n']} result(s)" + (f" ERROR {a['error']}" if a.get("error") else "") for a in ctx.action_log[-10:]]
        parts.append("Actions already taken:\n" + ("\n".join(hist) if hist else "  none"))
        parts.append("Choose the next action.")
        try:
            data, u = complete_json("decide", DECIDE_SYSTEM, "\n\n".join(parts), DECISION_SCHEMA, max_tokens=400, effort="low")
            ctx.add_usage(u)
            data["arguments"] = {k: v for k, v in (data.get("arguments") or {}).items() if v not in ("", None, 0, False, [])}
            tb._record("orchestrator", "decide", "llm", {"action": data.get("action")}, f"{data.get('action')} {json.dumps(data.get('arguments'), ensure_ascii=False)[:120]}", 1, t0, tokens=u.total, rationale=data.get("rationale", ""))
            return data
        except Exception as e:  # noqa: BLE001
            tb._record("orchestrator", "decide", "llm", {}, "decision error; using top suggestion", 0, t0, error=str(e)[:200])
            if gaps:
                s = gaps[0]["suggested_actions"][0]
                return {**s, "rationale": "fallback to top suggestion after decision error"}
            return {"action": "finish", "arguments": {"reason": "decision error"}, "rationale": ""}

    def _state_summary(self, st: InvestigationState) -> str:
        lines = []
        if st.events:
            lines.append(f"  resolved events ({len(st.events)}): " + "; ".join(f"{e['title']} [{e['event_id']}] venue='{e.get('venue_raw')}' date='{e.get('date_raw')}'" for e in st.events[:8]))
        if st.date_match:
            lines.append(f"  date match: {len(st.date_match['best'])} best ({st.date_match['reason']})")
        if st.prev_event:
            lines.append(f"  predecessor: {st.prev_event['title']} via {st.prev_sources}")
        elif st.ref_event and st.ref_event.get("_prev_checked"):
            lines.append("  predecessor: none in graph" + (f"; previous corpus edition is {st.prev_edition.get('edition_id') or st.prev_edition.get('v_id')}" if st.prev_edition else ""))
        for eid, fr in list(st.facts_for.items())[:4]:
            e = fr.get("event") or {}
            lines.append(f"  facts[{eid}]: gold='{e.get('gold_raw')}' nations='{e.get('nations_raw')}' competitors='{e.get('competitors_raw')}'")
        for name, r in (("aggregate", st.aggregate), ("superlative", st.superlative)):
            if r:
                unk = [c.title for c in r["manifest"] if c.status == "unknown"]
                head = f"count={r['count']}" if name == "aggregate" else f"max={r['max_value']} winners={[w['title'] for w in r['winners']]}"
                lines.append(f"  {name}: {head}; {r['n_candidates']} candidates, {len(unk)} unknown" + (f": {unk[:4]}" if unk else ""))
        for eid, rep in st.repairs.items():
            lines.append(f"  repair[{eid}]: " + (f"value={rep.get('value')} verified={rep.get('verified')} quote='{str(rep.get('quote'))[:80]}'" if rep.get("found") else "not found in prose"))
        if st.chunks:
            lines.append(f"  passages retrieved: {len(st.chunks)} (" + "; ".join(c.get("title", "")[:50] for c in st.chunks[:5]) + ")")
        if st.entity_docs:
            lines.append("  linked documents: " + "; ".join(d.get("title", "") for d in st.entity_docs[:5]))
        if st.notes:
            lines.append("  notes: " + " | ".join(st.notes[-3:]))
        return "\n".join(lines) if lines else "  nothing yet"

    # ------------------------------------------------------------------------------------
    def _apply(self, ctx: RunContext, tb: Toolbox, st: InvestigationState, action: str, a: dict, rationale: str, strategy_change: bool) -> bool:
        """Execute one action; return True when it added new evidence."""
        p = st.interp
        n_before = (len(ctx.events), len(ctx.chunks), len(ctx.facts), len(st.repairs))
        try:
            if action == "resolve_event":
                mode = a.get("mode") or "exact"
                sport, year, season, name = a.get("sport", p.get("sport", "")), int(a.get("year") or p.get("year") or 0), a.get("season", p.get("season", "")), a.get("event_name", p.get("event_name", ""))
                if mode == "exact":
                    evs = tb.find_events(sport, year, season, name, rationale=rationale)
                elif mode == "substring":
                    evs = tb.events_like(name, year, season, rationale=rationale)
                    evs = self._rank_by_name(evs, name, sport)
                else:
                    ch = tb.search_chunks(f"{name} {sport} {year} {season} Olympics", k=8, sport=sport, year=year, season=season, rationale=rationale)
                    ids = []
                    for c in ch:
                        if c["doc_id"] not in ids:
                            ids.append(c["doc_id"])
                    evs = []
                    for did in ids[:4]:
                        fr = tb.event_facts(did, rationale="inspect semantic candidate")
                        if fr["event"]:
                            st.facts_for[did] = fr
                            evs.append(fr["event"])
                    evs = self._rank_by_name(evs, name, sport)
                if evs:
                    st.events = evs
                    if st.intent == "previous_edition_winner":
                        if year and year != p.get("year") and st.ref_event:
                            # searched the previous edition directly
                            st.prev_event = evs[0]
                            st.prev_sources = ["direct_search_previous_edition"]
                            st.events = [st.ref_event]
                        else:
                            st.ref_event = evs[0] if len(evs) == 1 else None
                if strategy_change:
                    ctx.trace.events[-1].strategy_change = True
            elif action == "resolve_venue_events":
                evs, venues = tb.events_at_venue(a.get("venue", p.get("venue", "")), int(a.get("year") or p.get("year") or 0), rationale=rationale, fuzzy=bool(a.get("fuzzy")))
                st.venues = venues
                if evs:
                    st.events = evs
                    st.date_match = None
                if strategy_change:
                    ctx.trace.events[-1].strategy_change = True
                if evs and p.get("date"):
                    # the date filter is part of the same linking step (deterministic, no extra decision)
                    st.date_match = tb.match_date(st.events, p.get("date", ""), rationale="filter the venue's events by the question date")
                    st.ambiguous = st.date_match["ambiguous"]
            elif action == "match_date":
                st.date_match = tb.match_date(st.events, a.get("date", p.get("date", "")), rationale=rationale)
                st.ambiguous = st.date_match["ambiguous"]
            elif action == "aggregate":
                st.aggregate = tb.aggregate_competitors(a.get("sport", p.get("sport", "")), int(a.get("year") or p.get("year") or 0), a.get("season", p.get("season", "")), int(a.get("threshold", p.get("threshold", 0))), a.get("op") or p.get("operator") or ">", rationale=rationale)
            elif action == "superlative":
                st.superlative = tb.max_competitors(a.get("sport", p.get("sport", "")), int(a.get("year") or p.get("year") or 0), a.get("season", p.get("season", "")), rationale=rationale)
            elif action == "previous_event":
                eid = a.get("event_id") or (st.events[0]["event_id"] if st.events else "")
                pv = tb.previous_event(eid, rationale=rationale)
                if st.ref_event is None and st.events:
                    st.ref_event = st.events[0]
                if st.ref_event:
                    st.ref_event["_prev_checked"] = True
                st.prev_edition = pv["previous_edition"]
                if pv["previous"]:
                    st.prev_event = pv["previous"][0]
                    st.prev_sources = pv["sources"]
            elif action == "event_facts":
                eid = a.get("event_id") or (st.events[0]["event_id"] if st.events else "")
                fr = tb.event_facts(eid, rationale=rationale)
                st.facts_for[eid] = fr
            elif action == "repair_missing_count":
                ids = a.get("event_ids") or ([a["event_id"]] if a.get("event_id") else [])
                ids = [i for i in ids if i not in st.repairs][:4]
                for eid in ids:
                    ev = ctx.events.get(eid) or {"event_id": eid, "title": eid}
                    rep = tb.extract_count_from_prose(ev, "competitors", rationale=rationale)
                    st.repairs[eid] = rep
                    ctx.trace.events[-1].strategy_change = True
                if ids:
                    self._recompute_after_repair(ctx, tb, st)
            elif action == "search_chunks":
                q = a.get("query") or st.question
                if a.get("scoped"):
                    ch = tb.search_chunks(q, k=int(a.get("k") or 8), sport=p.get("sport", ""), year=p.get("year", 0), season=p.get("season", ""), rationale=rationale)
                else:
                    ch = tb.search_chunks(q, k=int(a.get("k") or 8), rationale=rationale)
                st.chunks.extend(c for c in ch if c["chunk_id"] not in {x["chunk_id"] for x in st.chunks})
                if strategy_change:
                    ctx.trace.events[-1].strategy_change = True
                # semantic evidence may identify events for structured intents (venue must still be compatible)
                if st.intent == "event_by_venue_date" and not st.events:
                    from ..tools.toolbox import venue_compatible

                    evs = []
                    for c in ch:
                        did = c["doc_id"]
                        if did in ctx.events and ctx.events[did] not in evs:
                            evs.append(ctx.events[did])
                        elif did not in ctx.events:
                            fr = tb.event_facts(did, rationale="inspect semantic candidate")
                            if fr["event"]:
                                st.facts_for[did] = fr
                                evs.append(fr["event"])
                    evs = [e for e in evs if venue_compatible(p.get("venue", ""), e.get("venue_id", ""))]
                    if evs:
                        st.events = evs
                        st.date_match = None
            elif action == "fetch_document":
                ch = tb.fetch_document(a.get("doc_id", ""), rationale=rationale)
                st.chunks.extend(c for c in ch[:6] if c["chunk_id"] not in {x["chunk_id"] for x in st.chunks})
            elif action == "resolve_entity":
                ent = tb.resolve_entity(a.get("name", ""), rationale=rationale)
                st.entity_docs.extend(ent["documents"])
                for d in ent["documents"][:1]:
                    ch = tb.fetch_document(d["doc_id"], rationale="fetch the linked document")
                    st.chunks.extend(c for c in ch[:4] if c["chunk_id"] not in {x["chunk_id"] for x in st.chunks})
            elif action == "disambiguate":
                self._disambiguate(ctx, tb, st, a.get("hint", ""))
                ctx.trace.events[-1].strategy_change = True
            elif action == "reinterpret":
                if st.reinterpreted >= 1:
                    tb._record("orchestrator", "reinterpret", "deterministic", a, "blocked: already re-interpreted once", 0, time.time())
                    return False
                st.reinterpreted += 1
                interp, u = interpret(st.question + "\n(Hint: " + str(a.get("hint", "")) + ")")
                ctx.add_usage(u)
                st.interp = interp
                st.requirements = build_requirements(interp)
                st.aggregate = st.superlative = None
                tb._record("orchestrator", "reinterpret", "llm", {"hint": a.get("hint", "")}, f"intent={interp['intent']} sport={interp.get('sport')} year={interp.get('year')}", 1, time.time(), tokens=u.total, rationale=rationale, strategy_change=True)
            else:
                tb._record("orchestrator", action, "deterministic", a, "unknown action", 0, time.time(), error="unknown action")
                return False
        except Exception as e:  # noqa: BLE001
            if not ctx.trace.events or ctx.trace.events[-1].error is None:
                tb._record("orchestrator", action, "deterministic", a, "error", 0, time.time(), error=f"{type(e).__name__}: {str(e)[:200]}")
            st.notes.append(f"{action} failed: {type(e).__name__}")
            return False
        n_after = (len(ctx.events), len(ctx.chunks), len(ctx.facts), len(st.repairs))
        return n_after != n_before or action in ("match_date", "aggregate", "superlative", "previous_event", "disambiguate", "reinterpret")

    @staticmethod
    def _rank_by_name(evs: list[dict], name: str, sport: str) -> list[dict]:
        from ..corpus.parse import event_norm

        target = set(event_norm(name).split())
        scored = []
        for e in evs:
            toks = set((e.get("event_norm") or "").split())
            j = len(target & toks) / max(1, len(target | toks))
            if sport and norm_text(sport) not in norm_text(e.get("sport", "")):
                j *= 0.5
            scored.append((j, e))
        scored.sort(key=lambda x: -x[0])
        best = [e for j, e in scored if j >= 0.6]
        return best[:1] if len(best) >= 1 and scored[0][0] >= 0.75 and (len(scored) == 1 or scored[1][0] < scored[0][0]) else best[:3]

    def _disambiguate(self, ctx: RunContext, tb: Toolbox, st: InvestigationState, hint: str) -> None:
        """Deterministic disambiguation by question wording; otherwise keep all candidates (ambiguous)."""
        t0 = time.time()
        cands = st.events if st.intent != "event_by_venue_date" else [ctx.events[b["event_id"]] for b in (st.date_match or {}).get("best", [])]
        q = norm_text(st.question)
        scored = []
        for e in cands:
            toks = [t for t in (e.get("event_norm") or "").split() if len(t) > 2]
            score = sum(1 for t in toks if t in q) / max(1, len(toks))
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.6 and (len(scored) == 1 or scored[0][0] > scored[1][0] + 0.2):
            keep = [scored[0][1]]
            st.events = keep if st.intent != "event_by_venue_date" else st.events
            if st.intent == "event_by_venue_date":
                st.date_match["best"] = [b for b in st.date_match["best"] if b["event_id"] == keep[0]["event_id"]]
                st.date_match["ambiguous"] = False
            if st.intent == "previous_edition_winner":
                st.ref_event = keep[0]
            summ = f"selected '{keep[0]['title']}' by question wording"
        else:
            st.ambiguous = True
            summ = f"{len(cands)} candidates remain; question wording does not distinguish them (reported as ambiguous)"
            for r in st.requirements:
                if r.status == "missing" and r.req_id in ("R1", "R2") and st.intent in ("event_by_venue_date", "lookup_field", "previous_edition_winner"):
                    r.status = "conflicting"
        tb._record("evidence_evaluator", "disambiguate", "deterministic", {"hint": hint, "n_candidates": len(cands)}, summ, len(cands), t0, [e["event_id"] for e in cands], strategy_change=True)

    def _recompute_after_repair(self, ctx: RunContext, tb: Toolbox, st: InvestigationState) -> None:
        """Re-evaluate the aggregate/maximum deterministically using verified prose values."""
        t0 = time.time()
        res = st.aggregate or st.superlative
        if not res:
            return
        changed = 0
        for c in res["manifest"]:
            rep = st.repairs.get(c.event_id)
            if rep and "implausible" in (c.reason or "") and c.status != "unknown":
                # verification of a suspicious infobox value: the infobox stays the primary source,
                # but a disagreement with the prose is recorded as an explicit conflict
                if rep.get("found") and rep.get("verified") and float(rep["value"]) != float(c.value or -1):
                    st.conflicts.append({"event_id": c.event_id, "title": c.title, "predicate": "competitors", "infobox_value": c.value_raw, "prose_value": rep["value"], "prose_quote": rep.get("quote", ""), "chunk_id": rep.get("chunk_id"), "char_start": rep.get("char_start"), "char_end": rep.get("char_end"), "policy": "infobox value used for the machine answer; conflict reported"})
                    c.reason = f"infobox says {c.value_raw}; prose says {rep['value']} ('{rep.get('quote','')[:60]}') - conflict reported, infobox value used"
                elif rep.get("found") and rep.get("verified"):
                    c.reason = c.reason.replace(" (implausible infobox value; verify against prose)", f" (prose agrees: {rep['value']})")
                else:
                    c.reason = c.reason.replace("verify against prose", "prose gives no total")
                continue
            if c.status == "unknown" and rep and rep.get("found") and rep.get("verified"):
                c.value = float(rep["value"])
                c.value_raw = str(rep["value"])
                c.source = "prose_extraction"
                c.chunk_id, c.char_start, c.char_end = rep.get("chunk_id"), rep.get("char_start"), rep.get("char_end")
                if st.aggregate:
                    thr, op = res["threshold"], res["op"]
                    ok = {">": c.value > thr, ">=": c.value >= thr, "<": c.value < thr, "<=": c.value <= thr, "==": c.value == thr}.get(op, False)
                    c.status = "included" if ok else "excluded"
                    c.reason = f"prose: '{rep.get('quote','')[:80]}' -> {int(c.value)} {op} {thr}"
                else:
                    c.status = "excluded"
                    c.reason = f"prose: '{rep.get('quote','')[:80]}' -> {int(c.value)}"
                changed += 1
        if changed == 0:
            tb._record("evidence_evaluator", "verify_values", "deterministic", {"conflicts": len(st.conflicts)}, f"{len(st.conflicts)} infobox/prose conflict(s) recorded; graph result unchanged", len(res["manifest"]), t0)
            return
        if st.aggregate:
            inc = [c for c in res["manifest"] if c.status == "included"]
            res["count"] = len(inc)
            res["n_unknown"] = len([c for c in res["manifest"] if c.status == "unknown"])
            summ = f"count={res['count']} after {changed} repaired value(s); {res['n_unknown']} still unknown"
        else:
            known = [c for c in res["manifest"] if c.value is not None]
            mx = max((c.value for c in known), default=None)
            winners = [c for c in known if c.value == mx]
            for c in res["manifest"]:
                if c.value is not None:
                    c.status = "included" if c.value == mx else "excluded"
                    c.reason = "maximum" if c.value == mx else c.reason or "below maximum"
            res["max_value"] = mx
            res["winners"] = [ctx.events.get(w.event_id, {"event_id": w.event_id, "title": w.title}) for w in winners]
            res["n_unknown"] = len([c for c in res["manifest"] if c.status == "unknown"])
            summ = f"max={mx} winners={[w.title for w in winners]} after {changed} repaired value(s)"
        tb._record("aggregator", "recompute_aggregate", "deterministic", {"repaired": changed}, summ, len(res["manifest"]), t0)

    # ------------------------------------------------------------------------------------
    def _synthesize(self, ctx: RunContext, tb: Toolbox, st: InvestigationState, res: AnswerResult, stop_reason: str, stop_expl: str) -> None:
        p = st.interp
        i = st.intent
        chunks: list[dict] = []
        graph_result: dict | None = None
        citations = []
        cov = CoverageReport(requirements=st.requirements)
        machine: list[str] = []
        if i == "count_events" and st.aggregate:
            r = st.aggregate
            graph_result = {k: v for k, v in r.items() if k != "manifest"} | {"manifest": [c.model_dump() for c in r["manifest"]]}
            machine = [str(r["count"])] if r["n_candidates"] > 0 else []
            if r["n_candidates"] == 0:
                cov.notes.append(f"the corpus contains no {r['sport'] or 'matching'} events for the {r['year']} {r['season']} Olympics; no count can be given")
            cov.candidate_scope = f"{r['sport']} events at the {r['year']} {r['season']} Olympics (corpus-defined)"
            cov.candidate_count, cov.unknown_values = r["n_candidates"], r["n_unknown"]
            cov.known_values = r["n_candidates"] - r["n_unknown"]
            cov.candidates = r["manifest"]
            cov.complete = r["n_unknown"] == 0 and r["n_candidates"] > 0
            if r["n_unknown"]:
                cov.bounds = {"lower": r["count"], "upper": r["count"] + r["n_unknown"]}
                cov.notes.append(f"{r['n_unknown']} candidate(s) remain unknown after repair attempts; count is a lower bound.")
            for c in r["manifest"]:
                if c.status == "included":
                    if c.source == "prose_extraction" and c.chunk_id:
                        ch = ctx.chunks.get(c.chunk_id)
                        rep = st.repairs.get(c.event_id) or {}
                        if ch:
                            cit = tb.citation_for_chunk(ch, "count", quote=rep.get("quote", ""))
                            cit.char_start, cit.char_end, cit.source = rep.get("char_start"), rep.get("char_end"), "document"
                            citations.append(cit)
                    else:
                        cit = tb.citation_for_fact(ctx.events[c.event_id], "competitors", "count")
                        if cit:
                            citations.append(cit)
            for eid, rep in st.repairs.items():
                if rep.get("verified") and rep.get("chunk_id") in ctx.chunks:
                    chunks.append(ctx.chunks[rep["chunk_id"]])
        elif i == "max_events" and st.superlative:
            r = st.superlative
            graph_result = {k: v for k, v in r.items() if k not in ("manifest", "winners")} | {"winners": [w["title"] for w in r["winners"]], "manifest": [c.model_dump() for c in r["manifest"]]}
            machine = [w["title"] for w in r["winners"]]
            cov.candidate_scope = f"{r['sport']} events at the {r['year']} {r['season']} Olympics (corpus-defined)"
            cov.candidate_count, cov.unknown_values = r["n_candidates"], r["n_unknown"]
            cov.known_values = r["n_candidates"] - r["n_unknown"]
            cov.candidates = r["manifest"]
            cov.complete = r["n_unknown"] == 0 and r["n_candidates"] > 0
            if r["n_unknown"]:
                cov.notes.append(f"{r['n_unknown']} candidate(s) have no comparable value; the maximum is over known values.")
            for w in r["winners"]:
                cit = tb.citation_for_fact(ctx.events.get(w["event_id"], w), "competitors", "max")
                if cit:
                    citations.append(cit)
        elif i == "lookup_field" and len(st.events) == 1:
            e = st.events[0]
            fr = st.facts_for.get(e["event_id"])
            e = (fr or {}).get("event") or e
            fld = p.get("field") or "nations"
            raw = e.get(f"{fld}_raw") if f"{fld}_raw" in e else e.get(fld)
            if fld in ("nations", "competitors") and e.get(f"has_{fld}"):
                machine = [str(e[fld])]
            elif raw:
                machine = [str(raw)]
            graph_result = {"event": e["title"], fld: raw, "venue": e.get("venue_raw"), "date": e.get("date_raw")}
            chunks = list((fr or {}).get("chunks", [])) + st.chunks
            cit = tb.citation_for_fact(e, fld, "lookup")
            if cit:
                citations.append(cit)
            cov.candidate_scope, cov.candidate_count, cov.known_values = "single event", 1, 1 if machine else 0
        elif i == "event_by_venue_date":
            best = (st.date_match or {}).get("best", [])
            targets = [ctx.events[b["event_id"]] for b in best]
            golds = []
            for t in targets[:3]:
                fr = st.facts_for.get(t["event_id"])
                e = (fr or {}).get("event") or t
                golds.append({"event": e["title"], "gold": e.get("gold_raw"), "gold_noc": e.get("gold_noc"), "venue": e.get("venue_raw"), "date": e.get("date_raw")})
                chunks.extend((fr or {}).get("chunks", []))
                cit = tb.citation_for_fact(e, "gold", "gold")
                if cit:
                    citations.append(cit)
            if golds:
                graph_result = {"matched_events": golds, "date_match": (st.date_match or {}).get("reason"), "venues_considered": st.venues}
                machine = [g["gold"] for g in golds if g.get("gold")]
            chunks.extend(st.chunks)
            cov.candidate_scope = f"events at venue '{p.get('venue')}'"
            cov.candidate_count, cov.known_values = len(st.events), len(targets)
            if len(targets) > 1:
                cov.notes.append("ambiguous: more than one corpus event has this venue and date; all gold medallists are listed")
        elif i == "previous_edition_winner":
            if st.prev_event:
                fr = st.facts_for.get(st.prev_event["event_id"])
                e = (fr or {}).get("event") or st.prev_event
                graph_result = {"reference_event": (st.ref_event or {}).get("title"), "predecessor_event": e["title"], "predecessor_source": st.prev_sources, "gold": e.get("gold_raw"), "gold_noc": e.get("gold_noc")}
                machine = [e["gold_raw"]] if e.get("gold_raw") else []
                chunks = list((fr or {}).get("chunks", []))
                cit = tb.citation_for_fact(e, "gold", "gold")
                if cit:
                    citations.append(cit)
            elif st.ref_event and st.ref_event.get("_prev_checked"):
                cov.notes.append("the corpus contains no document for this event at the previous edition")
            chunks.extend(st.chunks)
            cov.candidate_scope, cov.candidate_count, cov.known_values = "reference event and its predecessor", len(st.events), 1 if machine else 0
        else:
            chunks = st.chunks
            cov.candidate_scope, cov.candidate_count = "retrieved passages", len(chunks)

        data, seen, _ = generate_answer(ctx, tb, st.question, chunks, graph_result, extra_instructions=("Stop reason of the investigation: " + stop_expl) if stop_reason != "sufficient_evidence" else "")
        res.evidence_passages = seen
        res.explanation = data.get("explanation", "")
        llm_answer = clean_answer(data.get("answer", []))
        res.answer = machine or llm_answer
        if i in ("count_events", "max_events") and (st.aggregate or st.superlative) and (st.aggregate or st.superlative)["n_candidates"] == 0:
            res.answer = []  # empty cohort: no count/maximum exists in the corpus; never fall back to a model guess
        res.interpretation = {**st.interp, "llm_answer": llm_answer, "mode": self.mode}
        res.citations = citations or build_citations(ctx, tb, data.get("cited_chunk_ids", []), chunks)
        if i == "open_question":
            supported = bool(data.get("supported")) and not data.get("abstain")
            for r in st.requirements:
                if r.req_id == "R2":
                    r.status = "satisfied" if supported else "missing"
        cov.conflicts = st.conflicts
        for cf in st.conflicts:
            cov.notes.append(f"conflict: '{cf['title']}' infobox competitors={cf['infobox_value']} but prose says {cf['prose_value']}; infobox value used, conflict reported")
        res.coverage = cov
        res.stop_reason = stop_reason
        ctx.trace.stop_explanation = stop_expl
        if not res.answer or (data.get("abstain") and not machine):
            res.status = "abstained"
            res.stop_reason = stop_reason if stop_reason not in ("sufficient_evidence",) else "no_evidence"
            res.explanation = res.explanation or data.get("abstain_reason", "")
        elif any(r.status != "satisfied" for r in st.requirements):
            res.status = "partial"
        else:
            res.status = "answered"


def run(question: str, mode: str = "full") -> AnswerResult:
    return Agent(mode).run(question)
