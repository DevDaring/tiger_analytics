"""Coverage-directed investigation: explicit evidence requirements per question intent, a
deterministic assessment of what is satisfied, and *suggested* repair actions for each gap.

The orchestrator (LLM) chooses the next action; this module only tells it what is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import Candidate, Requirement


@dataclass
class InvestigationState:
    question: str
    interp: dict
    requirements: list[Requirement] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # resolved candidate events for the question
    venues: list[dict] = field(default_factory=list)
    date_match: dict | None = None
    ref_event: dict | None = None
    prev_event: dict | None = None
    prev_sources: list[str] = field(default_factory=list)
    prev_edition: dict | None = None
    facts_for: dict[str, dict] = field(default_factory=dict)  # event_id -> event_facts result
    aggregate: dict | None = None
    superlative: dict | None = None
    repairs: dict[str, dict] = field(default_factory=dict)  # event_id -> extraction result
    chunks: list[dict] = field(default_factory=list)  # passages retrieved for answering
    entity_docs: list[dict] = field(default_factory=list)
    machine_answer: list[str] = field(default_factory=list)
    graph_result: dict | None = None
    notes: list[str] = field(default_factory=list)
    reinterpreted: int = 0
    tried: dict[str, int] = field(default_factory=dict)  # action signature -> count
    ambiguous: bool = False
    verified: bool | None = None
    conflicts: list[dict] = field(default_factory=list)

    @property
    def intent(self) -> str:
        return self.interp.get("intent", "open_question")


def build_requirements(interp: dict) -> list[Requirement]:
    i = interp.get("intent")
    if i == "lookup_field":
        return [Requirement(req_id="R1", description="the named event is resolved unambiguously in the graph"), Requirement(req_id="R2", description=f"the '{interp.get('field') or 'nations'}' value is supported by an exact source span")]
    if i == "count_events":
        return [Requirement(req_id="R1", description="the complete corpus-defined cohort (sport + edition) is enumerated"), Requirement(req_id="R2", description="the predicate outcome is known for every candidate (no unknown competitor counts)"), Requirement(req_id="R3", description="the count is computed by a graph query over the cohort")]
    if i == "max_events":
        return [Requirement(req_id="R1", description="the complete corpus-defined cohort (sport + edition) is enumerated"), Requirement(req_id="R2", description="a comparable competitor value is known for every candidate"), Requirement(req_id="R3", description="the maximum is computed over the cohort with ties preserved")]
    if i == "event_by_venue_date":
        return [Requirement(req_id="R1", description="the venue is resolved to graph venue node(s)"), Requirement(req_id="R2", description="exactly one event matches both venue and date"), Requirement(req_id="R3", description="the gold medallist is supported by an exact source span")]
    if i == "previous_edition_winner":
        return [Requirement(req_id="R1", description="the reference-edition event is resolved"), Requirement(req_id="R2", description="the predecessor event is justified (explicit prev field or corpus edition order)"), Requirement(req_id="R3", description="the predecessor's gold medallist is supported by an exact source span")]
    return [Requirement(req_id="R1", description="relevant passages retrieved"), Requirement(req_id="R2", description="the answer is directly supported by a retrieved passage")]


def _has_fact(st: InvestigationState, event_id: str, predicate: str) -> bool:
    fr = st.facts_for.get(event_id)
    if not fr:
        return False
    return any(f.get("predicate") == predicate for f in fr.get("facts", []))


def assess(st: InvestigationState) -> list[dict[str, Any]]:
    """Update requirement statuses from the state; return gaps with suggested actions."""
    i = st.intent
    p = st.interp
    gaps: list[dict] = []
    R = {r.req_id: r for r in st.requirements}

    def gap(rid: str, *suggestions: dict) -> None:
        gaps.append({"req_id": rid, "description": R[rid].description, "suggested_actions": list(suggestions)})

    if i in ("lookup_field", "previous_edition_winner"):
        n = len(st.events)
        R["R1"].status = "satisfied" if n == 1 else "missing"
        R["R1"].note = f"{n} candidate event(s)"
        if n == 0:
            gap("R1",
                {"action": "resolve_event", "arguments": {"sport": p.get("sport", ""), "year": p.get("year", 0), "season": p.get("season", ""), "event_name": p.get("event_name", ""), "mode": "exact"}, "why": "exact title-derived match"},
                {"action": "resolve_event", "arguments": {"sport": p.get("sport", ""), "year": p.get("year", 0), "season": p.get("season", ""), "event_name": p.get("event_name", ""), "mode": "substring"}, "why": "title substring match if the exact event name differs"},
                {"action": "resolve_event", "arguments": {"sport": p.get("sport", ""), "year": p.get("year", 0), "season": p.get("season", ""), "event_name": p.get("event_name", ""), "mode": "semantic"}, "why": "scoped vector search over the sport/edition when names differ"})
        elif n > 1:
            gap("R1", {"action": "disambiguate", "arguments": {"hint": "choose the event whose title matches the question wording"}, "why": "several candidates; pick by wording or abstain as ambiguous"})
        if i == "lookup_field":
            fld = p.get("field") or "nations"
            ok = n == 1 and _has_fact(st, st.events[0]["event_id"], fld)
            R["R2"].status = "satisfied" if ok else "missing"
            if n == 1 and not ok:
                if st.facts_for.get(st.events[0]["event_id"]):
                    gap("R2", {"action": "fetch_document", "arguments": {"doc_id": st.events[0]["event_id"]}, "why": "the infobox has no such field; look for the value in the article prose"})
                else:
                    gap("R2", {"action": "event_facts", "arguments": {"event_id": st.events[0]["event_id"]}, "why": "fetch the field with its source span"})
        else:
            R["R2"].status = "satisfied" if st.prev_event else "missing"
            R["R2"].note = f"source={st.prev_sources}" if st.prev_event else ("no predecessor document in corpus" if st.prev_edition is not None and n == 1 and st.ref_event and st.ref_event.get("_prev_checked") else "")
            if n == 1 and not st.prev_event:
                if not (st.ref_event or {}).get("_prev_checked"):
                    gap("R2", {"action": "previous_event", "arguments": {"event_id": st.events[0]["event_id"]}, "why": "follow the PREVIOUS_EVENT edge (infobox prev or derived edition order)"})
                else:
                    yr = (st.prev_edition or {}).get("year") or (st.events[0].get("prev_year") or 0)
                    gap("R2", {"action": "resolve_event", "arguments": {"sport": p.get("sport", ""), "year": yr, "season": p.get("season", ""), "event_name": p.get("event_name", ""), "mode": "substring"}, "why": "no edge: search the previous edition directly by title"},
                        {"action": "resolve_event", "arguments": {"sport": p.get("sport", ""), "year": yr, "season": p.get("season", ""), "event_name": p.get("event_name", ""), "mode": "semantic"}, "why": "semantic search in the previous edition"})
            ok = bool(st.prev_event) and _has_fact(st, st.prev_event["event_id"], "gold")
            R["R3"].status = "satisfied" if ok else "missing"
            if st.prev_event and not ok:
                gap("R3", {"action": "event_facts", "arguments": {"event_id": st.prev_event["event_id"]}, "why": "fetch the gold medallist with its source span"})
    elif i in ("count_events", "max_events"):
        res = st.aggregate if i == "count_events" else st.superlative
        if not res:
            R["R1"].status = R["R2"].status = R["R3"].status = "missing"
            act = {"action": "aggregate", "arguments": {"sport": p.get("sport", ""), "year": p.get("year", 0), "season": p.get("season", ""), "threshold": p.get("threshold", 0), "op": p.get("operator") or ">"}} if i == "count_events" else {"action": "superlative", "arguments": {"sport": p.get("sport", ""), "year": p.get("year", 0), "season": p.get("season", "")}}
            act["why"] = "enumerate the complete cohort and compute in the graph"
            gap("R1", act)
        else:
            n = res["n_candidates"]
            R["R1"].status = "satisfied" if n > 0 else "missing"
            R["R1"].note = f"{n} candidates"
            if n == 0:
                gap("R1", {"action": "reinterpret", "arguments": {"hint": "the sport/edition produced no candidates; re-read the sport name and year"}, "why": "empty cohort suggests a mis-read sport or edition"})
            unknown = [c for c in res["manifest"] if c.status == "unknown" and c.event_id not in st.repairs]
            still_unknown = [c for c in res["manifest"] if c.status == "unknown"]
            R["R2"].status = "satisfied" if n > 0 and not still_unknown else ("unresolvable" if n > 0 and not unknown else "missing")
            R["R2"].note = f"{len(still_unknown)} unknown" if still_unknown else ""
            if unknown:
                names = "; ".join(c.title.split(" – ")[-1] for c in unknown[:4])
                gap("R2", {"action": "repair_missing_count", "arguments": {"event_ids": [c.event_id for c in unknown[:4]]}, "why": f"{len(unknown)} candidate(s) have no parseable competitors field ({names}); recover the values from the article prose with a verified quote, in one batch"})
            suspicious = [c for c in res["manifest"] if "implausible" in (c.reason or "") and c.event_id not in st.repairs]
            if suspicious and n > 0:
                R["R2"].status = "missing"
                R["R2"].note = (R["R2"].note + "; " if R["R2"].note else "") + f"{len(suspicious)} implausible value(s) unverified"
                names = "; ".join(c.title.split(" – ")[-1] for c in suspicious[:4])
                gap("R2", {"action": "repair_missing_count", "arguments": {"event_ids": [c.event_id for c in suspicious[:4]]}, "why": f"implausible infobox value(s) for {names}; verify against the article prose and report a conflict if they disagree"})
            R["R3"].status = "satisfied" if n > 0 else "missing"
    elif i == "event_by_venue_date":
        R["R1"].status = "satisfied" if st.events else "missing"
        if not st.events:
            gap("R1",
                {"action": "resolve_venue_events", "arguments": {"venue": p.get("venue", ""), "year": p.get("year", 0), "fuzzy": False}, "why": "exact normalised venue name"},
                {"action": "resolve_venue_events", "arguments": {"venue": p.get("venue", ""), "year": p.get("year", 0), "fuzzy": True}, "why": "token-based venue search when the name is written differently"},
                {"action": "search_chunks", "arguments": {"query": f"{p.get('venue','')} {p.get('date','')}", "k": 8, "scoped": False}, "why": "semantic search for the venue and date"})
        best = (st.date_match or {}).get("best", [])
        if st.events and st.date_match is None:
            R["R2"].status = "missing"
            gap("R2", {"action": "match_date", "arguments": {"date": p.get("date", "")}, "why": "filter the venue's events by the question date"})
        elif st.events:
            R["R2"].status = "satisfied" if len(best) == 1 else ("conflicting" if len(best) > 1 else "missing")
            R["R2"].note = f"{len(best)} match(es), {st.date_match.get('reason')}"
            if len(best) == 0:
                gap("R2", {"action": "resolve_venue_events", "arguments": {"venue": p.get("venue", ""), "year": p.get("year", 0), "fuzzy": True}, "why": "no event at this venue matches the date; widen the venue search"},
                    {"action": "search_chunks", "arguments": {"query": f"{p.get('venue','')} {p.get('date','')}", "k": 8, "scoped": False}, "why": "semantic search for the venue and date"})
            elif len(best) > 1:
                gap("R2", {"action": "disambiguate", "arguments": {"hint": "more than one event has this venue and date"}, "why": "report both medallists as an ambiguous answer, or use extra wording from the question"})
        targets = [e for e in st.events if any(b["event_id"] == e["event_id"] for b in best)]
        ok = bool(targets) and all(_has_fact(st, t["event_id"], "gold") for t in targets)
        R["R3"].status = "satisfied" if ok else "missing"
        for t in targets:
            if not _has_fact(st, t["event_id"], "gold"):
                gap("R3", {"action": "event_facts", "arguments": {"event_id": t["event_id"]}, "why": "fetch the gold medallist with its source span"})
    else:
        R["R1"].status = "satisfied" if st.chunks else "missing"
        if not st.chunks:
            gap("R1", {"action": "search_chunks", "arguments": {"query": st.question, "k": 8, "scoped": False}, "why": "retrieve relevant passages"})
            for name in (p.get("entities") or [])[:2]:
                gaps[-1]["suggested_actions"].append({"action": "resolve_entity", "arguments": {"name": name}, "why": "link the named entity to its document"})
        R["R2"].status = "satisfied" if st.verified else "missing"
        if st.chunks and not st.verified:
            gap("R2", {"action": "finish", "arguments": {}, "why": "generate and verify the answer from the passages"})
    return gaps
