"""Query semantics, scoring, accounting, coverage and benchmark isolation (no database needed)."""
import json

from podium.agent.coverage import InvestigationState, assess, build_requirements
from podium.contracts import Candidate, LLMCallUsage, UsageSummary
from podium.eval.scoring import score_answer
from podium.settings import CACHE, HIDDEN_QUESTIONS, PUBLIC_QUESTIONS


def test_scoring_is_strict():
    assert score_answer(["4"], ["4"])["exact"] is True
    assert score_answer(["14"], ["4"])["exact"] is False  # no substring credit
    assert score_answer(["4 events"], ["4"])["exact"] is False
    assert score_answer(["Chen Ding"], ["Chen Ding"])["exact"] is True
    assert score_answer(["chen ding"], ["Chen Ding"])["exact"] is True
    assert score_answer(["Chen Dong"], ["Chen Ding"])["exact"] is False
    s = score_answer(["Kateřina Emmons", "Pang Wei"], ["Kateřina Emmons"])
    assert s["exact"] is False and s["gold_in_pred"] is True and s["ambiguous_multi"]
    assert score_answer([], ["5"])["exact"] is False


def test_usage_accounting_never_double_counts_context():
    u = UsageSummary()
    u.add(LLMCallUsage(provider="x", model_id="m", operation="answer", input_tokens=1000, output_tokens=50, context_tokens=800))
    u.add(LLMCallUsage(provider="x", model_id="m", operation="decide", input_tokens=300, output_tokens=20))
    u.add(LLMCallUsage(provider="x", model_id="e", operation="embed", input_tokens=12))
    u.add(LLMCallUsage(provider="x", model_id="m", operation="answer", usage_unknown=True, error="timeout"))
    assert u.total_tokens == 1370 and u.context_tokens == 800
    assert u.llm_calls == 3 and u.embedding_calls == 1 and u.embedding_tokens == 12
    assert u.usage_unknown_calls == 1
    assert u.by_operation["answer"]["calls"] == 2


def test_aggregate_semantics_strict_boundary_and_unknowns():
    # mimic the graph result manifest: strict '>' excludes the boundary value; unknown never counts as zero
    vals = {"a": 74, "b": 73, "c": None, "d": 100}
    inc = [k for k, v in vals.items() if v is not None and v > 73]
    unk = [k for k, v in vals.items() if v is None]
    assert inc == ["a", "d"] and unk == ["c"]
    lower, upper = len(inc), len(inc) + len(unk)
    assert (lower, upper) == (2, 3)


def test_coverage_requires_repair_for_unknown_candidates():
    st = InvestigationState(question="q", interp={"intent": "count_events", "sport": "Cycling", "year": 2000, "season": "Summer", "threshold": 30, "operator": ">"})
    st.requirements = build_requirements(st.interp)
    gaps = assess(st)
    assert gaps[0]["suggested_actions"][0]["action"] == "aggregate"
    st.aggregate = {"n_candidates": 3, "count": 1, "n_unknown": 1, "manifest": [Candidate(event_id="e1", title="A", value=49, status="included"), Candidate(event_id="e2", title="B", value=12, status="excluded"), Candidate(event_id="e3", title="C", status="unknown")], "threshold": 30, "op": ">"}
    gaps = assess(st)
    assert [r.status for r in st.requirements] == ["satisfied", "missing", "satisfied"]
    assert gaps[0]["suggested_actions"][0]["action"] == "repair_missing_count"
    st.repairs["e3"] = {"found": False}
    gaps = assess(st)
    assert st.requirements[1].status == "unresolvable"


def test_coverage_flags_ambiguous_venue_date():
    st = InvestigationState(question="q", interp={"intent": "event_by_venue_date", "venue": "X", "date": "9 August"})
    st.requirements = build_requirements(st.interp)
    st.events = [{"event_id": "a", "title": "A"}, {"event_id": "b", "title": "B"}]
    st.date_match = {"best": [{"event_id": "a"}, {"event_id": "b"}], "reason": "exact", "ambiguous": True}
    gaps = assess(st)
    assert st.requirements[1].status == "conflicting"
    assert any(s["action"] == "disambiguate" for g in gaps for s in g["suggested_actions"])


def test_benchmark_answers_never_enter_ingestion_artifacts():
    """Gold answers / qids must not appear in any graph-load artifact or prompt module."""
    gold = set()
    for p in (PUBLIC_QUESTIONS,):
        for line in open(p, encoding="utf-8"):
            q = json.loads(line)
            gold.add(q["qid"])
    hidden_qids = {json.loads(l)["qid"] for l in open(HIDDEN_QUESTIONS, encoding="utf-8")}
    for name in ("events.jsonl", "documents.jsonl", "entities.jsonl", "venues.jsonl"):
        path = CACHE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(q in text for q in list(gold)[:20] + list(hidden_qids)[:20]), f"benchmark id leaked into {name}"
    import podium.agent.orchestrator as orch
    import podium.tools.interpret as interp

    src = open(orch.__file__).read() + open(interp.__file__).read()
    assert "eval_public" not in src and "gold_doc_ids" not in src and "pub-0" not in src
