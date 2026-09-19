"""Pipeline A: RAG. Fixed similarity retrieval from TigerGraph chunk vectors + one grounded
answer generation step. No graph traversal, no full-set aggregation, no second retrieval.
"""
from __future__ import annotations

from ..contracts import AnswerResult, CoverageReport, Requirement
from ..settings import settings
from ..tools.toolbox import RunContext, Toolbox
from .answer import build_citations, clean_answer, finalize, generate_answer


def run(question: str, top_k: int | None = None) -> AnswerResult:
    ctx = RunContext(question, "rag")
    tb = Toolbox(ctx)
    res = AnswerResult(stop_reason="single_pass")
    try:
        chunks = tb.search_chunks(question, k=top_k or settings.rag_top_k, rationale="fixed top-k similarity search")
        data, seen, _ = generate_answer(ctx, tb, question, chunks)
        res.answer = clean_answer(data.get("answer", []))
        res.explanation = data.get("explanation", "")
        res.citations = build_citations(ctx, tb, data.get("cited_chunk_ids", []), chunks)
        res.evidence_passages = seen
        supported = bool(data.get("supported")) and not data.get("abstain")
        res.coverage = CoverageReport(
            requirements=[Requirement(req_id="R1", description="answer supported by a retrieved passage", status="satisfied" if supported else "missing", evidence_ids=[c.chunk_id for c in res.citations if c.chunk_id])],
            candidate_scope="top-k retrieved chunks (not a complete set)", candidate_count=len(chunks), known_values=len(chunks), complete=False,
            notes=["RAG cannot enumerate a complete candidate set; counts and superlatives are unverified."],
        )
        if data.get("abstain") or not res.answer:
            res.status = "abstained"
            res.stop_reason = "no_evidence"
            res.explanation = res.explanation or data.get("abstain_reason", "")
        elif not supported:
            res.status = "partial"
    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.stop_reason = "error"
        res.error = f"{type(e).__name__}: {str(e)[:300]}"
    return finalize(res, ctx)
