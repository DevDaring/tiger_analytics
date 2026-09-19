"""Grounded answer generation shared by all pipelines.

The model sees only corpus passages (and, when available, the graph computation result).
Machine answers for graph-computed results are enforced deterministically; the model writes
the explanation and may abstain.
"""
from __future__ import annotations

import re
from typing import Any

from ..contracts import AnswerResult, Citation
from ..llm.client import complete_json
from ..settings import settings
from ..tools.toolbox import RunContext, Toolbox

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "array", "items": {"type": "string"}, "description": "compact machine answer(s): a number, a name exactly as written in the source, or an event title; empty if abstaining"},
        "explanation": {"type": "string", "description": "2-4 sentences grounded in the passages, mention the source titles"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "supported": {"type": "boolean", "description": "true only if the passages/graph result directly support the answer"},
        "abstain": {"type": "boolean"},
        "abstain_reason": {"type": "string"},
    },
}

SYSTEM = """You answer questions strictly from the supplied corpus passages (frozen Wikipedia-derived text) and,
when present, a GRAPH RESULT computed by the database over the complete corpus. Rules:
- The corpus is the only source of truth. Never use outside knowledge or memory. If the passages do not
  contain the answer, set abstain=true and explain what is missing.
- Copy names and values exactly as written in the source (keep the source's spelling; team names may appear
  concatenated - copy them as they appear, and you may list the members in the explanation).
- For numbers answer with the bare number. For 'which event' answer with the full event title.
- If a GRAPH RESULT is provided it was computed over every matching document; use its value as the answer
  and explain it using the candidate list. Report unknown candidates honestly.
- If two different events both satisfy the question, list both answers and say so.
- The infobox field is the primary structured source. If the GRAPH RESULT marks a value as implausible or
  conflicting with the prose, keep the graph result as the answer and state the conflict plainly in the
  explanation (what the infobox says, what the prose says). Never silently exclude a candidate.
- cited_chunk_ids must only contain chunk ids from the passages you actually relied on."""


def pack_passages(ctx: RunContext, chunks: list[dict], budget: int | None = None) -> tuple[str, list[dict], int]:
    budget = budget or settings.context_char_budget
    out, used, seen = [], 0, []
    for c in chunks:
        if used >= budget:
            break
        t = c.get("text", "")[: budget - used]
        head = f"[chunk {c['chunk_id']}] {c.get('title','')}" + (f" > {c['section']}" if c.get("section") else "")
        out.append(f"{head}\n{t}")
        used += len(t)
        seen.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c.get("title", ""), "section": c.get("section", ""), "chars": len(t)})
        ctx.llm_seen_chunks.add(c["chunk_id"])
    return "\n\n".join(out), seen, used // 4


def generate_answer(ctx: RunContext, tb: Toolbox, question: str, chunks: list[dict], graph_result: dict | None = None, extra_instructions: str = "") -> tuple[dict, list[dict], int]:
    passages, seen, ctx_tokens = pack_passages(ctx, chunks)
    parts = [f"Question: {question}"]
    if graph_result:
        gtxt = _fmt_graph_result(graph_result)
        parts.append("GRAPH RESULT (computed by TigerGraph over the complete corpus-defined candidate set):\n" + gtxt)
        ctx_tokens += len(gtxt) // 4  # graph evidence transmitted to the model counts as context
    if extra_instructions:
        parts.append(extra_instructions)
    parts.append("Passages:\n" + (passages or "(no passages retrieved)"))
    data, u = complete_json("answer", SYSTEM, "\n\n".join(parts), ANSWER_SCHEMA, max_tokens=900, context_tokens=ctx_tokens, effort="low")
    ctx.add_usage(u)
    tb._record("answer_generator", "generate_answer", "llm", {"n_passages": len(seen), "graph_result": bool(graph_result)}, ("abstain" if data.get("abstain") else "answer=" + " | ".join(data.get("answer", [])))[:200], len(data.get("answer", [])), ctx.t0 + ctx.elapsed() - u.latency_ms / 1000, data.get("cited_chunk_ids", []), tokens=u.total)
    return data, seen, ctx_tokens


def _fmt_graph_result(g: dict) -> str:
    lines = []
    for k, v in g.items():
        if k == "manifest":
            lines.append("candidates (event | competitors | status | reason):")
            for c in v[:60]:
                lines.append(f"  - {c['title']} | {c.get('value_raw') or 'unknown'} | {c['status']} | {c.get('reason','')}")
        elif isinstance(v, (list, dict)):
            lines.append(f"{k}: {str(v)[:1500]}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def build_citations(ctx: RunContext, tb: Toolbox, cited_ids: list[str], fallback_chunks: list[dict], claim_id: str = "c1") -> list[Citation]:
    cits: list[Citation] = []
    valid = [cid for cid in cited_ids if cid in ctx.chunks]
    for cid in valid[:8]:
        cits.append(tb.citation_for_chunk(ctx.chunks[cid], claim_id))
    if not cits:
        for c in fallback_chunks[:3]:
            cits.append(tb.citation_for_chunk(c, claim_id))
    return cits


def finalize(result: AnswerResult, ctx: RunContext) -> AnswerResult:
    result.trace = ctx.trace
    result.usage = ctx.usage
    result.latency_ms = ctx.elapsed() * 1000
    result.trace.n_chunks = len(ctx.llm_seen_chunks)
    result.trace.n_citations = len(result.citations)
    result.trace.stop_reason = result.stop_reason
    result.trace.strategy_changes = sum(1 for e in result.trace.events if e.strategy_change)
    result.trace.strategy_changed = result.trace.strategy_changes > 0
    result.graph_path = ctx.graph_path[:200]
    result.coverage.examined_doc_ids = sorted({c["doc_id"] for c in ctx.chunks.values()} | set(ctx.events.keys()))[:500]
    return result


def clean_answer(items: list[str]) -> list[str]:
    out = []
    for a in items or []:
        a = re.sub(r"\s+", " ", str(a)).strip()
        if a and a not in out:
            out.append(a)
    return out
