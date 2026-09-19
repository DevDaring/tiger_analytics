"""Typed contracts shared by all three pipelines, the agent harness, the benchmark runner
and the dashboard (plan section 10).

Inference receives *question text only*. `qid`, `qtype`, gold answers and gold document
IDs are attached by the benchmark runner outside of these contracts.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Pipeline = Literal["rag", "graphrag", "agentic_graphrag", "fixed_plan", "agentic_nocov"]
Status = Literal["answered", "partial", "abstained", "error"]
StopReason = Literal[
    "sufficient_evidence",
    "ambiguous_entity",
    "incomplete_scope",
    "unresolved_conflict",
    "no_new_evidence",
    "budget_exhausted",
    "tool_unavailable",
    "no_evidence",
    "single_pass",
    "error",
]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))[:16]


# --------------------------------------------------------------------------------------
# Usage accounting (plan section 10, "Token accounting")
# --------------------------------------------------------------------------------------
class LLMCallUsage(BaseModel):
    call_id: str = Field(default_factory=lambda: new_id("call"))
    provider: str
    model_id: str
    operation: str  # interpret | plan | decide | answer | extract | verify | judge | embed
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    context_tokens: int = 0  # evidence tokens *inside* input_tokens (diagnostic subset)
    latency_ms: float = 0.0
    usage_unknown: bool = False
    error: str | None = None
    request_id: str | None = None

    @property
    def total(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class UsageSummary(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    context_tokens: int = 0  # subset of input_tokens: evidence text transmitted to the model
    total_tokens: int = 0  # input + output (context is NOT added again)
    embedding_calls: int = 0
    embedding_tokens: int = 0
    usage_unknown_calls: int = 0
    by_operation: dict[str, dict[str, int]] = Field(default_factory=dict)
    calls: list[LLMCallUsage] = Field(default_factory=list)

    def add(self, u: LLMCallUsage) -> None:
        self.calls.append(u)
        if u.operation == "embed":
            self.embedding_calls += 1
            self.embedding_tokens += u.input_tokens or 0
            return
        self.llm_calls += 1
        if u.usage_unknown:
            self.usage_unknown_calls += 1
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cached_input_tokens += u.cached_input_tokens or 0
        self.reasoning_tokens += u.reasoning_tokens or 0
        self.context_tokens += u.context_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        op = self.by_operation.setdefault(u.operation, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        op["calls"] += 1
        op["input_tokens"] += u.input_tokens or 0
        op["output_tokens"] += u.output_tokens or 0


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------
class Citation(BaseModel):
    doc_id: str
    chunk_id: str | None = None
    title: str = ""
    url: str = ""
    char_start: int | None = None  # original-text character offsets
    char_end: int | None = None
    quote: str = ""  # exact source span (trimmed for display)
    claim_ids: list[str] = Field(default_factory=list)
    source: str = "graph"  # graph | vector | document | fact


class Requirement(BaseModel):
    """One evidence requirement the answer must satisfy (coverage-directed investigation)."""

    req_id: str
    description: str
    status: Literal["missing", "satisfied", "conflicting", "unresolvable"] = "missing"
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class Candidate(BaseModel):
    """One member of a candidate set (aggregation / superlative / disambiguation)."""

    event_id: str
    title: str
    value: float | None = None
    value_raw: str | None = None
    unit: str | None = None
    status: Literal["included", "excluded", "unknown", "ambiguous"] = "unknown"
    reason: str = ""
    source: str = "infobox"  # infobox | prose_extraction | none
    chunk_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class CoverageReport(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    candidate_scope: str = ""
    candidate_count: int = 0
    known_values: int = 0
    unknown_values: int = 0
    candidates: list[Candidate] = Field(default_factory=list)
    examined_doc_ids: list[str] = Field(default_factory=list)
    graph_snapshot_id: str = ""
    complete: bool = False
    bounds: dict[str, int] | None = None  # {"lower":..,"upper":..} when unknowns remain
    conflicts: list[dict[str, Any]] = Field(default_factory=list)  # competing assertions with provenance
    notes: list[str] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return all(r.status == "satisfied" for r in self.requirements) and bool(self.requirements)

    def missing(self) -> list[Requirement]:
        return [r for r in self.requirements if r.status != "satisfied"]


# --------------------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------------------
class TraceEvent(BaseModel):
    step: int
    ts: float = Field(default_factory=time.time)
    specialist: str  # orchestrator | entity_linker | graph_traverser | similarity_search | ...
    tool: str
    execution_kind: Literal["deterministic", "llm", "graph_query", "vector_search"] = "deterministic"
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    result_summary: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    n_results: int = 0
    coverage_before: int = 0  # number of missing requirements before this step
    coverage_after: int = 0
    latency_ms: float = 0.0
    tokens: int = 0
    error: str | None = None
    strategy_change: bool = False


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    pipeline: str = ""
    events: list[TraceEvent] = Field(default_factory=list)
    retrieval_methods: list[str] = Field(default_factory=list)
    specialists_invoked: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    n_retrieval_steps: int = 0
    n_reasoning_steps: int = 0
    n_llm_decisions: int = 0
    strategy_changed: bool = False
    strategy_changes: int = 0
    n_chunks: int = 0
    n_citations: int = 0
    stop_reason: str = ""
    stop_explanation: str = ""

    def record(self, ev: TraceEvent) -> None:
        self.events.append(ev)
        if ev.tool not in self.tools_called:
            self.tools_called.append(ev.tool)
        if ev.specialist not in self.specialists_invoked:
            self.specialists_invoked.append(ev.specialist)
        if ev.execution_kind in ("graph_query", "vector_search"):
            self.n_retrieval_steps += 1
            if ev.execution_kind not in self.retrieval_methods:
                self.retrieval_methods.append(ev.execution_kind)
        elif ev.tool != "decide":
            self.n_reasoning_steps += 1
        if ev.tool == "decide":
            self.n_llm_decisions += 1
        if ev.strategy_change:
            self.strategy_changed = True
            self.strategy_changes += 1


# --------------------------------------------------------------------------------------
# Answer
# --------------------------------------------------------------------------------------
class AnswerResult(BaseModel):
    answer: list[str] = Field(default_factory=list)
    explanation: str = ""
    status: Status = "answered"
    citations: list[Citation] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    trace: Trace = Field(default_factory=Trace)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    latency_ms: float = 0.0
    stop_reason: str = "single_pass"
    interpretation: dict[str, Any] = Field(default_factory=dict)
    graph_path: list[dict[str, Any]] = Field(default_factory=list)  # nodes/edges used by the answer
    evidence_passages: list[dict[str, Any]] = Field(default_factory=list)  # what the model actually saw
    error: str | None = None


class PredictionRecord(BaseModel):
    """Benchmark envelope (plan section 10). Identifiers are attached *after* inference."""

    run_id: str
    qid: str
    pipeline: str
    question: str
    qtype: str | None = None
    result: AnswerResult
    model_id: str
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    prompt_hash: str = ""
    corpus_hash: str = ""
    graph_snapshot_id: str = ""
    config_hash: str = ""
    code_commit: str = ""
    recorded_at: str = ""
