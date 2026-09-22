# Podium — Evidence-Driven Agentic GraphRAG on TigerGraph

*Agentic GraphRAG Hackathon (TigerGraph), Round 1 submission.*

**Live dashboard:** https://tigeranalytics.duckdns.org  ·  **Architecture diagram:** [`Codes/docs/architecture.svg`](Codes/docs/architecture.svg)  ·  **Demo video:** see the submission form

Podium answers questions over a frozen corpus (2,951 Wikipedia-derived articles, mostly Olympic event pages) through three independently runnable pipelines on the *same* TigerGraph graph — **RAG**, **GraphRAG** and **Agentic GraphRAG** — and measures where the extra agentic work is worth its token cost. Its distinguishing feature is an **evidence coverage report**: every answer carries the explicit evidence requirements it had to satisfy, the complete candidate set it examined (with included / excluded / unknown status), exact source spans for every cited fact, the full action trace, and the reason the system stopped.

> Pitch: *Podium investigates until its evidence is sufficient — and shows when an agent was worth the extra work.*

---

## 1. Headline results (100 public questions, frozen run)

The tables below are produced by `scripts/render_results.py` from the frozen run artifacts in `Codes/artifacts/runs/<run_id>/` (predictions, per-question scores, summary and config snapshot are all committed).

<!-- RESULTS:BEGIN -->
Frozen run **`final-savanna-20260919`** on TigerGraph Savanna 4.2.5 (TG-00 workspace); answer model `gpt-4.1`, judge `gemini-2.5-flash`, embeddings `text-embedding-3-small`. Artifacts: `Codes/artifacts/runs/final-savanna-20260919/` (submission package in `submission/`).

### public set — run `final-savanna-20260919`

| Pipeline | n | Exact match | Judge correct | Judge completeness | Evidence support | Citations valid | Mean tokens | Mean context tok | LLM calls | p50 latency | p95 latency | Steps | Strategy changes | Abstained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RAG** | 100 | 51.0% | 57.0% | 0.7 | 0.8 | 100.0% | 2642.7 | 1579.4 | 1 | 1.6s | 2.3s | 2 | 0.0% | 17 |
| **GraphRAG** | 100 | 99.0% | 99.0% | 1.0 | 0.9 | 100.0% | 1750.7 | 223.1 | 2 | 1.9s | 2.4s | 4.5 | 0.0% | 0 |
| **Agentic GraphRAG** | 100 | 99.0% | 99.0% | 1.0 | 0.9 | 100.0% | 3605.0 | 286.8 | 4.0 | 4.5s | 7.0s | 6.6 | 3.0% | 0 |
| **Fixed-plan control** | 100 | 99.0% | 100.0% | 1.0 | 0.9 | 100.0% | 1753.6 | 225.9 | 2 | 1.9s | 2.4s | 4.5 | 0.0% | 0 |

**By question type** (exact match · mean tokens)

| Type | n | RAG | GraphRAG | Agentic GraphRAG | Fixed-plan control |
|---|---:|---:|---:|---:|---:|
| aggregation | 21 | 4.8% · 2633.0 | 100.0% · 1928.5 | 100.0% · 3160.8 | 100.0% · 1933.0 |
| lookup | 19 | 89.5% · 3087.1 | 100.0% · 1620.7 | 100.0% · 3341.6 | 100.0% · 1620.7 |
| multi_hop | 28 | 39.3% · 2466.9 | 96.4% · 1723.0 | 96.4% · 3748.7 | 96.4% · 1729.2 |
| superlative | 10 | 0.0% · 2487.4 | 100.0% · 1868.1 | 100.0% · 2861.2 | 100.0% · 1867.9 |
| temporal | 22 | 100.0% · 2562.4 | 100.0% · 1675.1 | 100.0% · 4411.9 | 100.0% · 1676.2 |

**Paired GraphRAG → Agentic GraphRAG** (n=100): quality lift 0.0 pp (bootstrap 95% CI [0.0, 0.0]), mean token delta 1854.3, extra tokens per extra correct answer: N/A. Agent wins: none; losses: none.

Stop reasons (agentic): sufficient_evidence 99, ambiguous_entity 1

### robustness set — run `final-savanna-20260919`

| Pipeline | n | Exact match | Judge correct | Judge completeness | Evidence support | Citations valid | Mean tokens | Mean context tok | LLM calls | p50 latency | p95 latency | Steps | Strategy changes | Abstained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RAG** | 22 | 50.0% | 63.6% | 0.7 | 0.7 | 100.0% | 2878.8 | 1842.3 | 1 | 1.6s | 2.3s | 2 | 0.0% | 3 |
| **GraphRAG** | 22 | 77.3% | 81.8% | 0.8 | 0.7 | 100.0% | 2809 | 1231.9 | 2 | 2.3s | 2.7s | 4.7 | 0.0% | 3 |
| **Agentic GraphRAG** | 22 | 100.0% | 100.0% | 1.0 | 0.8 | 100.0% | 4995.3 | 891.6 | 5 | 5.3s | 7.9s | 8.4 | 36.4% | 3 |
| **Fixed-plan control** | 22 | 86.4% | 86.4% | 0.9 | 0.8 | 100.0% | 2488.9 | 950.4 | 2 | 2.1s | 3.2s | 4.8 | 9.1% | 4 |

**By question type** (exact match · mean tokens)

| Type | n | RAG | GraphRAG | Agentic GraphRAG | Fixed-plan control |
|---|---:|---:|---:|---:|---:|
| ambiguous | 2 | 0.0% · 2469 | 100.0% · 1944 | 100.0% · 5592 | 100.0% · 1784.5 |
| lookup_medal | 2 | 100.0% · 2354 | 100.0% · 1634 | 100.0% · 3361 | 100.0% · 1636.5 |
| missing_field | 2 | 0.0% · 3405.5 | 0.0% · 1896.5 | 100.0% · 9567 | 0.0% · 1907.5 |
| open_domain | 5 | 100.0% · 3332.2 | 100.0% · 5194 | 100.0% · 4549.4 | 100.0% · 4173.4 |
| paraphrase | 5 | 40.0% · 2922.8 | 100.0% · 1728.4 | 100.0% · 3772.6 | 100.0% · 1732.8 |
| unanswerable | 3 | 33.3% · 2985 | 33.3% · 3358.3 | 100.0% · 6419.3 | 100.0% · 2232.7 |
| venue_variant | 3 | 33.3% · 2215.7 | 66.7% · 2054 | 100.0% · 3996 | 66.7% · 2623.3 |

**Paired GraphRAG → Agentic GraphRAG** (n=22): quality lift 22.7 pp (bootstrap 95% CI [4.5, 40.9]), mean token delta 2186.3, extra tokens per extra correct answer: 9619.6. Agent wins: rob-006, rob-007, rob-011, rob-013, rob-015; losses: none.

Stop reasons (agentic): sufficient_evidence 17, no_new_evidence 2, ambiguous_entity 2, budget_exhausted 1

### hidden set — run `final-savanna-20260919`

| Pipeline | n | Answered | Partial | Abstained | Errors | Mean tokens | Mean context tok | LLM calls | p50 latency | p95 latency | Steps | Strategy changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RAG** | 50 | 46 | 0 | 4 | 0 | 2551.9 | 1506.1 | 1 | 1.5s | 2.5s | 2 | 0.0% |
| **GraphRAG** | 50 | 48 | 2 | 0 | 0 | 1776.0 | 256.9 | 2 | 1.8s | 2.2s | 4.4 | 0.0% |
| **Agentic GraphRAG** | 50 | 48 | 2 | 0 | 0 | 3333.0 | 271.6 | 3.7 | 4.2s | 6.1s | 6.2 | 2.0% |

**By question type** (mean tokens · answered)

| Type | n | RAG | GraphRAG | Agentic GraphRAG |
|---|---:|---:|---:|---:|
| aggregation | 15 | 2510.2 · 15 | 1898.5 · 14 | 2829.9 · 14 |
| lookup | 7 | 3062.9 · 7 | 1607.9 · 7 | 3312.3 · 7 |
| multi_hop | 10 | 2391.9 · 7 | 1757.2 · 9 | 3991 · 9 |
| superlative | 10 | 2657.8 · 9 | 1809.9 · 10 | 2589.5 · 10 |
| temporal | 8 | 2250.6 · 8 | 1674.4 · 8 | 4401 · 8 |
<!-- RESULTS:END -->

**What the numbers say**

* **RAG** (top-8 similarity search over chunk vectors stored in TigerGraph) answers lookups and temporal questions from the infobox chunk, but cannot enumerate a complete candidate set: it fails almost every aggregation and superlative question and most venue/date multi-hop questions (51% overall).
* **GraphRAG** (one interpretation, one fixed template of installed GSQL queries) is a strong baseline: full-cohort aggregates and edge traversals make it correct on 99% of the public set at the *lowest* token cost (≈1,750 tokens/answer). The single miss is a genuinely ambiguous question (two events share the venue *and* the date string; both medallists are listed).
* **Agentic GraphRAG** reaches the same 99% on the public set at ≈2× the tokens. On this benchmark the paired lift is **0.0 pp** — we report that honestly: when the graph already has complete coverage, the agent stops after one graph query, and the *fixed-plan control* (same planner, no replanning) matches it. The public set simply contains almost no evidence gaps.
* The **robustness set** (22 labelled questions kept separate from the official numbers) is where adaptation pays: **Agentic 100% vs GraphRAG 77% vs fixed-plan 86% vs RAG 50%**, with strategy changes on 36% of questions — missing infobox values recovered from prose with verbatim, character-checked quotes; venue spellings that differ from the corpus; correct abstention on unanswerable questions; entity questions outside the Olympic domain.
* Hidden set: 150 raw predictions with tokens and traces (no accuracy claim); GraphRAG and the agent agree on 50/50 questions, two of which are reported as bounded/ambiguous.
* A second complete frozen run on TigerGraph Community Edition with `claude-sonnet-5` as the answer model (`final-ce-20260919`) gives the same picture (public 47 / 99 / 99 %, robustness 59 / 86 / 100 %), so the result is not tied to one model.

---

## 2. What was built

```
question ─┬─► RAG            : embed → TigerGraph vectorSearch(top-k) → grounded answer
          ├─► GraphRAG       : interpret (LLM, once) → fixed GSQL template → grounded answer
          └─► Agentic GraphRAG:
                interpret → evidence requirements ledger
                loop: coverage check (deterministic) → orchestrator picks next specialist action (LLM)
                      → tool runs on TigerGraph → ledger updated → verify → stop when sufficient / budget
```

### 2.1 Graph model on TigerGraph Savanna (v4.2)

| Vertex | Key attributes |
|---|---|
| `Document` | title, url, infobox type, content hash |
| `Chunk` | text, section, original-text char offsets, **`embedding` VECTOR(1536, COSINE)** |
| `Event` | sport, edition year/season, event name (normalised), venue, date (raw + normalised), competitors / nations (value, raw string, unit, parse status, `has_*` flags), medallists + NOCs, prev/next |
| `Edition`, `Sport`, `Venue`, `Entity` (medallists, NOCs, films, people…), `Fact` (predicate, raw value, chunk id + char offsets) | |

Edges: `HAS_CHUNK`, `DESCRIBES_EVENT`, `DESCRIBES_ENTITY`, `IN_EDITION`, `IN_SPORT`, `AT_VENUE`, `HAS_FACT`, `SUPPORTED_BY` (fact → exact chunk span), `WON_MEDAL{medal,noc}`, `REPRESENTS`, `PREVIOUS_EVENT{source}` (explicit infobox `prev` or derived from corpus edition order — the derivation is recorded), `PREVIOUS_EDITION`.

**17 installed, parameterised GSQL queries** do the retrieval and reasoning (`Codes/podium/db/gsql/queries.gsql`): `search_chunks` / `search_chunks_scoped` (vectorSearch, optionally restricted to a sport/edition cohort), `find_events`, `events_at_venue`, `venues_like`, `events_like`, `aggregate_competitors` (count over the complete cohort with included/excluded/unknown sets), `max_competitors` (ties preserved), `previous_event`, `event_facts` (facts + supporting chunks), `cohort_facts` (span-backed facts for a whole cohort in one query), `event_subgraph`, `next_event`, `resolve_entity`, `doc_chunks`, `get_chunk`, `graph_stats`. The planner chooses tools and arguments; it never generates database code. A missing number is never treated as zero: aggregates return bounds when candidates are unknown.

### 2.2 Ingestion (deterministic, versioned)

`podium/corpus/parse.py` parses titles (`<Sport> at the <Year> <Season> Olympics – <Event>`), every infobox (with original-text offsets for every field), counts with units (`23 teams`, `32 (16 pairs)`, `41000000` → *suspicious*), dates in all corpus forms (`6 to 8 August`, `August 12, 2008`, `22 September 2000 (heats)25 September 2000 (final)`, `2008-08-10`…), venues (normalised keys that unify `TechnologyUniversity`/`Technology University`) and concatenated team names (split for display only). `podium/corpus/chunking.py` produces 17,904 structure-aware chunks (infobox intact, headings as context, tables with their header row, exact char offsets). Everything is reconciled against an ingestion manifest (`make verify-graph`), and the graph snapshot id is stamped on every prediction.

### 2.3 Agent harness (`podium/agent/`)

* **Requirements ledger** (`coverage.py`): each intent has explicit evidence requirements, e.g. *count*: cohort enumerated → predicate known for every candidate → count computed in the graph; *venue/date*: venue resolved → exactly one event matches venue and date → medallist supported by a source span. A deterministic `assess()` updates statuses and proposes candidate actions for each gap.
* **Orchestrator** (`orchestrator.py`): an LLM chooses ONE next action from the question, the evidence so far, the requirement statuses and the gaps (`resolve_event{exact|substring|semantic}`, `resolve_venue_events{fuzzy}`, `match_date`, `aggregate`, `superlative`, `previous_event`, `event_facts`, `repair_missing_count`, `search_chunks{scoped}`, `fetch_document`, `resolve_entity`, `disambiguate`, `reinterpret`, `finish`, `abstain`). Repeated no-progress actions are blocked; budgets: 8 decisions, 12 tool calls, 40k tokens, 90 s.
* **Specialists** (recorded with `execution_kind`): entity_linker, graph_traverser, similarity_search, document_retriever, aggregator, multi_hop_reasoner, evidence_evaluator (quote-verified prose extraction, value verification, disambiguation), answer_generator. Deterministic function calls are never dressed up as model reasoning.
* **Stop reasons**: `sufficient_evidence`, `ambiguous_entity` (all candidates reported), `incomplete_scope` (bounds reported), `no_new_evidence`, `budget_exhausted`, `no_evidence`.
* **Controls**: `fixed_plan` (same planner/tools, replanning disabled) and `agentic_nocov` (no ledger shown to the orchestrator).

### 2.4 Evidence and explainability

Every `AnswerResult` has: compact machine answer(s), grounded explanation, status, citations (doc, chunk, **char offsets**, verbatim quote), coverage report (requirements, candidate manifest, bounds, conflicts), trace (steps with tool, kind, args, latency, tokens, strategy-change flag, coverage before/after), usage (context / input / output / total, embeddings separate), latency and stop reason. The dashboard's *Inspect* screen shows the requirement ledger, the sortable candidate table, the action timeline, the graph path traversed, and a source viewer that highlights the exact cited span in the original document.

---

## 3. Findings worth knowing (from the corpus, not from us)

* **Corpus-native evidence repair.** *Cycling at the 2000 Summer Olympics*: 14 events, 4 without an infobox (women's/men's individual pursuit, men's keirin, women's points race). The agent recovers 12, 20, 17 and 17 riders from prose with verbatim quotes. For the official question (>30) the count stays 4; for a >15 threshold the infobox-only answer (9, bounds 9–13) is wrong and the repaired answer is 12.
* **Conflicting facts inside the corpus.** *Fencing at the 2008 Summer Olympics – Men's épée* has `competitors: 41000000` in its infobox while its prose says *"There were 41 competitors from 23 nations."* The infobox is the primary structured source (the corpus is the truth), so the machine answer follows it; the agent verifies the value against the prose and reports the conflict with the exact span. This is the Round-2 problem appearing in Round-1 data.
* **Genuine ambiguity.** Two events share venue *and* date string (e.g. Laura Biathlon & Ski Complex, 22 February 2014: men's biathlon relay and women's 30 km cross-country). Podium lists both medallists and stops with `ambiguous_entity` instead of guessing.
* **Labels ≠ difficulty.** All 28 public *multi_hop* questions resolve to one document; all 22 *temporal* questions to two. We report the supplied labels and, separately, measured retrieval steps and documents.

---

## 4. Evaluation methodology

* **Inference boundary.** A pipeline receives question text only. `qid`, `qtype`, gold answers and gold document ids are attached by the runner after inference; gold answers are read only by the offline scorer (`tests/test_semantics.py` checks that benchmark ids never enter graph-load artifacts or prompts).
* **Exact / numeric match** with a versioned, conservative normalisation (unicode, whitespace, dashes, case, numeric formatting). No substring credit: "14" ≠ "4". Multi-candidate answers count as wrong for exact match and are reported separately (`gold_in_pred`).
* **LLM judge** (`gpt-4.1`, a different model family from the answer model, blind to pipeline labels): correctness, completeness (0–1) and evidence support (0–1) against the reference and the cited spans; cached and reported with the judged denominator.
* **Citation validity**: every cited span is re-checked against the frozen corpus text. **Gold-document recall** distinguishes examined vs cited documents.
* **Tokens**: `context_tokens` (evidence + graph result transmitted to the model) ⊂ `input_tokens`; `total = input + output`; embedding tokens separate; failed calls recorded as `usage_unknown`, never as zero. Offline ingestion cost (≈5.4M embedding tokens once) is kept out of the online comparison.
* **Paired comparison** GraphRAG → Agentic with bootstrap CI; per-type tables; fixed-plan ablation; failure explorer.
* **Hidden set**: 50 × 3 raw predictions with tokens and traces, exported unscored.

---

## 5. Reproduce

```bash
cd Codes
make setup                      # Python 3.12 venv + dependencies
cp .env.example .env            # add TG_HOST + TG_SECRET (Savanna database secret) and an LLM key
make audit-data                 # checksums / counts of the supplied dataset
make ingest                     # parse + chunk → artifacts/cache/*.jsonl + ingestion manifest
make embed                      # chunk embeddings (cached, resumable)
make load                       # schema, vertices/edges/vectors, install GSQL queries on TigerGraph
make verify-graph               # graph counts reconciled with the manifest
make test                       # unit tests (no DB / LLM)
make benchmark-public RUN=my-run && make evaluate-public RUN=my-run
make benchmark-hidden RUN=my-run
make benchmark-robustness RUN=my-run && make evaluate-robustness RUN=my-run
make export-submission RUN=my-run
make dashboard                  # http://127.0.0.1:8120
```

TigerGraph Community Edition works as a drop-in (`scripts/env_local.sh`), which is how the pipelines were developed before the frozen run on Savanna.

Models (frozen, `configs/models.yaml`): answer/planning `gpt-4.1` (structured JSON output, one model for all pipelines), judge `gemini-2.5-flash` (different model family), embeddings `text-embedding-3-small` (1536). Development and the Community-Edition frozen run (`final-ce-20260919`) used `claude-sonnet-5` with `gpt-4.1` as judge; the Anthropic account ran out of credits before the Savanna run, so the headline run uses `gpt-4.1`. Both runs are committed, so the effect of the model swap is visible (it is small: the graph tools carry the factual load). Any OpenAI-compatible or Anthropic model can be substituted via `PODIUM_ANSWER_MODEL`.

---

## 6. Repository layout

```
README.md                      this file
Codes/
  podium/
    corpus/      parse.py, chunking.py            deterministic parsing + structure-aware chunks
    ingest/      build.py, embed.py, load_tg.py   artifacts, embeddings, TigerGraph load
    db/          tg.py, gsql/schema.gsql, gsql/queries.gsql
    tools/       toolbox.py, interpret.py         typed tools over the graph + question interpretation
    pipelines/   rag.py, graphrag.py, answer.py
    agent/       coverage.py, orchestrator.py     requirements ledger + orchestrator + controls
    eval/        runner.py, scoring.py, judge.py, reports.py
    api.py       FastAPI: live three-way inference, replay, artifacts, dashboard
    contracts.py, settings.py
  web/index.html               dashboard (Compare · Inspect · Benchmark · Architecture)
  scripts/                     audit_data, benchmark, evaluate, export_submission, render_results, doctor
  configs/                     models.yaml, robustness_questions.jsonl
  tests/                       parsing, query semantics, accounting, coverage, benchmark isolation
  artifacts/manifests/         ingestion manifest + graph verification
  artifacts/runs/<run_id>/     predictions_{public,hidden,robustness}.jsonl, scores, summary, config, submission/
  docs/architecture.svg
  Makefile, pyproject.toml, requirements.txt, .env.example, deploy/
```

---

## 7. Limitations

* The corpus is a set of article snapshots; a count is "complete" only relative to the corpus, the ingestion checks and the query scope. Podium says "within the supplied corpus" and returns bounds when values are unknown.
* Public multi-hop/temporal questions are single- or two-document; the agent's advantage on the official set is small by construction and is shown honestly (paired lift with CI). Its value shows on the robustness set and on the corpus's own missing/corrupt values.
* Team medallists are kept as the raw (sometimes concatenated) source string; splitting is display-only.
* Prose extraction recovers only values stated in the text with a verbatim quote; it does not count table rows.
* One answer model was used for all pipelines; a stronger/cheaper model sweep was not run.

---

## 8. Artifacts and attribution

* Frozen run id and the submission package: `Codes/artifacts/runs/<run_id>/submission/` (`final-savanna-20260919`).
* Corpus text is derived from English Wikipedia (CC BY-SA 4.0); every citation carries the source URL. The TigerGraph GraphRAG starter repository was used as a reference for engine concepts; all code here is original. Third-party libraries: pyTigerGraph/httpx, Anthropic and OpenAI SDKs, FastAPI, Chart.js, vis-network.
