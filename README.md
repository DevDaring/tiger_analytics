# Podium — Evidence-Driven Agentic GraphRAG on TigerGraph

*Agentic GraphRAG Hackathon (TigerGraph), Round 1 submission.*

**Live dashboard:** https://tigeranalytics.duckdns.org  ·  **Architecture diagram:** [`Codes/docs/architecture.svg`](Codes/docs/architecture.svg)  ·  **Demo video:** see the submission form

Podium answers questions over a frozen corpus (2,951 Wikipedia-derived articles, mostly Olympic event pages) through three independently runnable pipelines on the *same* TigerGraph graph — **RAG**, **GraphRAG** and **Agentic GraphRAG** — and measures where the extra agentic work is worth its token cost. Its distinguishing feature is an **evidence coverage report**: every answer carries the explicit evidence requirements it had to satisfy, the complete candidate set it examined (with included / excluded / unknown status), exact source spans for every cited fact, the full action trace, and the reason the system stopped.

> Pitch: *Podium investigates until its evidence is sufficient — and shows when an agent was worth the extra work.*

---

## 1. Headline results (100 public questions, frozen run)

The tables below are produced by `scripts/render_results.py` from the frozen run artifacts in `Codes/artifacts/runs/<run_id>/` (predictions, per-question scores, summary and config snapshot are all committed).

<!-- RESULTS:BEGIN -->
_The final frozen results table is inserted here by `scripts/render_results.py` (see section 8 for the run id)._
<!-- RESULTS:END -->

**What the numbers say**

* **RAG** (top-8 similarity search over chunk vectors stored in TigerGraph) answers lookups and most temporal questions from the infobox chunk, but cannot enumerate a complete candidate set: it fails almost every aggregation and superlative question and many venue/date multi-hop questions.
* **GraphRAG** (one interpretation, one fixed template of installed GSQL queries) is a strong baseline: full-cohort aggregates and edge traversals make it correct on ~96% of the public set at the *lowest* token cost of the three.
* **Agentic GraphRAG** uses the same tools but decides its next action from the evidence gaps. On the public set it recovers the cases GraphRAG cannot: missing infobox values (recovered from prose with a verbatim, character-checked quote), implausible infobox values (verified against the prose and reported as an explicit conflict), unresolved event names (substring / semantic fallback), and venue spellings that differ from the corpus. It costs roughly 2× the tokens of GraphRAG. Where GraphRAG already has complete coverage, the agent stops after one graph query — the *fixed-plan control* shows that the gain comes from adaptation, not from a different planner.
* The supplemental **robustness set** (22 labelled questions: paraphrases, missing fields, ambiguity, venue variants, unanswerable, open-domain) is where adaptation matters most; it is reported separately and never mixed with the official numbers.

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

**16 installed, parameterised GSQL queries** do the retrieval and reasoning (`Codes/podium/db/gsql/queries.gsql`): `search_chunks` / `search_chunks_scoped` (vectorSearch, optionally restricted to a sport/edition cohort), `find_events`, `events_at_venue`, `venues_like`, `events_like`, `aggregate_competitors` (count over the complete cohort with included/excluded/unknown sets), `max_competitors` (ties preserved), `previous_event`, `event_facts` (facts + supporting chunks), `event_subgraph`, `resolve_entity`, `doc_chunks`, `get_chunk`, `graph_stats`. The planner chooses tools and arguments; it never generates database code. A missing number is never treated as zero: aggregates return bounds when candidates are unknown.

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

Models (frozen, `configs/models.yaml`): answer/planning `claude-sonnet-5` (effort=low, structured JSON), judge `gpt-4.1`, embeddings `text-embedding-3-small` (1536). Any OpenAI-compatible or Anthropic model can be substituted via `PODIUM_ANSWER_MODEL`.

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

* Frozen run id and the submission package: `Codes/artifacts/runs/<run_id>/submission/` (see the results section above for the id).
* Corpus text is derived from English Wikipedia (CC BY-SA 4.0); every citation carries the source URL. The TigerGraph GraphRAG starter repository was used as a reference for engine concepts; all code here is original. Third-party libraries: pyTigerGraph/httpx, Anthropic and OpenAI SDKs, FastAPI, Chart.js, vis-network.
