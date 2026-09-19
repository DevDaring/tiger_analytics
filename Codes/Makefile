# Podium — operational commands (plan section 15). Every command is safe to re-run.
PY=.venv/bin/python
RUN?=run-$(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: setup doctor audit-data ingest embed load verify-graph test benchmark-public evaluate-public benchmark-hidden benchmark-robustness evaluate-robustness ablations dashboard export-submission

setup:            ## create the virtualenv and install dependencies
	python3.12 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt && .venv/bin/pip install -q -e .

doctor:           ## report credentials (presence only) and endpoint status
	$(PY) scripts/doctor.py

audit-data:       ## verify dataset checksums, counts and schema
	$(PY) scripts/audit_data.py

ingest:           ## parse + chunk the corpus into graph-load artifacts (no DB needed)
	$(PY) -m podium.ingest.build

embed:            ## embed all chunks (resumable, cached)
	$(PY) -m podium.ingest.embed

load:             ## create schema, load vertices/edges/vectors, install GSQL queries on TigerGraph
	$(PY) -m podium.ingest.load_tg --all

verify-graph:     ## reconcile graph counts against the ingestion manifest
	$(PY) -m podium.ingest.load_tg --verify

test:             ## unit tests (no DB / no LLM)
	$(PY) -m pytest -q tests

benchmark-public: ## run the three headline pipelines on the 100 public questions
	$(PY) scripts/benchmark.py --run $(RUN) --split public

evaluate-public:  ## score a public run (exact/numeric match, citation validity, LLM judge)
	$(PY) scripts/evaluate.py --run $(RUN) --split public

benchmark-hidden: ## run the three pipelines on the 50 hidden questions (raw outputs only)
	$(PY) scripts/benchmark.py --run $(RUN) --split hidden

benchmark-robustness: ## supplemental labelled robustness set (agent vs fixed-plan control too)
	$(PY) scripts/benchmark.py --run $(RUN) --split robustness --pipelines rag graphrag agentic_graphrag fixed_plan

evaluate-robustness:
	$(PY) scripts/evaluate.py --run $(RUN) --split robustness

ablations:        ## fixed-plan and no-coverage controls on the public set
	$(PY) scripts/benchmark.py --run $(RUN) --split public --pipelines fixed_plan agentic_nocov

dashboard:        ## serve API + dashboard on 127.0.0.1:8120
	.venv/bin/uvicorn podium.api:app --host 127.0.0.1 --port 8120

export-submission: ## validate and package predictions, traces and metrics for submission
	$(PY) scripts/export_submission.py --run $(RUN)
