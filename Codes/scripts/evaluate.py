"""CLI: score a run offline (gold answers are read only here)."""
import argparse
import json

from podium.eval.reports import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--split", default="public")
ap.add_argument("--no-judge", action="store_true")
a = ap.parse_args()
s = evaluate(a.run, a.split, use_judge=not a.no_judge)
print(json.dumps({k: v for k, v in s.items() if k in ("pipelines", "paired_graphrag_vs_agentic")}, indent=1, ensure_ascii=False))
