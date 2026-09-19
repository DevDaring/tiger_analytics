"""CLI: run pipelines on a question split. Only question text crosses the inference boundary."""
import argparse

from podium.eval.runner import HEADLINE, run_benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--split", default="public")
ap.add_argument("--pipelines", nargs="*", default=HEADLINE)
ap.add_argument("--qids", nargs="*", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--concurrency", type=int, default=3)
a = ap.parse_args()
print(run_benchmark(a.run, a.split, a.pipelines, a.qids, a.concurrency, a.limit))
