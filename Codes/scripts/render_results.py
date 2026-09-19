"""Render benchmark summaries as Markdown tables (for the README / write-up)."""
import argparse
import json

from podium.settings import RUNS

NAMES = {"rag": "RAG", "graphrag": "GraphRAG", "agentic_graphrag": "Agentic GraphRAG", "fixed_plan": "Fixed-plan control", "agentic_nocov": "No-coverage control"}
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--split", default="public")
a = ap.parse_args()
s = json.load(open(RUNS / a.run / f"summary_{a.split}.json"))
P = s["pipelines"]
order = [p for p in NAMES if p in P]
scored = a.split != "hidden"


def f(v, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}" + ("%" if pct else "")
    return str(v) + ("%" if pct else "")


print(f"### {a.split} set — run `{a.run}`\n")
if scored:
    print("| Pipeline | n | Exact match | Judge correct | Judge completeness | Evidence support | Citations valid | Mean tokens | Mean context tok | LLM calls | p50 latency | p95 latency | Steps | Strategy changes | Abstained |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in order:
        x = P[p]
        print(f"| **{NAMES[p]}** | {x['n']} | {f(x.get('accuracy_exact'), True)} | {f(x.get('accuracy_judge'), True)} | {f(x.get('completeness_judge'))} | {f(x.get('evidence_support_judge'))} | {f(x.get('citations_all_valid_rate'), True)} | {f(x['mean_tokens'])} | {f(x['mean_context_tokens'])} | {f(x['mean_llm_calls'])} | {x['p50_latency_ms']/1000:.1f}s | {x['p95_latency_ms']/1000:.1f}s | {f(x['mean_steps'])} | {f(x['strategy_change_rate'], True)} | {x['abstained']} |")
else:
    print("| Pipeline | n | Answered | Partial | Abstained | Errors | Mean tokens | Mean context tok | LLM calls | p50 latency | p95 latency | Steps | Strategy changes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in order:
        x = P[p]
        print(f"| **{NAMES[p]}** | {x['n']} | {x['answered']} | {x['partial']} | {x['abstained']} | {x['errors']} | {f(x['mean_tokens'])} | {f(x['mean_context_tokens'])} | {f(x['mean_llm_calls'])} | {x['p50_latency_ms']/1000:.1f}s | {x['p95_latency_ms']/1000:.1f}s | {f(x['mean_steps'])} | {f(x['strategy_change_rate'], True)} |")
if s.get("by_qtype"):
    print(f"\n**By question type** ({'exact match · mean tokens' if scored else 'mean tokens · answered'})\n")
    print("| Type | n | " + " | ".join(NAMES[p] for p in order) + " |")
    print("|---|---:|" + "---:|" * len(order))
    for qt, d in s["by_qtype"].items():
        n = d[order[0]]["n"]
        cells = [f"{f(d[p].get('accuracy_exact'), True)} · {f(d[p]['mean_tokens'])}" if scored else f"{f(d[p]['mean_tokens'])} · {d[p]['answered']}" for p in order]
        print(f"| {qt} | {n} | " + " | ".join(cells) + " |")
pr = s.get("paired_graphrag_vs_agentic")
if pr:
    print(f"\n**Paired GraphRAG → Agentic GraphRAG** (n={pr['n']}): quality lift {pr['quality_lift_pct_points']} pp (bootstrap 95% CI {pr['bootstrap_95ci_lift']}), mean token delta {pr['mean_token_delta']}, extra tokens per extra correct answer: {pr['extra_tokens_per_extra_correct']}. Agent wins: {', '.join(pr['agent_wins']) or 'none'}; losses: {', '.join(pr['agent_losses']) or 'none'}.")
    print("\nStop reasons (agentic): " + ", ".join(f"{k} {v}" for k, v in P.get("agentic_graphrag", {}).get("stop_reasons", {}).items()))
