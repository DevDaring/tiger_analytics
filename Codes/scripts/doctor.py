"""Report credential presence and endpoint status without printing secrets."""
import json

from podium.settings import MANIFESTS, settings

print(json.dumps(settings.redacted(), indent=1))
try:
    from podium.db.tg import get_tg

    tg = get_tg()
    print("tigergraph:", tg.version().get("version"), "reachable" if tg.ping() else "unreachable")
    print("graph_stats:", tg.run_query("graph_stats"))
except Exception as e:  # noqa: BLE001
    print("tigergraph: ERROR", str(e)[:200])
try:
    from podium.llm.client import complete_json, embed_query

    d, u = complete_json("doctor", "Reply in JSON.", "Return {\"ok\": true}", {"type": "object", "properties": {"ok": {"type": "boolean"}}}, max_tokens=50)
    print("answer model:", settings.answer_model, d, f"{u.input_tokens}+{u.output_tokens} tokens")
    v, u2 = embed_query("doctor")
    print("embedding model:", settings.embed_model, len(v), "dims")
except Exception as e:  # noqa: BLE001
    print("llm: ERROR", str(e)[:200])
m = MANIFESTS / "ingestion_manifest.json"
print("ingestion manifest:", "present" if m.exists() else "missing (run make ingest)")
