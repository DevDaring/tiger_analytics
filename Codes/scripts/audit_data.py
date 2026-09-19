"""Verify the supplied dataset: checksums, counts, schema, gold-id integrity."""
import hashlib
import json

from podium.settings import CORPUS_PATH, DATASET_DIR, HIDDEN_QUESTIONS, PUBLIC_QUESTIONS

EXPECTED = {
    "corpus/corpus.jsonl": "27aef30bbe32474df782f5164262d5d765af14efc9f0288320486cee585e191a",
    "questions/eval_public.jsonl": "abddb7d18a6d8ed908f514a7e560fe4950ebe479ebb2cdb75a7456887c10c6e5",
    "questions/eval_hidden.jsonl": "a2742f449765bb66c7ccc87a281560de3327894fe656a3b2f415f68d990c7d0a",
}
ok = True
for rel, exp in EXPECTED.items():
    h = hashlib.sha256(open(DATASET_DIR / rel, "rb").read()).hexdigest()
    print(f"{rel}: {'OK' if h == exp else 'MISMATCH'} {h}")
    ok &= h == exp
docs = [json.loads(l) for l in open(CORPUS_PATH, encoding="utf-8")]
ids = {d["doc_id"] for d in docs}
pub = [json.loads(l) for l in open(PUBLIC_QUESTIONS, encoding="utf-8")]
hid = [json.loads(l) for l in open(HIDDEN_QUESTIONS, encoding="utf-8")]
print(f"documents={len(docs)} unique_ids={len(ids)} public={len(pub)} hidden={len(hid)} approx_tokens={sum(d.get('approx_tokens', 0) for d in docs)}")
missing = [g for q in pub for g in q["gold_doc_ids"] if g not in ids]
print("public gold_doc_ids missing from corpus:", len(missing))
print("hidden schema:", sorted(hid[0].keys()))
print("AUDIT", "PASSED" if ok and len(ids) == len(docs) == 2951 and not missing else "FAILED")
