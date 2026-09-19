"""Embed chunks with one fixed embedding model; cache by (content_hash, model, dims).

Resumable: re-running only embeds chunks missing from the cache.
Output: artifacts/cache/embeddings.<model>.jsonl  {chunk_id, content_hash, vector}
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..llm.client import embed_texts
from ..settings import CACHE, settings

BATCH = 64
WORKERS = 4


def cache_path(model_id: str | None = None) -> Path:
    mid = (model_id or settings.embed_model.model_id).replace("/", "_")
    return CACHE / f"embeddings.{mid}.{settings.embed_dim}.jsonl"


def load_cache(path: Path) -> dict[str, list[float]]:
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["content_hash"]] = r["vector"]
    return out


def run(chunks_path: Path = CACHE / "chunks.jsonl") -> dict:
    t0 = time.time()
    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    path = cache_path()
    cache = load_cache(path)
    todo = {}
    for c in chunks:
        if c["content_hash"] not in cache:
            todo.setdefault(c["content_hash"], c["embed_text"])
    items = list(todo.items())
    print(f"{len(chunks)} chunks, {len(cache)} cached, {len(items)} to embed", flush=True)
    batches = [items[i : i + BATCH] for i in range(0, len(items), BATCH)]
    total_tokens = 0
    with open(path, "a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(embed_texts, [t for _, t in b]): b for b in batches}
        done = 0
        for fut in as_completed(futs):
            b = futs[fut]
            vectors, usage = fut.result()
            total_tokens += usage.input_tokens or 0
            for (h, _), v in zip(b, vectors):
                out.write(json.dumps({"content_hash": h, "vector": v}) + "\n")
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(batches)} batches, {total_tokens} tokens, {time.time()-t0:.0f}s", flush=True)
            out.flush()
    return {"embedded": len(items), "tokens": total_tokens, "seconds": round(time.time() - t0, 1), "path": str(path)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
    sys.stdout.flush()
