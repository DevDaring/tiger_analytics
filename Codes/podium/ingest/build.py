"""Build graph-load artifacts from the immutable corpus (plan section 6).

Outputs (artifacts/cache/):
  documents.jsonl  chunks.jsonl  events.jsonl  facts.jsonl  entities.jsonl
  venues.jsonl  editions.json  sports.json
and artifacts/manifests/ingestion_manifest.json with hashes and counts.

The question files are never read here.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..contracts import stable_hash
from ..corpus.chunking import chunk_document
from ..corpus.parse import (
    event_norm,
    norm_key,
    norm_text,
    parse_count,
    parse_dates,
    parse_infoboxes,
    parse_title,
    sport_norm,
    split_concatenated_names,
)
from ..settings import CACHE, CHUNKER_VERSION, CORPUS_PATH, EXTRACTION_VERSION, MANIFESTS, PARSER_VERSION

MEDALS = ("gold", "silver", "bronze")
YEAR_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def _fact(subject: str, predicate: str, raw: str, chunk_id: str, cs: int, ce: int, value=None, unit: str = "", status: str = "ok", note: str = "") -> dict:
    fid = stable_hash([subject, predicate, raw, cs, ce, EXTRACTION_VERSION])
    return {
        "fact_id": f"f-{fid}",
        "subject_id": subject,
        "predicate": predicate,
        "value_raw": raw,
        "value_num": value,
        "unit": unit,
        "status": status,
        "note": note,
        "method": "infobox",
        "chunk_id": chunk_id,
        "char_start": cs,
        "char_end": ce,
    }


def _year_from_prev(raw: str | None) -> int | None:
    if not raw:
        return None
    m = YEAR_RE.search(raw)
    return int(m.group(1)) if m else None


def build(corpus_path: Path = CORPUS_PATH, out_dir: Path = CACHE) -> dict:
    t0 = time.time()
    docs = load_corpus(corpus_path)
    corpus_hash = file_sha256(corpus_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents, chunks, events, facts, entities, venues = [], [], [], [], {}, {}
    editions: dict[str, dict] = {}
    sports: Counter = Counter()
    stats = Counter()
    ids = set()

    for d in docs:
        doc_id = d["doc_id"]
        assert doc_id not in ids, f"duplicate doc_id {doc_id}"
        ids.add(doc_id)
        text = d["text"]
        boxes, _ = parse_infoboxes(text)
        title = parse_title(d["title"])
        doc_chunks = chunk_document(doc_id, text)
        infobox_chunk = next((c for c in doc_chunks if c.kind == "infobox"), None)
        is_event = bool(title and title.kind == "Olympics" and title.event)
        documents.append(
            {
                "doc_id": doc_id,
                "title": d["title"],
                "url": d.get("url", ""),
                "wikidata_qid": d.get("wikidata_qid", ""),
                "wikipedia_pageid": d.get("wikipedia_pageid"),
                "approx_tokens": d.get("approx_tokens"),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "infobox_type": boxes[0].type if boxes else "",
                "is_olympic_event": is_event,
                "n_chunks": len(doc_chunks),
                "n_chars": len(text),
            }
        )
        for c in doc_chunks:
            chunks.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": doc_id,
                    "ordinal": c.ordinal,
                    "section": c.section,
                    "kind": c.kind,
                    "text": c.text,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "tokens": c.tokens,
                    "content_hash": c.content_hash,
                    "embed_text": c.embed_text(d["title"]),
                }
            )
        stats["chunks"] += len(doc_chunks)

        # ---- generic entity for every non-event document (films, people, ...)
        if not is_event:
            etype = boxes[0].type if boxes else "topic"
            entities[f"doc:{doc_id}"] = {
                "entity_id": f"doc:{doc_id}",
                "name": d["title"],
                "name_norm": norm_key(d["title"]),
                "type": etype,
                "doc_id": doc_id,
                "noc": "",
            }
            stats["non_event_docs"] += 1
            continue

        # ---- Olympic event
        stats["event_docs"] += 1
        ev_box = next((b for b in boxes if b.type == "Olympic event"), None)
        f = ev_box.fields if ev_box else {}
        edition_id = title.edition_id
        editions.setdefault(edition_id, {"edition_id": edition_id, "year": title.year, "season": title.season, "label": f"{title.year} {title.season} Olympics", "n_events": 0})
        editions[edition_id]["n_events"] += 1
        sports[title.sport] += 1

        def fv(k: str) -> str | None:
            return f[k].value if k in f else None

        comp = parse_count(fv("competitors"))
        nat = parse_count(fv("nations"))
        venue_raw = fv("venue") or fv("venues") or ""
        date_raw = fv("date") or fv("dates") or ""
        ds = parse_dates(date_raw)
        games_raw = fv("games") or ""
        prev_raw, next_raw = fv("prev"), fv("next")
        venue_id = norm_key(venue_raw) if venue_raw else ""
        if venue_id:
            v = venues.setdefault(venue_id, {"venue_id": venue_id, "name": venue_raw, "aliases": set(), "n_events": 0})
            v["aliases"].add(venue_raw)
            v["n_events"] += 1
        parse_status = "ok" if ev_box else ("no_event_infobox" if boxes else "no_infobox")
        ev = {
            "event_id": doc_id,
            "doc_id": doc_id,
            "title": d["title"],
            "sport": title.sport,
            "sport_norm": sport_norm(title.sport),
            "season": title.season,
            "edition_year": title.year,
            "edition_id": edition_id,
            "event_name": title.event,
            "event_norm": event_norm(title.event),
            "event_field": fv("event") or "",
            "games_raw": games_raw,
            "venue_raw": venue_raw,
            "venue_id": venue_id,
            "date_raw": date_raw,
            "date_norm": norm_text(date_raw),
            "date_days": sorted(f"{m:02d}-{dd:02d}" for m, dd in ds.days),
            "date_years": sorted(ds.years),
            "competitors_raw": comp.raw or "",
            "competitors": comp.value if comp.value is not None else -1,
            "has_competitors": comp.value is not None and comp.status in ("ok", "suspicious"),
            "competitors_unit": comp.unit,
            "competitors_status": comp.status,
            "nations_raw": nat.raw or "",
            "nations": nat.value if nat.value is not None else -1,
            "has_nations": nat.value is not None and nat.status == "ok",
            "nations_status": nat.status,
            "gold_raw": fv("gold") or "",
            "gold_noc": fv("goldNOC") or "",
            "silver_raw": fv("silver") or "",
            "silver_noc": fv("silverNOC") or "",
            "bronze_raw": fv("bronze") or "",
            "bronze_noc": fv("bronzeNOC") or "",
            "bronze2_raw": fv("bronze2") or "",
            "bronze2_noc": fv("bronzeNOC2") or "",
            "prev_raw": prev_raw or "",
            "next_raw": next_raw or "",
            "prev_year": _year_from_prev(prev_raw) or -1,
            "next_year": _year_from_prev(next_raw) or -1,
            "win_value": fv("win_value") or "",
            "infobox_type": ev_box.type if ev_box else (boxes[0].type if boxes else ""),
            "parse_status": parse_status,
            "infobox_chunk_id": infobox_chunk.chunk_id if infobox_chunk else "",
        }
        events.append(ev)
        stats[f"competitors_{comp.status}"] += 1

        # ---- facts with source spans (all inside the infobox chunk)
        if ev_box and infobox_chunk:
            cid = infobox_chunk.chunk_id
            for key, pred in (("competitors", "competitors"), ("nations", "nations")):
                if key in f:
                    cv = parse_count(f[key].value)
                    facts.append(_fact(doc_id, pred, f[key].value, cid, f[key].value_start, f[key].value_end, cv.value, cv.unit, cv.status, cv.note))
            for key, pred in (("venue", "venue"), ("venues", "venue"), ("date", "date"), ("dates", "date"), ("prev", "prev"), ("next", "next"), ("games", "games"), ("win_value", "win_value")):
                if key in f:
                    facts.append(_fact(doc_id, pred, f[key].value, cid, f[key].value_start, f[key].value_end))
            for medal in MEDALS:
                if medal in f:
                    noc = fv(medal + "NOC") or ""
                    fact = _fact(doc_id, medal, f[medal].value, cid, f[medal].value_start, f[medal].value_end, unit=noc)
                    facts.append(fact)
                    # medallist entities (raw string preserved; names may be concatenated for teams)
                    raw = f[medal].value
                    eid = "ath:" + norm_key(raw)
                    if raw and eid != "ath:":
                        e = entities.setdefault(
                            eid,
                            {"entity_id": eid, "name": raw, "name_norm": norm_key(raw), "type": "medallist", "doc_id": "", "noc": noc, "members": split_concatenated_names(raw), "medals": []},
                        )
                        e["medals"].append({"event_id": doc_id, "medal": medal, "noc": noc, "fact_id": fact["fact_id"]})
                    if noc:
                        nid = "noc:" + noc
                        entities.setdefault(nid, {"entity_id": nid, "name": noc, "name_norm": norm_key(noc), "type": "noc", "doc_id": "", "noc": noc})
            if "bronze2" in f:
                facts.append(_fact(doc_id, "bronze", f["bronze2"].value, cid, f["bronze2"].value_start, f["bronze2"].value_end, unit=fv("bronzeNOC2") or ""))

    # ---- derived temporal relations: previous edition (season-specific ordering of corpus editions)
    by_season: dict[str, list[int]] = defaultdict(list)
    for e in editions.values():
        by_season[e["season"]].append(e["year"])
    prev_edition = {}
    for season, years in by_season.items():
        ys = sorted(years)
        for i, y in enumerate(ys):
            prev_edition[f"{y} {season}"] = f"{ys[i-1]} {season}" if i > 0 else ""
    # previous event: same sport + event_norm, target edition = prev field (explicit) else edition order (derived)
    key_index: dict[tuple[str, str, str, int], str] = {}
    for ev in events:
        key_index[(ev["sport_norm"], ev["event_norm"], ev["season"], ev["edition_year"])] = ev["event_id"]
    prev_links = []
    for ev in events:
        target = None
        source = ""
        if ev["prev_year"] > 0:
            target = key_index.get((ev["sport_norm"], ev["event_norm"], ev["season"], ev["prev_year"]))
            source = "infobox_prev"
        if target is None:
            pe = prev_edition.get(ev["edition_id"], "")
            if pe:
                target = key_index.get((ev["sport_norm"], ev["event_norm"], ev["season"], int(pe.split()[0])))
                source = "derived_edition_order" if target else ""
        ev["prev_event_id"] = target or ""
        ev["prev_event_source"] = source if target else ""
        if target:
            prev_links.append({"from": ev["event_id"], "to": target, "source": source, "prev_raw": ev["prev_raw"]})

    for v in venues.values():
        v["aliases"] = sorted(v["aliases"])

    def dump(name: str, rows) -> None:
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("documents.jsonl", documents)
    dump("chunks.jsonl", chunks)
    dump("events.jsonl", events)
    dump("facts.jsonl", facts)
    dump("entities.jsonl", entities.values())
    dump("venues.jsonl", venues.values())
    dump("prev_links.jsonl", prev_links)
    json.dump({"editions": editions, "prev_edition": prev_edition}, open(out_dir / "editions.json", "w"), indent=1)
    json.dump(dict(sports), open(out_dir / "sports.json", "w"), indent=1, ensure_ascii=False)

    manifest = {
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_hash,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "graph_snapshot_id": f"podium-{corpus_hash[:8]}-{PARSER_VERSION}-{CHUNKER_VERSION}",
        "documents": len(documents),
        "chunks": len(chunks),
        "events": len(events),
        "facts": len(facts),
        "entities": len(entities),
        "venues": len(venues),
        "editions": len(editions),
        "sports": len(sports),
        "prev_links": len(prev_links),
        "prev_links_by_source": dict(Counter(p["source"] for p in prev_links)),
        "stats": dict(stats),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
    }
    json.dump(manifest, open(MANIFESTS / "ingestion_manifest.json", "w"), indent=1)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
