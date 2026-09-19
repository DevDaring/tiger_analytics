"""Structure-aware chunking that preserves evidence (plan section 6, step B).

* infobox blocks stay intact (split at field boundaries only if very large)
* headings are detected and carried as context; tables keep their header row
* every chunk records original-text character offsets
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .parse import approx_tokens, parse_infoboxes

TARGET_TOKENS = 450
MAX_TOKENS = 700
HEADING_RE = re.compile(r"^[A-Z0-9;!'\"(][^|\n]{0,70}$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    section: str
    kind: str  # infobox | prose | table
    text: str
    char_start: int
    char_end: int
    tokens: int
    content_hash: str

    def embed_text(self, title: str) -> str:
        head = title if not self.section else f"{title} > {self.section}"
        return f"{head}\n{self.text}"


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or " | " in s or len(s) > 70:
        return False
    if s.endswith((".", ":", ",", ";")):
        return False
    if s.startswith(("[Infobox", "-", "!", ";")):
        return False
    words = s.split()
    return len(words) <= 8 and not re.search(r"\d{4}\s*\|", s)


def _blocks(text: str, start: int) -> list[tuple[int, int, str]]:
    """Split body into (start, end, kind) blocks on blank lines; kind is prose|table|heading."""
    out = []
    pos = start
    n = len(text)
    while pos < n:
        nxt = text.find("\n\n", pos)
        if nxt == -1:
            nxt = n
        seg = text[pos:nxt]
        if seg.strip():
            lines = [ln for ln in seg.split("\n") if ln.strip()]
            if all(" | " in ln for ln in lines):
                kind = "table"
            elif len(lines) == 1 and _is_heading(lines[0]):
                kind = "heading"
            else:
                kind = "prose"
            out.append((pos, nxt, kind))
        pos = nxt + 2
    return out


def _split_long(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """Split text[s:e] into pieces of roughly TARGET_TOKENS at newline or sentence boundaries."""
    out = []
    cur = s
    limit = TARGET_TOKENS * 4  # characters
    while e - cur > MAX_TOKENS * 4:
        window_end = cur + limit
        cut = text.rfind("\n", cur + limit // 2, window_end)
        if cut == -1:
            cut = text.rfind(". ", cur + limit // 2, window_end)
            cut = cut + 1 if cut != -1 else window_end
        out.append((cur, cut))
        cur = cut
    out.append((cur, e))
    return [(a, b) for a, b in out if text[a:b].strip()]


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    boxes, body_start = parse_infoboxes(text)

    def add(kind: str, section: str, s: int, e: int) -> None:
        t = text[s:e].strip("\n")
        # trim offsets to the stripped text
        lead = len(text[s:e]) - len(text[s:e].lstrip("\n"))
        s2 = s + lead
        e2 = s2 + len(t)
        if not t.strip():
            return
        if approx_tokens(t) > MAX_TOKENS and kind != "infobox":
            # oversized single block: split at line / sentence boundaries near the target
            for ps, pe in _split_long(text, s2, e2):
                add(kind, section, ps, pe)
            return
        cid = f"{doc_id}#{len(chunks)}"
        chunks.append(
            Chunk(cid, doc_id, len(chunks), section, kind, t, s2, e2, approx_tokens(t), hashlib.sha256(t.encode()).hexdigest()[:16])
        )

    if boxes:
        s, e = boxes[0].start, boxes[-1].end
        if approx_tokens(text[s:e]) <= MAX_TOKENS * 2:
            add("infobox", "Infobox", s, e)
        else:  # split at field boundaries
            cur_s = s
            for box in boxes:
                lines = sorted(box.fields.values(), key=lambda f: f.line_start)
                for fv in lines:
                    if approx_tokens(text[cur_s : fv.line_end]) > MAX_TOKENS:
                        add("infobox", "Infobox", cur_s, fv.line_start)
                        cur_s = fv.line_start
            add("infobox", "Infobox", cur_s, e)

    section = ""  # heading in force at the start of the current buffer
    current_heading = ""
    buf_start: int | None = None
    buf_end: int | None = None
    buf_kind = "prose"

    def flush() -> None:
        nonlocal buf_start, buf_end
        if buf_start is not None and buf_end is not None:
            add(buf_kind, section, buf_start, buf_end)
        buf_start = buf_end = None

    def buf_tokens(upto: int) -> int:
        return approx_tokens(text[buf_start:upto]) if buf_start is not None else 0

    for s, e, kind in _blocks(text, body_start):
        if kind == "heading":
            head = text[s:e].strip()
            # a heading is a preferred boundary once the buffer is reasonably full
            if buf_start is not None and buf_tokens(buf_end) >= TARGET_TOKENS // 2:
                flush()
            current_heading = head
            if buf_start is None:
                section = head
                buf_start, buf_end, buf_kind = s, e, "prose"
            else:
                buf_end = e
            continue
        if kind == "table" and approx_tokens(text[s:e]) > MAX_TOKENS:
            flush()
            rows = text[s:e].split("\n")
            header = rows[0]
            pos = s
            cur_s = s
            for i, row in enumerate(rows):
                row_end = pos + len(row)
                if approx_tokens(text[cur_s:row_end]) > TARGET_TOKENS and i > 0:
                    add("table", (current_heading + " | " + header)[:120], cur_s, pos - 1)
                    cur_s = pos
                pos = row_end + 1
            add("table", (current_heading + " | " + header)[:120], cur_s, e)
            section = current_heading
            continue
        if buf_start is None:
            section = current_heading
            buf_start, buf_end, buf_kind = s, e, kind
            continue
        if buf_tokens(e) > TARGET_TOKENS and buf_tokens(buf_end) >= 120:
            flush()
            section = current_heading
            buf_start, buf_end, buf_kind = s, e, kind
        else:
            buf_end = e
            if kind == "table" and buf_kind != "table":
                buf_kind = "prose"
    flush()
    return chunks
