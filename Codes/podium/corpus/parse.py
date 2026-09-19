"""Deterministic corpus parsing: titles, infoboxes, counts, dates, venues, names.

Everything here is pure and testable. Offsets are *original-text character offsets*.
The parser never sees the question files.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        start=1,
    )
}
MONTH_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
MONTHS.update(MONTH_ABBR)
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) (?P<kind>Olympics|Paralympics)(?: – (?P<event>.+))?$")


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def norm_text(s: str) -> str:
    """Conservative normalisation: unicode, dashes, apostrophes, case, whitespace."""
    s = nfkc(s or "")
    s = s.replace("–", "-").replace("—", "-").replace("‐", "-").replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def norm_key(s: str) -> str:
    """Alphanumeric-only key used for venue / entity identity (accents stripped)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def event_norm(s: str) -> str:
    s = norm_text(s)
    s = s.replace("kilometres", "km").replace("kilometre", "km").replace("metres", "m").replace("metre", "m")
    s = s.replace("women's", "women").replace("men's", "men").replace("'", "")
    s = re.sub(r"[^a-z0-9+ ]+", " ", s)  # keep '+' so "80 kg" and "+80 kg" stay distinct
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sport_norm(s: str) -> str:
    s = norm_text(s)
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------------------
@dataclass
class TitleInfo:
    sport: str
    year: int
    season: str
    kind: str
    event: str | None

    @property
    def edition_id(self) -> str:
        return f"{self.year} {self.season}"


def parse_title(title: str) -> TitleInfo | None:
    m = TITLE_RE.match(nfkc(title).replace(" - ", " – ") if " – " not in title else title)
    if not m:
        return None
    return TitleInfo(m["sport"], int(m["year"]), m["season"], m["kind"], m["event"])


# --------------------------------------------------------------------------------------
# Infobox
# --------------------------------------------------------------------------------------
@dataclass
class FieldValue:
    value: str
    line_start: int  # offset of the line start in original text
    line_end: int
    value_start: int
    value_end: int


@dataclass
class Infobox:
    type: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    start: int = 0
    end: int = 0


INFOBOX_HEAD = re.compile(r"^\[Infobox ([^\]]*)\]$")
FIELD_LINE = re.compile(r"^(\s+)([^:\n]+?): ?(.*)$")


def parse_infoboxes(text: str) -> tuple[list[Infobox], int]:
    """Parse leading infobox blocks. Returns (infoboxes, offset where the body starts)."""
    boxes: list[Infobox] = []
    pos = 0
    n = len(text)
    while pos < n:
        line_end = text.find("\n", pos)
        if line_end == -1:
            line_end = n
        line = text[pos:line_end]
        m = INFOBOX_HEAD.match(line)
        if not m:
            break
        box = Infobox(type=m.group(1).strip(), start=pos)
        pos = line_end + 1
        while pos < n:
            le = text.find("\n", pos)
            if le == -1:
                le = n
            ln = text[pos:le]
            fm = FIELD_LINE.match(ln)
            if not fm:
                break
            indent, key, val = fm.group(1), fm.group(2).strip(), fm.group(3)
            vstart = pos + len(indent) + len(fm.group(2)) + 1  # after ':'
            if text[vstart : vstart + 1] == " ":
                vstart += 1
            box.fields[key] = FieldValue(val.strip(), pos, le, vstart, le)
            pos = le + 1
        box.end = pos
        boxes.append(box)
    # skip blank lines
    while pos < n and text[pos] == "\n":
        pos += 1
    return boxes, pos


# --------------------------------------------------------------------------------------
# Counts (competitors / nations)
# --------------------------------------------------------------------------------------
@dataclass
class CountValue:
    raw: str | None
    value: int | None
    unit: str  # athletes | teams | pairs | unknown
    status: str  # ok | unit | missing | unparsed | suspicious
    note: str = ""


def parse_count(raw: str | None) -> CountValue:
    if raw is None or raw.strip() == "":
        return CountValue(raw, None, "unknown", "missing")
    s = nfkc(raw).strip()
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    if re.fullmatch(r"\d+", s):
        v = int(s)
        if v > 5000:
            return CountValue(raw, v, "athletes", "suspicious", "implausibly large value")
        return CountValue(raw, v, "athletes", "ok")
    m = re.fullmatch(r"(\d+)\s*\((\d+)\s*(teams|pairs)\)", s)
    if m:  # "32 (16 pairs)" -> 32 athletes forming 16 pairs
        return CountValue(raw, int(m.group(1)), "athletes", "ok", f"{m.group(2)} {m.group(3)}")
    m = re.fullmatch(r"(\d+)\s*(teams|pairs|boats|crews)", s)
    if m:
        return CountValue(raw, int(m.group(1)), m.group(2), "unit", "count is in " + m.group(2) + ", not athletes")
    m = re.match(r"(\d+)", s)
    if m:
        return CountValue(raw, int(m.group(1)), "unknown", "unparsed", "leading number with unrecognised suffix")
    return CountValue(raw, None, "unknown", "unparsed")


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------
@dataclass
class DateSpec:
    raw: str
    days: set[tuple[int, int]] = field(default_factory=set)  # (month, day)
    years: set[int] = field(default_factory=set)
    months: set[int] = field(default_factory=set)

    @property
    def empty(self) -> bool:
        return not self.days and not self.months and not self.years


def _expand(m1: int, d1: int, m2: int, d2: int) -> set[tuple[int, int]]:
    out = set()
    if m1 == m2:
        lo, hi = sorted((d1, d2))
        return {(m1, d) for d in range(lo, hi + 1)}
    # cross-month range
    for d in range(d1, 32):
        out.add((m1, d))
    for d in range(1, d2 + 1):
        out.add((m2, d))
    return out


DAY_LIST = r"((?<!\d)\d{1,2}(?!\d)(?:\s*(?:,|and|&)\s*(?<!\d)\d{1,2}(?!\d))*)"
RANGE_SEP = r"\s*(?:-|to|until|through|/)\s*"


def parse_dates(raw: str | None) -> DateSpec:
    spec = DateSpec(raw or "")
    if not raw:
        return spec
    s = norm_text(raw)
    s = re.sub(r"\(([^)]*)\)", " ; ", s)  # "(heats)" labels become separators
    s = re.sub(r"(\d)(?=[a-z])", r"\1 ", s)  # "2021(final)" glue, "12august"
    s = re.sub(r"(?<=[a-z])(?=\d)", " ", s)
    # ISO dates: 2008-08-10
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", s):
        y, mo, d = (int(x) for x in m.groups())
        spec.days.add((mo, d)); spec.months.add(mo); spec.years.add(y)
    s = re.sub(r"\d{4}-\d{2}-\d{2}", " ; ", s)
    # cross-month ranges: "18 september - 1 october 2000" / "29 july - 5 august"
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s+({MONTH_RE}){RANGE_SEP}(\d{{1,2}})\s+({MONTH_RE})\b(?:\s+(\d{{4}}))?", s):
        d1, m1, d2, m2, y = m.groups()
        spec.days |= _expand(MONTHS[m1], int(d1), MONTHS[m2], int(d2))
        spec.months |= {MONTHS[m1], MONTHS[m2]}
        if y:
            spec.years.add(int(y))
    for m in re.finditer(rf"\b({MONTH_RE})\s+(\d{{1,2}})(?!\d){RANGE_SEP}({MONTH_RE})\s+(\d{{1,2}})(?!\d)(?:,?\s+(\d{{4}}))?", s):
        m1, d1, m2, d2, y = m.groups()
        spec.days |= _expand(MONTHS[m1], int(d1), MONTHS[m2], int(d2))
        spec.months |= {MONTHS[m1], MONTHS[m2]}
        if y:
            spec.years.add(int(y))
    # day(s) month (year): "6-8 august", "13 february 2010", "13, 14 february 2022"
    for m in re.finditer(rf"{DAY_LIST}(?:{RANGE_SEP}(\d{{1,2}})(?!\d))?\s+({MONTH_RE})\b(?:,?\s+(\d{{4}}))?", s):
        days, d2, mon, y = m.groups()
        mi = MONTHS[mon]
        ds = [int(x) for x in re.findall(r"\d{1,2}", days)]
        if d2:
            spec.days |= _expand(mi, ds[-1], mi, int(d2))
            ds = ds[:-1]
        spec.days |= {(mi, d) for d in ds}
        spec.months.add(mi)
        if y:
            spec.years.add(int(y))
    # month day(s) (, year): "august 12, 2008", "february 20-21, 1994", "august 9"
    for m in re.finditer(rf"\b({MONTH_RE})\s+{DAY_LIST}(?:{RANGE_SEP}(\d{{1,2}})(?!\d))?(?:,?\s+(\d{{4}}))?", s):
        mon, days, d2, y = m.groups()
        mi = MONTHS[mon]
        ds = [int(x) for x in re.findall(r"\d{1,2}", days)]
        if d2:
            spec.days |= _expand(mi, ds[-1], mi, int(d2))
            ds = ds[:-1]
        spec.days |= {(mi, d) for d in ds}
        spec.months.add(mi)
        if y:
            spec.years.add(int(y))
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", s):
        spec.years.add(int(m.group(1)))
    for m in re.finditer(rf"\b({MONTH_RE})\b", s):
        spec.months.add(MONTHS[m.group(1)])
    spec.days = {(mo, d) for mo, d in spec.days if 1 <= d <= 31}
    return spec


def date_match(question_date: str, field_date: str | None) -> tuple[int, str]:
    """Return (score, reason). 3 exact, 2 normalised-equal, 1 day overlap, 0 no match."""
    if not field_date:
        return 0, "no date field"
    if question_date == field_date:
        return 3, "exact"
    if norm_text(question_date) == norm_text(field_date):
        return 2, "normalised-equal"
    a, b = parse_dates(question_date), parse_dates(field_date)
    if a.years and b.years and not (a.years & b.years):
        return 0, "different year"
    if a.days and b.days and (a.days & b.days):
        return 1, "day overlap"
    if not a.days and not b.days and a.months and b.months and (a.months & b.months):
        return 1, "month overlap"
    return 0, "no overlap"


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def split_concatenated_names(raw: str) -> list[str]:
    """Display-only heuristic: 'Dani KingLaura Trott' -> ['Dani King', 'Laura Trott'].
    Splits where a lowercase letter is directly followed by an uppercase letter."""
    if not raw:
        return []
    parts = re.split(r"(?<=[a-zà-ÿøœßğıšžčćđłńśź])(?=[A-ZÀ-ÝØŒĞİŠŽČĆĐŁŃŚŹ])", raw)
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)
