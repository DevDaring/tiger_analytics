from podium.corpus.chunking import chunk_document
from podium.corpus.parse import (
    date_match,
    event_norm,
    norm_key,
    parse_count,
    parse_dates,
    parse_infoboxes,
    parse_title,
    split_concatenated_names,
    sport_norm,
)

DOC = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  date: 6 to 8 August
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  goldNOC: HUN
  prev: 2008
  next: 2016

The men's canoe sprint K-2 1,000 metres competition took place between 6 and 8 August at Eton Dorney.

Competition format
The competition comprised heats, semifinals, and a final round.

Results
Rank | Canoer | Country | Time
1 | Martin HollsteinAndreas Ihle | 3:15.263 | Q
2 | 3:19.073
"""


def test_title_parsing():
    t = parse_title("Athletics at the 2016 Summer Olympics – Men's 20 kilometres walk")
    assert t and (t.sport, t.year, t.season, t.event) == ("Athletics", 2016, "Summer", "Men's 20 kilometres walk")
    assert parse_title("Jab We Met") is None
    assert parse_title("Cycling at the 1900 Summer Olympics – Men's sprint").year == 1900


def test_infobox_offsets_are_original_text_coordinates():
    boxes, body = parse_infoboxes(DOC)
    assert len(boxes) == 1 and boxes[0].type == "Olympic event"
    f = boxes[0].fields
    assert f["competitors"].value == "24"
    assert DOC[f["competitors"].value_start : f["competitors"].value_end] == "24"
    assert DOC[f["gold"].value_start : f["gold"].value_end] == "Rudolf DombiRoland Kökény"
    assert DOC[body:].startswith("The men's canoe sprint")


def test_count_parsing_distinguishes_missing_zero_units():
    assert parse_count(None).status == "missing" and parse_count(None).value is None
    assert parse_count("0").value == 0 and parse_count("0").status == "ok"
    assert parse_count("1,234").value == 1234
    c = parse_count("23 teams")
    assert c.value == 23 and c.unit == "teams" and c.status == "unit"
    c = parse_count("32 (16 pairs)")
    assert c.value == 32 and c.unit == "athletes" and c.status == "ok"
    assert parse_count("41000000").status == "suspicious"
    assert parse_count("many").value is None


def test_dates():
    assert parse_dates("6 to 8 August").days == {(8, 6), (8, 7), (8, 8)}
    assert parse_dates("13 February 2010").days == {(2, 13)} and parse_dates("13 February 2010").years == {2010}
    assert parse_dates("August 12, 2008").days == {(8, 12)}
    assert parse_dates("18 September to 1 October 2000").days >= {(9, 18), (9, 30), (10, 1)}
    assert parse_dates("22 September 2000 (heats)25 September 2000 (final)").days == {(9, 22), (9, 25)}
    assert parse_dates("2008-08-10").days == {(8, 10)}
    assert date_match("16 February 1992", "16 February 1992") == (3, "exact")
    assert date_match("12 August 2008", "August 12, 2008")[0] == 2  # same day set
    assert date_match("6–9 August", "6-8 August")[0] == 1  # partial overlap
    assert date_match("12 August 2008", "August 12, 2004")[0] == 0
    assert date_match("9 August", "August 10")[0] == 0


def test_normalisation():
    assert event_norm("Men's 20 kilometres walk") == "men 20 km walk"
    assert event_norm("Individual normal hill/10 km") == event_norm("individual normal hill/10 km")
    assert event_norm("Men's Greco-Roman 96 kg") == event_norm("men's greco-roman 96 kg")
    assert sport_norm("Short-track speed skating") == "short track speed skating"
    assert norm_key("Beijing Science and TechnologyUniversity Gymnasium") == norm_key("Beijing Science and Technology University Gymnasium")
    assert norm_key("Val-d'Isère") == norm_key("Val d'Isere")


def test_name_split_is_display_only():
    assert split_concatenated_names("Dani KingLaura TrottJoanna Rowsell") == ["Dani King", "Laura Trott", "Joanna Rowsell"]
    assert split_concatenated_names("Naim Süleymanoğlu") == ["Naim Süleymanoğlu"]


def test_chunks_cover_exact_spans():
    chunks = chunk_document("Q1", DOC)
    assert chunks[0].kind == "infobox"
    for c in chunks:
        assert DOC[c.char_start : c.char_end] == c.text
    assert any("Competition format" in c.text for c in chunks)
