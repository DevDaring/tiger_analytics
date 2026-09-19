"""Question interpretation (LLM, structured) shared by GraphRAG and the agent.

The interpretation is a *proposal*; every slot is validated against the graph afterwards.
"""
from __future__ import annotations

from ..contracts import LLMCallUsage
from ..llm.client import complete_json

INTENTS = ["lookup_field", "count_events", "max_events", "event_by_venue_date", "previous_edition_winner", "open_question"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "sport": {"type": "string", "description": "sport as written in corpus titles, e.g. 'Athletics', 'Cross-country skiing'; empty if unknown"},
        "year": {"type": "integer", "description": "edition year mentioned in the question (0 if none)"},
        "season": {"type": "string", "enum": ["Summer", "Winter", ""]},
        "event_name": {"type": "string", "description": "event name as it would appear after the dash in a title, e.g. \"Men's 20 kilometres walk\"; empty if none"},
        "venue": {"type": "string", "description": "venue string exactly as written in the question, or empty"},
        "date": {"type": "string", "description": "date string exactly as written in the question, or empty"},
        "field": {"type": "string", "enum": ["nations", "competitors", "gold", "silver", "bronze", "venue", "date", "win_value", ""]},
        "threshold": {"type": "integer", "description": "numeric threshold for count questions (0 if none)"},
        "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "==", ""]},
        "temporal_relation": {"type": "string", "enum": ["previous_edition", "next_edition", "same", ""]},
        "entities": {"type": "array", "items": {"type": "string"}, "description": "other named entities (people, films, organisations) mentioned"},
        "answer_type": {"type": "string", "enum": ["number", "person_or_team", "event_title", "text"]},
        "notes": {"type": "string"},
    },
}

SYSTEM = """You interpret questions about a frozen corpus of Wikipedia articles (mostly Olympic event pages titled
'<Sport> at the <Year> <Summer|Winter> Olympics – <Event>', plus films, people and organisations).
Map the question to ONE intent and fill the slots literally from the question. Do not answer the question.

Intents:
- lookup_field: a single field of one named event (e.g. 'How many nations competed in <title>?' -> field=nations).
- count_events: 'how many <sport> events at the <year> <season> Olympics had more than N competitors' -> threshold, operator.
- max_events: 'which <sport> event at the <year> <season> Olympics had the highest number of competitors'.
- event_by_venue_date: 'who won the gold medal in the event held at <venue> on <date> [at the <year> <season> Olympics]' -> venue, date, field=gold.
- previous_edition_winner: 'who won the gold medal in the <event> <sport> event at the <season> Olympics held immediately before <year>'
  -> sport, event_name, year (the reference year written in the question), temporal_relation=previous_edition, field=gold.
- open_question: anything else (films, people, general prose questions).

Slot rules: copy venue and date strings verbatim (including odd spacing); for event_name give the event part only
(e.g. "Men's 20 kilometres walk", "Women's 57 kg", "Individual normal hill/10 km"); sport is the sport name only
(e.g. 'Athletics', 'Judo', 'Nordic combined', 'Cross-country skiing', 'Short-track speed skating')."""


def interpret(question: str) -> tuple[dict, LLMCallUsage]:
    data, usage = complete_json("interpret", SYSTEM, f"Question: {question}", SCHEMA, max_tokens=600, effort="low")
    data.setdefault("intent", "open_question")
    data["year"] = int(data.get("year") or 0)
    data["threshold"] = int(data.get("threshold") or 0)
    return data, usage
