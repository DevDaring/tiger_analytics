"""Central configuration. Secrets come from `.env`; everything else has a default.

No question/answer file is ever read here: the settings only know where the corpus
lives and where artifacts are written.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATASET_DIR = ROOT / "Dataset" / "hackathon-resources"
CORPUS_PATH = DATASET_DIR / "corpus" / "corpus.jsonl"
PUBLIC_QUESTIONS = DATASET_DIR / "questions" / "eval_public.jsonl"
HIDDEN_QUESTIONS = DATASET_DIR / "questions" / "eval_hidden.jsonl"
ARTIFACTS = ROOT / "artifacts"
MANIFESTS = ARTIFACTS / "manifests"
RUNS = ARTIFACTS / "runs"
CACHE = ARTIFACTS / "cache"
GSQL_DIR = ROOT / "podium" / "db" / "gsql"

for _p in (MANIFESTS, RUNS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

PARSER_VERSION = "parser-v3"
CHUNKER_VERSION = "chunker-v2"
EXTRACTION_VERSION = "extract-v3"


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


@dataclass
class ModelSpec:
    """`provider:model_id` strings, e.g. `anthropic:claude-sonnet-5`."""

    provider: str
    model_id: str

    @classmethod
    def parse(cls, s: str) -> "ModelSpec":
        if ":" not in s:
            raise ValueError(f"model spec must be provider:model, got {s!r}")
        p, m = s.split(":", 1)
        return cls(p.strip(), m.strip())

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.provider}:{self.model_id}"


@dataclass
class Settings:
    tg_host: str | None = field(default_factory=lambda: env("TG_HOST"))
    tg_secret: str | None = field(default_factory=lambda: env("TG_SECRET"))
    tg_username: str | None = field(default_factory=lambda: env("TG_USERNAME"))
    tg_password: str | None = field(default_factory=lambda: env("TG_PASSWORD"))
    tg_graph: str = field(default_factory=lambda: env("TG_GRAPH", "Podium") or "Podium")

    answer_model: ModelSpec = field(
        default_factory=lambda: ModelSpec.parse(env("PODIUM_ANSWER_MODEL", "anthropic:claude-sonnet-5"))
    )
    judge_model: ModelSpec = field(default_factory=lambda: ModelSpec.parse(env("PODIUM_JUDGE_MODEL", "openai:gpt-4.1")))
    embed_model: ModelSpec = field(
        default_factory=lambda: ModelSpec.parse(env("PODIUM_EMBED_MODEL", "openai:text-embedding-3-small"))
    )
    embed_dim: int = int(env("PODIUM_EMBED_DIM", "1536") or 1536)

    openai_key: str | None = field(default_factory=lambda: env("Open_AI_2009_Key") or env("OPENAI_API_KEY"))
    anthropic_key: str | None = field(default_factory=lambda: env("Calude_2009_API_Key") or env("ANTHROPIC_API_KEY"))
    anthropic_workspace: str | None = field(default_factory=lambda: env("Claude_workspace_id"))
    deepseek_key: str | None = field(default_factory=lambda: env("DEEPSEEK_API_KEY_1"))
    gemini_key: str | None = field(default_factory=lambda: env("GEMINI_API_KEY_1"))

    # Agent limits (plan section 9); calibrated on the pilot.
    agent_max_decisions: int = 8
    agent_max_tool_calls: int = 12
    agent_max_repairs: int = 2
    agent_token_budget: int = 40_000
    agent_wall_clock_s: float = 90.0
    rag_top_k: int = 8
    context_char_budget: int = 18_000  # ~4.5k tokens of evidence per model call

    def redacted(self) -> dict:
        return {
            "tg_host": self.tg_host,
            "tg_graph": self.tg_graph,
            "tg_secret": "set" if self.tg_secret else "missing",
            "tg_password": "set" if self.tg_password else "missing",
            "answer_model": str(self.answer_model),
            "judge_model": str(self.judge_model),
            "embed_model": str(self.embed_model),
            "openai_key": "set" if self.openai_key else "missing",
            "anthropic_key": "set" if self.anthropic_key else "missing",
        }


settings = Settings()
