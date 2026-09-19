"""Thin provider adapter: structured JSON completions + embeddings with usage capture.

Providers: `anthropic` (official SDK), `openai` / `deepseek` / `gemini` (OpenAI-compatible).
Every call returns an `LLMCallUsage` so pipelines account for all tokens, including
retries and failures (`usage_unknown=True`, never a fabricated zero).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from ..contracts import LLMCallUsage
from ..settings import ModelSpec, settings

_clients: dict[str, Any] = {}
_lock = threading.Lock()


def _anthropic():
    import anthropic

    with _lock:
        if "anthropic" not in _clients:
            headers = {}
            if settings.anthropic_workspace:
                headers["anthropic-workspace-id"] = settings.anthropic_workspace
            _clients["anthropic"] = anthropic.Anthropic(api_key=settings.anthropic_key, default_headers=headers, max_retries=3, timeout=120.0)
        return _clients["anthropic"]


def _openai_compatible(provider: str):
    from openai import OpenAI

    with _lock:
        if provider not in _clients:
            if provider == "openai":
                _clients[provider] = OpenAI(api_key=settings.openai_key, max_retries=3, timeout=120.0)
            elif provider == "deepseek":
                _clients[provider] = OpenAI(api_key=settings.deepseek_key, base_url="https://api.deepseek.com", max_retries=3, timeout=120.0)
            elif provider == "gemini":
                _clients[provider] = OpenAI(
                    api_key=settings.gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", max_retries=3, timeout=120.0
                )
            else:
                raise ValueError(f"unknown provider {provider}")
        return _clients[provider]


def _strictify(schema: dict) -> dict:
    """OpenAI strict mode needs additionalProperties=false and all keys required."""
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        schema = {**schema, "additionalProperties": False, "required": list(props.keys()), "properties": {k: _strictify(v) for k, v in props.items()}}
    elif schema.get("type") == "array" and "items" in schema:
        schema = {**schema, "items": _strictify(schema["items"])}
    return schema


def complete_json(
    operation: str,
    system: str,
    user: str,
    schema: dict,
    *,
    model: ModelSpec | None = None,
    max_tokens: int = 2000,
    context_tokens: int = 0,
    effort: str = "low",
    temperature: float = 0.0,
) -> tuple[dict, LLMCallUsage]:
    """Return (parsed_json, usage). Raises RuntimeError after retries with usage recorded."""
    spec = model or settings.answer_model
    schema = _strictify(schema)
    t0 = time.time()
    usage = LLMCallUsage(provider=spec.provider, model_id=spec.model_id, operation=operation, context_tokens=context_tokens)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            if spec.provider == "anthropic":
                client = _anthropic()
                kwargs: dict[str, Any] = dict(
                    model=spec.model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                )
                if "haiku" not in spec.model_id:
                    kwargs["output_config"]["effort"] = effort
                r = client.messages.create(**kwargs)
                text = next((b.text for b in r.content if b.type == "text"), "")
                usage.input_tokens = r.usage.input_tokens
                usage.output_tokens = r.usage.output_tokens
                usage.cached_input_tokens = getattr(r.usage, "cache_read_input_tokens", 0) or 0
                usage.request_id = getattr(r, "id", None)
                if r.stop_reason == "max_tokens":
                    raise RuntimeError("max_tokens reached")
            else:
                client = _openai_compatible(spec.provider)
                kwargs = dict(
                    model=spec.model_id,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": operation, "schema": schema, "strict": True}},
                )
                if spec.model_id.startswith(("gpt-5", "o3", "o4")):
                    kwargs["max_completion_tokens"] = max_tokens + 4000
                    if spec.provider == "openai" and not spec.model_id.endswith("chat-latest"):
                        kwargs["reasoning_effort"] = "low" if effort == "low" else "medium"
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["temperature"] = temperature
                r = client.chat.completions.create(**kwargs)
                text = r.choices[0].message.content or ""
                if r.usage:
                    usage.input_tokens = r.usage.prompt_tokens
                    usage.output_tokens = r.usage.completion_tokens
                    ptd = getattr(r.usage, "prompt_tokens_details", None)
                    ctd = getattr(r.usage, "completion_tokens_details", None)
                    usage.cached_input_tokens = getattr(ptd, "cached_tokens", 0) or 0 if ptd else 0
                    usage.reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0 if ctd else 0
                usage.request_id = getattr(r, "id", None)
            usage.latency_ms = (time.time() - t0) * 1000
            return json.loads(text), usage
        except Exception as e:  # noqa: BLE001 - we record and retry
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    usage.latency_ms = (time.time() - t0) * 1000
    usage.usage_unknown = usage.input_tokens is None
    usage.error = f"{type(last_err).__name__}: {str(last_err)[:300]}"
    raise RuntimeError(usage.error) from last_err


def embed_texts(texts: list[str], model: ModelSpec | None = None) -> tuple[list[list[float]], LLMCallUsage]:
    spec = model or settings.embed_model
    t0 = time.time()
    usage = LLMCallUsage(provider=spec.provider, model_id=spec.model_id, operation="embed")
    client = _openai_compatible(spec.provider)
    for attempt in range(4):
        try:
            r = client.embeddings.create(model=spec.model_id, input=texts, dimensions=settings.embed_dim)
            usage.input_tokens = r.usage.prompt_tokens if r.usage else None
            usage.output_tokens = 0
            usage.latency_ms = (time.time() - t0) * 1000
            vecs = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
            return vecs, usage
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                usage.usage_unknown = True
                usage.error = f"{type(e).__name__}: {str(e)[:200]}"
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def embed_query(text: str) -> tuple[list[float], LLMCallUsage]:
    vecs, usage = embed_texts([text])
    return vecs[0], usage
