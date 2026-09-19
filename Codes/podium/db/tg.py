"""Small explicit TigerGraph adapter (REST++ + GSQL endpoints) with typed helpers.

Authentication: a Savanna *database secret* is exchanged for a bearer token; username /
password basic auth is the fallback. Runtime queries are read-only installed queries.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx

from ..settings import settings


class TigerGraphError(RuntimeError):
    pass


class TG:
    def __init__(self, host: str | None = None, graph: str | None = None, secret: str | None = None, username: str | None = None, password: str | None = None):
        self.host = (host or settings.tg_host or "").rstrip("/")
        self.graph = graph or settings.tg_graph
        self.secret = secret if secret is not None else settings.tg_secret
        self.username = username or settings.tg_username
        self.password = password or settings.tg_password
        self._token: str | None = None
        self._token_exp: float = 0
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0))
        if not self.host:
            raise TigerGraphError("TG_HOST is not configured")

    # ---- auth ---------------------------------------------------------------------------
    def _basic(self) -> tuple[str, str] | None:
        if self.username and self.password:
            return (self.username, self.password)
        return None

    def token(self) -> str | None:
        if not self.secret:
            return None
        with self._lock:
            if self._token and time.time() < self._token_exp - 300:
                return self._token
            last = None
            for url, body in (
                (f"{self.host}/gsql/v1/tokens", {"secret": self.secret, "lifetime": 86400 * 7, "graph": self.graph}),
                (f"{self.host}/restpp/requesttoken", {"secret": self.secret, "lifetime": 86400 * 7}),
            ):
                try:
                    r = self._client.post(url, json=body, headers={"Content-Type": "application/json"})
                    d = r.json()
                    tok = d.get("token") or (d.get("results") or {}).get("token")
                    if tok:
                        self._token = tok
                        self._token_exp = time.time() + 86400 * 7
                        return tok
                    last = d
                except Exception as e:  # noqa: BLE001
                    last = str(e)
            raise TigerGraphError(f"could not obtain token from secret: {last}")

    def _headers(self) -> dict[str, str]:
        tok = self.token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def _auth_kwargs(self) -> dict[str, Any]:
        tok = self.token()
        if tok:
            return {"headers": {"Authorization": f"Bearer {tok}"}}
        b = self._basic()
        if b:
            return {"auth": b}
        raise TigerGraphError("no TG credentials: set TG_SECRET or TG_USERNAME/TG_PASSWORD")

    # ---- endpoints --------------------------------------------------------------------
    def gsql(self, command: str, graph: str | None = None) -> str:
        kw = self._auth_kwargs()
        headers = dict(kw.pop("headers", {}))
        headers["Content-Type"] = "text/plain"
        params = {}
        if graph:
            params["graph"] = graph
        r = self._client.post(f"{self.host}/gsql/v1/statements", content=command.encode("utf-8"), headers=headers, params=params, timeout=1800.0, **kw)
        text = r.text
        if r.status_code >= 400:
            raise TigerGraphError(f"GSQL HTTP {r.status_code}: {text[:2000]}")
        return text

    def rest(self, method: str, path: str, *, json_body: Any = None, params: dict | None = None, timeout: float = 180.0) -> dict:
        kw = self._auth_kwargs()
        r = self._client.request(method, f"{self.host}{path}", json=json_body, params=params, timeout=timeout, **kw)
        try:
            d = r.json()
        except Exception as e:  # noqa: BLE001
            raise TigerGraphError(f"non-JSON response {r.status_code}: {r.text[:500]}") from e
        if r.status_code >= 400 or d.get("error"):
            raise TigerGraphError(f"{method} {path}: {d.get('message') or d}"[:2000])
        return d

    def version(self) -> dict:
        return self.rest("GET", "/restpp/version")

    def ping(self) -> bool:
        try:
            self.rest("GET", "/restpp/echo")
            return True
        except Exception:  # noqa: BLE001
            return False

    def run_query(self, name: str, params: dict | None = None, timeout: float = 120.0) -> list[dict]:
        """Run an installed query; returns the list of PRINT results."""
        d = self.rest("POST", f"/restpp/query/{self.graph}/{name}", json_body=params or {}, timeout=timeout)
        return d.get("results", [])

    def upsert(self, vertices: dict | None = None, edges: dict | None = None, timeout: float = 600.0) -> dict:
        payload: dict[str, Any] = {}
        if vertices:
            payload["vertices"] = vertices
        if edges:
            payload["edges"] = edges
        return self.rest("POST", f"/restpp/graph/{self.graph}", json_body=payload, timeout=timeout)

    def vertex_count(self, vtype: str) -> int:
        d = self.rest("POST", f"/restpp/builtins/{self.graph}", json_body={"function": "stat_vertex_number", "type": vtype})
        res = d.get("results", [])
        return int(res[0]["count"]) if res else 0

    def edge_count(self, etype: str) -> int:
        d = self.rest("POST", f"/restpp/builtins/{self.graph}", json_body={"function": "stat_edge_number", "type": etype})
        res = d.get("results", [])
        return int(res[0]["count"]) if res else 0


_default: TG | None = None


def get_tg() -> TG:
    global _default
    if _default is None:
        _default = TG()
    return _default


def results_by_name(results: list[dict]) -> dict[str, Any]:
    """Flatten the list of PRINT dicts into one dict keyed by alias."""
    out: dict[str, Any] = {}
    for r in results:
        out.update(r)
    return out


def vattrs(v: dict) -> dict:
    """Vertex JSON -> flat attribute dict including the id."""
    a = {}
    for k, val in v.get("attributes", {}).items():
        # attribute-selected prints are prefixed with the alias ("v.chunk_id"); strip it
        a[k.split(".", 1)[1] if "." in k and " " not in k else k] = val
    a.setdefault("v_id", v.get("v_id"))
    a["_type"] = v.get("v_type")
    return a


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
