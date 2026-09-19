#!/usr/bin/env bash
# Start (or keep alive) the Podium API + dashboard on 127.0.0.1:8120 (proxied by Caddy at
# https://tigeranalytics.duckdns.org). Exits if already listening, so it is safe to re-run.
set -u
cd "$(dirname "$0")/.."
if curl -sf -m 5 http://127.0.0.1:8120/api/health >/dev/null 2>&1; then exit 0; fi
pkill -f "uvicorn podium.api:app" 2>/dev/null || true
mkdir -p artifacts/cache
if [ -f scripts/env_active.sh ]; then . scripts/env_active.sh; fi
nohup .venv/bin/uvicorn podium.api:app --host 127.0.0.1 --port 8120 --workers 1 >> artifacts/cache/api.log 2>&1 &
echo "started podium api pid $!"
