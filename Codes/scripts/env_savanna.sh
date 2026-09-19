# source this to point Podium at the TigerGraph Savanna workspace (values come from Codes/.env:
# TG_HOST, TG_SECRET, TG_GRAPH). Requires a Database Secret created in Savanna -> Database Secrets.
unset TG_USERNAME TG_PASSWORD
export TG_HOST="$(grep '^TG_HOST=' "$(dirname "${BASH_SOURCE[0]}")/../.env" | cut -d= -f2- | tr -d '"\r ')"
export TG_SECRET="$(grep '^TG_SECRET=' "$(dirname "${BASH_SOURCE[0]}")/../.env" | cut -d= -f2- | tr -d '"\r ')"
