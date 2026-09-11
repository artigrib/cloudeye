#!/bin/bash
# Dev entrypoint. Production runs uvicorn from deploy/video-api.service instead.
#
# .env is required: every setting has a hardcoded fallback in app/config.py, so a missing
# .env would silently start against those defaults rather than your configuration.
# The OpenRouter secret file is optional - without it, VLM calls are simply disabled.
set -a
source .env
[ -f "${CLOUDEYE_SECRETS_FILE:-$HOME/.secrets/openrouter.env}" ] && \
  source "${CLOUDEYE_SECRETS_FILE:-$HOME/.secrets/openrouter.env}"
set +a
exec uv run uvicorn app.main:app --host 127.0.0.1 --port "${API_PORT:-8000}"
