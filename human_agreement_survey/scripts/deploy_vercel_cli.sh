#!/bin/bash
# CLI-only deploy. No Git integration.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
[ -f .key ] && . ./.key
set +a
python3 scripts/write_config_from_key.py
if ! command -v vercel >/dev/null 2>&1; then
  echo "vercel CLI not found. Install once with: npm i -g vercel" >&2
  exit 1
fi
if [ -n "${VERCEL_TOKEN:-}" ]; then
  if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SECRET_KEY:-}" ]; then
    for env_name in SUPABASE_URL SUPABASE_SECRET_KEY; do
      vercel env rm "$env_name" production --yes --token "$VERCEL_TOKEN" >/dev/null 2>&1 || true
      printf "%s" "${!env_name}" | vercel env add "$env_name" production --token "$VERCEL_TOKEN" >/dev/null
    done
  fi
  vercel deploy --prod --yes --token "$VERCEL_TOKEN"
else
  cat >&2 <<EOF
VERCEL_TOKEN is not set in .key.
The Vercel AI Gateway key is not a Vercel deploy token. Create a Vercel Account Token
(Account Settings -> Tokens), put it in .key as VERCEL_TOKEN=..., then rerun this script.
EOF
  exit 2
fi
