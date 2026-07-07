#!/usr/bin/env bash
# One-shot setup: python venv + deps, node deps for root and web/.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

npm install

if [ ! -e model/gemma-4-E4B-it/config.json ] && [ ! -e model/gemma-4-E4B/config.json ]; then
  echo "no model found — downloading official google/gemma-4-E4B-it (~16 GB),"
  echo "the instruct model the chat UI uses by default."
  echo "(base model: npm run import-model | offline: npm run import-model -- --source ollama)"
  .venv/bin/python scripts/import_model.py --source hf \
    --repo google/gemma-4-E4B-it --out model/gemma-4-E4B-it
else
  ls -d model/gemma-4-E4B* 2>/dev/null | sed 's/^/model present: /'
fi

echo "setup done. run: npm run dev"
