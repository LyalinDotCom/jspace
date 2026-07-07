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

if [ ! -e model/gemma-4-E4B/config.json ]; then
  echo "model missing — downloading official google/gemma-4-E4B (~16 GB)."
  echo "(offline alternative: npm run import-model -- --source ollama)"
  .venv/bin/python scripts/import_model.py --source hf
else
  echo "model present: model/gemma-4-E4B"
fi

echo "setup done. run: npm run dev"
