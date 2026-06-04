#!/usr/bin/env bash
# GraphRAG one-command setup (macOS / Linux)
#
#   bash scripts/setup.sh
#
# Creates the project virtualenv, installs backend deps into it, and seeds
# .env from .env.example. Every other command must use the venv interpreter
# at .venv/bin/python.

set -euo pipefail

# Resolve repo root (parent of this scripts/ dir) and run from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

VENV_PY="$REPO_ROOT/.venv/bin/python"

# Pick a bootstrap python.
bootstrap_py=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        bootstrap_py="$cand"
        break
    fi
done

# 1. Create venv if missing.
if [ ! -x "$VENV_PY" ]; then
    echo "[1/3] Creating virtualenv at .venv ..."
    if [ -z "$bootstrap_py" ]; then
        echo "ERROR: No python found on PATH to bootstrap the venv. Install Python 3.11+." >&2
        exit 1
    fi
    "$bootstrap_py" -m venv .venv
else
    echo "[1/3] virtualenv already exists, skipping."
fi

# 2. Install backend requirements into the venv.
echo "[2/3] Installing backend requirements ..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r backend/requirements.txt

# 3. Seed .env from .env.example if absent.
if [ ! -f .env ]; then
    echo "[3/3] Creating .env from .env.example ..."
    cp .env.example .env
    echo "      -> edit .env and set GROQ_API_KEY (only Groq is wired up today)."
else
    echo "[3/3] .env already exists, leaving it untouched."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Put your GROQ_API_KEY in .env"
echo "  2. make corpus     (or: .venv/bin/python scripts/download_corpus.py)"
echo "  3. make ingest     (or: .venv/bin/python scripts/ingest_corpus.py)"
echo "  4. make api        (FastAPI on :8000)"
echo "  5. make ui         (Next.js on :3000)"
