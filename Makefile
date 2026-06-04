# GraphRAG — Jurisdiction-Aware Regulatory Navigator
#
# One-command setup and run targets. Every Python target uses the project
# virtualenv interpreter (NOT system python — the MS Store stub is broken).
#
# Windows (PowerShell / cmd with GNU make) vs. *nix is auto-detected below.
# If you don't have `make` on Windows, run scripts\setup.ps1 instead — it
# covers the `setup` target end-to-end.

ifeq ($(OS),Windows_NT)
    VENV_PY := .venv\Scripts\python.exe
    NPM     := npm
else
    VENV_PY := .venv/bin/python
    NPM     := npm
endif

.PHONY: help setup corpus ingest api ui eval test

help:
	@echo "Targets:"
	@echo "  make setup    create .venv, install backend deps, create .env from .env.example"
	@echo "  make corpus   download the USCIS PDF corpus"
	@echo "  make ingest   build vectors + typed knowledge graph"
	@echo "  make api      run the FastAPI backend on :8000"
	@echo "  make ui       run the Next.js frontend on :3000"
	@echo "  make eval     reproduce the evaluation table"
	@echo "  make test     run the e2e + integration tests"

# --- one-command setup ------------------------------------------------------
# On Windows this delegates to the PowerShell script; on *nix to the bash one.
setup:
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
else
	bash scripts/setup.sh
endif

# --- pipeline ---------------------------------------------------------------
corpus:
	$(VENV_PY) scripts/download_corpus.py

# Pass extra args with GLOB, e.g. `make ingest GLOB="*.pdf"`.
ingest:
	$(VENV_PY) scripts/ingest_corpus.py $(if $(GLOB),--glob "$(GLOB)",)

# --- run --------------------------------------------------------------------
api:
	$(VENV_PY) -m uvicorn backend.app:app --port 8000

ui:
	$(NPM) --prefix frontend install
	$(NPM) --prefix frontend run dev

# --- eval & test ------------------------------------------------------------
eval:
	$(VENV_PY) -m backend.eval.run_eval

test:
	$(VENV_PY) -m pytest tests/ -v
