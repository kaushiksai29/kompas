# GraphRAG one-command setup (Windows / PowerShell)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Creates the project virtualenv, installs backend deps into it, and seeds
# .env from .env.example. The venv interpreter is what every other command
# must use — the system / MS Store python is a broken stub.

$ErrorActionPreference = "Stop"

# Resolve repo root (parent of this scripts/ dir) and run from there.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# Find a real python to bootstrap the venv (prefer the launcher).
function Get-Bootstrap-Python {
    foreach ($cand in @("py -3", "python", "python3")) {
        $parts = $cand.Split(" ")
        $exe = $parts[0]
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            return $cand
        }
    }
    return $null
}

# 1. Create venv if missing.
if (-not (Test-Path $VenvPy)) {
    Write-Host "[1/3] Creating virtualenv at .venv ..."
    $boot = Get-Bootstrap-Python
    if ($null -eq $boot) {
        Write-Error "No python found on PATH to bootstrap the venv. Install Python 3.11+ from python.org (not the Microsoft Store)."
        exit 1
    }
    $bp = $boot.Split(" ")
    & $bp[0] $bp[1..($bp.Length-1)] -m venv .venv
} else {
    Write-Host "[1/3] virtualenv already exists, skipping."
}

# 2. Install backend requirements into the venv.
Write-Host "[2/3] Installing backend requirements ..."
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r backend/requirements.txt

# 3. Seed .env from .env.example if absent.
if (-not (Test-Path ".env")) {
    Write-Host "[3/3] Creating .env from .env.example ..."
    Copy-Item ".env.example" ".env"
    Write-Host "      -> edit .env and set GROQ_API_KEY (only Groq is wired up today)."
} else {
    Write-Host "[3/3] .env already exists, leaving it untouched."
}

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. Put your GROQ_API_KEY in .env"
Write-Host "  2. make corpus     (or: .venv\Scripts\python.exe scripts\download_corpus.py)"
Write-Host "  3. make ingest     (or: .venv\Scripts\python.exe scripts\ingest_corpus.py)"
Write-Host "  4. make api        (FastAPI on :8000)"
Write-Host "  5. make ui         (Next.js on :3000)"
