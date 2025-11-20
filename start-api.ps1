# Quick launcher for the API on Windows PowerShell.
# Assumes .venv exists and dependencies are installed.

$venvActivate = ".\.venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "Virtual env not found. Create it first: python -m venv .venv"
    exit 1
}

& $venvActivate
uvicorn api:app --reload
