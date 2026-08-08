$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
docker compose ps
& ".\.venv\Scripts\python.exe" -m scripts.verify_stores
