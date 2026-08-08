$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& ".\.venv\Scripts\python.exe" -m compileall -q frontend src scripts tests
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
docker compose config --quiet
Write-Host "Project checks passed." -ForegroundColor Green
