$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
docker compose up -d
Write-Host "Cho Docker khoi dong 20-40 giay, sau do chay .\INIT_DB_AND_MILVUS.ps1"

