$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
docker compose up -d
docker compose exec -T postgres psql -U aic -d aic2026 -f /docker-entrypoint-initdb.d/001_schema.sql
& ".\.venv\Scripts\python.exe" -m scripts.init_milvus

