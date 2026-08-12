$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
docker compose up -d postgres
docker compose exec -T postgres psql -U aic -d aic2026 -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/002_ocr_objects.sql
Write-Host "Migration OCR/Object completed. Existing SigLIP2 and keyframe rows were preserved."
