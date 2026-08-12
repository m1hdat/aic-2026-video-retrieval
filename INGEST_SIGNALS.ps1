param(
  [Parameter(Mandatory=$false)][string[]]$Objects = @(),
  [Parameter(Mandatory=$false)][string[]]$Ocr = @()
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if ($Objects.Count -eq 0 -and $Ocr.Count -eq 0) { throw "Provide -Objects and/or -Ocr paths." }
& ".\.venv\Scripts\python.exe" -m scripts.ingest_signals --objects $Objects --ocr $Ocr

