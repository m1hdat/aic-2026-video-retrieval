param(
  [Parameter(Mandatory=$true)][string[]]$Features,
  [Parameter(Mandatory=$true)][string]$Maps
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$ArgsList = @("-m","scripts.ingest")
foreach ($Path in $Features) { $ArgsList += @("--features",$Path) }
$ArgsList += @("--maps",$Maps)
& ".\.venv\Scripts\python.exe" @ArgsList

