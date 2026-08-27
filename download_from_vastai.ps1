# PowerShell Downloader for Vast.ai Artifacts & Telemetry
param (
    [string]$SshCmd = "",
    [string]$Port = "",
    [string]$HostName = "",
    [switch]$TelemetryOnly,
    [switch]$CheckpointsOnly,
    [switch]$All
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "   📥 Vast.ai -> Local PC Artifact Downloader (PowerShell)    " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

if ($SshCmd -ne "") {
    if ($SshCmd -match "-p\s+(\d+)") { $Port = $Matches[1] }
    if ($SshCmd -match "([a-zA-Z0-9_\-]+@[a-zA-Z0-9\.\-]+)") { $HostName = $Matches[1] }
}

if (-not $Port -or -not $HostName) {
    $inputStr = Read-Host "Paste your Vast.ai SSH command (e.g. ssh -p 12345 root@ssh5.vast.ai)"
    if ($inputStr -match "-p\s+(\d+)") { $Port = $Matches[1] }
    if ($inputStr -match "([a-zA-Z0-9_\-]+@[a-zA-Z0-9\.\-]+)") { $HostName = $Matches[1] }
}

if (-not $Port -or -not $HostName) {
    Write-Host "Error: Could not extract Port and Host." -ForegroundColor Red
    exit 1
}

$destRuns = Join-Path $PSScriptRoot "runs"
$destCheckpoints = Join-Path $PSScriptRoot "checkpoints"
New-Item -ItemType Directory -Force -Path $destRuns | Out-Null
New-Item -ItemType Directory -Force -Path $destCheckpoints | Out-Null

Write-Host "`nTarget Instance: $HostName (Port: $Port)" -ForegroundColor Yellow

# 1. Telemetry CSV
if ($TelemetryOnly -or $All -or (-not $CheckpointsOnly)) {
    $csvTarget = Join-Path $destRuns "training_telemetry.csv"
    Write-Host "--> Downloading training_telemetry.csv..." -ForegroundColor Green
    scp -P $Port "${HostName}:/workspace/runs/training_telemetry.csv" "$csvTarget"
}

# 2. Checkpoints
if ($CheckpointsOnly -or $All -or (-not $TelemetryOnly)) {
    Write-Host "--> Downloading best & latest checkpoints..." -ForegroundColor Green
    scp -P $Port "${HostName}:/workspace/checkpoints/ppo_carla_best.pth" "$destCheckpoints\ppo_carla_best.pth"
    scp -P $Port "${HostName}:/workspace/checkpoints/ppo_carla_latest.pth" "$destCheckpoints\ppo_carla_latest.pth"
    scp -P $Port "${HostName}:/workspace/checkpoints/train_state.json" "$destCheckpoints\train_state.json"
}

Write-Host "`n✓ Finished downloading to $PSScriptRoot!" -ForegroundColor Cyan
