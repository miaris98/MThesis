# PowerShell Helper to create an SSH Port Forward Tunnel for MLflow & TensorBoard
param (
    [string]$SshCmd = "",
    [string]$Port = "",
    [string]$HostName = ""
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "   🌐 Vast.ai SSH Tunnel for MLflow (Port 10100) & TensorBoard " -ForegroundColor Cyan
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
    Write-Host "Error: Could not extract Port and Host from input." -ForegroundColor Red
    exit 1
}

Write-Host "`nTarget Instance: $HostName (Port: $Port)" -ForegroundColor Yellow
Write-Host "Forwarding ports:" -ForegroundColor Green
Write-Host "  👉 MLflow UI:     http://localhost:10100" -ForegroundColor Green
Write-Host "  👉 TensorBoard:   http://localhost:6006" -ForegroundColor Green
Write-Host "`nOpening http://localhost:10100 in your browser..." -ForegroundColor Cyan

Start-Process "http://localhost:10100"

Write-Host "`n[NOTE] Keep this window OPEN while viewing the dashboards." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the tunnel.`n" -ForegroundColor DarkGray

# Launch SSH port forward
ssh -p $Port $HostName -L 10100:127.0.0.1:10100 -L 6006:127.0.0.1:6006 -N
