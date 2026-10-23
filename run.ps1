# Start the AI Platform GeoAI server.
#   Works from any directory:   C:\path\to\AI_Platfrom\run.ps1   (or double-click run.bat)
#   Picks the next free port if the requested one is taken (e.g. by Docker), then opens the browser.
param([int]$Port = 8000, [string]$BindHost = "127.0.0.1", [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "First run: creating the Python environment (this takes a few minutes)..." -ForegroundColor Cyan
    python -m venv (Join-Path $root ".venv")
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $root "backend\requirements.txt")
}

function Test-PortInUse([int]$p) {
    # Busy if anything accepts connections on IPv4 or IPv6 loopback (Docker/WSL often hold ::1),
    # or if the port cannot be bound.
    foreach ($addr in "127.0.0.1", "::1") {
        $client = [System.Net.Sockets.TcpClient]::new([System.Net.Sockets.AddressFamily]$(if ($addr -eq "::1") { "InterNetworkV6" } else { "InterNetwork" }))
        try {
            if ($client.ConnectAsync($addr, $p).Wait(300) -and $client.Connected) { return $true }
        } catch { } finally { $client.Dispose() }
    }
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($BindHost), $p)
        $listener.ExclusiveAddressUse = $true
        $listener.Start(); $listener.Stop()
        return $false
    } catch { return $true }
}

$chosen = $null
foreach ($p in $Port..($Port + 20)) {
    if (-not (Test-PortInUse $p)) { $chosen = $p; break }
    Write-Host "Port $p is in use by another program, trying the next one..." -ForegroundColor Yellow
}
if (-not $chosen) { throw "No free port between $Port and $($Port + 20)." }

$url = "http://${BindHost}:$chosen"
Write-Host ""
Write-Host "  AI Platform GeoAI  ->  $url" -ForegroundColor Green
Write-Host "  Keep this window open while you use the app. Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    # open the browser once the API answers
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 120; $i++) {
            try { Invoke-RestMethod "$u/api/health" -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep -Milliseconds 500 }
        }
    } -ArgumentList $url | Out-Null
}

& $py -m uvicorn geoai.api:app --app-dir (Join-Path $root "backend") --host $BindHost --port $chosen
