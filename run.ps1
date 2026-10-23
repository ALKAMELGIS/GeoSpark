param([int]$Port = 8000, [string]$BindHost = "127.0.0.1")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$root\.venv\Scripts\python.exe" -m uvicorn geoai.api:app --app-dir "$root\backend" --host $BindHost --port $Port
