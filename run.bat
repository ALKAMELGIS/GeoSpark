@echo off
rem Double-click to start AI Platform GeoAI (opens the browser automatically).
where pwsh >nul 2>nul && (pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*) || (powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*)
pause
