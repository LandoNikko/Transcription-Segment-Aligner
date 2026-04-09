@echo off
setlocal EnableDelayedExpansion
rem Always run from this script's folder (fixes wrong venv/python if launched elsewhere)
cd /d "%~dp0"

title Transcription Segment Aligner
echo ==============================================
echo   Starting Transcription Segment Aligner backend
echo ==============================================

echo Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
  if not "%%a"=="" if not "%%a"=="0" (
    echo Stopping old process PID %%a ...
    taskkill /PID %%a /F >nul 2>&1
  )
)
timeout /t 1 /nobreak >nul

rem Use venv python directly — activate.bat hardcodes the path from venv creation and breaks after rename
start "Transcription Segment Aligner — Backend" /D "%~dp0" cmd /k "venv\Scripts\python.exe app.py"

set "FRONTEND_URL=http://127.0.0.1:8000/"
set /a MAX_ATTEMPTS=120
set /a N=0

echo Waiting until the backend answers at %FRONTEND_URL% ...

:wait_ready
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%FRONTEND_URL%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) { exit 0 } } catch { } exit 1"
if !ERRORLEVEL! EQU 0 goto open_browser

set /a N+=1
if !N! GEQ %MAX_ATTEMPTS% (
  echo.
  echo Gave up after %MAX_ATTEMPTS% checks — server never became ready.
  echo Check the "Transcription Segment Aligner — Backend" window for errors.
  pause
  exit /b 1
)
echo   ... still waiting (attempt !N! / %MAX_ATTEMPTS%^)
timeout /t 1 /nobreak >nul
goto wait_ready

:open_browser
echo Opening the web interface...
start "" "%FRONTEND_URL%"
