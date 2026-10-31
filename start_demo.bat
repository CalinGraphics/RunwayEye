@echo off
rem RunwayEye: start the server and open the Command Center in the browser (double-click).
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title RunwayEye

rem First run on this computer: venv + dependencies + .env + demo clip (backend\scripts\setup.py).
if not exist "venv\Scripts\python.exe" (
  echo [RunwayEye] Prima pornire: instalez mediul Python ^(cateva minute, o singura data^).
  py -3 backend\scripts\setup.py
  if not exist "venv\Scripts\python.exe" python backend\scripts\setup.py
  if not exist "venv\Scripts\python.exe" (
    echo [RunwayEye] Nu gasesc Python 3.11+. Instalati-l de pe python.org ^(bifati "Add to PATH"^), apoi porniti din nou.
    pause
    exit /b 1
  )
)
rem Relative path without spaces on purpose: it can be used unquoted inside for /f below.
set "PY=venv\Scripts\python.exe"
rem The Python code lives in backend\ (package "app").
set "PYTHONPATH=%~dp0backend"

if not exist ".env" copy ".env.example" ".env" >nul

if not exist "data\videos\synthetic_demo.mp4" (
  echo [RunwayEye] Desenez clipul demo folosit cand nu exista camera ^(o singura data, ~1 minut^)...
  %PY% backend\scripts\make_demo_clip.py
)

set "RE_PORT="
for /f "usebackq delims=" %%i in (`%PY% -c "from app.config import get_settings as g; print(g().port)"`) do set "RE_PORT=%%i"
if "%RE_PORT%"=="" set "RE_PORT=8000"

rem Already running (another window, or started earlier)? Then only open the dashboard.
%PY% -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', %RE_PORT%)) == 0 else 1)"
if not errorlevel 1 (
  echo [RunwayEye] Serverul ruleaza deja pe portul %RE_PORT%. Deschid panoul.
  start "" http://localhost:%RE_PORT%/
  ping -n 4 127.0.0.1 >nul
  exit /b 0
)

rem Open the dashboard a few seconds after the server starts (ping = portable wait).
start "" /min cmd /c "ping -n 6 127.0.0.1 >nul & start http://localhost:%RE_PORT%/"

%PY% -m app

echo.
echo [RunwayEye] Serverul s-a oprit.
pause
