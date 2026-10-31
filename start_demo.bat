@echo off
rem RunwayEye: start the server and open the Command Center in the browser.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title RunwayEye

if not exist "venv\Scripts\python.exe" (
  echo [RunwayEye] Prima pornire: instalez mediul Python ^(cateva minute^).
  py -3 scripts\setup.py || python scripts\setup.py || (
    echo [RunwayEye] Nu gasesc Python 3.11+. Instalati-l de pe python.org ^(bifati "Add to PATH"^).
    pause
    exit /b 1
  )
)

if not exist ".env" (
  copy ".env.example" ".env" >nul 2>&1 && (
    echo [RunwayEye] Am creat .env din .env.example. Verifica VIDEO_SOURCE in .env.
  ) || (
    echo [RunwayEye] Lipseste .env.example, pornesc cu valorile implicite.
  )
)

if not exist "dataideos\synthetic_demo.mp4" "venv\Scripts\python.exe" scripts\make_demo_clip.py

if not exist "app\static\css\app.css" (
  echo [RunwayEye] Lipseste CSS-ul compilat. Ruleaza o data: npm install
)

call "venv\Scripts\activate.bat"

for /f "usebackq delims=" %%i in (`python -c "from app.config import get_settings as g; print(g().port)"`) do set "RE_PORT=%%i"
if "%RE_PORT%"=="" set "RE_PORT=8000"

rem Open the dashboard a few seconds after the server starts (ping = portable 4 s wait).
start "" /min cmd /c "ping -n 5 127.0.0.1 >nul & start http://localhost:%RE_PORT%/"

python -m app

echo.
echo [RunwayEye] Serverul s-a oprit.
pause
