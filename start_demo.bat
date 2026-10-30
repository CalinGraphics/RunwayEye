@echo off
rem RunwayEye: start the server and open the Command Center in the browser.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title RunwayEye

if not exist "venv\Scripts\python.exe" (
  echo [RunwayEye] Lipseste mediul Python ^(venv^). Urmeaza pasii din README.md:
  echo     py -3.14 -m venv venv
  echo     venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo [RunwayEye] Am creat .env din .env.example. Verifica VIDEO_SOURCE in .env.
)

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
