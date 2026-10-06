@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" main.py --gui %*
) else (
  py -3 main.py --gui %*
)
if errorlevel 1 pause
