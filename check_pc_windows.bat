@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tools\check_pc.py
) else (
  python tools\check_pc.py
  if errorlevel 1 pause
)
