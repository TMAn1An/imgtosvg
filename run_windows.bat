@echo off
chcp 65001 >nul
cd /d "%~dp0"
title icon2svg

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python paoa jayni. https://www.python.org/downloads/ theke Python 3.10+ install korun
  echo  ^(install korar somoy "Add python.exe to PATH" tick diben^).
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo  Prothom bar setup hocche, 1-2 minute lagbe...
  python -m venv .venv
)
if not exist ".venv\installed.ok" (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
  echo ok> ".venv\installed.ok"
)

".venv\Scripts\python.exe" app.py
pause
