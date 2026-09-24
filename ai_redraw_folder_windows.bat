@echo off
chcp 65001 >nul
rem Drag and drop a folder (or image files) onto this file to redraw them with AI.
rem SVGs are written to an "Output_ai" folder next to the input.
rem Set your key once in the web app (run_windows.bat, AI redraw, Save) or run:
rem   .venv\Scripts\python.exe ai_redraw.py --key YOUR_KEY --save-key --list-models
cd /d "%~dp0"
if not exist ".venv\installed.ok" (
  echo  Age ekbar run_windows.bat chalan (setup er jonno).
  pause
  exit /b 1
)
if "%~1"=="" (
  echo  Ekta folder ba image file ei .bat file er upor drag ^& drop korun.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" ai_redraw.py %*
pause
