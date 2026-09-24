@echo off
chcp 65001 >nul
rem Drag and drop a folder (or image files) onto this file.
rem SVGs are written to an "Output" folder next to the input folder.
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
".venv\Scripts\python.exe" convert.py %*
pause
