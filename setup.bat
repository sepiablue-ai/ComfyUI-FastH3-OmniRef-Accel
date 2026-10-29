@echo off
setlocal
set "COMFY_ROOT=%~1"
set "COMFY_PYTHON=%~2"
if not defined COMFY_ROOT if exist "%~dp0..\..\main.py" for %%I in ("%~dp0..\..") do set "COMFY_ROOT=%%~fI"
if not defined COMFY_ROOT set /p "COMFY_ROOT=ComfyUI folder containing main.py: "
if not exist "%COMFY_ROOT%\main.py" (
  echo ERROR: ComfyUI main.py was not found.
  exit /b 1
)
if defined COMFY_PYTHON goto run
set "PYTHON_COUNT=0"
call :candidate "%COMFY_ROOT%\..\python_embeded\python.exe"
call :candidate "%COMFY_ROOT%\.venv\Scripts\python.exe"
call :candidate "%COMFY_ROOT%\..\.venv\Scripts\python.exe"
call :candidate "%COMFY_ROOT%\venv\Scripts\python.exe"
if not "%PYTHON_COUNT%"=="1" (
  echo ERROR: Specify the ComfyUI Python explicitly as the second argument.
  echo Usage: setup.bat "COMFY_ROOT" "COMFY_PYTHON" [--check] [--sol]
  exit /b 1
)
:run
if not exist "%COMFY_PYTHON%" (
  echo ERROR: The specified Python does not exist.
  exit /b 1
)
set "FASTH3_ACCEL_SETUP_ROOT=%COMFY_ROOT%"
"%COMFY_PYTHON%" -X utf8 "%~dp0setup.py" %*
exit /b %errorlevel%
:candidate
if not exist "%~1" exit /b 0
set /a PYTHON_COUNT+=1 >nul
set "COMFY_PYTHON=%~1"
exit /b 0
