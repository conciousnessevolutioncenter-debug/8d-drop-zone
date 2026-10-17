@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-studio\Scripts\python.exe" goto setup_needed
".venv-studio\Scripts\python.exe" -m studio.launch
if errorlevel 1 goto failed
exit /b 0
:setup_needed
echo Please run SETUP_STUDIO_WINDOWS.cmd once before starting Studio.
pause
exit /b 1
:failed
echo Studio could not start. Read the message above.
pause
exit /b 1
