@echo off
setlocal
cd /d "%~dp0"
echo 8D Engine Studio - first-time setup
echo.
py -3.12 -c "import sys; assert sys.version_info >= (3,12)" >nul 2>&1
if errorlevel 1 goto missing_python
py -3.12 -m venv ".venv-studio"
if errorlevel 1 goto failed
".venv-studio\Scripts\python.exe" -m pip install -r requirements-studio.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click START_STUDIO_WINDOWS.cmd to open Studio.
pause
exit /b 0
:missing_python
echo Python 3.12 is required. Install 64-bit Python 3.12 from python.org with the Python launcher enabled.
echo Then run this setup file again.
pause
exit /b 1
:failed
echo.
echo Setup did not finish. Check the error above and your internet connection.
echo Keep this window open or send a screenshot for help.
pause
exit /b 1
