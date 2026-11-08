@echo off
setlocal EnableExtensions DisableDelayedExpansion
title EEG Project Console
cd /d "%~dp0"
if exist "pyproject.toml" if exist "eeg_transfer\console.py" goto project_ready

echo EEG Project Console - Windows launcher
echo Put this file in the About-EEG project folder, or enter that folder below.
echo Example: E:\About-EEG
set "EEG_PROJECT_DIR="
set /p "EEG_PROJECT_DIR=Project folder: "
if not defined EEG_PROJECT_DIR goto bad_project
set "EEG_PROJECT_DIR=%EEG_PROJECT_DIR:"=%"
cd /d "%EEG_PROJECT_DIR%" 2>nul
if errorlevel 1 goto bad_project
if not exist "pyproject.toml" goto bad_project
if not exist "eeg_transfer\console.py" goto bad_project

:project_ready
echo.
echo Project: "%CD%"
echo This launcher uses the project's .venv, independently of bci_env.
echo No formal experiments are started automatically.
echo.
if exist ".venv\Scripts\python.exe" goto check_python
if exist ".venv" goto bad_venv

echo [1/3] Creating a separate Python environment...
py -3.12 -m venv .venv 2>nul
if not errorlevel 1 goto check_python
py -3 -m venv .venv 2>nul
if not errorlevel 1 goto check_python
python -m venv .venv
if errorlevel 1 goto python_missing

:check_python
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
if errorlevel 1 goto bad_venv

echo [2/3] Checking project dependencies...
".venv\Scripts\python.exe" -c "import torch, numpy, scipy, mne, sklearn, pytest" >nul 2>&1
if not errorlevel 1 goto start_console
echo Installing dependencies. First launch requires internet and may take several minutes.
echo This does not change the old bci_env environment.
".venv\Scripts\python.exe" -m pip install -e ".[test]"
if errorlevel 1 goto install_failed

:start_console
echo [3/3] Starting the console...
echo.
echo Open http://127.0.0.1:8765 in your browser.
echo Copy the Access token printed below into the web page.
echo Keep this window open. Press Ctrl+C to exit the console.
echo.
".venv\Scripts\python.exe" -m eeg_transfer.console
if errorlevel 1 goto console_failed
goto done

:bad_project
echo ERROR: Project files not found. Update the work branch and select the folder
echo containing pyproject.toml and eeg_transfer\console.py.
goto failed

:bad_venv
echo ERROR: The existing .venv is not a working Windows Python 3.10+ environment.
echo A Linux .venv cannot be copied to Windows.
echo Rename the project's .venv folder, then run this launcher again.
echo Your old bci_env is not modified.
goto failed

:python_missing
echo ERROR: Could not create the environment. Install Python 3.12 for Windows
echo with the Python launcher or python on PATH, then run this file again.
goto failed

:install_failed
echo ERROR: Dependency installation failed. Check the messages above and your network.
echo Run this launcher again after fixing the issue.
goto failed

:console_failed
echo ERROR: Console stopped with an error. Check the messages above.
echo If port 8765 is already in use, close the previous console first.
goto failed

:done
echo Console stopped.
pause
exit /b 0

:failed
pause
exit /b 1
