@echo off
REM Start the flAICheckBot AI Engine on Windows
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%src\ai"

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    echo Error: .venv not found in project root. Please run the setup first.
    pause
    exit /b 1
)

echo Starting AI Engine using root .venv...
"%SCRIPT_DIR%.venv\Scripts\python.exe" icr_prototype.py
pause
