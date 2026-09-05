@echo off
echo =========================================================
echo       STARTING BAS EXPERIMENT COPILOT (OFFLINE)
echo =========================================================
cd /d "%~dp0"

IF EXIST "C:\Users\Sanjay\bvenv\Scripts\python.exe" (
    "C:\Users\Sanjay\bvenv\Scripts\python.exe" run_copilot.py
) ELSE (
    python run_copilot.py
)
pause
