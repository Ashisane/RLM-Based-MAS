@echo off
REM Training Pipeline - One-Click Run

echo ======================================================================
echo  NARRATIVE CONSISTENCY CHECKER - TRAINING
echo ======================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] .env file not found
    pause
    exit /b 1
)

if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [SETUP] Installing dependencies...
pip install -q -r requirements.txt

echo.
echo ======================================================================
echo  RUNNING TRAINING PIPELINE
echo ======================================================================
echo.

python run_training.py --extract --output training_results.csv

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  COMPLETE
echo ======================================================================
echo Results: training_results.csv
echo Report: agent\report\training_report.md
echo.

pause
