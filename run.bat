@echo off
REM Narrative Consistency Checker - One-Click Run Script
REM Creates environment, installs dependencies, runs full pipeline

echo ======================================================================
echo  NARRATIVE CONSISTENCY CHECKER
echo  One-Click Setup and Run
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo Please create .env with: GEMINI_API_KEY=your_api_key
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [SETUP] Installing dependencies...
pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  RUNNING PIPELINE
echo ======================================================================
echo.

REM Run the test pipeline
python run_test.py --extract --output results.csv

REM Check if successful
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline failed. Check logs in cache\logs\
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  COMPLETE
echo ======================================================================
echo.
echo Results saved to: results.csv
echo Logs saved to: cache\logs\
echo.

REM Show results preview
echo --- Results Preview ---
if exist "results.csv" (
    type results.csv | more
)

pause
