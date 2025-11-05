@echo off
REM Viper ML Bot - Windows Setup Script
REM Run this once: setup.bat

echo.
echo =========================================
echo 🐍 VIPER ML TRADING BOT - SETUP SCRIPT
echo =========================================

REM Check Python
echo.
echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% found

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated

REM Upgrade pip
echo.
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo ✓ pip upgraded

REM Install dependencies
echo.
echo [5/5] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo ✓ All dependencies installed successfully
) else (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo =========================================
echo ✅ SETUP COMPLETE!
echo =========================================
echo.
echo Next steps:
echo 1. Activate virtual environment:
echo    .venv\Scripts\activate
echo.
echo 2. Run the bot:
echo    python viper_ml_bot.py
echo.
echo 3. Select 'Configure Settings' and start backtesting
echo.
pause
