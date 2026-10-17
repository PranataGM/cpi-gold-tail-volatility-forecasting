@echo off
chcp 65001 >nul
cd /d "%~dp0"
title CPI Volatility Forecasting Engine - XAU/USD
echo ======================================================================
echo    Loading CPI Macro Volatility Prediction Engine (XAU/USD)...
echo ======================================================================
echo.
python predict_cpi.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to run Python script.
    echo Please make sure Python and dependencies are installed.
    pause
    exit /b
)
echo.
pause >nul
