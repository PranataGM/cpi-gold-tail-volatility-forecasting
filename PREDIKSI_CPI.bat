@echo off
cd /d "%~dp0"
title Model Prediksi Volatilitas CPI - Emas XAU/USD
echo ======================================================================
echo    Memuat Engine Kuantitatif Prediksi Volatilitas CPI (XAU/USD)...
echo ======================================================================
echo.
python jalankan_prediksi_cpi.py
if %errorlevel% neq 0 (
    echo.
    echo [!] Terjadi kendala saat menjalankan program Python.
    echo Pastikan Python terpasang di komputer Anda.
    pause
    exit /b
)
echo.
echo ======================================================================
echo Selesai. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul
