@echo off
setlocal
cd /d "%~dp0"
title Anime Studio - Install

echo ==========================================================
echo    Anime Studio installer (Windows)
echo ==========================================================
echo.

py -3.12 --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 was not found.
    echo.
    echo 1. Download Python 3.12 from https://www.python.org/downloads/windows/
    echo    ^(pick the latest "Python 3.12.x" Windows installer, 64-bit^)
    echo 2. Run it and tick "Add python.exe to PATH".
    echo 3. Run this install_windows.bat again.
    echo.
    pause
    exit /b 1
)

echo Which graphics card do you have?
echo.
echo   1 = AMD Radeon RX 7000 series  (RX 7600, 7600 XT, 7700 XT, 7800 XT, 7900 ...)
echo   2 = AMD Radeon RX 9000 series  (RX 9060 XT, 9070, 9070 XT)
echo   3 = NVIDIA GeForce             (RTX 20/30/40/50 series)
echo   4 = No graphics card           (CPU only, very slow)
echo.
set /p GPU="Type 1, 2, 3 or 4 and press Enter: "

set TORCH_INDEX=
if "%GPU%"=="1" set TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx110X-all/
if "%GPU%"=="2" set TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx120X-all/
if "%GPU%"=="3" set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if "%GPU%"=="4" set TORCH_INDEX=https://download.pytorch.org/whl/cpu
if "%TORCH_INDEX%"=="" (
    echo Please run the installer again and type 1, 2, 3 or 4.
    pause
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo.
    echo Creating the Python environment...
    py -3.12 -m venv venv
    if errorlevel 1 goto :fail
)
call venv\Scripts\activate.bat

python -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing PyTorch for your graphics card (this is a big download)...
python -m pip install --index-url %TORCH_INDEX% torch torchvision torchaudio
if errorlevel 1 goto :fail

rem Pin the PyTorch build we just installed so nothing replaces it.
python -c "import importlib.metadata as m; print(chr(10).join(p + '==' + m.version(p) for p in ('torch', 'torchvision', 'torchaudio')))" > constraints.txt
if errorlevel 1 goto :fail

echo.
echo Installing Anime Studio...
python -m pip install -r requirements.txt -c constraints.txt
if errorlevel 1 goto :fail

echo.
echo Checking your graphics card...
python -c "import torch; ok = torch.cuda.is_available(); print('GPU ready:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND (running on CPU)')"

echo.
echo ==========================================================
echo    Done! Double-click run_windows.bat to start.
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo Something went wrong. Scroll up to see the error, and check the
echo "Troubleshooting" part of README.md.
pause
exit /b 1
