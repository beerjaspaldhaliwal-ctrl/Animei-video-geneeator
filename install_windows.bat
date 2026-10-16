@echo off
setlocal
cd /d "%~dp0"
title Anime Studio - Install

rem Everything (Python, libraries, download cache) stays inside this folder.
set "UV_VERSION=0.12.20"
set "TOOLS=%~dp0tools"
set "UV=%TOOLS%\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_CACHE_DIR=%TOOLS%\cache"
set "PY=%~dp0venv\Scripts\python.exe"

echo ==========================================================
echo    Anime Studio installer (Windows)
echo ==========================================================
echo.

rem The Setup program passes the graphics card choice (1-4). Otherwise ask.
set "GPU=%~1"
if not "%GPU%"=="" goto :have_gpu
echo Which graphics card do you have?
echo.
echo   1 = AMD Radeon RX 7000 series  (RX 7600, 7600 XT, 7700 XT, 7800 XT, 7900 ...)
echo   2 = AMD Radeon RX 9000 series  (RX 9060 XT, 9070, 9070 XT)
echo   3 = NVIDIA GeForce             (RTX 20/30/40/50 series)
echo   4 = No graphics card           (CPU only, very slow)
echo.
set /p GPU="Type 1, 2, 3 or 4 and press Enter: "
:have_gpu

set "TORCH_INDEX="
if "%GPU%"=="1" set "TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx110X-all/"
if "%GPU%"=="2" set "TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx120X-all/"
if "%GPU%"=="3" set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if "%GPU%"=="4" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if "%TORCH_INDEX%"=="" (
    echo Please run the installer again and type 1, 2, 3 or 4.
    goto :fail
)

rem 1. uv: a small tool that downloads Python and the libraries.
if exist "%UV%" goto :have_uv
echo Downloading the installer helper (uv %UV_VERSION%)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip' -OutFile '%TEMP%\anime-studio-uv.zip'; Expand-Archive -Force '%TEMP%\anime-studio-uv.zip' '%TOOLS%\uv'"
if errorlevel 1 goto :fail
if not exist "%UV%" goto :fail
:have_uv

rem 2. A private copy of Python 3.12, only for Anime Studio.
if exist "%PY%" goto :have_venv
echo.
echo Getting Python 3.12...
"%UV%" venv --seed --python 3.12 venv
if errorlevel 1 goto :fail
:have_venv

rem 3. PyTorch built for your graphics card (a big download).
echo.
echo Installing PyTorch for your graphics card (a few GB, please wait)...
"%UV%" pip install --python "%PY%" --index-url %TORCH_INDEX% torch torchvision torchaudio
if errorlevel 1 goto :fail

rem Pin the PyTorch build we just installed so nothing replaces it.
"%PY%" -c "import importlib.metadata as m; print(chr(10).join(p + '==' + m.version(p) for p in ('torch', 'torchvision', 'torchaudio')))" > constraints.txt
if errorlevel 1 goto :fail

rem 4. Everything else.
echo.
echo Installing Anime Studio...
"%UV%" pip install --python "%PY%" -r requirements.txt -c constraints.txt
if errorlevel 1 goto :fail

rem The download cache is no longer needed (frees several GB).
"%UV%" cache clean >nul 2>nul

echo.
echo Checking your graphics card...
"%PY%" -c "import torch; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; print('GPUs found:', ', '.join(torch.cuda.get_device_name(i) for i in range(n)) if n else 'NONE (running on CPU)')"

echo.
echo ==========================================================
echo    Done! Start Anime Studio from the desktop shortcut,
echo    or double-click run_windows.bat in this folder.
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo Something went wrong. Scroll up to see the error, check your internet
echo connection, and run this again. See "Troubleshooting" in README.md.
pause
exit /b 1
