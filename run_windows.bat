@echo off
cd /d "%~dp0"
title Anime Studio
if not exist venv\Scripts\python.exe (
    echo Anime Studio is not installed yet. Double-click install_windows.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo Starting Anime Studio... your browser will open in a moment.
echo Keep this window open while you use the app. Close it to quit.
python app.py %*
pause
