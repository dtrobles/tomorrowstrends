@echo off
title Running Tomorrow's Trends
echo Starting Tomorrow's Trends project...
echo.

:: Start frontend
echo Starting Frontend...
start cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 2

:: Start public (tileserver-gl)
echo Starting Tileserver-GL...
start cmd /k "cd /d %~dp0frontend\public && tileserver-gl --verbose"
timeout /t 2

:: Start backend
echo Starting Backend...
start cmd /k "cd /d %~dp0backend && py app.py"
timeout /t 2

:: Open Chrome with the frontend URL
echo Opening Chrome...
start chrome "http://localhost:5173/"

echo All services started successfully!
exit
