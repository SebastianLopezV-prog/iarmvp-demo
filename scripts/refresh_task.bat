@echo off
REM Wrapper invoked by Windows Task Scheduler ("IaR Refresh", every 15 min).
REM Keeps the IaR database current: live IaR + realised prices + backtest, all areas.
REM Output is appended to data\refresh.log.
cd /d "%~dp0.."
".\venv\Scripts\python.exe" scripts\refresh.py >> "data\refresh.log" 2>&1
