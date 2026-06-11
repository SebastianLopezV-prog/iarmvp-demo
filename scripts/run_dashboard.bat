@echo off
REM Launch the IaR dashboard in your default web browser.
REM Double-click this file (or run it) to start the dashboard; Streamlit opens the
REM browser automatically. Close the console window to stop it.
cd /d "%~dp0.."
".\venv\Scripts\python.exe" -m streamlit run app\dashboard.py --server.port 8501
