@echo off
echo Starting S&P 500 Pattern Scanner...
cd /d %~dp0
streamlit run sp500_scanner_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
