@echo off
title Adaptive Recipe Recommendation Engine
echo ==============================================================================
echo Adaptive Personalized Recipe Recommendation Engine
echo ==============================================================================
echo Team Members:
echo   - Krrish Jha (24STUCHH010116)
echo   - Rasineni Charishma (24STUCHH010662)
echo   - Ande Hiranmai (24STUCHH010931)
echo   - Chirra Chaithanya (24STUCHH010120)
echo ==============================================================================
echo Checking dependencies...
python -m pip install -r requirements.txt --quiet

echo.
echo Starting FastAPI & Web Dashboard on http://127.0.0.1:8000 ...
echo Press Ctrl+C in this terminal window to stop the server anytime.
echo.

start "" http://127.0.0.1:8000
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
pause
