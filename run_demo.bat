@echo off
echo ==============================================================================
echo Adaptive Personalized Recipe Recommendation Engine
echo ==============================================================================
echo 1. Run Complete Data Preprocessing & Model Training Pipeline
echo 2. Launch Streamlit Interactive UI (Explorer, Adaptive Feedback, Benchmarks)
echo 3. Launch FastAPI Production REST API Server
echo 4. Run Pytest Test Suite
echo 5. Exit
echo ==============================================================================
set /p opt="Select an option (1-5): "

if "%opt%"=="1" (
    echo Running end-to-end pipeline...
    python scripts\run_pipeline.py
    pause
) else if "%opt%"=="2" (
    echo Starting Streamlit app...
    streamlit run src\frontend\app.py
) else if "%opt%"=="3" (
    echo Starting FastAPI server at http://127.0.0.1:8000 ...
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
) else if "%opt%"=="4" (
    echo Running unit and integration tests...
    pytest tests\ -v
    pause
) else (
    echo Exiting.
)
