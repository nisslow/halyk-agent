@echo off
echo ===================================================
echo Halyk AI Agent - Final Submission Run
echo ===================================================

REM 1. Set the dataset directory (assuming agentic-bank-hidden is in the root)
set DATASET_DIR=E:\AntigravityProjects\halyk-agent\agentic-bank-hidden

REM 2. Ensure LLM provider is set to OpenRouter
set LLM_PROVIDER=openrouter
set LLM_MODEL=deepseek/deepseek-v4-flash-0731
REM Insert your OpenRouter API key here
set OPENROUTER_API_KEY=your_openrouter_api_key_here

REM 3. Concurrency
set MAX_WORKERS=12
set SCENARIO_FILTER=

echo.
echo Dataset Directory: %DATASET_DIR%
echo Provider: %LLM_PROVIDER%
echo Model: %LLM_MODEL%
echo.

echo [1/2] Mapping documents to scenarios...
python scripts\map_documents.py
if %ERRORLEVEL% neq 0 (
    echo Error during mapping. Exiting...
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Running main agent (this may take 5-10 minutes)...
python scripts\run_hackathon_agent.py
if %ERRORLEVEL% neq 0 (
    echo Error during agent execution. Exiting...
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo Run complete! Results saved to submission.json
echo ===================================================
pause
