@echo off
echo ----------------------------------------
echo FAKR: Quick Start Initialization
echo ----------------------------------------

:: Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

:: Check Ollama server availability
echo Checking Ollama backend...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama server not found!
    echo - Start Ollama: ollama serve
    echo - Pull models if needed: ollama pull phi3-mini:3.8b
) else (
    echo Ollama server is running.
)

:: Run the main runtime
echo Starting FAKR runtime...
py main.py

:: Post-run message
echo.
echo ----------------------------------------
echo FAKR is running!
echo Now you just need to connect your APIs:
echo Fill in your API_KEY, API_BASE_URL, WORKSPACE_SLUG
echo Or install and configure Ollama backend if you want local LLMs
echo ----------------------------------------
pause
