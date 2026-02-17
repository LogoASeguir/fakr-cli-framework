#!/bin/bash
echo "----------------------------------------"
echo "FAKR: Quick Start Initialization"
echo "----------------------------------------"

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check Ollama server availability
echo "Checking Ollama backend..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama server is running."
else
    echo "WARNING: Ollama server not found!"
    echo "- Start Ollama: ollama serve"
    echo "- Pull models if needed: ollama pull phi3-mini:3.8b"
fi

# Run main runtime
echo "Starting FAKR runtime..."
python3 main.py

# Post-run message
echo
echo "----------------------------------------"
echo "FAKR is running!"
echo "Now you just need to connect your APIs:"
echo "- Edit API.env.template -> fill in your API_KEY, API_BASE_URL, WORKSPACE_SLUG"
echo "- Or install and configure Ollama backend if you want local LLMs"
echo "----------------------------------------"