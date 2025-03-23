#!/bin/bash
# start-app.sh - Shell script to start the Document Chat application

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    # Not in a virtual environment, check if venv/env directory exists
    if [[ -d "venv" ]]; then
        echo "Found venv directory, activating virtual environment..."
        source venv/bin/activate
    elif [[ -d "env" ]]; then
        echo "Found env directory, activating virtual environment..."
        source env/bin/activate
    elif [[ -d ".venv" ]]; then
        echo "Found .venv directory, activating virtual environment..."
        source .venv/bin/activate
    elif [[ -d ".env" ]]; then
        echo "Found .env directory, activating virtual environment..."
        source .env/bin/activate
    elif [[ -d "backend/venv" ]]; then
        echo "Found backend/venv directory, activating virtual environment..."
        source backend/venv/bin/activate
    elif [[ -d "backend/env" ]]; then
        echo "Found backend/env directory, activating virtual environment..."
        source backend/env/bin/activate
    elif [[ -d "chatbot/backend/venv" ]]; then
        echo "Found chatbot/backend/venv directory, activating virtual environment..."
        source chatbot/backend/venv/bin/activate
    elif [[ -d "chatbot/backend/env" ]]; then
        echo "Found chatbot/backend/env directory, activating virtual environment..."
        source chatbot/backend/env/bin/activate
    else
        echo "Warning: No virtual environment found. Using system Python."
        echo "It's recommended to use a virtual environment for this application."
        echo "You can create one with: python -m venv venv"
        echo ""
    fi
else
    echo "Already in virtual environment: $VIRTUAL_ENV"
fi

# Check for required Python packages
echo "Checking for required packages..."
if ! python -c "import langchain_chroma" &> /dev/null; then
    echo "The langchain-chroma package is not installed. Installing it now..."
    pip install langchain-chroma
fi

if ! python -c "import fastapi" &> /dev/null; then
    echo "The fastapi package is not installed. Please run 'pip install -r backend/requirements.txt' first."
    exit 1
fi

# Execute the Python start script
echo "Starting the application..."
python start_app.py