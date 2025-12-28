#!/bin/bash
# Start the flAICheckBot AI Engine manually (Linux/Mac)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -d "$SCRIPT_DIR/ai" ]; then
    AI_DIR="$SCRIPT_DIR/ai"
else
    AI_DIR="$SCRIPT_DIR/src/ai"
fi

# Check for root .venv
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: .venv not found in project root. Please run the setup first."
    exit 1
fi

echo "Starting AI Engine..."

# Kill existing process on port 8000 if it exists
EXISTING_PID=$(lsof -t -i :8000)
if [ ! -z "$EXISTING_PID" ]; then
    echo "Stopping existing AI Engine (PID: $EXISTING_PID)..."
    kill -9 $EXISTING_PID
    sleep 1
fi

cd "$AI_DIR"
"$SCRIPT_DIR/.venv/bin/python" "$AI_DIR/icr_prototype.py"
