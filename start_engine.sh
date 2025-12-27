#!/bin/bash
# Start the flAICheckBot AI Engine manually (Linux/Mac)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for root .venv
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: .venv not found in project root. Please run the setup first."
    exit 1
fi

echo "Starting AI Engine using root .venv..."
# Execute python within the project root context but pointing to the script in src/ai
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/src/ai/icr_prototype.py"
