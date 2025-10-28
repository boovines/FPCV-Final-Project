#!/bin/bash

# Vision-Prompt Glasses Prototype Setup Script
echo "Setting up Vision-Prompt Glasses Prototype..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

echo "Installing dependencies..."

# Install required packages
pip3 install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one with your OpenAI API key:"
    echo "echo 'OPENAI_API_KEY=your_api_key_here' > .env"
fi

# Create snapshots directory if it doesn't exist
mkdir -p snapshots

echo "Running tests..."
python3 test_prototype.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup complete! The prototype is ready to run."
    echo ""
    echo "To start the application:"
    echo "  python3 main.py"
    echo ""
    echo "To test the implementation:"
    echo "  python3 test_prototype.py"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
