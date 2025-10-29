#!/bin/bash
# Setup script for TTS testing

echo "Setting up Text-to-Speech testing environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To use the TTS script:"
echo "1. Set your ElevenLabs API key: export ELEVENLABS_API_KEY='your_api_key_here'"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the script: python text_to_speech.py"
