#!/bin/bash
# Run Vision-Prompt Glasses with TTS Integration

echo "🎤 Vision-Prompt Glasses with TTS Integration"
echo "============================================="
echo ""

# Check if API key is set
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo "❌ ELEVENLABS_API_KEY not set"
    echo "Please set your API key:"
    echo "export ELEVENLABS_API_KEY='your_api_key_here'"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "ai_response_tts.py" ]; then
    echo "❌ Please run this script from the tts_test directory"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the main application with TTS pipe
echo "🚀 Starting Vision-Prompt Glasses with TTS..."
echo "The AI responses will be spoken aloud automatically!"
echo ""

# Run the main application and pipe output to TTS
python ../main.py 2>&1 | python pipe_ai_to_tts.py --interactive
