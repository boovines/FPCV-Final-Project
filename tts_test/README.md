# Text-to-Speech Testing

This folder contains a simple text-to-speech testing script using ElevenLabs API.

## Setup

1. **Install dependencies:**
   ```bash
   ./setup.sh
   ```

2. **Set your ElevenLabs API key:**
   ```bash
   export ELEVENLABS_API_KEY='your_api_key_here'
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

## Usage

### Interactive Mode
Run the text-to-speech script:
```bash
python text_to_speech.py
```

The script will:
- Prompt you to enter text
- Show available voices (first 10)
- Let you select a voice or use the default
- Ask if you want to play audio immediately
- Generate speech and save it as an MP3 file
- Play the generated audio (if requested)

### Quick Speak Mode
For immediate speech without interaction:
```bash
python speak.py "Your text here"
```

Example:
```bash
python speak.py "Hello, how are you today?"
```

## Features

- **Interactive text input** with voice selection
- **Immediate audio playback** using ffmpeg
- **Quick speak mode** for command-line usage
- **Voice selection** from available ElevenLabs voices
- **Audio generation** with customizable voice settings
- **MP3 output files** for saving generated speech
- **Error handling** and user-friendly interface

## Getting ElevenLabs API Key

1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Sign up for an account
3. Go to your profile settings
4. Copy your API key
5. Set it as an environment variable: `export ELEVENLABS_API_KEY='your_key'`

## Output

Generated audio files will be saved in the current directory with names like:
- `tts_output_0.mp3`
- `tts_output_1.mp3`
- etc.

You can play them using:
```bash
open tts_output_0.mp3  # macOS
# or
mpv tts_output_0.mp3   # Linux
```
