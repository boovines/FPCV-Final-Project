#!/usr/bin/env python3
"""
Simple TTS script that automatically plays audio
Usage: python speak.py "Your text here"
"""

import os
import sys
from elevenlabs import Voice, VoiceSettings, generate, save, play, set_api_key

def speak_text(text, voice_id="2EiwWnXFnvU5JabPnv8n"):
    """Convert text to speech and play it immediately"""
    try:
        # Set API key
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            print("Error: ELEVENLABS_API_KEY environment variable not set")
            return False
        
        set_api_key(api_key)
        
        print(f"🔊 Speaking: {text}")
        
        # Generate audio
        audio = generate(
            text=text,
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
        )
        
        # Play audio immediately
        play(audio)
        print("✅ Speech completed!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python speak.py 'Your text here'")
        print("Example: python speak.py 'Hello, how are you today?'")
        return
    
    text = " ".join(sys.argv[1:])
    speak_text(text)

if __name__ == "__main__":
    main()
