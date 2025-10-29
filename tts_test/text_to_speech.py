#!/usr/bin/env python3
"""
Simple Text-to-Speech script using ElevenLabs API
Takes manual input text and converts it to speech
"""

import os
import sys
from elevenlabs import Voice, VoiceSettings, generate, save, set_api_key, play

def setup_client():
    """Initialize ElevenLabs client with API key"""
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        print("Please set your ElevenLabs API key:")
        print("export ELEVENLABS_API_KEY='your_api_key_here'")
        return None
    
    set_api_key(api_key)
    return True

def get_voices():
    """Get available voices from ElevenLabs"""
    try:
        from elevenlabs import voices
        voices_list = voices()
        return voices_list
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return None

def text_to_speech(text, voice_id=None, output_file="output.mp3", play_audio=True):
    """Convert text to speech using ElevenLabs"""
    try:
        if voice_id is None:
            # Use default voice (Rachel)
            voice_id = "21m00Tcm4TlvDq8ikWAM"
        
        print(f"Converting text to speech...")
        print(f"Text: {text}")
        print(f"Voice ID: {voice_id}")
        
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
        
        # Save audio to file
        save(audio, output_file)
        print(f"Audio saved to: {output_file}")
        
        # Play audio if requested
        if play_audio:
            print("🔊 Playing audio...")
            play(audio)
            print("✅ Audio playback completed!")
        
        return True
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False

def main():
    """Main function to handle user input and TTS conversion"""
    print("=== ElevenLabs Text-to-Speech Test ===")
    print()
    
    # Setup client
    client = setup_client()
    if not client:
        return
    
    # Get available voices
    print("Fetching available voices...")
    voices = get_voices()
    if voices:
        print(f"Found {len(voices)} available voices:")
        for i, voice in enumerate(voices[:10]):  # Show first 10 voices
            print(f"  {i+1}. {voice.name} (ID: {voice.voice_id})")
        print()
    
    # Interactive loop
    while True:
        print("Enter text to convert to speech (or 'quit' to exit):")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        # Ask for voice selection
        if voices:
            print(f"\nSelect a voice (1-{min(10, len(voices))}) or press Enter for default:")
            voice_choice = input("Voice: ").strip()
            
            voice_id = None
            if voice_choice.isdigit():
                choice_idx = int(voice_choice) - 1
                if 0 <= choice_idx < min(10, len(voices)):
                    voice_id = voices[choice_idx].voice_id
                    print(f"Selected voice: {voices[choice_idx].name}")
        
        # Ask if user wants to play audio
        print("\nPlay audio immediately? (y/n, default: y):")
        play_choice = input("Play: ").strip().lower()
        play_audio = play_choice != 'n'
        
        # Convert to speech
        output_file = f"tts_output_{len(os.listdir('.'))}.mp3"
        success = text_to_speech(text, voice_id, output_file, play_audio)
        
        if success:
            print(f"✅ Successfully generated speech!")
            print(f"📁 Output file: {output_file}")
            if not play_audio:
                print(f"🔊 You can play the file with: open {output_file}")
        else:
            print("❌ Failed to generate speech")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
