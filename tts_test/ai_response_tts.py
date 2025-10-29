#!/usr/bin/env python3
"""
AI Response TTS Integration
Captures AI responses from the vision-prompt glasses pipeline and speaks them aloud
"""

import os
import sys
import re
from elevenlabs import Voice, VoiceSettings, generate, play, set_api_key

class AIResponseTTS:
    def __init__(self, voice_id="2EiwWnXFnvU5JabPnv8n"):
        """Initialize TTS for AI responses."""
        self.voice_id = voice_id
        self.setup_elevenlabs()
    
    def setup_elevenlabs(self):
        """Setup ElevenLabs API key."""
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        
        set_api_key(api_key)
        print("✅ ElevenLabs TTS initialized")
    
    def speak_ai_response(self, response_text: str):
        """Convert AI response to speech and play it."""
        if not response_text or not response_text.strip():
            print("No response text to speak")
            return False
        
        try:
            # Clean up the response text
            cleaned_text = self.clean_response_text(response_text)
            
            print(f"🔊 Speaking AI response: {cleaned_text[:100]}...")
            
            # Generate audio
            audio = generate(
                text=cleaned_text,
                voice=Voice(
                    voice_id=self.voice_id,
                    settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.5,
                        style=0.0,
                        use_speaker_boost=True
                    )
                )
            )
            
            # Play audio
            play(audio)
            print("✅ AI response spoken successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error speaking AI response: {e}")
            return False
    
    def clean_response_text(self, text: str) -> str:
        """Clean up AI response text for better speech."""
        # Remove common prefixes
        text = re.sub(r'^AI Response:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Response:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        
        # Clean up bullet points
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
        
        # Ensure proper sentence endings
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()
    
    def speak_from_stdin(self):
        """Read AI responses from stdin and speak them."""
        print("🎤 AI Response TTS - Listening for responses...")
        print("Paste AI responses here (Ctrl+C to exit):")
        print("-" * 50)
        
        try:
            while True:
                line = input()
                
                # Check if line contains AI response
                if self.is_ai_response_line(line):
                    response_text = self.extract_response_text(line)
                    if response_text:
                        self.speak_ai_response(response_text)
                        print("-" * 30)
                
        except KeyboardInterrupt:
            print("\n👋 AI Response TTS stopped")
    
    def is_ai_response_line(self, line: str) -> bool:
        """Check if line contains an AI response."""
        ai_indicators = [
            "AI Response:",
            "Response:",
            "AI:",
            "Assistant:",
            "GPT:",
            "OpenAI:"
        ]
        
        line_lower = line.lower()
        return any(indicator.lower() in line_lower for indicator in ai_indicators)
    
    def extract_response_text(self, line: str) -> str:
        """Extract the actual response text from a line."""
        # Try to find text after common prefixes
        patterns = [
            r'AI Response:\s*(.*)',
            r'Response:\s*(.*)',
            r'AI:\s*(.*)',
            r'Assistant:\s*(.*)',
            r'GPT:\s*(.*)',
            r'OpenAI:\s*(.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the whole line
        return line.strip()

def main():
    """Main function for command-line usage."""
    if len(sys.argv) > 1:
        # Direct text input
        response_text = " ".join(sys.argv[1:])
        tts = AIResponseTTS()
        tts.speak_ai_response(response_text)
    else:
        # Interactive mode
        tts = AIResponseTTS()
        tts.speak_from_stdin()

if __name__ == "__main__":
    main()
