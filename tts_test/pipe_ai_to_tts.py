#!/usr/bin/env python3
"""
Pipe AI Responses to TTS
Monitors terminal output for AI responses and speaks them aloud
"""

import sys
import re
import subprocess
import threading
import time
from ai_response_tts import AIResponseTTS

class AIPipeTTS:
    def __init__(self):
        """Initialize the AI pipe TTS system."""
        try:
            self.tts = AIResponseTTS()
            print("✅ AI Pipe TTS initialized")
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            sys.exit(1)
    
    def monitor_terminal_output(self, command):
        """Run a command and monitor its output for AI responses."""
        print(f"🚀 Running command: {command}")
        print("🎤 Monitoring for AI responses...")
        print("-" * 50)
        
        try:
            # Run the command and capture output in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Print the line
                    
                    # Check if this line contains an AI response
                    if self.tts.is_ai_response_line(line):
                        response_text = self.tts.extract_response_text(line)
                        if response_text:
                            # Speak the response in a separate thread to avoid blocking
                            threading.Thread(
                                target=self.tts.speak_ai_response,
                                args=(response_text,),
                                daemon=True
                            ).start()
            
            # Wait for process to complete
            process.wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping AI pipe TTS...")
            if 'process' in locals():
                process.terminate()
        except Exception as e:
            print(f"❌ Error running command: {e}")
    
    def interactive_mode(self):
        """Interactive mode for manual input."""
        print("🎤 AI Response TTS - Interactive Mode")
        print("Paste AI responses here (Ctrl+C to exit):")
        print("-" * 50)
        
        try:
            while True:
                line = input()
                
                # Check if line contains AI response
                if self.tts.is_ai_response_line(line):
                    response_text = self.tts.extract_response_text(line)
                    if response_text:
                        self.tts.speak_ai_response(response_text)
                        print("-" * 30)
                
        except KeyboardInterrupt:
            print("\n👋 AI Pipe TTS stopped")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipe_ai_to_tts.py 'command to run'")
        print("  python pipe_ai_to_tts.py --interactive")
        print()
        print("Examples:")
        print("  python pipe_ai_to_tts.py 'python ../main.py'")
        print("  python pipe_ai_to_tts.py --interactive")
        sys.exit(1)
    
    pipe_tts = AIPipeTTS()
    
    if sys.argv[1] == "--interactive":
        pipe_tts.interactive_mode()
    else:
        command = " ".join(sys.argv[1:])
        pipe_tts.monitor_terminal_output(command)

if __name__ == "__main__":
    main()
