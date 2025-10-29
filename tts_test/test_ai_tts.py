#!/usr/bin/env python3
"""
Test AI Response TTS Integration
Simulates AI responses and tests the TTS system
"""

import time
from ai_response_tts import AIResponseTTS

def test_ai_responses():
    """Test various AI response formats."""
    try:
        tts = AIResponseTTS()
        print("🧪 Testing AI Response TTS Integration")
        print("=" * 50)
        
        # Test responses
        test_responses = [
            "AI Response: I can see a person holding a smartphone in their hand. The phone appears to be a modern device with a dark screen.",
            "Response: The image shows a laptop computer on a wooden desk with a coffee cup next to it.",
            "AI: This is a test of the text-to-speech system for AI responses.",
            "Assistant: The object in the image is a red car parked on the street.",
            "GPT: I can identify several objects including a book, a pen, and a notebook on the table.",
            "OpenAI: The scene depicts a beautiful sunset over the ocean with waves gently lapping the shore."
        ]
        
        for i, response in enumerate(test_responses, 1):
            print(f"\n🔊 Test {i}: {response[:50]}...")
            tts.speak_ai_response(response)
            
            if i < len(test_responses):
                print("⏳ Waiting 3 seconds before next test...")
                time.sleep(3)
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_response_cleaning():
    """Test response text cleaning."""
    try:
        tts = AIResponseTTS()
        print("\n🧹 Testing Response Text Cleaning")
        print("=" * 40)
        
        test_cases = [
            "AI Response: This is a **bold** text with *italics* and `code`.",
            "Response: - Bullet point 1\n- Bullet point 2\n- Bullet point 3",
            "AI:   Multiple    spaces    and    formatting   ",
            "Assistant: Text without proper ending",
            "GPT: **Bold** and *italic* and `code` formatting"
        ]
        
        for i, test_text in enumerate(test_cases, 1):
            cleaned = tts.clean_response_text(test_text)
            print(f"\nTest {i}:")
            print(f"Original: {test_text}")
            print(f"Cleaned:  {cleaned}")
        
        print("\n✅ Text cleaning tests completed!")
        
    except Exception as e:
        print(f"❌ Text cleaning test failed: {e}")

def main():
    """Main test function."""
    print("🎯 AI Response TTS Integration Tests")
    print("=" * 50)
    
    # Test response cleaning first
    test_response_cleaning()
    
    # Ask user if they want to test actual TTS
    print("\n" + "=" * 50)
    response = input("Do you want to test actual TTS playback? (y/n): ").strip().lower()
    
    if response == 'y':
        test_ai_responses()
    else:
        print("Skipping TTS playback tests.")

if __name__ == "__main__":
    main()
