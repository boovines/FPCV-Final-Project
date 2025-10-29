#!/usr/bin/env python3
"""
Vision-Prompt Glasses with TTS Integration
Modified version that speaks AI responses aloud
"""

import io
import cv2
import numpy as np
import os
import sys
import time
import requests
from datetime import datetime
from typing import Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient
from dotenv import load_dotenv

# Import TTS functionality
from ai_response_tts import AIResponseTTS

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    ElevenLabs = None

class VisionPromptGlassesWithTTS:
    def __init__(self):
        """Initialize the Vision-Prompt Glasses prototype with TTS."""
        # Initialize components
        self.hand_detector = HandDetector()
        self.frame_detector = FrameDetector()
        self.crop_utils = CropUtils()
        self.openai_client = OpenAIClient()
        
        # Initialize TTS
        try:
            self.tts = AIResponseTTS()
            print("✅ TTS system initialized")
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            self.tts = None
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Create snapshots directory
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # State tracking
        self.last_capture_time = 0
        self.capture_cooldown = 3.0  # seconds between captures

        # Load environment variables
        load_dotenv()

        # Verify speech recognition dependency
        if sr is None:
            raise ImportError(
                "speech_recognition package is required for voice input. "
                "Install it with 'pip install SpeechRecognition' and ensure PyAudio dependencies are available."
            )

        # Speech recognition configuration
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.non_speaking_duration = 0.4
        self.recognizer.dynamic_energy_threshold = True
        self.audio_sample_rate = 16000
        self.listen_timeout = float(os.getenv("ELEVENLABS_LISTEN_TIMEOUT", 12))
        self.max_listen_duration = float(os.getenv("ELEVENLABS_MAX_LISTEN_SECONDS", 25))

        # ElevenLabs configuration
        self.eleven_api_key = (
            os.getenv("ELEVENLABS_API_KEY")
            or os.getenv("ELEVEN_API_KEY")
        )
        if not self.eleven_api_key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY or ELEVEN_API_KEY in your environment."
            )

        self.eleven_stt_model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")
        self.eleven_stt_language = os.getenv("ELEVENLABS_STT_LANGUAGE", "en")
        self.eleven_stt_endpoint = os.getenv(
            "ELEVENLABS_STT_ENDPOINT",
            "https://api.elevenlabs.io/v1/speech-to-text",
        )

        # Initialize ElevenLabs SDK if available
        self.eleven_client = None
        if ElevenLabs is not None:
            try:
                self.eleven_client = ElevenLabs(api_key=self.eleven_api_key)
            except Exception as sdk_error:
                print(f"Warning: ElevenLabs SDK initialization failed ({sdk_error}). Falling back to REST API calls.")
    
    def initialize_camera(self) -> bool:
        """Initialize webcam capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_cached_snapshots(self) -> list:
        """Get list of cached snapshot files."""
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    snapshot_files.append(os.path.join(self.snapshots_dir, filename))
            return sorted(snapshot_files, key=os.path.getmtime, reverse=True)
        except Exception as e:
            print(f"Error reading snapshots directory: {e}")
            return []
    
    def select_cached_snapshot(self) -> Optional[str]:
        """Allow user to select a cached snapshot for analysis."""
        snapshots = self.get_cached_snapshots()
        
        if not snapshots:
            print("No cached snapshots found.")
            return None
        
        print("\nAvailable cached snapshots:")
        for i, snapshot in enumerate(snapshots[:10]):
            filename = os.path.basename(snapshot)
            print(f"{i + 1}. {filename}")
        
        try:
            choice = input("\nSelect snapshot number (or press Enter to cancel): ").strip()
            if not choice:
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(snapshots):
                return snapshots[index]
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None
    
    def process_cached_snapshot(self, snapshot_path: str):
        """Process a cached snapshot with OpenAI."""
        try:
            # Load image
            image = cv2.imread(snapshot_path)
            if image is None:
                print(f"Could not load image: {snapshot_path}")
                return
            
            # Encode to base64
            image_base64 = self.crop_utils.encode_image_to_base64(image)
            if not image_base64:
                print("Failed to encode image")
                return
            
            # Get user prompt
            prompt = input(f"\nEnter your question about the image '{os.path.basename(snapshot_path)}': ")
            if not prompt.strip():
                print("No prompt provided.")
                return
            
            # Send to OpenAI
            print("Analyzing image...")
            response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
            
            if response:
                print(f"\nAI Response:\n{response}\n")
                
                # Speak the AI response
                if self.tts:
                    self.tts.speak_ai_response(response)
            else:
                print("Failed to get response from AI.")
                
        except Exception as e:
            print(f"Error processing cached snapshot: {e}")
    
    def capture_and_analyze(self, frame: np.ndarray, corners: list):
        """Capture the framed region and send to OpenAI for analysis."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_capture_time < self.capture_cooldown:
            return
        
        try:
            # Crop the framed region
            cropped_image = self.crop_utils.crop_frame_region(frame, corners)
            if cropped_image is None:
                print("Failed to crop image")
                return
            
            # Validate crop quality
            if not self.crop_utils.validate_crop_quality(cropped_image):
                print("Cropped image quality is too poor")
                return
            
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(self.snapshots_dir, f"snapshot_{timestamp}.jpg")
            
            if not self.crop_utils.save_cropped_image(cropped_image, snapshot_path):
                print("Failed to save snapshot")
                return
            
            print(f"\nSnapshot saved: {snapshot_path}")
            
            # Encode image for API
            image_base64 = self.crop_utils.encode_image_to_base64(cropped_image)
            if not image_base64:
                print("Failed to encode image")
                return
            
            # Get spoken user prompt via ElevenLabs speech-to-text
            prompt = self.capture_spoken_prompt()
            if not prompt:
                print("No spoken prompt captured.")
                return
            
            # Send to OpenAI
            print("Analyzing image...")
            response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
            
            if response:
                print(f"\nAI Response:\n{response}\n")
                
                # Speak the AI response
                if self.tts:
                    self.tts.speak_ai_response(response)
            else:
                print("Failed to get response from AI.")
            
            # Update cooldown
            self.last_capture_time = current_time
            
        except Exception as e:
            print(f"Error in capture and analyze: {e}")

    def capture_spoken_prompt(self) -> Optional[str]:
        """Record the user's spoken question and transcribe it via ElevenLabs."""
        try:
            with sr.Microphone(sample_rate=self.audio_sample_rate) as source:
                print("\nListening... (speak your question)")
                # Calibrate to ambient noise for more reliable detection
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.max_listen_duration,
                    )
                except sr.WaitTimeoutError:
                    print("No speech detected within the listening window.")
                    return None

            print("Processing speech...")
            audio_bytes = audio.get_wav_data()
            transcript = self.transcribe_audio_with_elevenlabs(audio_bytes)

            if transcript:
                print(f"You said: {transcript}")
                return transcript

            print("Transcription failed or returned empty text.")
            return None

        except sr.UnknownValueError:
            print("Speech was unintelligible. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error capturing audio: {e}")
            return None

    def transcribe_audio_with_elevenlabs(self, audio_bytes: bytes) -> Optional[str]:
        """Send recorded audio to ElevenLabs Speech-to-Text and return the transcript."""
        if not audio_bytes:
            return None

        # Attempt SDK transcription first
        if self.eleven_client is not None:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "question.wav"
            try:
                result = self.eleven_client.speech_to_text.convert(
                    file=audio_buffer,
                    model_id=self.eleven_stt_model_id,
                    language_code=self.eleven_stt_language,
                )

                transcript = self._extract_transcript_from_result(result)
                if transcript:
                    return transcript
            except Exception as sdk_error:
                print(f"ElevenLabs SDK transcription error: {sdk_error}. Falling back to REST API.")

        # REST API fallback using requests
        try:
            response = requests.post(
                self.eleven_stt_endpoint,
                headers={
                    "xi-api-key": self.eleven_api_key,
                },
                data={
                    "model_id": self.eleven_stt_model_id,
                    "language_code": self.eleven_stt_language,
                },
                files={
                    "file": ("question.wav", audio_bytes, "audio/wav"),
                },
                timeout=90,
            )
            response.raise_for_status()

            result = response.json()
            transcript = self._extract_transcript_from_result(result)
            if transcript:
                return transcript

            print("ElevenLabs STT response did not include a transcript.")
            return None

        except requests.RequestException as rest_error:
            print(f"ElevenLabs REST transcription error: {rest_error}")
            return None

    @staticmethod
    def _extract_transcript_from_result(result: Optional[object]) -> Optional[str]:
        """Normalize various ElevenLabs STT response styles to a plain string."""
        if result is None:
            return None

        # Handle SDK response objects
        for attr in ("text", "transcription", "transcript"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Handle mapping/dict-style responses
        if isinstance(result, dict):
            for key in ("text", "transcription", "transcript"):
                if key in result and isinstance(result[key], str) and result[key].strip():
                    return result[key].strip()

            # Some responses may include segment arrays
            segments = result.get("segments") if isinstance(result.get("segments"), list) else None
            if segments:
                combined = " ".join(
                    segment.get("text", "").strip()
                    for segment in segments
                    if isinstance(segment, dict)
                ).strip()
                if combined:
                    return combined

        return None
    
    def run(self):
        """Main application loop."""
        print("Vision-Prompt Glasses with TTS")
        print("=============================")
        print("Instructions:")
        print("- Form a rectangle with both hands (thumb + index extended, others curled)")
        print("- Hold the gesture steady for 2 seconds")
        print("- Press 'q' to quit, 'r' to reset, 's' to select cached snapshot")
        print("- Press 't' to test OpenAI connection")
        print("- Press 'v' to test TTS")
        print()
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Test OpenAI connection
        print("Testing OpenAI connection...")
        if not self.openai_client.test_connection():
            print("Warning: OpenAI connection test failed. Check your API key.")
        else:
            print("OpenAI connection successful!")
        
        print("\nStarting camera feed...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame with MediaPipe
                annotated_frame, hand_data = self.hand_detector.process_frame(frame)
                
                # Get finger tips
                finger_tips = self.hand_detector.get_finger_tips(hand_data)
                
                # Detect frame gesture
                gesture_detected, corners, progress = self.frame_detector.detect_frame_gesture(
                    hand_data, finger_tips
                )
                
                # Draw overlay
                overlay_frame = self.crop_utils.draw_frame_overlay(annotated_frame, corners, progress)
                
                # Add instructions overlay
                cv2.putText(overlay_frame, "Form rectangle with both hands (thumb+index)", 
                           (10, overlay_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay_frame, "Hold steady for 2 seconds", 
                           (10, overlay_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Capture if gesture detected
                if gesture_detected and corners:
                    self.capture_and_analyze(frame, corners)
                    self.frame_detector.reset()
                
                # Display frame
                cv2.imshow('Vision-Prompt Glasses with TTS', overlay_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.frame_detector.reset()
                    print("Gesture detection reset")
                elif key == ord('s'):
                    snapshot_path = self.select_cached_snapshot()
                    if snapshot_path:
                        self.process_cached_snapshot(snapshot_path)
                elif key == ord('t'):
                    if self.openai_client.test_connection():
                        print("OpenAI connection test: SUCCESS")
                    else:
                        print("OpenAI connection test: FAILED")
                elif key == ord('v'):
                    if self.tts:
                        self.tts.speak_ai_response("TTS test successful! The text-to-speech system is working.")
                    else:
                        print("TTS not available")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.hand_detector.cleanup()
        
        print("Cleanup completed")

def main():
    """Main entry point."""
    try:
        app = VisionPromptGlassesWithTTS()
        app.run()
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    main()
