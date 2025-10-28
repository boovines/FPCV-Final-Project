import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Optional

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient

class VisionPromptGlasses:
    def __init__(self):
        """Initialize the Vision-Prompt Glasses prototype."""
        # Initialize components
        self.hand_detector = HandDetector()
        self.frame_detector = FrameDetector()
        self.crop_utils = CropUtils()
        self.openai_client = OpenAIClient()
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Create snapshots directory
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # State tracking
        self.last_capture_time = 0
        self.capture_cooldown = 3.0  # seconds between captures
        
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
        for i, snapshot in enumerate(snapshots[:10]):  # Show last 10
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
            
            # Get user prompt
            prompt = input("\nEnter your question about the captured image: ")
            if not prompt.strip():
                print("No prompt provided.")
                return
            
            # Send to OpenAI
            print("Analyzing image...")
            response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
            
            if response:
                print(f"\nAI Response:\n{response}\n")
            else:
                print("Failed to get response from AI.")
            
            # Update cooldown
            self.last_capture_time = current_time
            
        except Exception as e:
            print(f"Error in capture and analyze: {e}")
    
    def run(self):
        """Main application loop."""
        print("Vision-Prompt Glasses Prototype")
        print("==============================")
        print("Instructions:")
        print("- Form a rectangle with both hands (thumb + index extended, others curled)")
        print("- Hold the gesture steady for 2 seconds")
        print("- Press 'q' to quit, 'r' to reset, 's' to select cached snapshot")
        print("- Press 't' to test OpenAI connection")
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
                cv2.imshow('Vision-Prompt Glasses', overlay_frame)
                
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
        app = VisionPromptGlasses()
        app.run()
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    main()
