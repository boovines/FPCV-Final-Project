#!/usr/bin/env python3
"""
Demo script to show enhanced hand detection with visible knuckles and nodes.
This script runs the camera feed with enhanced hand landmark visualization.
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mediapipe_utils import HandDetector

def main():
    """Run hand detection demo with enhanced visualization."""
    print("Hand Detection Demo - Enhanced Visualization")
    print("=" * 50)
    print("This demo shows enhanced hand landmarks with:")
    print("• Green circles: Finger tips")
    print("• Yellow circles: PIP joints (middle finger joints)")
    print("• Red squares: MCP joints (knuckles)")
    print("• Magenta circle: Wrist")
    print("• T/I labels: Thumb and Index finger tips")
    print("\nPress 'q' to quit, 'r' to reset")
    print("=" * 50)
    
    # Initialize hand detector
    detector = HandDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting camera feed...")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Process frame with enhanced hand detection
            annotated_frame, hand_data = detector.process_frame(frame)
            
            # Add instruction overlay
            cv2.putText(annotated_frame, "Enhanced Hand Detection Demo", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Green: Tips, Yellow: PIP, Red: Knuckles, Magenta: Wrist", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit, 'r' to reset", 
                       (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Enhanced Hand Detection Demo', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reset hand detection")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("Demo completed")

if __name__ == "__main__":
    main()
