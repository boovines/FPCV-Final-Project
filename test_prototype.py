#!/usr/bin/env python3
"""
Test script for Vision-Prompt Glasses Prototype
This script tests the core functionality without requiring camera access.
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient

def test_hand_detector():
    """Test MediaPipe hand detection."""
    print("Testing HandDetector...")
    
    try:
        detector = HandDetector()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_frame, hand_data = detector.process_frame(dummy_frame)
        
        print(f"✓ HandDetector initialized successfully")
        print(f"✓ Processed frame shape: {annotated_frame.shape}")
        print(f"✓ Hand data: {len(hand_data)} hands detected")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ HandDetector test failed: {e}")
        return False

def test_frame_detector():
    """Test frame detection."""
    print("\nTesting FrameDetector...")
    
    try:
        detector = FrameDetector()
        dummy_hand_data = []
        dummy_finger_tips = {'left': None, 'right': None}
        
        gesture_detected, corners, progress = detector.detect_frame_gesture(
            dummy_hand_data, dummy_finger_tips
        )
        
        print(f"✓ FrameDetector initialized successfully")
        print(f"✓ Gesture detection: {gesture_detected}")
        print(f"✓ Progress: {progress:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ FrameDetector test failed: {e}")
        return False

def test_crop_utils():
    """Test image cropping and encoding."""
    print("\nTesting CropUtils...")
    
    try:
        crop_utils = CropUtils()
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_corners = [(100, 100), (200, 100), (200, 200), (100, 200)]
        cropped = crop_utils.crop_frame_region(dummy_frame, dummy_corners)
        
        if cropped is not None:
            print(f"✓ CropUtils initialized successfully")
            print(f"✓ Cropped image shape: {cropped.shape}")
            
            encoded = crop_utils.encode_image_to_base64(cropped)
            if encoded:
                print(f"✓ Base64 encoding successful (length: {len(encoded)})")
            else:
                print("✗ Base64 encoding failed")
                return False
        else:
            print("✗ Image cropping failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ CropUtils test failed: {e}")
        return False

def test_openai_client():
    """Test OpenAI client."""
    print("\nTesting OpenAIClient...")
    
    try:
        client = OpenAIClient()
        
        if client.test_connection():
            print("✓ OpenAIClient initialized successfully")
            print("✓ OpenAI API connection successful")
            return True
        else:
            print("✗ OpenAI API connection failed")
            return False
            
    except Exception as e:
        print(f"✗ OpenAIClient test failed: {e}")
        return False

def test_snapshots_directory():
    """Test snapshots directory."""
    print("\nTesting snapshots directory...")
    
    try:
        snapshots_dir = "snapshots"
        
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir, exist_ok=True)
            print(f"✓ Created snapshots directory: {snapshots_dir}")
        else:
            print(f"✓ Snapshots directory exists: {snapshots_dir}")
        
        test_file = os.path.join(snapshots_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        print("✓ Snapshots directory is writable")
        return True
        
    except Exception as e:
        print(f"✗ Snapshots directory test failed: {e}")
        return False

def main():
    print("Vision-Prompt Glasses Prototype - Test Suite")
    print("=" * 50)
    
    tests = [
        test_hand_detector,
        test_frame_detector,
        test_crop_utils,
        test_openai_client,
        test_snapshots_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The prototype is ready to run.")
        print("\nTo start the application, run:")
        print("  python3 main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
