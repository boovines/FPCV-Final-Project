import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from mediapipe_utils import HandDetector
from frame_detector import FrameDetector
from crop_utils import CropUtils
from openai_client import OpenAIClient
from mode_switch_detector import ModeSwitchDetector
from audio_service import AudioService
from mode_handlers import ModeHandlers


class VisionPromptGlasses:
    def __init__(self):
        load_dotenv()
        
        self.hand_detector = HandDetector()
        self.frame_detector = FrameDetector()
        self.mode_detector = ModeSwitchDetector(hold_duration=1.0)
        self.crop_utils = CropUtils()
        self.openai_client = OpenAIClient()
        self.audio_service = AudioService()
        self.mode_handlers = ModeHandlers(self.openai_client, self.audio_service)
        
        self.cap = None
        self.is_running = False
        self.is_processing = False
        self.processing_frame = None
        self.processing_lock = threading.Lock()
        self.snapshots_dir = "snapshots"
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        self.last_capture_time = 0
        self.capture_cooldown = 3.0
    
    def initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Couldn't open webcam")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def get_cached_snapshots(self) -> list:
        try:
            snapshot_files = []
            for filename in os.listdir(self.snapshots_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    snapshot_files.append(os.path.join(self.snapshots_dir, filename))
            return sorted(snapshot_files, key=os.path.getmtime, reverse=True)
        except Exception as e:
            print(f"Couldn't read snapshots: {e}")
            return []
    
    def select_cached_snapshot(self) -> Optional[str]:
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
        try:
            image = cv2.imread(snapshot_path)
            if image is None:
                print(f"Could not load image: {snapshot_path}")
                return
            
            image_base64 = self.crop_utils.encode_image_to_base64(image)
            if not image_base64:
                print("Couldn't process image")
                return
            
            prompt = input(f"\nEnter your question about the image '{os.path.basename(snapshot_path)}': ")
            if not prompt.strip():
                print("No prompt provided.")
                return
            
            print("Analyzing image...")
            response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
            
            if response:
                self.audio_service.output_ai_response(response)
            else:
                print("Couldn't analyze the image")
                
        except Exception as e:
            print(f"Analysis error: {e}")
    
    def _process_in_background(self, cropped_image: np.ndarray, snapshot_path: str, 
                               image_base64: str, mode: int):
        try:
            self.mode_handlers.handle_snapshot(
                cropped_image=cropped_image,
                snapshot_path=snapshot_path,
                image_base64=image_base64,
                mode=mode
            )
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            with self.processing_lock:
                self.is_processing = False
    
    def capture_and_analyze(self, frame: np.ndarray, corners: list, mode: int = 0):
        current_time = time.time()
        
        with self.processing_lock:
            if self.is_processing:
                return
            
            if current_time - self.last_capture_time < self.capture_cooldown:
                return
        
        try:
            cropped_image = self.crop_utils.crop_frame_region(frame, corners)
            if cropped_image is None:
                print("Couldn't crop image")
                return
            
            if not self.crop_utils.validate_crop_quality(cropped_image):
                print("Cropped image quality is too poor")
                return
            
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(self.snapshots_dir, f"snapshot_{timestamp}.jpg")
            
            if not self.crop_utils.save_cropped_image(cropped_image, snapshot_path):
                print("Couldn't save snapshot")
                return
            
            print(f"\nSnapshot saved: {snapshot_path}")
            
            image_base64 = self.crop_utils.encode_image_to_base64(cropped_image)
            if not image_base64:
                print("Couldn't process image")
                return
            
            with self.processing_lock:
                self.processing_frame = frame.copy()
                self.is_processing = True
                self.last_capture_time = current_time
            
            processing_thread = threading.Thread(
                target=self._process_in_background,
                args=(cropped_image, snapshot_path, image_base64, mode),
                daemon=True
            )
            processing_thread.start()
            
        except Exception as e:
            print(f"Capture error: {e}")
            with self.processing_lock:
                self.is_processing = False
    
    def run(self):
        print("Vision-Prompt Glasses Prototype")
        print("==============================")
        print("Instructions:")
        print("- Form a rectangle with both hands (thumb + index extended, others curled)")
        print("- Hold the gesture steady for 2 seconds")
        print("- Press 'q' to quit, 'r' to reset, 's' to select cached snapshot")
        print("- Press 't' to test OpenAI connection")
        print()
        
        if not self.initialize_camera():
            return
        
        print("Testing OpenAI connection...")
        if not self.openai_client.test_connection():
            print("OpenAI connection failed - check your API key")
        else:
            print("OpenAI connected!")
        
        print("\nStarting camera feed...")
        
        cv2.namedWindow('Vision-Prompt Glasses', cv2.WINDOW_AUTOSIZE)
        
        self.is_running = True
        
        try:
            frame_count = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera disconnected")
                    break
                
                frame_count += 1
                
                with self.processing_lock:
                    currently_processing = self.is_processing
                    frozen_frame = self.processing_frame.copy() if self.processing_frame is not None else None
                
                if not currently_processing:
                    annotated_frame, hand_data = self.hand_detector.process_frame(frame)
                    finger_tips = self.hand_detector.get_finger_tips(hand_data)
                    
                    gesture_detected, corners, progress = self.frame_detector.detect_frame_gesture(
                        hand_data, finger_tips
                    )
                    
                    mode_state = self.mode_detector.update(hand_data)
                    
                    overlay_frame = self.crop_utils.draw_frame_overlay(annotated_frame, corners, progress)
                    overlay_frame = self.crop_utils.draw_mode_switch_progress(
                        overlay_frame, mode_state.progress, mode_state.pending_mode
                    )
                    
                    cv2.putText(overlay_frame, "Form rectangle with both hands (thumb+index)", 
                               (10, overlay_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(overlay_frame, "Hold steady for 2 seconds", 
                               (10, overlay_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    mode_text = f"Mode: {mode_state.current_mode}"
                    text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = overlay_frame.shape[1] - text_size[0] - 20
                    text_y = overlay_frame.shape[0] - 20
                    cv2.putText(overlay_frame, mode_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if gesture_detected and corners:
                        self.capture_and_analyze(frame, corners, mode_state.current_mode)
                        self.frame_detector.reset()
                else:
                    if frozen_frame is not None:
                        overlay_frame = frozen_frame.copy()
                    else:
                        overlay_frame = frame.copy()
                    
                    overlay = overlay_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (overlay_frame.shape[1], overlay_frame.shape[0]), 
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, overlay_frame, 0.7, 0, overlay_frame)
                    
                    text = "Processing..."
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = (overlay_frame.shape[1] - text_size[0]) // 2
                    text_y = (overlay_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(overlay_frame, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                cv2.imshow('Vision-Prompt Glasses', overlay_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('r') and not currently_processing:
                    self.frame_detector.reset()
                    print("Gesture detection reset")
                elif key == ord('s') and not currently_processing:
                    snapshot_path = self.select_cached_snapshot()
                    if snapshot_path:
                        self.process_cached_snapshot(snapshot_path)
                elif key == ord('t') and not currently_processing:
                    if self.openai_client.test_connection():
                        print("OpenAI connected!")
                    else:
                        print("OpenAI connection failed")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nUnexpected error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.hand_detector.cleanup()


def main():
    try:
        app = VisionPromptGlasses()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()
