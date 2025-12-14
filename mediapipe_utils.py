import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process a video frame and return annotated frame with hand landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        hand_data = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                self._draw_enhanced_landmarks(annotated_frame, hand_landmarks, frame.shape)
                
                landmarks = []
                h, w, _ = frame.shape
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'z': landmark.z
                    })
                
                handedness = results.multi_handedness[idx].classification[0].label
                
                hand_data.append({
                    'landmarks': landmarks,
                    'handedness': handedness,
                    'landmark_object': hand_landmarks
                })
        
        return annotated_frame, hand_data
    
    def get_finger_tips(self, hand_data: List[dict]) -> dict:
        """Extract thumb and index finger tip coordinates for both hands."""
        finger_tips = {'left': None, 'right': None}
        
        for hand in hand_data:
            handedness = hand['handedness'].lower()
            landmarks = hand['landmarks']
            
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            finger_tips[handedness] = {
                'thumb': thumb_tip,
                'index': index_tip
            }
        
        return finger_tips
    
    def _draw_enhanced_landmarks(self, frame: np.ndarray, hand_landmarks, frame_shape: tuple):
        """Draw enhanced landmarks with larger markers."""
        h, w = frame_shape[:2]
        
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcps = [2, 5, 9, 13, 17]
        
        for idx in finger_tips:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        
        for idx in finger_pips:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
        
        for idx in finger_mcps:
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.rectangle(frame, (x-6, y-6), (x+6, y+6), (255, 0, 0), -1)
            cv2.rectangle(frame, (x-8, y-8), (x+8, y+8), (255, 255, 255), 2)
        
        wrist = hand_landmarks.landmark[0]
        x = int(wrist.x * w)
        y = int(wrist.y * h)
        cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)
        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)
        
        cv2.putText(frame, "T", (thumb_x + 12, thumb_y - 5), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, "I", (index_x + 12, index_y - 5), font, font_scale, (255, 255, 255), thickness)
    
    def cleanup(self):
        self.hands.close()
