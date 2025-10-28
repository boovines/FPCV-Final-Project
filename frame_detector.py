import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import time

class FrameDetector:
    def __init__(self, stability_threshold: float = 2.0, stability_tolerance: float = 30.0):
        """
        Initialize frame detector for hand gesture recognition.
        
        Args:
            stability_threshold: Time in seconds to hold gesture before triggering
            stability_tolerance: Pixel distance tolerance for "steady" detection
        """
        self.stability_threshold = stability_threshold
        self.stability_tolerance = stability_tolerance
        
        # Track gesture state
        self.gesture_start_time = None
        self.last_frame_corners = None
        self.is_gesture_active = False
        
    def is_proper_hand_pose(self, hand_data: List[dict]) -> bool:
        """
        Check if both hands are in the correct pose (thumb + index extended, others curled).
        
        Args:
            hand_data: List of hand data from MediaPipe
            
        Returns:
            True if both hands are in correct pose
        """
        if len(hand_data) != 2:
            return False
        
        for hand in hand_data:
            landmarks = hand['landmarks']
            
            # Check if thumb and index are extended (tips are furthest from palm)
            # Check if middle, ring, and pinky are curled (tips are closer to palm)
            
            # Palm center (wrist landmark)
            wrist = landmarks[0]
            
            # Finger tips
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Calculate distances from palm center
            thumb_dist = self._distance(wrist, thumb_tip)
            index_dist = self._distance(wrist, index_tip)
            middle_dist = self._distance(wrist, middle_tip)
            ring_dist = self._distance(wrist, ring_tip)
            pinky_dist = self._distance(wrist, pinky_tip)
            
            # Thumb and index should be extended (further from palm)
            # Middle, ring, pinky should be curled (closer to palm)
            if not (thumb_dist > middle_dist and thumb_dist > ring_dist and thumb_dist > pinky_dist):
                return False
            if not (index_dist > middle_dist and index_dist > ring_dist and index_dist > pinky_dist):
                return False
                
        return True
    
    def get_frame_corners(self, finger_tips: dict) -> Optional[List[Tuple[float, float]]]:
        """
        Extract the four corner points from hand frame gesture.
        
        Args:
            finger_tips: Dict with 'left' and 'right' keys containing thumb/index coordinates
            
        Returns:
            List of 4 corner points [(x,y), ...] or None if invalid frame
        """
        if not finger_tips['left'] or not finger_tips['right']:
            return None
        
        # Extract corner points
        left_thumb = finger_tips['left']['thumb']
        left_index = finger_tips['left']['index']
        right_thumb = finger_tips['right']['thumb']
        right_index = finger_tips['right']['index']
        
        corners = [
            (left_thumb['x'], left_thumb['y']),
            (left_index['x'], left_index['y']),
            (right_index['x'], right_index['y']),
            (right_thumb['x'], right_thumb['y'])
        ]
        
        # Validate frame geometry
        if self._is_valid_frame(corners):
            return corners
        
        return None
    
    def _is_valid_frame(self, corners: List[Tuple[float, float]]) -> bool:
        """
        Validate that the four corners form a reasonable frame.
        
        Args:
            corners: List of 4 corner points
            
        Returns:
            True if frame is valid
        """
        if len(corners) != 4:
            return False
        
        # Check minimum frame size
        min_size = 50  # pixels
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.sqrt((corners[i][0] - corners[j][0])**2 + (corners[i][1] - corners[j][1])**2)
                if dist < min_size:
                    return False
        
        # Check if frame is roughly rectangular (not too skewed)
        # Calculate area using shoelace formula
        area = 0
        for i in range(4):
            j = (i + 1) % 4
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = abs(area) / 2
        
        # Minimum area threshold
        if area < 1000:  # pixels squared
            return False
        
        return True
    
    def _distance(self, point1: dict, point2: dict) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
    
    def check_gesture_stability(self, corners: List[Tuple[float, float]]) -> Tuple[bool, float]:
        """
        Check if the gesture has been held stable for the required duration.
        
        Args:
            corners: Current frame corner points
            
        Returns:
            Tuple of (is_stable, progress_percentage)
        """
        current_time = time.time()
        
        if corners is None:
            self.gesture_start_time = None
            self.last_frame_corners = None
            self.is_gesture_active = False
            return False, 0.0
        
        # Check if corners have moved significantly
        if self.last_frame_corners is not None:
            max_movement = 0
            for i in range(4):
                movement = np.sqrt(
                    (corners[i][0] - self.last_frame_corners[i][0])**2 + 
                    (corners[i][1] - self.last_frame_corners[i][1])**2
                )
                max_movement = max(max_movement, movement)
            
            if max_movement > self.stability_tolerance:
                # Gesture moved too much, reset
                self.gesture_start_time = None
                self.last_frame_corners = None
                self.is_gesture_active = False
                return False, 0.0
        
        # Update tracking
        self.last_frame_corners = corners.copy()
        
        if self.gesture_start_time is None:
            self.gesture_start_time = current_time
        
        # Calculate progress
        elapsed_time = current_time - self.gesture_start_time
        progress = min(elapsed_time / self.stability_threshold, 1.0)
        
        # Check if stable for required duration
        is_stable = elapsed_time >= self.stability_threshold
        
        return is_stable, progress
    
    def detect_frame_gesture(self, hand_data: List[dict], finger_tips: dict) -> Tuple[bool, List[Tuple[float, float]], float]:
        """
        Main detection function that combines pose detection and stability tracking.
        
        Args:
            hand_data: List of hand data from MediaPipe
            finger_tips: Finger tip coordinates
            
        Returns:
            Tuple of (gesture_detected, corners, progress_percentage)
        """
        # Check if both hands are in correct pose
        if not self.is_proper_hand_pose(hand_data):
            self.gesture_start_time = None
            self.last_frame_corners = None
            self.is_gesture_active = False
            return False, None, 0.0
        
        # Get frame corners
        corners = self.get_frame_corners(finger_tips)
        if corners is None:
            self.gesture_start_time = None
            self.last_frame_corners = None
            self.is_gesture_active = False
            return False, None, 0.0
        
        # Check stability
        is_stable, progress = self.check_gesture_stability(corners)
        
        return is_stable, corners, progress
    
    def reset(self):
        """Reset the detector state."""
        self.gesture_start_time = None
        self.last_frame_corners = None
        self.is_gesture_active = False
