import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModeState:
    """Snapshot of the current mode switching state."""

    current_mode: int
    pending_mode: Optional[int]
    progress: float
    changed: bool


class ModeSwitchDetector:
    """
    Detects right-hand gestures to toggle between application modes.
    Modes are sticky - once set, they persist until explicitly changed.

    Gestures:
        • Thumbs up              → Mode 0
        • Thumb + Index pinch   → Mode 1
        • Thumb + Middle pinch  → Mode 2
        • Thumb + Ring pinch    → Mode 3
        • Thumb + Pinky pinch   → Mode 4
    """

    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    INDEX_MCP = 5
    PINKY_MCP = 17
    WRIST = 0

    def __init__(
        self,
        hold_duration: float = 2.0,
        contact_threshold_ratio: float = 0.28,
        lost_contact_grace_period: float = 0.25,
    ):
        self.hold_duration = hold_duration
        self.contact_threshold_ratio = contact_threshold_ratio
        self.lost_contact_grace_period = lost_contact_grace_period

        self.current_mode: int = 0
        self.pending_mode: Optional[int] = None
        self.gesture_start_time: Optional[float] = None
        self.last_switch_time: Optional[float] = None
        self.last_detection_time: Optional[float] = None
        self.last_detected_mode: Optional[int] = None

    def update(self, hand_data: List[Dict]) -> ModeState:
        """Process the latest hand landmarks and update the mode state."""
        now = time.time()
        active_hand = self._select_active_hand(hand_data)

        if not active_hand:
            self._reset_pending()
            return ModeState(self.current_mode, None, 0.0, False)

        landmarks = active_hand["landmarks"]
        palm_size = self._estimate_palm_size(landmarks)
        contact_threshold = palm_size * self.contact_threshold_ratio

        # Check for thumbs up first (mode 0), then pinch gestures (modes 1-4)
        target_mode = self._detect_thumbs_up(landmarks)
        if target_mode is None:
            target_mode = self._detect_pinched_mode(landmarks, contact_threshold)

        if target_mode is not None:
            self.last_detected_mode = target_mode
            self.last_detection_time = now
        elif (
            self.pending_mode is not None
            and self.last_detection_time is not None
            and now - self.last_detection_time <= self.lost_contact_grace_period
        ):
            target_mode = self.pending_mode
        else:
            self._reset_pending()
            return ModeState(self.current_mode, None, 0.0, False)

        if target_mode == self.current_mode:
            # Already in this mode – require release before re-triggering.
            self._reset_pending()
            return ModeState(self.current_mode, None, 0.0, False)

        if self.pending_mode != target_mode:
            self.pending_mode = target_mode
            self.gesture_start_time = now

        assert self.gesture_start_time is not None
        elapsed = now - self.gesture_start_time
        progress = max(0.0, min(elapsed / self.hold_duration, 1.0))

        if elapsed >= self.hold_duration:
            changed = self._set_mode(target_mode, now)
            self._reset_pending()
            return ModeState(self.current_mode, None, 0.0, changed)

        return ModeState(self.current_mode, self.pending_mode, progress, False)

    def _select_active_hand(self, hand_data: List[Dict]) -> Optional[Dict]:
        for hand in hand_data:
            handedness = hand.get("handedness", "").lower()
            if handedness == "right":
                return hand
        for hand in hand_data:
            handedness = hand.get("handedness", "").lower()
            if handedness == "left":
                return hand
        return hand_data[0] if hand_data else None

    def _set_mode(self, mode: int, timestamp: float) -> bool:
        if mode != self.current_mode:
            self.current_mode = mode
            self.last_switch_time = timestamp
            return True
        return False

    def _reset_pending(self):
        self.pending_mode = None
        self.gesture_start_time = None
        self.last_detection_time = None
        self.last_detected_mode = None

    def _estimate_palm_size(self, landmarks: List[Dict]) -> float:
        width = self._distance(landmarks[self.INDEX_MCP], landmarks[self.PINKY_MCP])
        if width <= 0.0:
            width = self._distance(landmarks[self.WRIST], landmarks[self.MIDDLE_TIP])
        return max(width, 1.0)

    def _detect_pinched_mode(self, landmarks: List[Dict], threshold: float) -> Optional[int]:
        thumb_tip = landmarks[self.THUMB_TIP]
        finger_indices = {
            1: landmarks[self.INDEX_TIP],
            2: landmarks[self.MIDDLE_TIP],
            3: landmarks[self.RING_TIP],
            4: landmarks[self.PINKY_TIP],
        }

        closest_mode = None
        closest_distance = math.inf

        for mode, tip in finger_indices.items():
            distance = self._distance(thumb_tip, tip)
            if distance <= threshold and distance < closest_distance:
                closest_distance = distance
                closest_mode = mode

        return closest_mode

    def _detect_thumbs_up(self, landmarks: List[Dict]) -> Optional[int]:
        """Detect thumbs up gesture for mode 0."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        
        index_tip = landmarks[self.INDEX_TIP]
        index_pip = landmarks[self.INDEX_PIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        middle_pip = landmarks[self.MIDDLE_PIP]
        ring_tip = landmarks[self.RING_TIP]
        ring_pip = landmarks[self.RING_PIP]
        pinky_tip = landmarks[self.PINKY_TIP]
        pinky_pip = landmarks[self.PINKY_PIP]
        
        # Thumb should be extended upward (tip Y < IP Y and MCP Y)
        thumb_extended = (thumb_tip["y"] < thumb_ip["y"]) and (thumb_tip["y"] < thumb_mcp["y"])
        
        # Other fingers should be curled (tips Y > PIP Y)
        fingers_curled = (
            index_tip["y"] > index_pip["y"] and
            middle_tip["y"] > middle_pip["y"] and
            ring_tip["y"] > ring_pip["y"] and
            pinky_tip["y"] > pinky_pip["y"]
        )
        
        if thumb_extended and fingers_curled:
            return 0
        
        return None

    @staticmethod
    def _distance(point_a: Dict, point_b: Dict) -> float:
        dx = point_a["x"] - point_b["x"]
        dy = point_a["y"] - point_b["y"]
        return math.hypot(dx, dy)

