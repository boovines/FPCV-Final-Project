import cv2
import numpy as np
import base64
from typing import List, Tuple, Optional
from PIL import Image
import io

class CropUtils:
    def __init__(self, output_width: int = 512, output_height: int = 512):
        self.output_width = output_width
        self.output_height = output_height
    
    def crop_frame_region(self, frame: np.ndarray, corners: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        if len(corners) != 4:
            return None
        
        src_points = np.array(corners, dtype=np.float32)
        dst_points = np.array([
            [0, 0],
            [self.output_width - 1, 0],
            [self.output_width - 1, self.output_height - 1],
            [0, self.output_height - 1]
        ], dtype=np.float32)
        
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            cropped = cv2.warpPerspective(frame, matrix, (self.output_width, self.output_height))
            
            return cropped
        except cv2.error as e:
            print(f"Crop failed: {e}")
            return None
    
    def encode_image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> Optional[str]:
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
        except Exception as e:
            print(f"Encoding error: {e}")
            return None
    
    def save_cropped_image(self, cropped_image: np.ndarray, filename: str) -> bool:
        try:
            success = cv2.imwrite(filename, cropped_image)
            return success
        except Exception as e:
            print(f"Couldn't save image: {e}")
            return False
    
    def draw_frame_overlay(self, frame: np.ndarray, corners: List[Tuple[float, float]], 
                          progress: float = 0.0) -> np.ndarray:
        overlay_frame = frame.copy()
        
        if corners is None or len(corners) != 4:
            return overlay_frame
        
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        if progress < 0.5:
            color = (0, 0, 255)
        elif progress < 1.0:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        
        cv2.polylines(overlay_frame, [pts], True, color, 3)
        
        for corner in corners:
            cv2.circle(overlay_frame, (int(corner[0]), int(corner[1])), 8, color, -1)
        
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 10
        
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        progress_width = int(bar_width * progress)
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        progress_text = f"Frame: {progress:.1%}"
        cv2.putText(overlay_frame, progress_text, (bar_x, bar_y + bar_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay_frame
    
    def draw_mode_switch_progress(self, frame: np.ndarray, progress: float, pending_mode: Optional[int]) -> np.ndarray:
        overlay_frame = frame.copy()
        
        if pending_mode is None or progress <= 0.0:
            return overlay_frame
        
        if progress < 0.5:
            color = (0, 0, 255)
        elif progress < 1.0:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 50
        
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        progress_width = int(bar_width * progress)
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        progress_text = f"Mode {pending_mode}: {progress:.1%}"
        cv2.putText(overlay_frame, progress_text, (bar_x, bar_y + bar_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay_frame
    
    def validate_crop_quality(self, cropped_image: np.ndarray) -> bool:
        if cropped_image is None:
            return False
        
        if cropped_image.shape[0] < 100 or cropped_image.shape[1] < 100:
            return False
        
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 30)
        total_pixels = gray.shape[0] * gray.shape[1]
        black_ratio = black_pixels / total_pixels
        
        if black_ratio > 0.8:
            return False
        
        return True
