import cv2
import numpy as np
import base64
from typing import List, Tuple, Optional
from PIL import Image
import io

class CropUtils:
    def __init__(self, output_width: int = 512, output_height: int = 512):
        """
        Initialize crop utilities for perspective transformation.
        
        Args:
            output_width: Width of output cropped image
            output_height: Height of output cropped image
        """
        self.output_width = output_width
        self.output_height = output_height
    
    def crop_frame_region(self, frame: np.ndarray, corners: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Crop the region inside the hand frame using perspective transformation.
        
        Args:
            frame: Input video frame
            corners: List of 4 corner points [(x,y), ...]
            
        Returns:
            Cropped image or None if transformation fails
        """
        if len(corners) != 4:
            return None
        
        # Convert corners to numpy array
        src_points = np.array(corners, dtype=np.float32)
        
        # Define destination points (rectangular output)
        dst_points = np.array([
            [0, 0],
            [self.output_width - 1, 0],
            [self.output_width - 1, self.output_height - 1],
            [0, self.output_height - 1]
        ], dtype=np.float32)
        
        try:
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transformation
            cropped = cv2.warpPerspective(frame, matrix, (self.output_width, self.output_height))
            
            return cropped
        except cv2.error as e:
            print(f"Perspective transformation failed: {e}")
            return None
    
    def encode_image_to_base64(self, image: np.ndarray, format: str = 'JPEG') -> Optional[str]:
        """
        Encode image to base64 string for API transmission.
        
        Args:
            image: Input image array
            format: Image format ('JPEG' or 'PNG')
            
        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Encode to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
        except Exception as e:
            print(f"Image encoding failed: {e}")
            return None
    
    def save_cropped_image(self, cropped_image: np.ndarray, filename: str) -> bool:
        """
        Save cropped image to file.
        
        Args:
            cropped_image: Cropped image array
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(filename, cropped_image)
            return success
        except Exception as e:
            print(f"Failed to save image {filename}: {e}")
            return False
    
    def draw_frame_overlay(self, frame: np.ndarray, corners: List[Tuple[float, float]], 
                          progress: float = 0.0) -> np.ndarray:
        """
        Draw frame overlay on the video frame for visual feedback.
        
        Args:
            frame: Input video frame
            corners: Frame corner points
            progress: Progress percentage (0.0 to 1.0)
            
        Returns:
            Frame with overlay drawn
        """
        overlay_frame = frame.copy()
        
        if corners is None or len(corners) != 4:
            return overlay_frame
        
        # Draw frame outline
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Color based on progress (red -> yellow -> green)
        if progress < 0.5:
            color = (0, 0, 255)  # Red
        elif progress < 1.0:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green
        
        # Draw frame outline
        cv2.polylines(overlay_frame, [pts], True, color, 3)
        
        # Draw corner points
        for corner in corners:
            cv2.circle(overlay_frame, (int(corner[0]), int(corner[1])), 8, color, -1)
        
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 10
        
        # Background bar
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        # Progress text
        progress_text = f"Progress: {progress:.1%}"
        cv2.putText(overlay_frame, progress_text, (bar_x, bar_y + bar_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay_frame
    
    def validate_crop_quality(self, cropped_image: np.ndarray) -> bool:
        """
        Validate that the cropped image is of good quality.
        
        Args:
            cropped_image: Cropped image to validate
            
        Returns:
            True if image quality is acceptable
        """
        if cropped_image is None:
            return False
        
        # Check image dimensions
        if cropped_image.shape[0] < 100 or cropped_image.shape[1] < 100:
            return False
        
        # Check for too much black/empty space
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 30)
        total_pixels = gray.shape[0] * gray.shape[1]
        black_ratio = black_pixels / total_pixels
        
        # If more than 80% is black, likely a bad crop
        if black_ratio > 0.8:
            return False
        
        return True
