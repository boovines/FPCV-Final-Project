"""
Mode Handlers Module

Contains all mode-specific functionality for the Vision-Prompt Glasses:
- Mode 0: AI Processing with STT/TTS
- Mode 1: Google Drive Upload
- Mode 2: Google Drive Upload and iMessage Link
- Mode 3: Location Detection
- Mode 4: Visual Product Search
"""

import os
import numpy as np
from typing import Optional

# Import optional dependencies
try:
    from google_drive_uploader import upload_snapshot_to_drive, upload_snapshot_and_get_link
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    upload_snapshot_to_drive = None
    upload_snapshot_and_get_link = None

try:
    from imessage_sender import send_text_via_imessage
    IMESSAGE_AVAILABLE = True
except ImportError:
    IMESSAGE_AVAILABLE = False
    send_text_via_imessage = None

try:
    from product_search import search_similar_products
    PRODUCT_SEARCH_AVAILABLE = True
except ImportError:
    PRODUCT_SEARCH_AVAILABLE = False
    search_similar_products = None


class ModeHandlers:
    """Handles all mode-specific operations."""
    
    def __init__(self, openai_client, audio_service):
        """
        Initialize mode handlers with required services.
        
        Args:
            openai_client: OpenAI API client for image analysis
            audio_service: Audio service for TTS/STT
        """
        self.openai_client = openai_client
        self.audio_service = audio_service
    
    def handle_mode_0(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """Mode 0: AI Processing with STT/TTS"""
        prompt = self.audio_service.capture_spoken_prompt()
        if not prompt:
            return
        
        print("Analyzing image...")
        response = self.openai_client.analyze_with_default_prompt(image_base64, prompt)
        
        if response:
            self.audio_service.output_ai_response(response)
        else:
            print("Couldn't analyze the image")
    
    def handle_mode_1(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """Mode 1: Google Drive Upload"""
        if not GOOGLE_DRIVE_AVAILABLE or upload_snapshot_to_drive is None:
            print("Install Google Drive packages to use this mode")
            return
        
        print("Uploading to Google Drive...")
        
        upload_success = upload_snapshot_to_drive(snapshot_path)
        
        if upload_success:
            self.audio_service.output_ai_response("Upload completed.")
        else:
            print("Upload failed")
    
    def handle_mode_2(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """Mode 2: Google Drive Upload and iMessage Link"""
        if not GOOGLE_DRIVE_AVAILABLE or upload_snapshot_and_get_link is None:
            print("Install Google Drive packages to use this mode")
            return
        
        if not IMESSAGE_AVAILABLE or send_text_via_imessage is None:
            print("This mode requires macOS and Messages.app")
            return
        
        print("Uploading to Google Drive...")
        
        shareable_link = upload_snapshot_and_get_link(snapshot_path)
        
        if not shareable_link:
            print("Upload failed")
            self.audio_service.output_ai_response("Upload failed")
            return
        
        print("Sending via iMessage...")
        
        send_success = send_text_via_imessage(shareable_link)
        
        if send_success:
            self.audio_service.output_ai_response("Photo sent")
        else:
            print("Message send failed")
            self.audio_service.output_ai_response("Photo uploaded but couldn't send")
    
    def handle_mode_3(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """Mode 3: Location Detection"""
        print("Detecting location...")
        
        location_prompt = "Look at the contextual image. Guess where this is. Give a short answer, limited to 5 words, that just includes the region, city, country, etc. If you know where it is, just say \"The location is probably...\", and if you don't know, say \"I'm sorry, I'm not sure where this is.\""
        
        try:
            location = self.openai_client.analyze_image(
                image_base64=image_base64,
                prompt=location_prompt
            )
            
            if location:
                location = location.strip()
                self.audio_service.output_ai_response(location)
            else:
                self.audio_service.output_ai_response("Can't determine location")
        except Exception as e:
            print(f"Location detection error: {e}")
            self.audio_service.output_ai_response("Can't determine location")
    
    def handle_mode_4(self, cropped_image: np.ndarray, snapshot_path: str, image_base64: str):
        """Mode 4: Visual Product Search with SerpAPI Google Shopping"""
        if not PRODUCT_SEARCH_AVAILABLE or search_similar_products is None:
            print("Product search not available")
            self.audio_service.output_ai_response("Product search isn't available")
            return
        
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not serpapi_api_key:
            print("Set SERPAPI_API_KEY to use product search")
            self.audio_service.output_ai_response("Need SerpAPI key")
            return
        
        if not openai_api_key:
            print("Set OPENAI_API_KEY to use product search")
            self.audio_service.output_ai_response("Need OpenAI key")
            return
        
        try:
            image_url = f"data:image/jpeg;base64,{image_base64}"
            
            result = search_similar_products(
                image_url=image_url,
                serpapi_api_key=serpapi_api_key,
                openai_api_key=openai_api_key
            )
            
            products = result.get("results", [])
            
            if not products:
                self.audio_service.output_ai_response("Couldn't find similar products")
                return
            
            print("\nTop Similar Products:")
            
            for i, product in enumerate(products, 1):
                print(f"{i}. {product.get('product_name', 'Unknown')} by {product.get('brand', 'Unknown')}")
                print(f"   Similarity: {product.get('similarity_score', 0.0):.0%} - {product.get('product_url', 'N/A')}")
            
            num_products = len(products)
            if num_products == 1:
                top_product = products[0]
                tts_message = f"Found 1 similar product: {top_product.get('product_name', 'product')} by {top_product.get('brand', 'unknown brand')}."
            else:
                tts_message = f"Found {num_products} similar products. Top match: {products[0].get('product_name', 'product')}."
            
            self.audio_service.output_ai_response(tts_message)
            
        except Exception:
            self.audio_service.output_ai_response("Product search failed")
    
    def handle_snapshot(self, cropped_image: np.ndarray, snapshot_path: str, 
                       image_base64: str, mode: int):
        """
        Route snapshot to appropriate mode handler.
        
        Args:
            cropped_image: The cropped image as numpy array
            snapshot_path: Path where snapshot is saved
            image_base64: Base64-encoded image string
            mode: Current mode (0-4)
        """
        if mode == 0:
            self.handle_mode_0(cropped_image, snapshot_path, image_base64)
        elif mode == 1:
            self.handle_mode_1(cropped_image, snapshot_path, image_base64)
        elif mode == 2:
            self.handle_mode_2(cropped_image, snapshot_path, image_base64)
        elif mode == 3:
            self.handle_mode_3(cropped_image, snapshot_path, image_base64)
        elif mode == 4:
            self.handle_mode_4(cropped_image, snapshot_path, image_base64)
        else:
            print(f"Unknown mode {mode} - using mode 0")
            self.handle_mode_0(cropped_image, snapshot_path, image_base64)
