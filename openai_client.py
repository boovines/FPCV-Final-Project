import os
import openai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

class OpenAIClient:
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.max_tokens = 1000
        self.temperature = 0.7
    
    def analyze_image(self, image_base64: str, prompt: str, 
                     system_message: Optional[str] = None) -> Optional[str]:
        """Send image and prompt to OpenAI Vision API."""
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print("No response received from OpenAI API")
                return None
                
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error calling OpenAI API: {e}")
            return None
    
    def get_default_system_message(self) -> str:
        return """You are a helpful AI assistant that analyzes images. You will be given an image that may be blurry, distorted, and/or low quality - DO NOT COMMENT ON THE IMAGE QUALITY INCLUDING IT BEING DISTORTED OR BLURRY. Below, you will read a prompt from the user - respond to this prompt using the image as context."""
    
    def analyze_with_default_prompt(self, image_base64: str, 
                                  custom_prompt: Optional[str] = None) -> Optional[str]:
        """Analyze image with default or custom prompt."""
        if custom_prompt is None:
            custom_prompt = "Please describe what you see in this image. What are the main objects, text, or notable features?"
        
        return self.analyze_image(
            image_base64=image_base64,
            prompt=custom_prompt,
            system_message=self.get_default_system_message()
        )
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            print(f"OpenAI connection test failed: {e}")
            return False
