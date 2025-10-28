import os
import openai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

class OpenAIClient:
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Default model configuration
        self.model = "gpt-4o"
        self.max_tokens = 1000
        self.temperature = 0.7
    
    def analyze_image(self, image_base64: str, prompt: str, 
                     system_message: Optional[str] = None) -> Optional[str]:
        """
        Send image and prompt to OpenAI Vision API for analysis.
        
        Args:
            image_base64: Base64 encoded image
            prompt: User query about the image
            system_message: Optional system message to set context
            
        Returns:
            Model response text or None if API call fails
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            # Add user message with image and prompt
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
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response text
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
        """Get default system message for image analysis."""
        return """You are a helpful AI assistant that analyzes images. 
        When given an image, provide clear, concise, and accurate descriptions of what you see.
        Focus on the main objects, text, colors, and any notable details.
        If asked specific questions about the image, answer them directly and helpfully."""
    
    def analyze_with_default_prompt(self, image_base64: str, 
                                  custom_prompt: Optional[str] = None) -> Optional[str]:
        """
        Analyze image with default prompt or custom prompt.
        
        Args:
            image_base64: Base64 encoded image
            custom_prompt: Custom user prompt, or None for default
            
        Returns:
            Model response text or None if API call fails
        """
        if custom_prompt is None:
            custom_prompt = "Please describe what you see in this image. What are the main objects, text, or notable features?"
        
        return self.analyze_image(
            image_base64=image_base64,
            prompt=custom_prompt,
            system_message=self.get_default_system_message()
        )
    
    def set_model_config(self, model: str = None, max_tokens: int = None, 
                        temperature: float = None):
        """
        Update model configuration.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4-vision-preview')
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0 to 1.0)
        """
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
    
    def test_connection(self) -> bool:
        """
        Test OpenAI API connection with a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
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
