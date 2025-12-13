"""
InfoBip WhatsApp Sender Module

Handles sending image messages via WhatsApp using InfoBip's HTTPS API.
Implements WhatsApp Image Message as documented at:
https://www.infobip.com/docs/api/channels/whatsapp/whatsapp-outbound-messages/whatsapp-text-and-media-messages/send-whatsapp-image-message
"""

import os
import logging
import base64
import uuid
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_snapshot_via_whatsapp(snapshot_path: str) -> bool:
    """
    Send a snapshot image via WhatsApp using InfoBip's API.
    
    Reference: https://www.infobip.com/docs/api/channels/whatsapp/whatsapp-outbound-messages/whatsapp-text-and-media-messages/send-whatsapp-image-message
    
    Args:
        snapshot_path: Path to the snapshot image file to send
    
    Returns:
        True if message sent successfully, False otherwise
    """
    # Get configuration from environment variables
    base_url = os.getenv('INFOBIP_BASE_URL')
    api_key = os.getenv('INFOBIP_API_KEY')
    sender_number = os.getenv('INFOBIP_WHATSAPP_SENDER')
    recipient_number = os.getenv('INFOBIP_WHATSAPP_RECIPIENT')
    
    # Optional: Public URL for image (if InfoBip media upload is not available)
    public_image_url = os.getenv('INFOBIP_IMAGE_URL')
    
    # Validate required configuration
    if not base_url:
        logger.error("INFOBIP_BASE_URL environment variable not set")
        return False
    
    if not api_key:
        logger.error("INFOBIP_API_KEY environment variable not set")
        return False
    
    if not sender_number:
        logger.error("INFOBIP_WHATSAPP_SENDER environment variable not set")
        return False
    
    if not recipient_number:
        logger.error("INFOBIP_WHATSAPP_RECIPIENT environment variable not set")
        return False
    
    if not os.path.exists(snapshot_path):
        logger.error(f"Snapshot file not found: {snapshot_path}")
        return False
    
    try:
        # Determine media URL to use
        # Priority: 1) Public URL from env, 2) Upload to InfoBip, 3) Fail
        media_url = None
        
        if public_image_url:
            # Use provided public URL
            media_url = public_image_url
            logger.info(f"Using public image URL from configuration: {media_url}")
        else:
            # Try to upload image to InfoBip media storage
            # Read and encode image file to base64
            with open(snapshot_path, 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Determine MIME type from file extension
            mime_type = _get_mime_type(snapshot_path)
            
            # Upload image to InfoBip media storage to get a media URL
            # Reference: InfoBip requires media to be uploaded first, then referenced by URL
            media_url = _upload_media_to_infobip(base_url, api_key, image_base64, mime_type, snapshot_path)
            
            if not media_url:
                logger.error(
                    "Failed to upload image to InfoBip media storage. "
                    "InfoBip requires a publicly accessible image URL. "
                    "Either set INFOBIP_IMAGE_URL environment variable with a public URL, "
                    "or ensure InfoBip media upload is properly configured."
                )
                return False
        
        # Prepare WhatsApp message payload
        # Reference: https://www.infobip.com/docs/api/channels/whatsapp/whatsapp-outbound-messages/whatsapp-text-and-media-messages/send-whatsapp-image-message
        # Using the WhatsApp image message endpoint structure
        message_id = str(uuid.uuid4())
        
        payload = {
            "from": sender_number,
            "to": recipient_number,
            "messageId": message_id,
            "content": {
                "mediaUrl": media_url
            }
        }
        
        # Send WhatsApp message
        # Reference: InfoBip WhatsApp Image Message endpoint
        endpoint = f"{base_url.rstrip('/')}/whatsapp/1/message/image"
        
        headers = {
            "Authorization": f"App {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        logger.info(f"Sending WhatsApp image message to {recipient_number}")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        # Check response status
        if response.status_code in (200, 201):
            logger.info("WhatsApp image message sent successfully")
            return True
        else:
            logger.error(
                f"InfoBip API error: HTTP {response.status_code} - {response.text}"
            )
            return False
            
    except requests.RequestException as e:
        logger.error(f"Network error sending WhatsApp message: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending WhatsApp message: {e}")
        return False


def _upload_media_to_infobip(
    base_url: str,
    api_key: str,
    image_base64: str,
    mime_type: str,
    filename: str
) -> Optional[str]:
    """
    Upload image media to InfoBip's media storage and return the media URL.
    
    InfoBip requires media to be uploaded before it can be referenced in messages.
    This function uploads the image and returns a URL that can be used in the message.
    
    Args:
        base_url: InfoBip base URL
        api_key: InfoBip API key
        image_base64: Base64-encoded image data
        mime_type: MIME type of the image
        filename: Original filename
    
    Returns:
        Media URL if upload successful, None otherwise
    """
    try:
        # Try InfoBip Media API endpoint for uploading media
        # Reference: InfoBip may have a media upload endpoint
        endpoint = f"{base_url.rstrip('/')}/whatsapp/1/media"
        
        headers = {
            "Authorization": f"App {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prepare media upload payload
        payload = {
            "type": "IMAGE",
            "media": image_base64,
            "mimeType": mime_type,
            "fileName": os.path.basename(filename)
        }
        
        logger.info("Uploading image to InfoBip media storage")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        if response.status_code in (200, 201):
            result = response.json()
            # Try to get media URL from response
            media_url = result.get('url') or result.get('mediaUrl')
            if media_url:
                logger.info(f"Image uploaded successfully, media URL: {media_url}")
                return media_url
            
            # If URL not directly provided, try to construct it from media ID
            media_id = result.get('mediaId') or result.get('id')
            if media_id:
                # Construct media URL (InfoBip media URLs typically follow this pattern)
                constructed_url = f"{base_url.rstrip('/')}/media/{media_id}"
                logger.info(f"Image uploaded successfully, constructed URL: {constructed_url}")
                return constructed_url
            
            logger.error(f"Upload succeeded but no URL or ID in response: {result}")
            return None
        else:
            # If media upload endpoint doesn't exist or fails, log and return None
            # The caller will handle the error
            logger.warning(
                f"Media upload endpoint returned HTTP {response.status_code}: {response.text}. "
                "InfoBip may require a publicly accessible URL instead."
            )
            return None
            
    except requests.RequestException as e:
        logger.error(f"Network error uploading media: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error uploading media: {e}")
        return None


def _get_mime_type(file_path: str) -> str:
    """
    Infer MIME type from file extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        MIME type string
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }
    return mime_types.get(ext, 'image/jpeg')
