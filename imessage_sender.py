"""
iMessage Sender Module

Handles sending image messages via iMessage/SMS using macOS Messages.app and AppleScript.
Sends as iMessage if recipient is on Apple, otherwise falls back to SMS/MMS via paired iPhone.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_snapshot_via_imessage(snapshot_path: str) -> bool:
    """Send a text message "Hello" via iMessage/SMS."""
    recipient_phone = os.getenv('IMESSAGE_RECIPIENT')
    
    if not recipient_phone:
        logger.error(
            "IMESSAGE_RECIPIENT environment variable not set. "
            "Set IMESSAGE_RECIPIENT to the recipient's phone number (e.g., +1234567890)"
        )
        return False
    
    try:
        script = f'''
        tell application "Messages"
            activate
            
            -- Try to find or create the chat and send
            try
                -- First try iMessage service
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{recipient_phone}" of targetService
                
                -- Try to get existing chat, or send to buddy to create one
                try
                    set targetChat to chat id (get id of targetBuddy)
                    send "Hello" to targetChat
                on error
                    -- No existing chat, send directly to buddy (creates chat and sends)
                    send "Hello" to targetBuddy
                end try
                
            on error iMessageError
                -- If iMessage fails, try SMS service
                try
                    set targetService to 1st service whose service type = SMS
                    set targetBuddy to buddy "{recipient_phone}" of targetService
                    
                    try
                        set targetChat to chat id (get id of targetBuddy)
                        send "Hello" to targetChat
                    on error
                        send "Hello" to targetBuddy
                    end try
                    
                on error smsError
                    -- Both failed, log and re-raise
                    error "Failed to send via iMessage or SMS: " & iMessageError & " / " & smsError
                end try
            end try
        end tell
        
        -- Wait for message to appear, then ensure it's sent using System Events
        -- This addresses the known issue where Messages.app places messages in outbox but doesn't send them
        delay 2.0
        tell application "System Events"
            tell process "Messages"
                set frontmost to true
                delay 0.5
                -- Focus the message input field and send
                try
                    -- Try to find and click the message input area
                    set messageWindow to window 1
                    set messageField to text field 1 of scroll area 1 of splitter group 1 of messageWindow
                    click messageField
                    delay 0.3
                end try
                -- Press Enter/Return to actually send the message
                keystroke return
                delay 1.0
            end tell
        end tell
        '''
        
        logger.info(f"Sending text message 'Hello' via iMessage/SMS to {recipient_phone}")
        
        # Execute AppleScript
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("Text message 'Hello' sent successfully via iMessage/SMS")
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Failed to send iMessage: {error_msg}")
            
            # Provide helpful error messages for common issues
            if "not allowed assistive access" in error_msg.lower() or "not authorized" in error_msg.lower():
                logger.error(
                    "macOS automation permission required. "
                    "Go to System Settings > Privacy & Security > Automation "
                    "and allow Terminal/Python to control Messages.app"
                )
            elif "not signed in" in error_msg.lower():
                logger.error(
                    "Messages.app is not signed in. "
                    "Please sign in to Messages.app with your Apple ID."
                )
            
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("iMessage send timed out after 30 seconds")
        return False
    except FileNotFoundError:
        logger.error(
            "osascript not found. This script requires macOS. "
            "iMessage sending is only available on macOS systems."
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending iMessage: {e}")
        return False


def send_text_via_imessage(text: str) -> bool:
    """Send a text message via iMessage/SMS."""
    recipient_phone = os.getenv('IMESSAGE_RECIPIENT')
    
    if not recipient_phone:
        logger.error(
            "IMESSAGE_RECIPIENT environment variable not set. "
            "Set IMESSAGE_RECIPIENT to the recipient's phone number (e.g., +1234567890)"
        )
        return False
    
    escaped_text = text.replace('"', '\\"')
    
    try:
        script = f'''
        tell application "Messages"
            activate
            
            -- Try to find or create the chat and send
            try
                -- First try iMessage service
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{recipient_phone}" of targetService
                
                -- Try to get existing chat, or send to buddy to create one
                try
                    set targetChat to chat id (get id of targetBuddy)
                    send "{escaped_text}" to targetChat
                on error
                    -- No existing chat, send directly to buddy (creates chat and sends)
                    send "{escaped_text}" to targetBuddy
                end try
                
            on error iMessageError
                -- If iMessage fails, try SMS service
                try
                    set targetService to 1st service whose service type = SMS
                    set targetBuddy to buddy "{recipient_phone}" of targetService
                    
                    try
                        set targetChat to chat id (get id of targetBuddy)
                        send "{escaped_text}" to targetChat
                    on error
                        send "{escaped_text}" to targetBuddy
                    end try
                    
                on error smsError
                    -- Both failed, log and re-raise
                    error "Failed to send via iMessage or SMS: " & iMessageError & " / " & smsError
                end try
            end try
        end tell
        
        -- Wait for message to appear, then ensure it's sent using System Events
        -- This addresses the known issue where Messages.app places messages in outbox but doesn't send them
        delay 2.0
        tell application "System Events"
            tell process "Messages"
                set frontmost to true
                delay 0.5
                -- Focus the message input field and send
                try
                    -- Try to find and click the message input area
                    set messageWindow to window 1
                    set messageField to text field 1 of scroll area 1 of splitter group 1 of messageWindow
                    click messageField
                    delay 0.3
                end try
                -- Press Enter/Return to actually send the message
                keystroke return
                delay 1.0
            end tell
        end tell
        '''
        
        logger.info(f"Sending text message via iMessage/SMS to {recipient_phone}")
        
        # Execute AppleScript
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("Text message sent successfully via iMessage/SMS")
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Failed to send iMessage: {error_msg}")
            
            # Provide helpful error messages for common issues
            if "not allowed assistive access" in error_msg.lower() or "not authorized" in error_msg.lower():
                logger.error(
                    "macOS automation permission required. "
                    "Go to System Settings > Privacy & Security > Automation "
                    "and allow Terminal/Python to control Messages.app"
                )
            elif "not signed in" in error_msg.lower():
                logger.error(
                    "Messages.app is not signed in. "
                    "Please sign in to Messages.app with your Apple ID."
                )
            
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("iMessage send timed out after 30 seconds")
        return False
    except FileNotFoundError:
        logger.error(
            "osascript not found. This script requires macOS. "
            "iMessage sending is only available on macOS systems."
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending iMessage: {e}")
        return False


def send_text_and_image(phone: str, text: str, image_path: str) -> bool:
    """Send both text and image via iMessage/SMS."""
    image_path_resolved = Path(image_path).resolve()
    
    try:
        script = f'''
        tell application "Messages"
            activate
            set imageFile to POSIX file "{image_path_resolved}"
            
            -- Try to find or create the chat and send
            try
                -- First try iMessage service
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{phone}" of targetService
                
                -- Try to get existing chat, or send to buddy to create one
                try
                    set targetChat to chat id (get id of targetBuddy)
                    send "{text}" to targetChat
                    delay 0.5
                    send imageFile to targetChat
                on error
                    -- No existing chat, send directly to buddy (creates chat and sends)
                    send "{text}" to targetBuddy
                    delay 0.5
                    send imageFile to targetBuddy
                end try
                
            on error iMessageError
                -- If iMessage fails, try SMS service
                try
                    set targetService to 1st service whose service type = SMS
                    set targetBuddy to buddy "{phone}" of targetService
                    
                    try
                        set targetChat to chat id (get id of targetBuddy)
                        send "{text}" to targetChat
                        delay 0.5
                        send imageFile to targetChat
                    on error
                        send "{text}" to targetBuddy
                        delay 0.5
                        send imageFile to targetBuddy
                    end try
                    
                on error smsError
                    -- Both failed, log and re-raise
                    error "Failed to send via iMessage or SMS: " & iMessageError & " / " & smsError
                end try
            end try
        end tell
        
        -- Wait for message to appear, then ensure it's sent using System Events
        -- This addresses the known issue where Messages.app places messages in outbox but doesn't send them
        delay 2.0
        tell application "System Events"
            tell process "Messages"
                set frontmost to true
                delay 0.5
                -- Focus the message input field and send
                try
                    -- Try to find and click the message input area
                    set messageWindow to window 1
                    set messageField to text field 1 of scroll area 1 of splitter group 1 of messageWindow
                    click messageField
                    delay 0.3
                end try
                -- Press Enter/Return to actually send the message
                keystroke return
                delay 1.0
            end tell
        end tell
        '''
        
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("Text and image sent successfully")
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Failed to send text and image: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error sending text and image: {e}")
        return False
