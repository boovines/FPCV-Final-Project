import os
import subprocess
from dotenv import load_dotenv

load_dotenv()


def send_text_via_imessage(text: str) -> bool:
    recipient_phone = os.getenv('IMESSAGE_RECIPIENT')
    
    if not recipient_phone:
        print("Set IMESSAGE_RECIPIENT in your environment to send messages")
        return False
    
    escaped_text = text.replace('"', '\\"')
    
    try:
        script = f'''
        tell application "Messages"
            activate
            
            try
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{recipient_phone}" of targetService
                
                try
                    set targetChat to chat id (get id of targetBuddy)
                    send "{escaped_text}" to targetChat
                on error
                    send "{escaped_text}" to targetBuddy
                end try
                
            on error iMessageError
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
                    error "Failed to send via iMessage or SMS: " & iMessageError & " / " & smsError
                end try
            end try
        end tell
        
        delay 2.0
        tell application "System Events"
            tell process "Messages"
                set frontmost to true
                delay 0.5
                try
                    set messageWindow to window 1
                    set messageField to text field 1 of scroll area 1 of splitter group 1 of messageWindow
                    click messageField
                    delay 0.3
                end try
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
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            
            if "not allowed assistive access" in error_msg.lower() or "not authorized" in error_msg.lower():
                print("Grant automation permission in System Settings > Privacy & Security > Automation")
            elif "not signed in" in error_msg.lower():
                print("Sign in to Messages.app with your Apple ID")
            else:
                print(f"Message send failed: {error_msg}")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("Message send timed out")
        return False
    except FileNotFoundError:
        print("This feature requires macOS")
        return False
    except Exception as e:
        print(f"Message send error: {e}")
        return False
