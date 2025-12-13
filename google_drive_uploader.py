"""
Google Drive Uploader Module

Handles uploading snapshots to Google Drive using the Google Drive API.
Implements Simple Upload as documented at:
https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive API scopes required for file upload
SCOPES = ['https://www.googleapis.com/auth/drive.file']


class GoogleDriveUploader:
    """
    Handles authentication and file uploads to Google Drive.
    
    Supports both OAuth 2.0 and service account authentication.
    """
    
    def __init__(self, folder_id: Optional[str] = None):
        """
        Initialize the Google Drive uploader.
        
        Args:
            folder_id: Google Drive folder ID where files should be uploaded.
                      If None, will be read from GOOGLE_DRIVE_FOLDER_ID env var.
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive API libraries not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        if not self.folder_id:
            raise ValueError(
                "Google Drive folder ID not configured. "
                "Set GOOGLE_DRIVE_FOLDER_ID environment variable or pass folder_id parameter."
            )
        
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """
        Authenticate with Google Drive API.
        
        Supports:
        1. Service account (via GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SERVICE_ACCOUNT_FILE)
        2. OAuth 2.0 (via environment variables or JSON files):
           - GOOGLE_DRIVE_CREDENTIALS_JSON (JSON string) or credentials.json file
           - GOOGLE_DRIVE_TOKEN_JSON (JSON string) or token.json file
        """
        creds = None
        
        # Try service account authentication first
        service_account_file = (
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or
            os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')
        )
        
        if service_account_file and os.path.exists(service_account_file):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    service_account_file, scopes=SCOPES
                )
                logger.info("Authenticated using service account")
            except Exception as e:
                logger.warning(f"Service account authentication failed: {e}")
        
        # Fall back to OAuth 2.0 if service account not available
        if creds is None or not creds.valid:
            # Try to load token from environment variable first, then file
            token_json = os.getenv('GOOGLE_DRIVE_TOKEN_JSON')
            token_path = os.getenv('GOOGLE_DRIVE_TOKEN_FILE', 'token.json')
            
            # Load existing token if available
            if token_json:
                try:
                    # Parse JSON string from environment variable
                    token_data = json.loads(token_json)
                    creds = Credentials.from_authorized_user_info(token_data, SCOPES)
                    logger.info("Loaded OAuth token from environment variable")
                except (json.JSONDecodeError, ValueError, Exception) as e:
                    logger.warning(f"Failed to load token from environment variable: {e}")
            elif os.path.exists(token_path):
                try:
                    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                    logger.info("Loaded OAuth token from file")
                except Exception as e:
                    logger.warning(f"Failed to load existing token file: {e}")
            
            # Refresh or get new token
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logger.info("Refreshed OAuth token")
                        # Save refreshed token (to file, since we can't modify env vars)
                        if os.path.exists(token_path) or not token_json:
                            with open(token_path, 'w') as token:
                                token.write(creds.to_json())
                            logger.info(f"Saved refreshed token to {token_path}")
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {e}")
                        creds = None
                
                # If still no valid credentials, start OAuth flow
                if not creds:
                    # Try to load credentials from environment variable first, then file
                    credentials_json = os.getenv('GOOGLE_DRIVE_CREDENTIALS_JSON')
                    credentials_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE', 'credentials.json')
                    
                    if credentials_json:
                        try:
                            # Parse JSON string from environment variable
                            client_config = json.loads(credentials_json)
                            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                            creds = flow.run_local_server(port=0)
                            logger.info("Completed OAuth flow using credentials from environment variable")
                        except (json.JSONDecodeError, ValueError, Exception) as e:
                            raise ValueError(
                                f"Failed to parse GOOGLE_DRIVE_CREDENTIALS_JSON: {e}. "
                                "Ensure it's a valid JSON string."
                            )
                    elif os.path.exists(credentials_path):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            credentials_path, SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                        logger.info("Completed OAuth flow using credentials file")
                    else:
                        raise FileNotFoundError(
                            "OAuth credentials not found. Set GOOGLE_DRIVE_CREDENTIALS_JSON "
                            f"environment variable or provide {credentials_path} file. "
                            "Download credentials.json from Google Cloud Console."
                        )
                
                # Save token for future use (to file, since we can't modify parent process env vars)
                if creds:
                    # Always save to file for persistence
                    # Note: If using env vars, you'll need to manually update GOOGLE_DRIVE_TOKEN_JSON
                    # with the new token after first authentication
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                    logger.info(f"Saved OAuth token to {token_path}")
                    if token_json:
                        logger.info(
                            "Note: Token saved to file. To use environment variable, "
                            f"update GOOGLE_DRIVE_TOKEN_JSON with the contents of {token_path}"
                        )
        
        if not creds or not creds.valid:
            raise ValueError("Failed to obtain valid Google Drive credentials")
        
        # Build the Drive service
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service initialized")
    
    def upload_file(self, file_path: str, file_name: Optional[str] = None) -> bool:
        """
        Upload a file to Google Drive using Simple Upload.
        
        Reference: https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple
        
        Args:
            file_path: Local path to the file to upload
            file_name: Optional custom name for the uploaded file.
                      If None, uses the original filename.
        
        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.service:
            logger.error("Google Drive service not initialized")
            return False
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Use original filename if not specified
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        try:
            # Determine MIME type from file extension
            mime_type = self._get_mime_type(file_path)
            
            # Prepare file metadata
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Create MediaFileUpload object
            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                resumable=False  # Simple upload (non-resumable for files < 5MB)
            )
            
            # Upload file using Simple Upload
            # Reference: https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            
            logger.info(f"Successfully uploaded {file_name} to Google Drive (ID: {file.get('id')})")
            return True
            
        except HttpError as error:
            logger.error(f"Google Drive API error: {error}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            return False
    
    @staticmethod
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
        return mime_types.get(ext, 'application/octet-stream')


def upload_snapshot_to_drive(snapshot_path: str) -> bool:
    """
    Convenience function to upload a snapshot to Google Drive.
    
    Args:
        snapshot_path: Path to the snapshot file to upload
    
    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        uploader = GoogleDriveUploader()
        return uploader.upload_file(snapshot_path)
    except Exception as e:
        logger.error(f"Failed to upload snapshot: {e}")
        return False
