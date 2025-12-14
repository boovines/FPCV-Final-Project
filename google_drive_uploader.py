"""
Google Drive Uploader Module

Handles uploading snapshots to Google Drive using OAuth user authentication.
Implements Simple Upload as documented at:
https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple

Uses OAuth 2.0 with user consent for personal Gmail accounts.
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

# Google Drive API scope for file upload
# https://www.googleapis.com/auth/drive.file - allows creating and accessing files created by this app
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Default paths for OAuth credentials and tokens
DEFAULT_CLIENT_SECRET_FILE = 'client_secret.json'
DEFAULT_TOKEN_FILE = 'token.json'


class GoogleDriveUploader:
    """
    Handles OAuth user authentication and file uploads to Google Drive.
    
    Uses OAuth 2.0 with user consent for personal Gmail accounts.
    Uploads files to the authenticated user's My Drive.
    """
    
    def __init__(self, folder_id: Optional[str] = None):
        """
        Initialize the Google Drive uploader.
        
        Args:
            folder_id: Optional Google Drive folder ID where files should be uploaded.
                      If None, files will be uploaded to My Drive root.
                      Can be set via GOOGLE_DRIVE_FOLDER_ID env var.
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive API libraries not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """
        Authenticate with Google Drive API using OAuth 2.0 user consent.
        
        Flow:
        1. Load existing token from token.json if it exists
        2. Refresh token if expired
        3. If no token exists, launch OAuth consent flow
        4. Save token to token.json for future use
        
        Client credentials can be provided via:
        - GOOGLE_DRIVE_CLIENT_SECRET_JSON (JSON string) - preferred
        - client_secret.json file - fallback
        
        Token is always persisted to token.json.
        """
        creds = None
        token_path = DEFAULT_TOKEN_FILE
        
        # Load existing token from token.json
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                logger.info("Loaded OAuth token from token.json")
            except Exception as e:
                logger.warning(f"Failed to load existing token: {e}")
                creds = None
        
        # Refresh token if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Refreshed OAuth token")
                # Save refreshed token
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved refreshed token to {token_path}")
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}")
                creds = None
        
        # If no valid credentials, start OAuth flow
        if not creds or not creds.valid:
            # Load client credentials
            client_secret_json = os.getenv('GOOGLE_DRIVE_CLIENT_SECRET_JSON')
            client_secret_path = os.getenv('GOOGLE_DRIVE_CLIENT_SECRET_FILE', DEFAULT_CLIENT_SECRET_FILE)
            
            if client_secret_json:
                # Use client credentials from environment variable
                try:
                    client_config = json.loads(client_secret_json)
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                    logger.info("Starting OAuth flow using client credentials from environment variable...")
                    creds = flow.run_local_server(port=0)
                    logger.info("OAuth consent completed")
                except (json.JSONDecodeError, ValueError, Exception) as e:
                    raise ValueError(
                        f"Failed to parse GOOGLE_DRIVE_CLIENT_SECRET_JSON: {e}. "
                        "Ensure it's a valid JSON string with 'installed' or 'web' key."
                    )
            elif os.path.exists(client_secret_path):
                # Use client credentials from file
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                logger.info(f"Starting OAuth flow using client_secret.json...")
                creds = flow.run_local_server(port=0)
                logger.info("OAuth consent completed")
            else:
                raise FileNotFoundError(
                    f"OAuth client credentials not found. "
                    f"Set GOOGLE_DRIVE_CLIENT_SECRET_JSON environment variable or provide {client_secret_path} file. "
                    "Download client_secret.json from Google Cloud Console (OAuth 2.0 Client ID)."
                )
            
            # Save token to token.json for future use
            if creds:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved OAuth token to {token_path} for future use")
        
        # Final validation
        if not creds or not creds.valid:
            raise ValueError("Failed to obtain valid Google Drive OAuth credentials")
        
        # Build the Drive service
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service initialized with OAuth user authentication")
    
    def upload_file(self, file_path: str, file_name: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive using Simple Upload.
        
        Reference: https://developers.google.com/workspace/drive/api/guides/manage-uploads#simple
        
        Args:
            file_path: Local path to the file to upload
            file_name: Optional custom name for the uploaded file.
                      If None, uses the original filename.
        
        Returns:
            File ID if upload succeeded, None otherwise
        """
        if not self.service:
            logger.error("Google Drive service not initialized")
            return None
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        # Use original filename if not specified
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        try:
            # Determine MIME type from file extension
            mime_type = self._get_mime_type(file_path)
            
            # Prepare file metadata
            # If folder_id is provided, upload to that folder; otherwise upload to My Drive root
            file_metadata = {
                'name': file_name
            }
            if self.folder_id:
                file_metadata['parents'] = [self.folder_id]
            
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
            
            file_id = file.get('id')
            logger.info(f"Successfully uploaded {file_name} to Google Drive (ID: {file_id})")
            return file_id
            
        except HttpError as error:
            logger.error(f"Google Drive API error: {error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            return None
    
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
    
    def get_shareable_link(self, file_id: str) -> Optional[str]:
        """
        Make a file publicly accessible and get a shareable link.
        
        Args:
            file_id: Google Drive file ID
        
        Returns:
            Shareable link URL if successful, None otherwise
        """
        if not self.service:
            logger.error("Google Drive service not initialized")
            return None
        
        try:
            # Make file publicly accessible (anyone with the link can view)
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            # Get the shareable link
            file = self.service.files().get(
                fileId=file_id,
                fields='webViewLink, webContentLink'
            ).execute()
            
            # Prefer webContentLink (direct download) over webViewLink (preview)
            shareable_link = file.get('webContentLink') or file.get('webViewLink')
            
            if shareable_link:
                logger.info(f"Created shareable link for file {file_id}")
                return shareable_link
            else:
                # Fallback: construct link manually
                shareable_link = f"https://drive.google.com/uc?export=view&id={file_id}"
                logger.info(f"Using constructed shareable link for file {file_id}")
                return shareable_link
                
        except HttpError as error:
            logger.error(f"Google Drive API error creating shareable link: {error}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating shareable link: {e}")
            return None


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
        file_id = uploader.upload_file(snapshot_path)
        return file_id is not None
    except Exception as e:
        logger.error(f"Failed to upload snapshot: {e}")
        return False


def upload_snapshot_and_get_link(snapshot_path: str) -> Optional[str]:
    """
    Upload a snapshot to Google Drive and return a publicly shareable link.
    
    Args:
        snapshot_path: Path to the snapshot file to upload
    
    Returns:
        Publicly accessible shareable link if successful, None otherwise
    """
    try:
        uploader = GoogleDriveUploader()
        file_id = uploader.upload_file(snapshot_path)
        
        if not file_id:
            return None
        
        # Get shareable link
        shareable_link = uploader.get_shareable_link(file_id)
        return shareable_link
        
    except Exception as e:
        logger.error(f"Failed to upload snapshot and get shareable link: {e}")
        return None
