import os
import json
from typing import Optional
from dotenv import load_dotenv

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

SCOPES = ['https://www.googleapis.com/auth/drive.file']
DEFAULT_CLIENT_SECRET_FILE = 'client_secret.json'
DEFAULT_TOKEN_FILE = 'token.json'


class GoogleDriveUploader:
    def __init__(self, folder_id: Optional[str] = None):
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive API libraries not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        creds = None
        token_path = DEFAULT_TOKEN_FILE
        
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except Exception:
                creds = None
        
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            except Exception:
                creds = None
        
        if not creds or not creds.valid:
            client_secret_json = os.getenv('GOOGLE_DRIVE_CLIENT_SECRET_JSON')
            client_secret_path = os.getenv('GOOGLE_DRIVE_CLIENT_SECRET_FILE', DEFAULT_CLIENT_SECRET_FILE)
            
            if client_secret_json:
                try:
                    client_config = json.loads(client_secret_json)
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                    creds = flow.run_local_server(port=0)
                except (json.JSONDecodeError, ValueError, Exception) as e:
                    raise ValueError(f"Couldn't parse Google Drive credentials: {e}")
            elif os.path.exists(client_secret_path):
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                raise FileNotFoundError("Google Drive credentials not found - set up client_secret.json")
            
            if creds:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
        
        if not creds or not creds.valid:
            raise ValueError("Couldn't authenticate with Google Drive")
        
        self.service = build('drive', 'v3', credentials=creds)
    
    def upload_file(self, file_path: str, file_name: Optional[str] = None) -> Optional[str]:
        if not self.service:
            print("Google Drive isn't set up yet")
            return None
        
        if not os.path.exists(file_path):
            print(f"Can't find file: {file_path}")
            return None
        
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        try:
            mime_type = self._get_mime_type(file_path)
            
            file_metadata = {'name': file_name}
            if self.folder_id:
                file_metadata['parents'] = [self.folder_id]
            
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            
            file_id = file.get('id')
            return file_id
            
        except HttpError as error:
            print(f"Google Drive upload failed: {error}")
            return None
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    @staticmethod
    def _get_mime_type(file_path: str) -> str:
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
        if not self.service:
            print("Google Drive isn't set up yet")
            return None
        
        try:
            permission = {'type': 'anyone', 'role': 'reader'}
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            file = self.service.files().get(
                fileId=file_id,
                fields='webViewLink, webContentLink'
            ).execute()
            
            shareable_link = file.get('webContentLink') or file.get('webViewLink')
            
            if shareable_link:
                return shareable_link
            else:
                shareable_link = f"https://drive.google.com/uc?export=view&id={file_id}"
                return shareable_link
                
        except HttpError as error:
            print(f"Couldn't create shareable link: {error}")
            return None
        except Exception as e:
            print(f"Link creation error: {e}")
            return None


def upload_snapshot_to_drive(snapshot_path: str) -> bool:
    try:
        uploader = GoogleDriveUploader()
        file_id = uploader.upload_file(snapshot_path)
        return file_id is not None
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def upload_snapshot_and_get_link(snapshot_path: str) -> Optional[str]:
    try:
        uploader = GoogleDriveUploader()
        file_id = uploader.upload_file(snapshot_path)
        
        if not file_id:
            return None
        
        shareable_link = uploader.get_shareable_link(file_id)
        return shareable_link
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return None
