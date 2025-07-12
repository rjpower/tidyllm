"""Google Drive source adapter implementation."""

import base64
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from pydantic_core import Url

from tidyllm.types.part.lib import PART_SOURCE_REGISTRY, BasicPart, Part


class GDriveSource(BaseModel):
    """Source backed by Google Drive file."""

    url: str = Field(description="URL to load from gdrive.")
    credentials_path: str | None = Field(default=None, description="Path to credentials file")
    token_path: str | None = Field(default=None, description="Path to token file")

    def model_post_init(self, _ctx: Any):
        self._service = None
        self._file_id = None
        self._file_content: bytes | None = None
        self._mime_type: str = ""

    @property
    def path(self):
        parsed = urlparse(self.url)
        return parsed.netloc + parsed.path

    @property
    def mime_type(self):
        self._load_file()
        return self._mime_type

    def _get_credentials(self) -> Credentials:
        """Get Google Drive API credentials."""
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

        # Default paths - look for credentials in project root first
        project_root = Path(__file__).parent.parent.parent.parent
        default_creds_path = project_root / 'credentials' / 'gdrive_client_secret.json'

        token_path = Path(self.token_path) if self.token_path else Path.home() / '.config' / 'tidyllm' / 'gdrive_token.json'
        credentials_path = Path(self.credentials_path) if self.credentials_path else default_creds_path

        creds = None

        # Load existing token
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("Refreshing Google Drive credentials...")
                creds.refresh(Request())
            else:
                if not credentials_path.exists():
                    raise ValueError(
                        f"Google Drive credentials not found at {credentials_path}. "
                        f"Please download credentials from Google Cloud Console and save there."
                    )

                print("Google Drive authentication required. Opening browser...")
                flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)
                print("Authentication successful!")

            # Save credentials for next run
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
            print(f"Credentials saved to {token_path}")

        return cast(Credentials, creds)

    def _get_service(self):
        """Get Google Drive API service."""
        if self._service is None:
            creds = self._get_credentials()
            self._service = build('drive', 'v3', credentials=creds)
        return self._service

    def _find_file_by_path(self, path: str) -> str:
        """Find file ID by path in Google Drive."""
        service = self._get_service()

        # Remove leading slash if present
        path = path.lstrip('/')

        # Split path into parts
        parts = path.split('/')

        # Start from root
        parent_id = 'root'

        print(f"Searching for file: {path}")

        # Navigate through folders
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1

            # Search for item with this name in current folder
            query = f"name='{part}' and '{parent_id}' in parents"
            if not is_last:
                query += " and mimeType='application/vnd.google-apps.folder'"

            print(f"Searching for {'file' if is_last else 'folder'}: {part}")
            results = service.files().list(q=query).execute()
            items = results.get("files", [])

            if not items:
                raise ValueError(f"File or folder '{part}' not found in path '{path}'")

            if len(items) > 1:
                print(f"Warning: Multiple files named '{part}' found, using first one")

            file_id = items[0]["id"]
            print(f"Found {'file' if is_last else 'folder'}: {part} (ID: {file_id})")

            if is_last:
                return file_id
            else:
                parent_id = file_id

        raise ValueError(f"File not found: {path}")

    def _load_file(self) -> bytes:
        if self._file_content is not None:
            return self._file_content

        if self._file_id is None:
            self._file_id = self._find_file_by_path(self.path)

        # Get file metadata to check if it's a Google Docs file
        print(f"Fetching file metadata for Google Drive file: {self.path}")
        service = self._get_service()
        file_metadata = service.files().get(fileId=self._file_id).execute()
        mime_type = file_metadata.get("mimeType", "")
        file_name = file_metadata.get("name", "unknown")
        file_size = file_metadata.get("size", "unknown")

        print(f"File: {file_name} ({file_size} bytes, {mime_type})")

        # Export Google Docs files as the appropriate format: PDF for documents, openxml for sheets
        if mime_type.startswith("application/vnd.google-apps."):
            if "document" in mime_type:
                export_mime_type = "application/pdf"
            elif "spreadsheet" in mime_type:
                export_mime_type = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif "presentation" in mime_type:
                export_mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            else:
                export_mime_type = "application/pdf"

            print(f"Exporting Google Docs file as {export_mime_type}")
            request = service.files().export_media(
                fileId=self._file_id, mimeType=export_mime_type
            )
            self._mime_type = export_mime_type
        else:
            # Regular file download
            print("Downloading file from Google Drive...")
            request = service.files().get_media(fileId=self._file_id)
            self._mime_type = mime_type

        self._file_content = request.execute()
        return self._file_content # type: ignore

    def read(self, size: int = -1) -> bytes:
        """Read data from the Google Drive file."""
        self._file_content = self._load_file()
        if size == -1:
            return self._file_content
        else:
            return self._file_content[:size]

    def close(self):
        """Close the connection (no-op for Google Drive)."""
        pass


class GDrivePartSource:
    """Stream Parts from Google Drive."""

    def __call__(self, url: Url):
        from tidyllm.types.linq import Table

        source = GDriveSource(url=str(url))
        part = BasicPart(mime_type=source.mime_type, data=base64.b64encode(source.read()))
        return Table.from_rows([part])


PART_SOURCE_REGISTRY.register_scheme("gdrive", GDrivePartSource())
