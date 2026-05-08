import os
import uuid
import cv2
import numpy as np
from datetime import datetime, timedelta


class StorageService:
    """File management service for uploads and artifacts."""

    def __init__(self, artifacts_dir: str, uploads_dir: str):
        self.artifacts_dir = artifacts_dir
        self.uploads_dir = uploads_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        """Save uploaded file, return saved path."""
        ext = os.path.splitext(filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(self.uploads_dir, unique_name)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path

    def save_artifact(self, image: np.ndarray, name: str) -> str:
        """Save artifact image, return URL path."""
        filename = f"{uuid.uuid4().hex}_{name}.png"
        path = os.path.join(self.artifacts_dir, filename)
        cv2.imwrite(path, image)
        return f"/artifacts/{filename}"

    def get_artifact_url(self, filename: str) -> str:
        """Return public URL for artifact."""
        return f"/artifacts/{filename}"

    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """Delete artifacts older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        for folder in [self.artifacts_dir, self.uploads_dir]:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if mtime < cutoff:
                    os.remove(fpath)