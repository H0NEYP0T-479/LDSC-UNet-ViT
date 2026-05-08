import os
import uuid
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from app.logging_config import get_logger


logger = get_logger(__name__)


class StorageService:
    def __init__(self, artifacts_dir: str, uploads_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.uploads_dir = Path(uploads_dir)
        
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized StorageService with artifacts: {artifacts_dir}, uploads: {uploads_dir}")
    
    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        try:
            name, ext = os.path.splitext(filename)
            unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            file_path = self.uploads_dir / unique_name
            
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            
            logger.info(f"Saved upload to {file_path}")
            
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            raise
    
    def save_artifact(self, image: np.ndarray, name: str) -> str:
        try:
            unique_name = f"{name}_{uuid.uuid4().hex[:8]}.png"
            file_path = self.artifacts_dir / unique_name
            
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            cv2.imwrite(str(file_path), image)
            
            logger.info(f"Saved artifact to {file_path}")
            
            return self.get_artifact_url(unique_name)
        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")
            raise
    
    def get_artifact_url(self, filename: str) -> str:
        return f"/artifacts/{filename}"
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for directory in [self.artifacts_dir, self.uploads_dir]:
                for file_path in directory.glob('*'):
                    if file_path.is_file():
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if mod_time < cutoff_time:
                            file_path.unlink()
                            logger.info(f"Deleted old file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
