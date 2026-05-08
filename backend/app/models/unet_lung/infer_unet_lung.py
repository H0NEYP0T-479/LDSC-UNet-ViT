import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

from app.models.unet.model import UNetLung
from app.models.unet_lung.datamodule import get_seg_transforms
from app.utils.imaging import load_image
from app.logging_config import get_logger


logger = get_logger(__name__)


class UNetInference:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None
        self.transform = get_seg_transforms(train=False, image_size=256)
        self.load_model()
    
    def load_model(self) -> None:
        self.model = UNetLung(
            in_channels=1,
            out_channels=1,
            features=[64, 128, 256, 512]
        )
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {self.checkpoint_path}")
    
    def predict(self, image_tensor: torch.Tensor, threshold: float = 0.5) -> dict:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            pred_prob = logits[0, 0].cpu().numpy()
        
        mask_binary = (pred_prob > threshold).astype(np.uint8)
        
        area_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.size
        area_percentage = (area_pixels / total_pixels) * 100.0
        
        tumor_detected = area_pixels > 0
        
        return {
            'mask': mask_binary,
            'tumor_detected': bool(tumor_detected),
            'area_percentage': float(area_percentage)
        }
    
    def predict_from_path(self, image_path: str, threshold: float = 0.5) -> dict:
        image = load_image(image_path)
        
        if len(image.shape) == 2:
            image = image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = image.astype(np.float32) / 255.0
        
        augmented = self.transform(image=image, mask=image)
        image_tensor = augmented['image']
        
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        result = self.predict(image_tensor, threshold=threshold)
        result['image_path'] = image_path
        
        logger.info(
            f"Segmentation detected: {result['tumor_detected']} "
            f"({result['area_percentage']:.2f}%) for {image_path}"
        )
        
        return result
