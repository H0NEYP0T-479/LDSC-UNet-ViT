import torch
import torch.nn.functional as F
from pathlib import Path

from app.models.vit.model import LDSCViT
from app.models.vit.datamodule import get_transforms
from app.utils.imaging import load_image
from app.logging_config import get_logger


logger = get_logger(__name__)

CLASS_NAMES = ['normal', 'pneumonia', 'covid19', 'tuberculosis']


class ViTInference:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None
        self.transform = get_transforms(train=False, image_size=224)
        self.load_model()
    
    def load_model(self) -> None:
        self.model = LDSCViT(num_classes=4, pretrained=False)
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {self.checkpoint_path}")
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
        
        predicted_idx = probabilities.argmax(dim=0).item()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        prob_dict = {
            CLASS_NAMES[i]: probabilities[i].item()
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_from_path(self, image_path: str) -> dict:
        image = load_image(image_path)
        
        if len(image.shape) == 2:
            image_rgb = torch.stack([torch.from_numpy(image)] * 3, dim=0)
        else:
            image_rgb = torch.from_numpy(image).permute(2, 0, 1)
        
        image_rgb = image_rgb.float() / 255.0
        
        image_tensor = self.transform(image_rgb)
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        result = self.predict(image_tensor)
        result['image_path'] = image_path
        
        logger.info(
            f"Predicted {result['predicted_class']} "
            f"({result['confidence']:.4f}) for {image_path}"
        )
        
        return result

