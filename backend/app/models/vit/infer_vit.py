import torch
import torch.nn.functional as F
from app.models.vit.model import LDSCViT
from app.utils.preprocessing import LDSCPreprocessor


CLASS_NAMES = [
    "COVID-19",
    "Normal",
    "Lung Opacity",
    "Viral Pneumonia"
]


class ViTInference:
    """ViT inference wrapper for lung disease classification."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.preprocessor = LDSCPreprocessor()

    def load_model(self) -> None:
        """Load ViT model from checkpoint."""
        self.model = LDSCViT(num_classes=4, pretrained=False)
        state = torch.load(self.checkpoint_path,
                           map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_tensor: torch.Tensor) -> dict:
        """Run inference on preprocessed tensor."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cls: float(probs[i])
                for i, cls in enumerate(CLASS_NAMES)
            }
        }

    def predict_from_path(self, image_path: str) -> dict:
        """Full pipeline from image path to prediction."""
        tensor = self.preprocessor.preprocess_for_vit(image_path)
        return self.predict(tensor)