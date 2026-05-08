import torch
import numpy as np
from app.models.unet_lung.model import UNetLung
from app.utils.preprocessing import LDSCPreprocessor


class UNetInference:
    """UNet inference wrapper for lung segmentation."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.preprocessor = LDSCPreprocessor()

    def load_model(self) -> None:
        """Load UNet model from checkpoint."""
        self.model = UNetLung(in_channels=1, out_channels=1)
        state = torch.load(
            self.checkpoint_path, map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_tensor: torch.Tensor,
        threshold: float = 0.5
    ) -> dict:
        """Run segmentation inference on preprocessed tensor."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)

        prob_map = output[0, 0].cpu().numpy()
        binary_mask = (prob_map > threshold).astype(np.uint8)
        area_pct = float(binary_mask.sum()) / binary_mask.size * 100

        return {
            "mask": binary_mask,
            "disease_detected": bool(area_pct > 1.0),
            "area_percentage": round(area_pct, 2)
        }

    def predict_from_path(self, image_path: str) -> dict:
        """Full pipeline from image path to segmentation."""
        tensor = self.preprocessor.preprocess_for_unet(image_path)
        return self.predict(tensor)