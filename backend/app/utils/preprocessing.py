import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from app.utils.ldsc_preprocess.clahe import apply_clahe
from app.utils.ldsc_preprocess.denoise import non_local_means
from app.utils.ldsc_preprocess.contrast import normalize_zscore
from app.utils.ldsc_preprocess.noise import sharpen_edges


class LDSCPreprocessor:
    """Main preprocessing pipeline for chest X-rays."""

    def __init__(self, vit_size: int = 224, unet_size: int = 256):
        self.vit_size = vit_size
        self.unet_size = unet_size

        self.vit_transform = transforms.Compose([
            transforms.Resize((vit_size, vit_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def _base_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Apply base preprocessing steps."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image
        denoised = non_local_means(gray)
        enhanced = apply_clahe(denoised)
        sharpened = sharpen_edges(enhanced)
        return sharpened

    def preprocess_for_vit(self, image_path: str) -> torch.Tensor:
        """Preprocess image for ViT. Returns (1,3,224,224) tensor."""
        image = cv2.imread(image_path)
        processed = self._base_pipeline(image)
        pil = Image.fromarray(processed).convert("RGB")
        tensor = self.vit_transform(pil)
        return tensor.unsqueeze(0)

    def preprocess_for_unet(self, image_path: str) -> torch.Tensor:
        """Preprocess image for UNet. Returns (1,1,256,256) tensor."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        denoised = non_local_means(image)
        enhanced = apply_clahe(denoised)
        sharpened = sharpen_edges(enhanced)
        resized = cv2.resize(
            sharpened, (self.unet_size, self.unet_size)
        )
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor

    def get_stages(self, image_path: str) -> dict:
        """Return all preprocessing stages as numpy arrays."""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = non_local_means(gray)
        enhanced = apply_clahe(denoised)
        sharpened = sharpen_edges(enhanced)
        normalized = normalize_zscore(sharpened)
        norm_display = (
            (normalized - normalized.min()) /
            (normalized.max() - normalized.min()) * 255
        ).astype(np.uint8)

        return {
            "original": image,
            "grayscale": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "sharpened": sharpened,
            "normalized": norm_display
        }