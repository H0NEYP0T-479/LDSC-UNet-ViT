import cv2
import numpy as np
from .clahe import apply_clahe


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization."""
    return cv2.equalizeHist(image.astype(np.uint8))


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """Z-score normalize image, returns float32."""
    image = image.astype(np.float32)
    mean, std = image.mean(), image.std()
    return (image - mean) / (std + 1e-8)


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Min-max normalize image to [0,1], returns float32."""
    image = image.astype(np.float32)
    mn, mx = image.min(), image.max()
    return (image - mn) / (mx - mn + 1e-8)


def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Enhance contrast using specified method."""
    if method == "clahe":
        return apply_clahe(image)
    elif method == "histeq":
        return histogram_equalization(image)
    else:
        raise ValueError(f"Unknown contrast method: {method}")