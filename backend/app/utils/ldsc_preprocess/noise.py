import cv2
import numpy as np


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Apply median filter for noise reduction."""
    return cv2.medianBlur(image, 3)


def motion_artifact_reduction(image: np.ndarray) -> np.ndarray:
    """Apply bilateral filter to reduce motion artifacts."""
    return cv2.bilateralFilter(image, 9, 75, 75)


def sharpen_edges(image: np.ndarray) -> np.ndarray:
    """Sharpen edges using unsharp masking."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)