import base64
import uuid
import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load image as grayscale uint8."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(image: np.ndarray, path: str) -> str:
    """Save image to disk, return path."""
    cv2.imwrite(path, image)
    return path


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_numpy(b64: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay colored mask on original image."""
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    colored_mask = np.zeros_like(original)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to given (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)