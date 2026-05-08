import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a grayscale image."""
    original_dtype = image.dtype
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)

    if original_dtype != np.uint8:
        enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced