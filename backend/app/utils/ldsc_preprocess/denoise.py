import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def non_local_means(
    image: np.ndarray,
    h: float = 10,
    template_window: int = 7,
    search_window: int = 21
) -> np.ndarray:
    """Apply Non-Local Means denoising."""
    return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)


def gaussian_denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian denoising."""
    return gaussian_filter(image, sigma=sigma).astype(np.uint8)


def apply_denoising(image: np.ndarray, method: str = "nlm") -> np.ndarray:
    """Apply denoising using specified method: 'nlm' or 'gaussian'."""
    if method == "nlm":
        return non_local_means(image)
    elif method == "gaussian":
        return gaussian_denoise(image)
    else:
        raise ValueError(f"Unknown denoising method: {method}")