"""CLAHE (Contrast Limited Adaptive Histogram Equalization) for medical image preprocessing"""
import numpy as np
import cv2
from typing import Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.
    
    CLAHE is an advanced histogram equalization technique that improves local contrast
    enhancement while limiting noise amplification. It divides the image into small tiles,
    applies histogram equalization to each tile, and interpolates the results. This method
    is particularly effective for medical imaging like chest X-rays where local contrast
    enhancement is critical.
    
    Args:
        image (np.ndarray): Input grayscale image. Must be 2D.
            - If dtype is floating point: Values should be in [0, 1] range
            - If dtype is uint8: Values in [0, 255] range
            - Other integer types are converted to uint8
        
        clip_limit (float, optional): Threshold for contrast limiting. Regulates the 
            strength of the contrast enhancement.
            - Lower values (e.g., 1.0): Subtle enhancement, less noise amplification
            - Higher values (e.g., 4.0): Stronger enhancement, may amplify noise
            Defaults to 2.0. Recommended range for medical images: 1.5 - 3.5
        
        tile_grid_size (Tuple[int, int], optional): Grid dimensions that divide the image
            into tiles for local histogram equalization.
            - Larger values (e.g., 16x16): Smoother results, broader context
            - Smaller values (e.g., 4x4): More local detail preservation
            Defaults to (8, 8). Both dimensions must be positive integers.
    
    Returns:
        np.ndarray: CLAHE-enhanced image with the same dtype as input.
            - If input was floating point: Returns float32 in [0, 1] range
            - If input was uint8: Returns uint8 in [0, 255] range
            - Shape is identical to input image
    
    Raises:
        ValueError: If image is not 2D (not grayscale)
        ValueError: If clip_limit is not positive
        ValueError: If tile_grid_size contains non-positive values
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.clahe import apply_clahe
        >>> 
        >>> # Example 1: uint8 chest X-ray image
        >>> xray_uint8 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> enhanced_uint8 = apply_clahe(xray_uint8, clip_limit=2.5, tile_grid_size=(8, 8))
        >>> assert enhanced_uint8.dtype == np.uint8
        >>> assert enhanced_uint8.shape == (512, 512)
        >>> 
        >>> # Example 2: float chest X-ray image normalized to [0, 1]
        >>> xray_float = xray_uint8.astype(np.float32) / 255.0
        >>> enhanced_float = apply_clahe(xray_float, clip_limit=2.0)
        >>> assert enhanced_float.dtype == np.float32
        >>> assert 0.0 <= enhanced_float.min() <= enhanced_float.max() <= 1.0
    
    Notes:
        - CLAHE is highly effective for chest X-ray enhancement without introducing
          significant artifacts compared to global histogram equalization
        - For optimal results with medical images:
          * Clip limit 2.0-2.5: Good balance between enhancement and noise control
          * Tile size 8x8: Good for most chest X-rays (default)
          * Tile size 4x4: Use for preserving fine anatomical details
          * Tile size 16x16: Use for smoother enhancement with less local variation
        - The method creates visible "block" artifacts at tile boundaries when tiles
          are too large; use appropriate tile_grid_size to minimize this
        - Processing time is independent of clip_limit but increases with image size
    
    Reference:
        Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations.
        Computer Vision, Graphics, and Image Processing, 39(3), 355-368.
    """
    # ==================== Input Validation ====================
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}. "
            f"Image must be 2D for CLAHE processing."
        )
    
    if clip_limit <= 0:
        raise ValueError(
            f"clip_limit must be positive, got {clip_limit}. "
            f"Recommended range: 1.0 - 4.0"
        )
    
    if len(tile_grid_size) != 2 or any(size <= 0 for size in tile_grid_size):
        raise ValueError(
            f"tile_grid_size must be tuple of positive integers, got {tile_grid_size}"
        )
    
    # ==================== Type Conversion ====================
    original_dtype = image.dtype
    is_float = np.issubdtype(original_dtype, np.floating)
    
    # Convert float to uint8 for OpenCV processing
    if is_float:
        # Assume float image is in [0, 1] range; clip for safety
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        # Convert any integer type to uint8
        image_uint8 = image.astype(np.uint8)
    
    # ==================== Apply CLAHE ====================
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced_uint8 = clahe.apply(image_uint8)
    
    # ==================== Convert Back to Original Type ====================
    if is_float:
        # Convert back to float32 in [0, 1] range
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
    else:
        # Convert back to original dtype (usually uint8)
        enhanced = enhanced_uint8.astype(original_dtype)
    
    return enhanced
