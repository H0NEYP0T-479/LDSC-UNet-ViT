"""Contrast enhancement techniques for medical image preprocessing"""
import numpy as np
import cv2
from typing import Literal


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image.
    
    Histogram equalization redistributes pixel intensities to achieve a more
    uniform histogram, thereby improving local contrast. This technique is
    effective for images with poor contrast but may introduce noise amplification
    and is less sophisticated than CLAHE for medical imaging.
    
    Args:
        image (np.ndarray): Input grayscale image. Can be uint8 or any integer type.
            Shape: (height, width). Values are assumed to be in standard ranges
            (uint8: 0-255, uint16: 0-65535, etc.).
    
    Returns:
        np.ndarray: Equalized image with enhanced contrast. Returned as uint8
            if input was uint8, otherwise as the same dtype as input.
    
    Raises:
        ValueError: If image is not 2D (not grayscale)
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.contrast import histogram_equalization
        >>> 
        >>> xray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        >>> equalized = histogram_equalization(xray)
        >>> assert equalized.dtype == np.uint8
        >>> assert equalized.shape == xray.shape
        >>> assert equalized.min() < xray.min() and equalized.max() > xray.max()
    
    Notes:
        - Simple and fast technique but may produce unnatural results
        - Can amplify noise in uniform regions
        - For medical images, prefer CLAHE which limits local contrast
        - Works best on images with bimodal or multimodal histograms
        - Global method that affects entire image uniformly
    
    Reference:
        Gonzalez, R. C., & Woods, R. E. (2008). Digital Image Processing (3rd ed.).
        Prentice Hall.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype == np.uint8:
        equalized = cv2.equalizeHist(image)
    else:
        image_uint8 = image.astype(np.uint8)
        equalized_uint8 = cv2.equalizeHist(image_uint8)
        equalized = equalized_uint8.astype(image.dtype)
    
    return equalized


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization to a grayscale image.
    
    Z-score normalization (standardization) transforms pixel values to have
    mean 0 and standard deviation 1. This normalization technique centers the
    data around zero and scales it by variance, making it suitable for neural
    networks and statistical methods that assume normalized input distributions.
    
    Args:
        image (np.ndarray): Input grayscale image. Can be any dtype (uint8, uint16,
            float, etc.). Shape: (height, width).
    
    Returns:
        np.ndarray: Z-score normalized image as float32. Values will have
            approximately zero mean and unit variance. Values typically range
            from -2 to 2 for normal distributions but can extend beyond.
    
    Raises:
        ValueError: If image is not 2D (not grayscale)
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.contrast import normalize_zscore
        >>> 
        >>> xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> normalized = normalize_zscore(xray)
        >>> 
        >>> assert normalized.dtype == np.float32
        >>> assert np.abs(normalized.mean()) < 0.1
        >>> assert np.abs(normalized.std() - 1.0) < 0.1
    
    Notes:
        - Also called standardization or zero-centering
        - Essential preprocessing for deep learning models
        - Doesn't bound output to specific range (unlike min-max normalization)
        - Preserves outliers while scaling to unit variance
        - Recommended for CNN and ViT inputs after min-max normalization
        - Handles images with zero std by returning zero array
    
    Formula:
        z = (x - mean) / std
    
    Reference:
        Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
        Springer.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    image_float = image.astype(np.float32)
    mean = image_float.mean()
    std = image_float.std()
    
    if std == 0:
        normalized = np.zeros_like(image_float, dtype=np.float32)
    else:
        normalized = ((image_float - mean) / std).astype(np.float32)
    
    return normalized


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to a grayscale image.
    
    Min-max normalization (feature scaling) rescales pixel values to the range
    [0, 1] by linearly transforming based on the minimum and maximum values
    in the image. This preserves the shape of the original distribution while
    constraining values to a fixed range, making it ideal for neural networks
    that expect normalized inputs.
    
    Args:
        image (np.ndarray): Input grayscale image. Can be any dtype (uint8, uint16,
            float, etc.). Shape: (height, width). If image is constant (all values
            identical), returns zeros.
    
    Returns:
        np.ndarray: Min-max normalized image as float32 with values in [0, 1].
            Constant images (min == max) return zero array.
    
    Raises:
        ValueError: If image is not 2D (not grayscale)
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.contrast import normalize_minmax
        >>> 
        >>> xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> normalized = normalize_minmax(xray)
        >>> 
        >>> assert normalized.dtype == np.float32
        >>> assert normalized.min() >= 0.0 and normalized.max() <= 1.0
        >>> assert np.isclose(normalized.min(), 0.0)
        >>> assert np.isclose(normalized.max(), 1.0)
    
    Notes:
        - Also called feature scaling or normalization (range 0-1)
        - Bounds output to [0, 1] making it interpretable as probability
        - Common preprocessing before deep learning models
        - Sensitive to outliers (one extreme value scales entire range)
        - Preserves distribution shape while constraining range
        - Recommended first step before Z-score normalization
        - Handles constant images gracefully by returning zeros
    
    Formula:
        x_normalized = (x - min) / (max - min)
    
    Reference:
        Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.).
        Springer.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    image_float = image.astype(np.float32)
    min_val = image_float.min()
    max_val = image_float.max()
    
    if max_val == min_val:
        normalized = np.zeros_like(image_float, dtype=np.float32)
    else:
        normalized = ((image_float - min_val) / (max_val - min_val)).astype(np.float32)
    
    return normalized


def enhance_contrast(
    image: np.ndarray,
    method: Literal["clahe", "histogram"] = "clahe",
    **kwargs
) -> np.ndarray:
    """
    Enhance contrast of a grayscale image using specified method.
    
    Dispatcher function that routes to different contrast enhancement algorithms.
    CLAHE (Contrast Limited Adaptive Histogram Equalization) is recommended for
    medical imaging as it provides local contrast enhancement while limiting
    noise amplification. Standard histogram equalization is faster but may
    produce less natural results for medical images.
    
    Args:
        image (np.ndarray): Input grayscale image. Can be uint8 or any integer
            type. Shape: (height, width).
        
        method (str, optional): Contrast enhancement method to apply.
            - "clahe": Contrast Limited Adaptive Histogram Equalization
              (recommended for medical images, slower, better quality)
            - "histogram": Global histogram equalization (faster, basic)
            Defaults to "clahe".
        
        **kwargs: Additional keyword arguments passed to the enhancement function.
            For method="clahe", supports: clip_limit, tile_grid_size
            For method="histogram", no additional parameters supported
    
    Returns:
        np.ndarray: Contrast-enhanced image with same dtype as input (uint8).
    
    Raises:
        ValueError: If image is not 2D (not grayscale)
        ValueError: If method is not recognized
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.contrast import enhance_contrast
        >>> 
        >>> xray = np.random.randint(50, 150, (512, 512), dtype=np.uint8)
        >>> 
        >>> enhanced_clahe = enhance_contrast(xray, method="clahe")
        >>> enhanced_clahe_custom = enhance_contrast(
        ...     xray,
        ...     method="clahe",
        ...     clip_limit=3.0,
        ...     tile_grid_size=(16, 16)
        ... )
        >>> enhanced_hist = enhance_contrast(xray, method="histogram")
        >>> 
        >>> assert enhanced_clahe.dtype == np.uint8
        >>> assert enhanced_clahe.shape == xray.shape
    
    Notes:
        - Use "clahe" for medical images (default, recommended)
        - Use "histogram" for speed when quality is less critical
        - CLAHE requires OpenCV with additional parameters
        - Histogram equalization is simple but may amplify noise
        - Can be combined with normalization for preprocessing pipeline
        - Returns uint8 to preserve medical image conventions
    
    Typical Pipeline:
        1. normalize_minmax() - scale to [0, 1]
        2. enhance_contrast() - improve local contrast
        3. normalize_zscore() - prepare for neural network
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if method.lower() == "clahe":
        try:
            from app.utils.ldsc_preprocess.clahe import apply_clahe
            clip_limit = kwargs.get("clip_limit", 2.0)
            tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
            enhanced_float = apply_clahe(image, clip_limit, tile_grid_size)
            if enhanced_float.dtype == np.float32:
                enhanced = (enhanced_float * 255).astype(np.uint8)
            else:
                enhanced = enhanced_float.astype(np.uint8)
        except ImportError:
            enhanced = histogram_equalization(image)
    
    elif method.lower() == "histogram":
        enhanced = histogram_equalization(image)
    
    else:
        raise ValueError(
            f"Unknown contrast enhancement method: '{method}'. "
            f"Supported methods: 'clahe', 'histogram'"
        )
    
    return enhanced
