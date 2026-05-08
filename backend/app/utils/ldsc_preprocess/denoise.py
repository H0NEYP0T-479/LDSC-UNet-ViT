"""Denoising techniques for medical image preprocessing"""
import numpy as np
import cv2
from typing import Literal, Optional


def non_local_means(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising to a grayscale image.
    
    Non-Local Means is an advanced denoising algorithm that compares patches
    across the entire image to find similar regions, then averages them to
    reduce noise while preserving edges and details. This is particularly
    effective for medical images like chest X-rays where detail preservation
    is critical.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width)
        
        h (float, optional): Filter strength. Higher h value removes more noise
            but also removes image details. 
            - Low values (3-8): Minimal smoothing, preserves details
            - Medium values (8-15): Balanced noise reduction
            - High values (15+): Aggressive denoising, may blur details
            Defaults to 10.0. Recommended for medical images: 8-12
        
        template_window (int, optional): Size of template patch (in pixels).
            Must be odd. Larger patches capture more context but are slower.
            - 5x5: Fast, fine detail preservation
            - 7x7: Good balance (default, recommended)
            - 9x9: Better pattern matching, slower
            Defaults to 7.
        
        search_window (int, optional): Size of search area around each pixel.
            Larger values provide better results but slower processing.
            - 11x11: Fast, local denoising only
            - 21x21: Good balance (default, recommended)
            - 41x41: Thorough search, significantly slower
            Defaults to 21.
    
    Returns:
        np.ndarray: Denoised grayscale image (uint8), same shape as input.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If template_window or search_window is not odd
        ValueError: If template_window or search_window <= 0
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.denoise import non_local_means
        >>> 
        >>> # Load noisy chest X-ray
        >>> noisy_xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> 
        >>> # Apply NLM denoising with default parameters
        >>> denoised = non_local_means(noisy_xray, h=10.0, template_window=7, search_window=21)
        >>> 
        >>> # Apply aggressive denoising for very noisy images
        >>> denoised_strong = non_local_means(noisy_xray, h=15.0, search_window=41)
        >>> 
        >>> assert denoised.dtype == np.uint8
        >>> assert denoised.shape == noisy_xray.shape
    
    Notes:
        - NLM is computationally expensive; expect longer processing times
        - For chest X-rays, h=10 and search_window=21 provide good balance
        - Increasing search_window beyond 21 yields diminishing returns
        - Processing time scales roughly with h, template_window, and search_window
        - For real-time applications, use gaussian_denoise() instead
    
    Reference:
        Buades, A., Coll, B., & Morel, J. M. (2005). A non-local algorithm for
        image denoising. IEEE Computer Society Conference on Computer Vision
        and Pattern Recognition (CVPR).
    """
    # ==================== Input Validation ====================
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before denoising."
        )
    
    if template_window <= 0 or template_window % 2 == 0:
        raise ValueError(
            f"template_window must be positive odd integer, got {template_window}"
        )
    
    if search_window <= 0 or search_window % 2 == 0:
        raise ValueError(
            f"search_window must be positive odd integer, got {search_window}"
        )
    
    # ==================== Apply NLM Denoising ====================
    denoised = cv2.fastNlMeansDenoising(
        src=image,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )
    
    return denoised


def gaussian_denoise(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur denoising to a grayscale image.
    
    Gaussian blur is a simple but effective denoising technique that reduces
    noise by blurring the image with a Gaussian kernel. While less sophisticated
    than NLM, it is very fast and suitable for real-time applications or as
    a preprocessing step before more advanced denoising.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width)
        
        sigma (float, optional): Standard deviation of the Gaussian kernel.
            Controls the amount of blur/smoothing.
            - Low values (0.5-1.0): Minimal blur, noise slightly reduced
            - Medium values (1.0-2.0): Balanced denoising (default)
            - High values (2.0+): Strong blur, detail loss
            Defaults to 1.0. Recommended for chest X-rays: 0.8-1.5
    
    Returns:
        np.ndarray: Denoised grayscale image (uint8), same shape as input.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If sigma is not positive
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.denoise import gaussian_denoise
        >>> 
        >>> # Load noisy chest X-ray
        >>> noisy_xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> 
        >>> # Apply Gaussian denoising
        >>> denoised = gaussian_denoise(noisy_xray, sigma=1.0)
        >>> 
        >>> # Aggressive denoising
        >>> denoised_strong = gaussian_denoise(noisy_xray, sigma=2.0)
        >>> 
        >>> assert denoised.dtype == np.uint8
        >>> assert denoised.shape == noisy_xray.shape
    
    Notes:
        - Very fast denoising suitable for real-time applications
        - Less effective at preserving fine details compared to NLM
        - Can introduce blur artifacts at high sigma values
        - Kernel size is automatically computed from sigma for optimal quality
        - Recommended as a first-pass preprocessing before NLM
    """
    # ==================== Input Validation ====================
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before denoising."
        )
    
    if sigma <= 0:
        raise ValueError(
            f"sigma must be positive, got {sigma}"
        )
    
    # ==================== Compute Kernel Size ====================
    # Kernel size should be approximately 6*sigma for 3-sigma coverage
    kernel_size = int(np.ceil(6 * sigma))
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Limit maximum kernel size to avoid excessive blur
    kernel_size = min(kernel_size, 31)
    
    # ==================== Apply Gaussian Denoising ====================
    denoised = cv2.GaussianBlur(
        src=image,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma
    )
    
    return denoised


def apply_denoising(
    image: np.ndarray,
    method: Literal["nlm", "gaussian"] = "nlm",
    **kwargs
) -> np.ndarray:
    """
    Apply denoising to a grayscale image using specified method.
    
    Dispatcher function that routes to the appropriate denoising algorithm
    based on the method parameter. Provides a unified interface for different
    denoising techniques.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width)
        
        method (str, optional): Denoising method to apply.
            - "nlm": Non-Local Means denoising (slower, better quality)
            - "gaussian": Gaussian blur denoising (faster, acceptable quality)
            Defaults to "nlm".
        
        **kwargs: Additional keyword arguments passed to the selected denoising function.
            For method="nlm", supports: h, template_window, search_window
            For method="gaussian", supports: sigma
    
    Returns:
        np.ndarray: Denoised grayscale image (uint8), same shape as input.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If method is not recognized
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.denoise import apply_denoising
        >>> 
        >>> # Load noisy chest X-ray
        >>> noisy_xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> 
        >>> # Use default NLM denoising
        >>> denoised_nlm = apply_denoising(noisy_xray)
        >>> 
        >>> # Use Gaussian denoising with custom sigma
        >>> denoised_gaussian = apply_denoising(noisy_xray, method="gaussian", sigma=1.5)
        >>> 
        >>> # Use aggressive NLM with custom parameters
        >>> denoised_aggressive = apply_denoising(
        ...     noisy_xray,
        ...     method="nlm",
        ...     h=15.0,
        ...     search_window=41
        ... )
        >>> 
        >>> assert denoised_nlm.dtype == np.uint8
    
    Notes:
        - Choose "nlm" for best quality (slower)
        - Choose "gaussian" for speed (acceptable quality)
        - Can chain multiple denoising passes for aggressive noise reduction
        - Consider using gaussian denoising first, then NLM for efficiency
    """
    # ==================== Input Validation ====================
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before denoising."
        )
    
    # ==================== Route to Appropriate Method ====================
    if method.lower() == "nlm":
        return non_local_means(image, **kwargs)
    
    elif method.lower() == "gaussian":
        return gaussian_denoise(image, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown denoising method: '{method}'. "
            f"Supported methods: 'nlm', 'gaussian'"
        )
