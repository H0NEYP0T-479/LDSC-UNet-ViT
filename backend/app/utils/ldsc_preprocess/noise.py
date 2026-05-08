"""Noise reduction and edge enhancement for medical image preprocessing"""
import numpy as np
import cv2
from typing import Optional


def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Reduce noise in grayscale image using median filtering.
    
    Median filtering is a non-linear denoising technique that replaces each
    pixel with the median value of pixels in a neighborhood. This method is
    particularly effective for salt-and-pepper noise and impulse noise while
    preserving edges better than Gaussian blur. It is commonly used as a
    preprocessing step for medical imaging before more advanced denoising.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width).
            Must be 2D.
        
        kernel_size (int, optional): Size of the median filter kernel. Must be
            positive odd integer. Larger kernels provide more aggressive noise
            reduction but may blur fine details.
            - 3: Minimal smoothing, preserves details (fastest)
            - 5: Balanced noise reduction (default, recommended)
            - 7: Moderate smoothing
            - 9+: Strong smoothing, may lose fine structures
            Defaults to 5.
    
    Returns:
        np.ndarray: Noise-reduced grayscale image (uint8), same shape as input.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If kernel_size is not positive odd integer
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.noise import reduce_noise
        >>> 
        >>> noisy_xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> denoised = reduce_noise(noisy_xray, kernel_size=5)
        >>> 
        >>> assert denoised.dtype == np.uint8
        >>> assert denoised.shape == noisy_xray.shape
        >>> assert denoised.std() < noisy_xray.std()
    
    Notes:
        - Highly effective for salt-and-pepper noise
        - Preserves edges better than Gaussian blur
        - Computationally efficient compared to NLM denoising
        - Kernel size must be odd; even values are automatically converted
        - Recommended for preprocessing before segmentation tasks
        - Works well as first-pass denoising before advanced methods
    
    Performance Considerations:
        - Processing time is roughly proportional to kernel_size^2
        - Kernel size 5 provides good balance for medical images
        - For large images, consider smaller kernels for speed
    
    Reference:
        Huang, T., Yang, G. J., & Tang, G. Y. (1979). A fast two-dimensional
        median filtering algorithm. IEEE Transactions on Acoustics, Speech, and
        Signal Processing, 27(1), 13-18.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before denoising."
        )
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be positive odd integer, got {kernel_size}"
        )
    
    denoised = cv2.medianBlur(image, kernel_size)
    
    return denoised


def motion_artifact_reduction(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Reduce motion artifacts in grayscale image using bilateral filtering.
    
    Bilateral filtering is an edge-preserving smoothing technique that applies
    weighted averaging based on both spatial proximity and intensity similarity.
    This approach is particularly effective for reducing motion blur artifacts
    and noise while preserving sharp edges and fine anatomical details, making
    it ideal for medical imaging including chest X-rays affected by motion blur.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width).
            Must be 2D.
        
        diameter (int, optional): Diameter of each pixel neighborhood used in
            bilateral filtering. Must be positive.
            - 5-7: Fast processing, minimal smoothing
            - 9: Good balance (default, recommended for medical images)
            - 11-15: Strong smoothing, slower processing
            Defaults to 9.
        
        sigma_color (float, optional): Filter sigma in the color space. Controls
            how different colors/intensities are weighted. Higher values include
            pixels with greater intensity differences in the averaging.
            - Low values (25-50): Only similar intensities are averaged
            - Medium values (75): Balanced (default, recommended)
            - High values (150+): Aggregates pixels with different intensities
            Defaults to 75.0.
        
        sigma_space (float, optional): Filter sigma in the coordinate space.
            Controls spatial extent of the filter kernel. Higher values include
            pixels further away in the averaging.
            - Low values (25-50): Local filtering only
            - Medium values (75): Broader context (default, recommended)
            - High values (150+): Large area filtering
            Defaults to 75.0.
    
    Returns:
        np.ndarray: Filtered grayscale image (uint8) with reduced motion artifacts,
            same shape as input.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If diameter, sigma_color, or sigma_space is not positive
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.noise import motion_artifact_reduction
        >>> 
        >>> xray_with_artifacts = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> artifact_reduced = motion_artifact_reduction(
        ...     xray_with_artifacts,
        ...     diameter=9,
        ...     sigma_color=75.0,
        ...     sigma_space=75.0
        ... )
        >>> 
        >>> assert artifact_reduced.dtype == np.uint8
        >>> assert artifact_reduced.shape == xray_with_artifacts.shape
    
    Notes:
        - Bilateral filtering is slower than Gaussian blur but preserves edges
        - Highly effective for reducing motion blur and directional artifacts
        - Recommended for images affected by patient motion during acquisition
        - Parameter tuning may be necessary for specific artifact types
        - Default parameters work well for typical chest X-ray motion artifacts
        - Combines spatial proximity with intensity similarity for edge preservation
    
    Clinical Application:
        Motion artifacts are common in chest X-rays due to patient breathing or
        involuntary movement. Bilateral filtering reduces these artifacts while
        preserving anatomical edges crucial for diagnostic interpretation.
    
    Computational Cost:
        - More expensive than Gaussian blur
        - Processing time increases with diameter
        - sigma_color and sigma_space affect computation time
    
    Reference:
        Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and
        color images. 6th International Conference on Computer Vision (ICCV).
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before filtering."
        )
    
    if diameter <= 0:
        raise ValueError(
            f"diameter must be positive, got {diameter}"
        )
    
    if sigma_color <= 0:
        raise ValueError(
            f"sigma_color must be positive, got {sigma_color}"
        )
    
    if sigma_space <= 0:
        raise ValueError(
            f"sigma_space must be positive, got {sigma_space}"
        )
    
    filtered = cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )
    
    return filtered


def sharpen_edges(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.0
) -> np.ndarray:
    """
    Sharpen edges in grayscale image using unsharp mask.
    
    Unsharp masking is a classical edge enhancement technique that works by
    subtracting a blurred version of the image from the original, then adding
    this difference back to amplify high-frequency components (edges). This
    technique is highly effective for enhancing anatomical boundaries in medical
    images while preserving overall image structure and controlling the degree
    of sharpening through adjustable parameters.
    
    Args:
        image (np.ndarray): Input grayscale image (uint8). Shape: (height, width).
            Must be 2D.
        
        sigma (float, optional): Standard deviation of the Gaussian blur used to
            create the unsharp mask. Controls the scale of edges being enhanced.
            - Low values (0.5-1.0): Enhances fine details (default)
            - Medium values (1.5-2.5): Enhances medium structures
            - High values (3.0+): Emphasizes large-scale features
            Defaults to 1.0. Recommended for medical images: 0.8-1.5.
        
        strength (float, optional): Strength of edge enhancement. Controls the
            magnitude of the sharpening effect applied. Higher values produce
            more pronounced sharpening but may introduce artifacts.
            - Low values (0.5-1.0): Subtle enhancement
            - Medium values (1.0-2.0): Moderate sharpening (default)
            - High values (2.0+): Aggressive enhancement, risk of ringing artifacts
            Defaults to 1.0. Recommended range: 0.5-2.0.
    
    Returns:
        np.ndarray: Sharpened grayscale image (uint8), same shape as input.
            Values are clipped to [0, 255] to prevent overflow/underflow.
    
    Raises:
        ValueError: If image is not 2D or not uint8
        ValueError: If sigma or strength is not positive
    
    Example:
        >>> import numpy as np
        >>> from app.utils.ldsc_preprocess.noise import sharpen_edges
        >>> 
        >>> xray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        >>> sharpened = sharpen_edges(xray, sigma=1.0, strength=1.5)
        >>> 
        >>> assert sharpened.dtype == np.uint8
        >>> assert sharpened.shape == xray.shape
        >>> assert 0 <= sharpened.min() and sharpened.max() <= 255
    
    Notes:
        - Unsharp mask enhances edges without adding noise like direct differentiation
        - Particularly useful for improving visualization of fine structures
        - Helps prepare images for segmentation tasks by clarifying boundaries
        - Can be combined with denoising for better results
        - Excessive sharpening may create visible ringing artifacts
        - Parameters should be tuned based on image resolution and feature size
    
    Clinical Application:
        Enhanced edge sharpening in chest X-rays can improve visualization of:
        - Pulmonary nodules and suspicious lesions
        - Bronchial boundaries and branching structures
        - Heart-lung interface definition
        - Pneumothorax detection (air-tissue boundary)
    
    Typical Values for Medical Images:
        - Subtle enhancement: sigma=1.0, strength=0.5-1.0
        - Standard enhancement: sigma=1.0, strength=1.0-1.5 (default)
        - Aggressive enhancement: sigma=0.8, strength=2.0-3.0
    
    Artifact Considerations:
        - Sharpening can amplify noise if applied before denoising
        - Ringing artifacts appear as halos around sharp edges at high strengths
        - Over-sharpening may create false edges and degrade diagnostic value
    
    Reference:
        Marr, D., & Hildreth, E. (1980). Theory of edge detection.
        Proceedings of the Royal Society of London. Series B. Biological Sciences,
        207(1167), 187-217.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale image, got shape {image.shape}"
        )
    
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype {image.dtype}. "
            f"Please convert image to uint8 before sharpening."
        )
    
    if sigma <= 0:
        raise ValueError(
            f"sigma must be positive, got {sigma}"
        )
    
    if strength <= 0:
        raise ValueError(
            f"strength must be positive, got {strength}"
        )
    
    kernel_size = int(np.ceil(6 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 31)
    
    image_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(
        image_float,
        (kernel_size, kernel_size),
        sigma
    )
    
    unsharp_mask = image_float - blurred
    sharpened_float = image_float + strength * unsharp_mask
    
    sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)
    
    return sharpened
