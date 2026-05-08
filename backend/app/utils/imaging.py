"""Image I/O and manipulation operations for LDSC-UNet-ViT"""
import numpy as np
import cv2
import base64
import io
from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file as grayscale uint8 array.
    
    Loads image using OpenCV and converts to grayscale if necessary. Returns
    image as uint8 numpy array suitable for preprocessing pipelines. Supports
    all common image formats (JPEG, PNG, TIFF, BMP, etc.).
    
    Args:
        path (Union[str, Path]): Path to image file.
    
    Returns:
        np.ndarray: Loaded grayscale image as uint8. Shape: (height, width).
    
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image cannot be loaded or is invalid
    
    Example:
        >>> from app.utils.imaging import load_image
        >>> import numpy as np
        >>> 
        >>> image = load_image("chest_xray.jpg")
        >>> assert image.dtype == np.uint8
        >>> assert image.ndim == 2
        >>> assert image.shape == (height, width)
    
    Notes:
        - Automatically converts color images to grayscale
        - Always returns uint8 regardless of input bit depth
        - Grayscale images are returned as-is
        - Raises exception if file doesn't exist or is corrupted
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {path}. File may be corrupted or invalid format.")
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    return image


def save_image(image: np.ndarray, path: Union[str, Path]) -> str:
    """
    Save image to file and return the path.
    
    Saves image to disk using OpenCV. Creates parent directories if they
    don't exist. Infers format from file extension. Supports uint8 and float
    images (float images converted to uint8 with range [0, 255]).
    
    Args:
        image (np.ndarray): Image to save. Can be uint8 or float.
            - uint8: Saved as-is, values [0, 255]
            - float: Scaled to [0, 255] if in range [0, 1], otherwise clipped
        
        path (Union[str, Path]): Output path for saved image. File extension
            determines format (e.g., .jpg, .png, .tiff).
    
    Returns:
        str: Absolute path to saved image as string.
    
    Raises:
        ValueError: If image is invalid or cannot be saved
        OSError: If parent directory cannot be created
    
    Example:
        >>> from app.utils.imaging import save_image
        >>> import numpy as np
        >>> 
        >>> image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> saved_path = save_image(image, "output/result.jpg")
        >>> assert Path(saved_path).exists()
    
    Notes:
        - Creates parent directories automatically
        - Returns absolute path as string
        - Supports uint8 (0-255) and float (0-1) images
        - Float values > 1 are clipped to valid range
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image_save = (image * 255).astype(np.uint8)
        else:
            image_save = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_save = image.astype(np.uint8)
    
    success = cv2.imwrite(str(path), image_save)
    
    if not success:
        raise ValueError(f"Failed to save image to {path}")
    
    return str(path.absolute())


def numpy_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64-encoded PNG string.
    
    Encodes image as PNG format and returns as base64 string suitable for
    embedding in JSON responses or HTML. Handles uint8 and float images
    automatically with appropriate range conversion.
    
    Args:
        image (np.ndarray): Image to encode. Can be uint8 or float.
            - uint8: Encoded as-is, values [0, 255]
            - float: Scaled to [0, 255] if in range [0, 1]
    
    Returns:
        str: Base64-encoded PNG string (UTF-8 decoded for JSON compatibility).
            Can be embedded in HTML img tags as: data:image/png;base64,{result}
    
    Raises:
        ValueError: If image cannot be encoded
    
    Example:
        >>> from app.utils.imaging import numpy_to_base64
        >>> import numpy as np
        >>> 
        >>> image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> b64_string = numpy_to_base64(image)
        >>> assert isinstance(b64_string, str)
        >>> assert len(b64_string) > 0
    
    Notes:
        - Returns base64 string without data URI prefix
        - Suitable for JSON serialization
        - Lossless PNG compression preserves image quality
        - String can be used in HTML as: <img src="data:image/png;base64,{b64_string}">
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    success, buffer = cv2.imencode('.png', image_uint8)
    
    if not success:
        raise ValueError("Failed to encode image to PNG")
    
    b64_bytes = base64.b64encode(buffer.tobytes())
    b64_string = b64_bytes.decode('utf-8')
    
    return b64_string


def base64_to_numpy(b64_string: str) -> np.ndarray:
    """
    Convert base64-encoded PNG string to numpy image.
    
    Decodes base64 string and reconstructs numpy array from PNG data. Inverse
    of numpy_to_base64. Handles both raw base64 and data URI formats.
    
    Args:
        b64_string (str): Base64-encoded PNG string. Can be:
            - Raw base64: "iVBORw0KGgo..."
            - Data URI: "data:image/png;base64,iVBORw0KGgo..."
    
    Returns:
        np.ndarray: Decoded grayscale image as uint8. Shape: (height, width).
    
    Raises:
        ValueError: If base64 string is invalid or cannot be decoded
    
    Example:
        >>> from app.utils.imaging import numpy_to_base64, base64_to_numpy
        >>> import numpy as np
        >>> 
        >>> original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> b64 = numpy_to_base64(original)
        >>> recovered = base64_to_numpy(b64)
        >>> 
        >>> assert np.allclose(original, recovered)
    
    Notes:
        - Automatically handles data URI prefix if present
        - Returns grayscale image even if PNG contains color
        - Lossless recovery from numpy_to_base64 encoding
    """
    if b64_string.startswith('data:image/png;base64,'):
        b64_string = b64_string.split(',')[1]
    
    try:
        b64_bytes = base64.b64decode(b64_string)
        buffer = io.BytesIO(b64_bytes)
        image = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {str(e)}")
    
    if image is None:
        raise ValueError("Failed to decode base64 PNG data")
    
    return image


def create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create colored overlay of mask on original grayscale image.
    
    Blends a binary or grayscale mask with the original image using a specified
    color and transparency. Useful for visualizing segmentation masks, region
    of interest, and detection results overlaid on medical images.
    
    Args:
        original (np.ndarray): Original grayscale image (uint8).
            Shape: (height, width).
        
        mask (np.ndarray): Mask image (uint8 or float) same size as original.
            - uint8: Values 0-255 where non-zero indicates mask region
            - float: Values 0-1 where values > 0 indicate mask region
            Shape must match original: (height, width).
        
        color (Tuple[int, int, int], optional): RGB color for overlay.
            Specified as (B, G, R) in OpenCV convention.
            - (255, 0, 0): Blue in BGR/OpenCV
            - (0, 255, 0): Green
            - (0, 0, 255): Red
            Defaults to (255, 0, 0) which appears as red.
        
        alpha (float, optional): Blending factor for overlay transparency.
            - 0.0: Original image only (no mask visible)
            - 0.5: 50% mask, 50% original
            - 1.0: Full mask, no original visible
            Defaults to 0.4 (40% mask, 60% original).
    
    Returns:
        np.ndarray: Blended RGB image (uint8) with colored mask overlay.
            Shape: (height, width, 3). Can be displayed or saved as color image.
    
    Raises:
        ValueError: If image shapes don't match or invalid parameters
    
    Example:
        >>> from app.utils.imaging import load_image, create_overlay
        >>> import numpy as np
        >>> 
        >>> original = load_image("chest_xray.jpg")
        >>> mask = np.zeros_like(original)
        >>> mask[100:200, 100:200] = 255
        >>> 
        >>> overlay = create_overlay(
        ...     original,
        ...     mask,
        ...     color=(0, 255, 0),
        ...     alpha=0.5
        ... )
        >>> assert overlay.shape == (original.shape[0], original.shape[1], 3)
        >>> assert overlay.dtype == np.uint8
    
    Notes:
        - Returns RGB image (3 channels) even if inputs are grayscale
        - Color specification uses BGR convention (OpenCV standard)
        - Alpha controls transparency of mask overlay
        - Useful for visualizing: segmentation masks, detections, ROI
        - Original image intensity is preserved and combined with colored mask
    
    Common Colors (BGR):
        - Red: (0, 0, 255)
        - Green: (0, 255, 0)
        - Blue: (255, 0, 0)
        - Yellow: (0, 255, 255)
        - Cyan: (255, 255, 0)
        - Magenta: (255, 0, 255)
    """
    if original.shape != mask.shape:
        raise ValueError(
            f"Original shape {original.shape} != mask shape {mask.shape}"
        )
    
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    
    if len(color) != 3:
        raise ValueError(f"color must be RGB tuple (3 values), got {len(color)}")
    
    original_uint8 = original.astype(np.uint8)
    
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    overlay_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR)
    
    color_mask = np.zeros_like(overlay_bgr)
    for c in range(3):
        color_mask[:, :, c] = color[c]
    
    mask_regions = mask_uint8 > 0
    
    result = overlay_bgr.copy().astype(np.float32)
    color_mask_float = color_mask.astype(np.float32)
    
    result[mask_regions] = (1 - alpha) * overlay_bgr[mask_regions].astype(np.float32) + alpha * color_mask_float[mask_regions]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def resize_image(
    image: np.ndarray,
    size: Union[Tuple[int, int], int]
) -> np.ndarray:
    """
    Resize image to specified dimensions.
    
    Resizes image using bilinear interpolation. Supports both explicit (H, W)
    dimensions and single dimension for square output. Preserves image dtype
    and number of channels.
    
    Args:
        image (np.ndarray): Image to resize. Can be 2D (grayscale) or 3D (color).
        
        size (Union[Tuple[int, int], int]): Target size specification.
            - Tuple (height, width): Explicit dimensions
            - Single int: Square output (size x size)
    
    Returns:
        np.ndarray: Resized image with shape matching specified size.
            - For size=(H, W): Returns (H, W) if 2D input, (H, W, C) if 3D
            - For size=N: Returns (N, N) if 2D input, (N, N, C) if 3D
            Dtype preserved from input.
    
    Raises:
        ValueError: If size is invalid or image has unexpected shape
    
    Example:
        >>> from app.utils.imaging import load_image, resize_image
        >>> 
        >>> image = load_image("chest_xray.jpg")
        >>> 
        >>> resized_square = resize_image(image, 256)
        >>> assert resized_square.shape == (256, 256)
        >>> 
        >>> resized_rect = resize_image(image, (512, 384))
        >>> assert resized_rect.shape == (512, 384)
    
    Notes:
        - Bilinear interpolation provides good balance of speed and quality
        - Preserves original dtype (uint8, float, etc.)
        - Works with both grayscale and color images
        - If size tuple is provided, interpreted as (height, width)
    """
    if isinstance(size, int):
        target_size = (size, size)
    elif isinstance(size, tuple) and len(size) == 2:
        target_size = size
    else:
        raise ValueError(
            f"size must be int or tuple (height, width), got {size}"
        )
    
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(
            f"size dimensions must be positive, got {target_size}"
        )
    
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return resized
