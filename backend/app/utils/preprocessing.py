"""Main preprocessing pipeline for chest X-ray images in LDSC-UNet-ViT"""
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from PIL import Image

from app.utils.ldsc_preprocess.denoise import non_local_means, apply_denoising
from app.utils.ldsc_preprocess.clahe import apply_clahe
from app.utils.ldsc_preprocess.contrast import normalize_zscore, normalize_minmax, enhance_contrast
from app.utils.ldsc_preprocess.noise import sharpen_edges


class LDSCPreprocessor:
    """
    Main preprocessing pipeline for chest X-ray images in LDSC-UNet-ViT.
    
    This class provides a unified interface for preprocessing chest X-ray images
    for two different models: Vision Transformer (ViT) for disease classification
    and UNet for lung segmentation. Each model requires specific input sizes and
    preprocessing steps optimized for the respective architecture.
    
    The preprocessing pipeline includes:
    1. Image loading and grayscale conversion
    2. Non-local means denoising for noise reduction
    3. CLAHE contrast enhancement for local feature enhancement
    4. Edge sharpening using unsharp mask
    5. Z-score normalization for neural network input
    6. Resizing to model-specific dimensions
    7. Conversion to PyTorch tensors with appropriate channel configuration
    
    Attributes:
        vit_size (int): Target image size for ViT model (default 224). ViT models
            typically use 224x224 inputs matching standard ImageNet dimensions.
        
        unet_size (int): Target image size for UNet model (default 256). UNet
            segmentation models often use 256x256 inputs for better boundary
            precision in segmentation tasks.
    
    Example:
        >>> from app.utils.preprocessing import LDSCPreprocessor
        >>> 
        >>> preprocessor = LDSCPreprocessor(vit_size=224, unet_size=256)
        >>> 
        >>> xray_path = "path/to/chest_xray.jpg"
        >>> vit_tensor = preprocessor.preprocess_for_vit(xray_path)
        >>> unet_tensor = preprocessor.preprocess_for_unet(xray_path)
        >>> 
        >>> stages = preprocessor.get_stages(xray_path)
        >>> original = stages['original']
        >>> enhanced = stages['enhanced']
    
    Notes:
        - All intermediate images in get_stages are returned as numpy arrays
        - ViT expects 3-channel (RGB-like) input: (1, 3, 224, 224)
        - UNet expects 1-channel (grayscale) input: (1, 1, 256, 256)
        - Pipeline is optimized for chest X-ray characteristics
        - Processing is single-image; batch processing requires external wrapping
        - All tensors are on CPU; move to device as needed for inference
    """
    
    def __init__(self, vit_size: int = 224, unet_size: int = 256):
        """
        Initialize the LDSC Preprocessor.
        
        Args:
            vit_size (int, optional): Target image size for ViT model.
                Defaults to 224 (standard ImageNet size).
            
            unet_size (int, optional): Target image size for UNet model.
                Defaults to 256 (common segmentation size).
        """
        self.vit_size = vit_size
        self.unet_size = unet_size
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Loads image using PIL and converts to numpy array. Automatically detects
        and handles various image formats (JPEG, PNG, etc.) and image modes
        (RGB, RGBA, grayscale, etc.).
        
        Args:
            image_path (Union[str, Path]): Path to image file.
        
        Returns:
            np.ndarray: Loaded image as numpy array. If image is color,
                returns (height, width, channels) array. Otherwise returns
                (height, width) for grayscale.
        
        Raises:
            FileNotFoundError: If image file does not exist
            IOError: If image cannot be loaded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            pil_image = Image.open(image_path)
            image_array = np.array(pil_image)
        except Exception as e:
            raise IOError(f"Failed to load image {image_path}: {str(e)}")
        
        return image_array
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        If image is already grayscale, returns as-is. If color (RGB/RGBA),
        converts using standard luminosity formula. Maintains uint8 dtype.
        
        Args:
            image (np.ndarray): Input image (grayscale or color).
        
        Returns:
            np.ndarray: Grayscale image (uint8), shape (height, width).
        """
        if len(image.shape) == 2:
            grayscale = image
        elif len(image.shape) == 3:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        if grayscale.dtype != np.uint8:
            grayscale = grayscale.astype(np.uint8)
        
        return grayscale
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising.
        
        Uses NLM algorithm optimized for medical imaging. Parameters chosen
        to balance noise reduction with detail preservation for chest X-rays.
        
        Args:
            image (np.ndarray): Grayscale image (uint8).
        
        Returns:
            np.ndarray: Denoised image (uint8), same shape as input.
        """
        denoised = non_local_means(
            image,
            h=10.0,
            template_window=7,
            search_window=21
        )
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast using CLAHE.
        
        Applies Contrast Limited Adaptive Histogram Equalization for local
        contrast enhancement. CLAHE parameters tuned for chest X-ray enhancement.
        
        Args:
            image (np.ndarray): Denoised grayscale image (uint8).
        
        Returns:
            np.ndarray: Contrast-enhanced image (uint8), same shape as input.
        """
        enhanced = apply_clahe(
            image,
            clip_limit=2.5,
            tile_grid_size=(8, 8)
        )
        return enhanced
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen edges using unsharp mask.
        
        Enhances fine anatomical details and boundaries through edge
        sharpening. Parameters chosen to highlight diagnostically relevant
        structures without excessive artifacts.
        
        Args:
            image (np.ndarray): Enhanced grayscale image (uint8).
        
        Returns:
            np.ndarray: Sharpened image (uint8), same shape as input.
        """
        sharpened = sharpen_edges(
            image,
            sigma=1.0,
            strength=1.2
        )
        return sharpened
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using Z-score standardization.
        
        Converts image to float32 with zero mean and unit variance, optimized
        for neural network input. Preserves anatomical detail by basing
        normalization on actual image statistics.
        
        Args:
            image (np.ndarray): Sharpened grayscale image (uint8).
        
        Returns:
            np.ndarray: Normalized image (float32) with mean≈0, std≈1.
        """
        normalized = normalize_zscore(image)
        return normalized
    
    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Resize image to target size using interpolation.
        
        Uses bilinear interpolation for smooth resizing. Works with both
        2D grayscale and 3D color images.
        
        Args:
            image (np.ndarray): Image to resize (uint8 or float).
            size (int): Target size (square: size x size).
        
        Returns:
            np.ndarray: Resized image, shape (size, size) or (size, size, channels).
        """
        if image.ndim == 2:
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        elif image.ndim == 3:
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        return resized
    
    def _to_tensor(
        self,
        image: np.ndarray,
        num_channels: int = 1
    ) -> torch.Tensor:
        """
        Convert image to PyTorch tensor.
        
        Handles conversion from numpy array to torch tensor with appropriate
        channel configuration. For 3-channel output, grayscale image is
        replicated across channels. Adds batch dimension for model inference.
        
        Args:
            image (np.ndarray): Image to convert (float32).
            num_channels (int, optional): Number of output channels.
                - 1: Returns (1, 1, H, W) for UNet (grayscale)
                - 3: Returns (1, 3, H, W) for ViT (replicated channels)
                Defaults to 1.
        
        Returns:
            torch.Tensor: Tensor with shape (batch=1, channels, height, width).
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")
        
        if num_channels == 1:
            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        elif num_channels == 3:
            image_3ch = np.stack([image, image, image], axis=2)
            tensor = torch.from_numpy(image_3ch).permute(2, 0, 1).unsqueeze(0).float()
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")
        
        return tensor
    
    def preprocess_for_vit(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess chest X-ray image for Vision Transformer (ViT) classification.
        
        Applies full preprocessing pipeline optimized for ViT disease classification:
        - Loads image and converts to grayscale
        - Denoise using non-local means
        - Enhance contrast with CLAHE
        - Sharpen edges for feature clarity
        - Normalize using Z-score standardization
        - Resize to ViT input size (224x224)
        - Convert to 3-channel tensor for ViT
        
        Output is ready for direct model inference without additional preprocessing.
        
        Args:
            image_path (Union[str, Path]): Path to chest X-ray image file.
        
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (1, 3, 224, 224).
                Tensor is float32 on CPU.
        
        Raises:
            FileNotFoundError: If image file does not exist
            IOError: If image cannot be loaded
            ValueError: If image has unexpected format
        
        Example:
            >>> from app.utils.preprocessing import LDSCPreprocessor
            >>> preprocessor = LDSCPreprocessor()
            >>> tensor = preprocessor.preprocess_for_vit("chest_xray.jpg")
            >>> assert tensor.shape == (1, 3, 224, 224)
            >>> assert tensor.dtype == torch.float32
        
        Notes:
            - Returns 3-channel tensor (channels replicated from grayscale)
            - Each channel contains identical grayscale data
            - Suitable for ViT models pre-trained on RGB images
            - Processing is deterministic (no augmentation)
        """
        image = self._load_image(image_path)
        grayscale = self._to_grayscale(image)
        denoised = self._denoise(grayscale)
        enhanced = self._enhance_contrast(denoised)
        sharpened = self._sharpen(enhanced)
        normalized = self._normalize(sharpened)
        resized = self._resize(normalized, self.vit_size)
        tensor = self._to_tensor(resized, num_channels=3)
        
        return tensor
    
    def preprocess_for_unet(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess chest X-ray image for UNet lung segmentation.
        
        Applies full preprocessing pipeline optimized for UNet segmentation:
        - Loads image and converts to grayscale
        - Denoise using non-local means
        - Enhance contrast with CLAHE
        - Sharpen edges for boundary clarity
        - Normalize using Z-score standardization
        - Resize to UNet input size (256x256)
        - Convert to single-channel tensor
        
        Output is ready for direct model inference for lung segmentation.
        
        Args:
            image_path (Union[str, Path]): Path to chest X-ray image file.
        
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (1, 1, 256, 256).
                Tensor is float32 on CPU.
        
        Raises:
            FileNotFoundError: If image file does not exist
            IOError: If image cannot be loaded
            ValueError: If image has unexpected format
        
        Example:
            >>> from app.utils.preprocessing import LDSCPreprocessor
            >>> preprocessor = LDSCPreprocessor()
            >>> tensor = preprocessor.preprocess_for_unet("chest_xray.jpg")
            >>> assert tensor.shape == (1, 1, 256, 256)
            >>> assert tensor.dtype == torch.float32
        
        Notes:
            - Returns single-channel tensor (grayscale)
            - Maintains grayscale representation for segmentation
            - Suitable for UNet medical image segmentation architectures
            - Processing is deterministic (no augmentation)
        """
        image = self._load_image(image_path)
        grayscale = self._to_grayscale(image)
        denoised = self._denoise(grayscale)
        enhanced = self._enhance_contrast(denoised)
        sharpened = self._sharpen(enhanced)
        normalized = self._normalize(sharpened)
        resized = self._resize(normalized, self.unet_size)
        tensor = self._to_tensor(resized, num_channels=1)
        
        return tensor
    
    def get_stages(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Get preprocessing stages for visualization and debugging.
        
        Executes full preprocessing pipeline and returns intermediate results
        at each stage. Useful for visualizing preprocessing effects, debugging
        artifacts, and understanding transformations applied to the image.
        
        Args:
            image_path (Union[str, Path]): Path to chest X-ray image file.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with keys:
                - 'original': Original grayscale image (uint8)
                - 'grayscale': Grayscale conversion result (uint8)
                - 'denoised': After NLM denoising (uint8)
                - 'enhanced': After CLAHE contrast enhancement (uint8)
                - 'sharpened': After edge sharpening (uint8)
                - 'normalized': After Z-score normalization (float32)
        
        Raises:
            FileNotFoundError: If image file does not exist
            IOError: If image cannot be loaded
            ValueError: If image has unexpected format
        
        Example:
            >>> from app.utils.preprocessing import LDSCPreprocessor
            >>> preprocessor = LDSCPreprocessor()
            >>> stages = preprocessor.get_stages("chest_xray.jpg")
            >>> original = stages['original']
            >>> enhanced = stages['enhanced']
            >>> normalized = stages['normalized']
        
        Notes:
            - All intermediate images are at original resolution before resizing
            - 'original' and 'grayscale' may be identical if input is grayscale
            - Useful for preprocessing pipeline validation and troubleshooting
            - Returns numpy arrays suitable for visualization with matplotlib/PIL
            - Does not return tensors or resized versions
        """
        image = self._load_image(image_path)
        grayscale = self._to_grayscale(image)
        denoised = self._denoise(grayscale)
        enhanced = self._enhance_contrast(denoised)
        sharpened = self._sharpen(enhanced)
        normalized = self._normalize(sharpened)
        
        stages = {
            'original': grayscale,
            'grayscale': grayscale,
            'denoised': denoised,
            'enhanced': enhanced,
            'sharpened': sharpened,
            'normalized': normalized
        }
        
        return stages
