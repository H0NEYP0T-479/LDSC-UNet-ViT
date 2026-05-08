"""Gradient-weighted Class Activation Map (GradCAM) for Vision Transformer"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple


class GradCAM:
    """
    Generate Grad-CAM heatmaps for model interpretability from Vision Transformer.
    
    Grad-CAM (Gradient-weighted Class Activation Map) is a technique for visualizing
    which regions of an input image contribute most to a model's prediction for a
    specific class. It combines spatial feature maps with their gradient weights,
    providing interpretable visualizations for understanding model decisions.
    
    This implementation is optimized for Vision Transformer (ViT) architectures,
    capturing activations and gradients from specified transformer layers. The
    method works by:
    1. Forward pass: Capturing feature maps from target layer
    2. Backward pass: Computing gradients with respect to target layer
    3. Weighting: Computing importance weights as global average of gradients
    4. Combination: Creating heatmap from weighted feature maps
    5. Visualization: Applying colormap and overlaying on original image
    
    Attributes:
        model (nn.Module): PyTorch model for which to generate Grad-CAM.
            Typically a Vision Transformer model from timm library.
        
        target_layer (nn.Module): Specific layer from which to extract
            activations and gradients. Common choices for ViT:
            - model.blocks[-1]  # Last transformer block (default)
            - model.blocks[-2]  # Second-to-last block
    
    Example:
        >>> import torch
        >>> import timm
        >>> from app.utils.gradcam import GradCAM
        >>> 
        >>> model = timm.create_model('vit_base_patch16_224', pretrained=True)
        >>> model.eval()
        >>> 
        >>> gradcam = GradCAM(model, target_layer=model.blocks[-1])
        >>> 
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> heatmap = gradcam.generate(input_tensor, class_idx=0)
        >>> 
        >>> original_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> overlay = gradcam.overlay_heatmap(original_image, heatmap, alpha=0.4)
    
    Notes:
        - Model should be set to eval mode before generating Grad-CAM
        - Input tensor should have gradient computation enabled (default)
        - class_idx=None uses predicted class (requires forward pass)
        - Works best with transformer models (ViT, DeiT, etc.)
        - Supports both single image and batch inputs
        - Heatmap is normalized to 0-255 range automatically
    
    Reference:
        Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
        Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via
        Gradient-based Localization. IEEE ICCV.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize GradCAM with model and target layer.
        
        Args:
            model (nn.Module): PyTorch neural network model. Should be set to
                eval mode before use (model.eval()).
            
            target_layer (nn.Module): Specific layer to extract activations and
                gradients from. For ViT, typically model.blocks[-1].
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward and backward hooks on target layer.
        
        Forward hook captures feature map activations during forward pass.
        Backward hook captures gradients during backward pass. These are
        used together to compute Grad-CAM weights.
        """
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image.
        
        Performs forward and backward passes to compute class activation map.
        The heatmap highlights regions that contribute to the model's
        classification decision for the specified class.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor. Expected shape:
                - (1, 3, H, W): Single image batch with 3 channels (RGB)
                - (B, 3, H, W): Batch of images (uses first image)
                Values should be normalized to [-1, 1] or [0, 1] range.
            
            class_idx (int, optional): Target class index for which to generate
                heatmap. If None, uses the class with highest predicted
                probability (argmax). Defaults to None.
        
        Returns:
            np.ndarray: Grad-CAM heatmap as uint8 numpy array.
                Shape: (height, width). Values in [0, 255].
                Spatial dimensions match input image size.
        
        Raises:
            RuntimeError: If model is not in eval mode or gradients not available
            ValueError: If class_idx is out of range
        
        Example:
            >>> import torch
            >>> import timm
            >>> from app.utils.gradcam import GradCAM
            >>> 
            >>> model = timm.create_model('vit_base_patch16_224', pretrained=True)
            >>> model.eval()
            >>> gradcam = GradCAM(model, model.blocks[-1])
            >>> 
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> heatmap = gradcam.generate(input_tensor, class_idx=0)
            >>> 
            >>> assert heatmap.dtype == np.uint8
            >>> assert heatmap.shape == (224, 224)
            >>> assert 0 <= heatmap.min() and heatmap.max() <= 255
        
        Notes:
            - Model must be in eval() mode for valid results
            - Input tensor should require gradients (default)
            - class_idx=None performs extra forward pass to get predictions
            - Heatmap returned is always uint8 in [0, 255] range
            - Spatial resolution matches input image dimensions
        """
        batch_size, channels, height, width = input_tensor.shape
        
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        target_logit = logits[0, class_idx]
        
        self.model.zero_grad()
        target_logit.backward(retain_graph=True)
        
        activations = self.activations[0]
        gradients = self.gradients[0]
        
        if activations.dim() == 3:
            b, n, c = activations.shape
            activations = activations.view(b, int(np.sqrt(n-1)), int(np.sqrt(n-1)), c)
            activations = activations.permute(0, 3, 1, 2)
        
        if gradients.dim() == 3:
            b, n, c = gradients.shape
            gradients = gradients.view(b, int(np.sqrt(n-1)), int(np.sqrt(n-1)), c)
            gradients = gradients.permute(0, 3, 1, 2)
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)
        
        heatmap = torch.relu(weighted_activations)
        
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        if heatmap.size == 0:
            heatmap = np.zeros((height, width), dtype=np.uint8)
        else:
            if heatmap.ndim == 0:
                heatmap = np.zeros((height, width), dtype=np.uint8)
            else:
                h, w = heatmap.shape
                if h != height or w != width:
                    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                
                heatmap_min = heatmap.min()
                heatmap_max = heatmap.max()
                
                if heatmap_max - heatmap_min > 1e-6:
                    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
                else:
                    heatmap = np.zeros_like(heatmap)
                
                heatmap = (heatmap * 255).astype(np.uint8)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create colored overlay of heatmap on original image.
        
        Converts grayscale heatmap to jet colormap and blends with original
        image for visualization. The jet colormap uses cool colors (blue) for
        low values and warm colors (red) for high values, making important
        regions easily identifiable.
        
        Args:
            original_image (np.ndarray): Original image for overlay base.
                Expected shape (height, width) for grayscale or
                (height, width, 3) for color. Values should be uint8 [0, 255].
            
            heatmap (np.ndarray): Grad-CAM heatmap to overlay. Expected shape
                (height, width) with uint8 values [0, 255]. Should be output
                from generate() method.
            
            alpha (float, optional): Blending factor for heatmap overlay.
                - 0.0: Original image only (no heatmap visible)
                - 0.5: 50% heatmap, 50% original
                - 1.0: Full heatmap, no original visible
                Defaults to 0.4 (40% heatmap transparency).
        
        Returns:
            np.ndarray: Blended RGB image with colored heatmap overlay.
                Shape: (height, width, 3). Values are uint8 [0, 255].
                Can be displayed using matplotlib or saved to file.
        
        Raises:
            ValueError: If image shapes don't match or invalid parameters
        
        Example:
            >>> from app.utils.gradcam import GradCAM
            >>> from app.utils.imaging import load_image
            >>> import numpy as np
            >>> 
            >>> original = load_image("chest_xray.jpg")
            >>> original_rgb = np.stack([original, original, original], axis=2)
            >>> 
            >>> heatmap = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            >>> overlay = gradcam.overlay_heatmap(original_rgb, heatmap, alpha=0.4)
            >>> 
            >>> assert overlay.shape == (224, 224, 3)
            >>> assert overlay.dtype == np.uint8
        
        Notes:
            - Automatically converts grayscale image to RGB for overlay
            - Jet colormap: blue (cold) -> green (medium) -> red (hot)
            - Alpha controls transparency of heatmap overlay
            - Output is always RGB/BGR 3-channel image
            - Can be directly displayed or saved as color image
        
        Jet Colormap Interpretation:
            - Dark blue: Low activation (less important for decision)
            - Green: Medium activation
            - Red/Yellow: High activation (very important for decision)
        """
        if heatmap.shape[0] != original_image.shape[0] or heatmap.shape[1] != original_image.shape[1]:
            raise ValueError(
                f"Image shapes don't match: original {original_image.shape}, heatmap {heatmap.shape}"
            )
        
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        if original_image.dtype != np.uint8:
            original_image = original_image.astype(np.uint8)
        
        if len(original_image.shape) == 2:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_bgr = original_image
        else:
            raise ValueError(f"Unexpected image shape: {original_image.shape}")
        
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
