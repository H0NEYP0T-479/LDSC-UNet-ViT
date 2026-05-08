"""Vision Transformer classifier for disease classification in LDSC-UNet-ViT"""
import torch
import torch.nn as nn
import timm
from typing import Optional


CLASS_NAMES = ['normal', 'pneumonia', 'covid19', 'tuberculosis']
NUM_CLASSES = len(CLASS_NAMES)
VIT_HIDDEN_DIM = 768
CLASSIFIER_HIDDEN_DIM = 256
DROPOUT_RATE = 0.3


class LDSCViT(nn.Module):
    """
    Vision Transformer classifier for lung disease classification.
    
    LDSCViT combines a pre-trained Vision Transformer base model with a custom
    classification head optimized for medical image analysis. The model processes
    224x224 RGB chest X-ray images and outputs predictions for four disease classes:
    normal, pneumonia, COVID-19, and tuberculosis.
    
    Architecture:
    - Backbone: ViT Base (patch16-224) from timm library
      * Pre-trained on ImageNet-21k (if pretrained=True)
      * 12 transformer blocks with 12 attention heads
      * Hidden dimension: 768
      * Processes image as 14x14 patches (224/16)
    
    - Classification Head (custom):
      * Input: 768-dimensional ViT output
      * Layer 1: Linear(768, 256)
      * Activation: ReLU
      * Regularization: Dropout(0.3)
      * Layer 2: Linear(256, 4) for 4-class classification
      * Output: Logits for [normal, pneumonia, covid19, tuberculosis]
    
    Features:
    - Transfer learning from ImageNet-21k pre-training
    - Optimized classifier head for medical imaging
    - Dropout for regularization (prevents overfitting)
    - Supports Grad-CAM visualization via get_last_attention_layer()
    - Compatible with both single image and batch inference
    
    Model Parameters:
    - Total parameters: ~86M (base ViT) + classifier
    - Trainable: All layers (fine-tuning or transfer learning)
    - Input size: (batch, 3, 224, 224)
    - Output size: (batch, 4)
    
    Attributes:
        num_classes (int): Number of output classes (default 4).
        
        pretrained (bool): Whether to use pre-trained ImageNet-21k weights.
            True: Loads pre-trained backbone for transfer learning (recommended)
            False: Random initialization (requires more training data)
        
        vit_model (nn.Module): Pre-trained ViT backbone from timm.
        
        classifier (nn.Module): Custom classification head.
    
    Example:
        >>> import torch
        >>> from app.models.vit.model import LDSCViT, CLASS_NAMES
        >>> 
        >>> model = LDSCViT(num_classes=4, pretrained=True)
        >>> model.eval()
        >>> 
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
        >>> 
        >>> assert logits.shape == (1, 4)
        >>> probabilities = torch.softmax(logits, dim=1)
        >>> predicted_class = logits.argmax(dim=1).item()
        >>> predicted_disease = CLASS_NAMES[predicted_class]
    
    Notes:
        - Model expects input normalized to ImageNet statistics
        - Default normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        - Grayscale X-ray images replicated to 3 channels for ViT compatibility
        - Pre-training significantly improves performance on medical imaging
        - Dropout in classifier head aids generalization
        - Suitable for fine-tuning on medical datasets
    
    References:
        - Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words:
          Transformers for Image Recognition at Scale. ICLR.
        - Vision Transformer (ViT) implementation in timm library
    
    Clinical Application:
        Input: Chest X-ray (224x224, 3-channel)
        Output: Disease probability distribution
        Diseases:
        - normal: Healthy lungs
        - pneumonia: Bacterial/viral pneumonia
        - covid19: COVID-19 infection
        - tuberculosis: Tuberculosis infection
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Initialize ViT classifier with custom head.
        
        Args:
            num_classes (int, optional): Number of output classes for disease
                classification. Defaults to 4 (normal, pneumonia, covid19, tuberculosis).
            
            pretrained (bool, optional): Load pre-trained ImageNet-21k weights.
                Significantly improves performance on medical images through
                transfer learning. Recommended for practical use.
                Defaults to True.
        """
        super(LDSCViT, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        self.vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(VIT_HIDDEN_DIM, CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT and classifier.
        
        Processes input image through ViT backbone to extract features,
        then passes through custom classifier head for disease prediction.
        
        Args:
            x (torch.Tensor): Input batch of images.
                Shape: (batch_size, 3, 224, 224)
                Expected values: Normalized to ImageNet statistics
                - mean=[0.485, 0.456, 0.406]
                - std=[0.229, 0.224, 0.225]
        
        Returns:
            torch.Tensor: Classification logits for each class.
                Shape: (batch_size, num_classes)
                Raw unnormalized scores. Apply softmax for probabilities.
                Suitable for cross-entropy loss during training.
        
        Example:
            >>> import torch
            >>> from app.models.vit.model import LDSCViT
            >>> 
            >>> model = LDSCViT()
            >>> model.eval()
            >>> 
            >>> batch = torch.randn(4, 3, 224, 224)
            >>> logits = model(batch)
            >>> 
            >>> assert logits.shape == (4, 4)
            >>> probabilities = torch.softmax(logits, dim=1)
            >>> predictions = logits.argmax(dim=1)
        
        Notes:
            - ViT backbone extracts 768-dimensional feature vector
            - Classifier head projects to num_classes logits
            - Output is pre-softmax (suitable for nn.CrossEntropyLoss)
            - For inference, apply torch.softmax(..., dim=1) for probabilities
        """
        features = self.vit_model(x)
        logits = self.classifier(features)
        return logits
    
    def get_last_attention_layer(self) -> nn.Module:
        """
        Get the last transformer block for Grad-CAM visualization.
        
        Returns the final transformer block from the ViT backbone. This layer
        is used as the target layer for Grad-CAM to visualize which image regions
        contribute most to the model's classification decision.
        
        Returns:
            nn.Module: Last transformer block (ViT.blocks[-1]). Contains multi-head
                self-attention and feedforward components. Produces feature maps
                that drive the classification decision.
        
        Example:
            >>> import torch
            >>> from app.models.vit.model import LDSCViT
            >>> from app.utils.gradcam import GradCAM
            >>> 
            >>> model = LDSCViT()
            >>> model.eval()
            >>> 
            >>> target_layer = model.get_last_attention_layer()
            >>> gradcam = GradCAM(model, target_layer)
            >>> 
            >>> x = torch.randn(1, 3, 224, 224)
            >>> heatmap = gradcam.generate(x, class_idx=0)
        
        Notes:
            - Used specifically for Grad-CAM visualization
            - Last layer chosen for most relevant feature visualization
            - Alternative layers (e.g., model.blocks[-2]) provide different perspectives
            - Grad-CAM with this layer highlights disease-relevant image regions
            - Useful for model interpretability in clinical settings
        
        Grad-CAM Application:
            1. Get attention layer: target_layer = model.get_last_attention_layer()
            2. Initialize GradCAM: gradcam = GradCAM(model, target_layer)
            3. Generate heatmap: heatmap = gradcam.generate(image, class_idx=disease_id)
            4. Overlay on image: overlay = gradcam.overlay_heatmap(image, heatmap)
            5. Visualize: Shows which lung regions triggered disease prediction
        """
        return self.vit_model.blocks[-1]
