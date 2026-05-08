import torch
import torch.nn as nn
import timm


class LDSCViT(nn.Module):
    """Vision Transformer classifier for lung disease classification."""

    CLASS_NAMES = ["normal", "pneumonia", "covid19", "tuberculosis"]

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_last_attention_layer(self) -> nn.Module:
        """Return last transformer block for GradCAM."""
        return self.backbone.blocks[-1]