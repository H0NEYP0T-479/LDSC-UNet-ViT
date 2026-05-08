import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """GradCAM heatmap generator for ViT models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap. Returns uint8 numpy (H,W)."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=0)
        cam = (weights.unsqueeze(0) * activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        h = int(input_tensor.shape[2])
        w = int(input_tensor.shape[3])
        cam = cv2.resize(cam, (w, h))
        return cam

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay GradCAM heatmap on original image."""
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(
                original_image, cv2.COLOR_GRAY2BGR
            )
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_image, 1 - alpha, colored, alpha, 0)