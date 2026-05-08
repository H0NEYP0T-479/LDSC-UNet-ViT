import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def save_seg_training_curve(
    train_losses: list,
    val_losses: list,
    train_dices: list,
    val_dices: list,
    save_path: str = "training_curves_unet.png"
) -> None:
    """Save UNet training loss and dice curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_dices, label="Train Dice")
    ax2.plot(val_dices, label="Val Dice")
    ax2.set_title("Dice Score")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    save_path: str = None
) -> np.ndarray:
    """Overlay segmentation mask on image for visualization."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    if save_path:
        cv2.imwrite(save_path, result)
    return result