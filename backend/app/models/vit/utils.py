import torch
import matplotlib.pyplot as plt
import os


def save_training_curve(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str = "training_curves_vit.png"
) -> None:
    """Save training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)