import torch
import torch.nn as nn
import os
from app.models.unet_lung.model import UNetLung
from app.models.unet_lung.datamodule import get_seg_dataloaders


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute Dice loss."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss."""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch, return loss and dice score."""
    model.train()
    total_loss, total_dice = 0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_bin = (preds > 0.5).float()
        intersection = (pred_bin * masks).sum()
        dice = (2 * intersection) / (pred_bin.sum() + masks.sum() + 1e-8)
        total_dice += dice.item()
    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, criterion, device):
    """Validate model, return loss and dice score."""
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            total_loss += loss.item()
            pred_bin = (preds > 0.5).float()
            intersection = (pred_bin * masks).sum()
            dice = (2 * intersection) / (pred_bin.sum() + masks.sum() + 1e-8)
            total_dice += dice.item()
    return total_loss / len(loader), total_dice / len(loader)


def train(config: dict):
    """Main UNet training loop."""
    device = config.get("device", "cpu")
    model = UNetLung(in_channels=1, out_channels=1).to(device)
    train_loader, val_loader = get_seg_dataloaders(
        config["image_dir"], config["mask_dir"], config["batch_size"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    best_dice = 0
    patience_counter = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, bce_dice_loss, device
        )
        val_loss, val_dice = validate(
            model, val_loader, bce_dice_loss, device
        )
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(config["checkpoint_dir"], "unet_lung_best.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 10):
                print("Early stopping triggered.")
                break

    torch.save(
        model.state_dict(),
        os.path.join(config["checkpoint_dir"], "unet_lung_last.pth")
    )


if __name__ == "__main__":
    train({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_dir": "backend/app/resources/dataset/segmentation/images",
        "mask_dir": "backend/app/resources/dataset/segmentation/masks",
        "checkpoint_dir": "backend/app/resources/checkpoints/unet_lung",
        "batch_size": 16,
        "epochs": 100,
        "patience": 10
    })