import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from app.models.vit.model import LDSCViT
from app.models.vit.datamodule import get_dataloaders
import os


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train(config: dict):
    """Main ViT training loop."""
    device = config.get("device", "cpu")
    model = LDSCViT(num_classes=4, pretrained=True).to(device)
    train_loader, val_loader = get_dataloaders(
        config["data_dir"], config["batch_size"]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    scaler = GradScaler()
    best_val_acc = 0
    patience_counter = 0

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(config["checkpoint_dir"], "vit_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 10):
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(),
               os.path.join(config["checkpoint_dir"], "vit_last.pth"))


if __name__ == "__main__":
    train({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "backend/app/resources/dataset/classification",
        "checkpoint_dir": "backend/app/resources/checkpoints/vit",
        "batch_size": 32,
        "epochs": 100,
        "patience": 10
    })