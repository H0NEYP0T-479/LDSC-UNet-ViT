import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path

from app.models.vit.model import LDSCViT
from app.models.vit.datamodule import get_dataloaders
from app.logging_config import get_logger
from app.utils.logger import log_inference, log_error


logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training', leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': correct / total
        })
    
    epoch_loss = total_loss / len(loader)
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
    
    epoch_loss = total_loss / len(loader)
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    model = LDSCViT(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    scaler = GradScaler()
    
    checkpoint_dir = Path(config['checkpoint_dir']) / 'vit'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        scheduler.step(val_loss)
        
        torch.save(model.state_dict(), checkpoint_dir / 'vit_last.pth')
        logger.info(f"Saved last model to {checkpoint_dir / 'vit_last.pth'}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'vit_best.pth')
            logger.info(f"Saved best model to {checkpoint_dir / 'vit_best.pth'}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{config['early_stopping_patience']}")
        
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    logger.info("Training completed")
    return model


if __name__ == '__main__':
    config = {
        'data_dir': 'data/chest_xrays',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'train_split': 0.8,
        'image_size': 224,
        'num_workers': 4,
        'early_stopping_patience': 10
    }
    
    train(config)
