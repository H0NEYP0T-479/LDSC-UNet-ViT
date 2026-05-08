import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path

from app.models.unet.model import UNetLung
from app.models.unet_lung.datamodule import get_seg_dataloaders
from app.utils.metrics import dice_score
from app.logging_config import get_logger


logger = get_logger(__name__)


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice


def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    
    return 0.5 * bce + 0.5 * dice


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    
    progress_bar = tqdm(loader, desc='Training', leave=False)
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            pred_binary = (logits > 0.5).float()
            dice = dice_score(pred_binary, masks)
        
        total_dice += dice.item()
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'dice': dice.item()
        })
    
    epoch_loss = total_loss / len(loader)
    epoch_dice = total_dice / len(loader)
    
    return epoch_loss, epoch_dice


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    progress_bar = tqdm(loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            total_loss += loss.item()
            
            pred_binary = (logits > 0.5).float()
            dice = dice_score(pred_binary, masks)
            total_dice += dice.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'dice': dice.item()
            })
    
    epoch_loss = total_loss / len(loader)
    epoch_dice = total_dice / len(loader)
    
    return epoch_loss, epoch_dice


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader = get_seg_dataloaders(
        image_dir=config['image_dir'],
        mask_dir=config['mask_dir'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    model = UNetLung(
        in_channels=1,
        out_channels=1,
        features=config['features']
    ).to(device)
    
    criterion = bce_dice_loss
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    checkpoint_dir = Path(config['checkpoint_dir']) / 'unet_lung'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
        )
        
        scheduler.step(val_dice)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'unet_lung_best.pth')
            logger.info(f"Saved best model to {checkpoint_dir / 'unet_lung_best.pth'}")
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
        'image_dir': 'data/lungs/images',
        'mask_dir': 'data/lungs/masks',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'train_split': 0.8,
        'image_size': 256,
        'num_workers': 4,
        'features': [64, 128, 256, 512],
        'early_stopping_patience': 10
    }
    
    train(config)
