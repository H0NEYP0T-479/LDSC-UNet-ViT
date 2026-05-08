import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


CLASS_NAMES = ['normal', 'pneumonia', 'covid19', 'tuberculosis']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LDSCDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = CLASS_TO_IDX
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                raise ValueError(f"Class directory missing: {class_dir}")
            
            image_files = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            ])
            
            if len(image_files) == 0:
                raise ValueError(f"No images found in {class_dir}")
            
            for image_path in image_files:
                self.image_paths.append(image_path)
                self.labels.append(class_idx)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(
    train: bool = True,
    image_size: int = 224
) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    if not (0 < train_split < 1):
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    
    train_transform = get_transforms(train=True, image_size=image_size)
    val_transform = get_transforms(train=False, image_size=image_size)
    
    full_dataset = LDSCDataset(root_dir=root_dir, transform=None)
    
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
