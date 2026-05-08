import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LungSegDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 256
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        self.image_paths = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        self.mask_paths = [
            self.mask_dir / img.name for img in self.image_paths
        ]
        
        for mask_path in self.mask_paths:
            if not mask_path.exists():
                raise ValueError(f"Mask not found: {mask_path}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        
        if mask is None:
            raise IOError(f"Failed to load mask: {mask_path}")
        
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) > 128).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


def get_seg_transforms(train: bool = True, image_size: int = 256) -> A.Compose:
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50),
            A.Normalize(
                mean=[0.485],
                std=[0.229],
                max_pixel_value=1.0
            )
        ], keypoint_params=A.KeypointParams(format='xy'))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485],
                std=[0.229],
                max_pixel_value=1.0
            )
        ])


def get_seg_dataloaders(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 16,
    train_split: float = 0.8,
    image_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_transform = get_seg_transforms(train=True, image_size=image_size)
    val_transform = get_seg_transforms(train=False, image_size=image_size)
    
    full_dataset = LungSegDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,
        image_size=image_size
    )
    
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
