import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LungSegDataset(Dataset):
    """Dataset for lung segmentation with images and binary masks."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        image_size: int = 256
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        image = cv2.imread(
            os.path.join(self.image_dir, fname),
            cv2.IMREAD_GRAYSCALE
        )
        mask_path = os.path.join(
            self.mask_dir,
            fname.replace(".jpg", ".png").replace(".jpeg", ".png")
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        return image, mask


def get_seg_transforms(train: bool = True) -> A.Compose:
    """Get albumentations transforms for segmentation."""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ElasticTransform(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])
    return A.Compose([
        A.Resize(256, 256),
    ])


def get_seg_dataloaders(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 16,
    train_split: float = 0.8
):
    """Return train and val dataloaders for segmentation."""
    full_dataset = LungSegDataset(
        image_dir, mask_dir,
        transform=get_seg_transforms(True)
    )
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_seg_transforms(False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader