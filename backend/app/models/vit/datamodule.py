import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


CLASS_NAMES = [
    "COVID-19", 
    "Normal", 
    "Lung Opacity", 
    "Viral Pneumonia"
]


class LDSCDataset(Dataset):
    """Dataset for lung disease classification."""

    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool = True, image_size: int = 224):
    """Get train or val transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    image_size: int = 224
):
    """Return train and val dataloaders."""
    full_dataset = LDSCDataset(root_dir, transform=get_transforms(True, image_size))
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_transforms(False, image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader