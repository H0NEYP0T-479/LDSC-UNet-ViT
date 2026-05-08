import os
from pathlib import Path


class DatasetService:
    """Service for dataset management and batch processing."""

    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root

    def get_class_counts(self) -> dict:
        """Return count of images per class."""
        counts = {}
        clf_dir = os.path.join(self.dataset_root, "classification")
        if not os.path.exists(clf_dir):
            return counts
        for cls in os.listdir(clf_dir):
            cls_path = os.path.join(clf_dir, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
        return counts

    def get_dataset_stats(self) -> dict:
        """Return full dataset statistics."""
        counts = self.get_class_counts()
        total = sum(counts.values())
        return {
            "total_images": total,
            "class_counts": counts,
            "class_balance": {
                cls: round(count / total * 100, 2)
                for cls, count in counts.items()
            } if total > 0 else {}
        }