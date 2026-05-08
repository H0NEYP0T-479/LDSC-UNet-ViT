import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice score between binary pred and target."""
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    intersection = (pred & target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU score between binary pred and target."""
    pred = pred.flatten().astype(bool)
    target = target.flatten().astype(bool)
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / (union + 1e-8)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-level accuracy."""
    return (pred == target).sum() / target.size


def classification_metrics(
    preds: list,
    targets: list,
    class_names: list
) -> dict:
    """Return per-class accuracy, precision, recall, f1."""
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(class_names))), zero_division=0
    )
    return {
        "accuracy": acc,
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i])
            }
            for i in range(len(class_names))
        }
    }