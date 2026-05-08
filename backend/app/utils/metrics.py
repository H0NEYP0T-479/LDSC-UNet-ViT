"""Evaluation metrics for segmentation and classification tasks in LDSC-UNet-ViT"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient (F1 score) for binary segmentation.
    
    Dice coefficient measures the overlap between predicted segmentation and
    ground truth. It ranges from 0 (no overlap) to 1 (perfect overlap). Dice
    score is less sensitive to class imbalance compared to pixel accuracy,
    making it suitable for medical image segmentation where foreground is
    typically smaller than background.
    
    Args:
        pred (np.ndarray): Predicted binary segmentation mask. Values should be
            0 (background) or 1 (foreground). Can be boolean or integer.
            Shape: (height, width).
        
        target (np.ndarray): Ground truth binary segmentation mask. Values should
            be 0 (background) or 1 (foreground). Can be boolean or integer.
            Shape must match pred: (height, width).
        
        smooth (float, optional): Small epsilon value to avoid division by zero
            when both prediction and target are empty. Prevents NaN when
            computing dice for regions with no foreground.
            Defaults to 1e-6.
    
    Returns:
        float: Dice coefficient in range [0, 1].
            - 0: No overlap between pred and target
            - 0.5: 50% overlap
            - 1: Perfect overlap (identical masks)
    
    Raises:
        ValueError: If pred and target shapes don't match
    
    Example:
        >>> import numpy as np
        >>> from app.utils.metrics import dice_score
        >>> 
        >>> pred = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        >>> target = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        >>> dice = dice_score(pred, target)
        >>> 
        >>> assert 0 <= dice <= 1
        >>> print(f"Dice Score: {dice:.4f}")  # Output: ~0.6667
    
    Notes:
        - Equivalent to F1 score for binary classification
        - Symmetric: dice(A, B) == dice(B, A)
        - Robust to class imbalance (common in medical imaging)
        - Smooth parameter prevents division by zero
        - Used extensively in medical image segmentation evaluation
    
    Formula:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
        where X is prediction, Y is ground truth
    
    Reference:
        Dice, L. R. (1945). Measures of the amount of ecologic association
        between species. Ecology, 26(3), 297-302.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    pred_binary = pred.flatten().astype(bool)
    target_binary = target.flatten().astype(bool)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    
    dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) for binary segmentation.
    
    IoU, also known as Jaccard index, measures the overlap between predicted
    and ground truth regions as a ratio of intersection to union. IoU is more
    stringent than Dice score and penalizes false positives more heavily.
    Standard metric for segmentation evaluation in computer vision.
    
    Args:
        pred (np.ndarray): Predicted binary segmentation mask. Values should be
            0 (background) or 1 (foreground). Can be boolean or integer.
            Shape: (height, width).
        
        target (np.ndarray): Ground truth binary segmentation mask. Values should
            be 0 (background) or 1 (foreground). Can be boolean or integer.
            Shape must match pred: (height, width).
        
        smooth (float, optional): Small epsilon value to avoid division by zero.
            Prevents NaN when both prediction and target are empty.
            Defaults to 1e-6.
    
    Returns:
        float: IoU score in range [0, 1].
            - 0: No overlap between pred and target
            - 0.5: 50% overlap
            - 1: Perfect overlap (identical masks)
    
    Raises:
        ValueError: If pred and target shapes don't match
    
    Example:
        >>> import numpy as np
        >>> from app.utils.metrics import iou_score
        >>> 
        >>> pred = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        >>> target = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        >>> iou = iou_score(pred, target)
        >>> 
        >>> assert 0 <= iou <= 1
        >>> assert iou < dice_score(pred, target)
        >>> print(f"IoU Score: {iou:.4f}")  # Output: ~0.5000
    
    Notes:
        - Also called Jaccard index or Jaccard similarity
        - More stringent than Dice score (always <= Dice)
        - Standard metric for COCO and other benchmarks
        - Symmetric: iou(A, B) == iou(B, A)
        - Smooth parameter prevents division by zero
        - Penalizes false positives more than Dice
    
    Relationship to Dice:
        IoU = Dice / (2 - Dice)
        Dice = 2 * IoU / (1 + IoU)
    
    Formula:
        IoU = (|X ∩ Y| + smooth) / (|X ∪ Y| + smooth)
        where X is prediction, Y is ground truth
    
    Reference:
        Jaccard, P. (1901). Étude comparative de la distribution florale dans
        une portion des Alpes et des Jura. Bulletin de la Société Vaudoise des
        Sciences Naturelles, 37, 547-579.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    pred_binary = pred.flatten().astype(bool)
    target_binary = target.flatten().astype(bool)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate pixel-level accuracy for segmentation.
    
    Pixel accuracy computes the fraction of correctly classified pixels across
    the entire image. While simple and intuitive, it is sensitive to class
    imbalance and may give misleadingly high scores when foreground is rare
    (as in medical imaging). Use Dice or IoU for medical segmentation instead.
    
    Args:
        pred (np.ndarray): Predicted binary segmentation mask. Values should be
            0 (background) or 1 (foreground). Can be boolean or integer.
            Shape: (height, width).
        
        target (np.ndarray): Ground truth binary segmentation mask. Values should
            be 0 (background) or 1 (foreground). Can be boolean or integer.
            Shape must match pred: (height, width).
    
    Returns:
        float: Pixel accuracy in range [0, 1].
            - 0: No pixels match
            - 0.5: 50% pixels correctly classified
            - 1: All pixels correctly classified
    
    Raises:
        ValueError: If pred and target shapes don't match
    
    Example:
        >>> import numpy as np
        >>> from app.utils.metrics import pixel_accuracy
        >>> 
        >>> pred = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        >>> target = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        >>> acc = pixel_accuracy(pred, target)
        >>> 
        >>> assert acc == 1.0
        >>> print(f"Pixel Accuracy: {acc:.4f}")  # Output: 1.0000
    
    Notes:
        - Simple and interpretable metric
        - Highly sensitive to class imbalance
        - Not recommended as primary metric for medical segmentation
        - Can be misleading when foreground is rare (e.g., 1% foreground)
        - Useful for monitoring overall performance alongside Dice/IoU
        - Computing time is linear with image size
    
    Limitations:
        - Example: 99% background image with perfect background prediction
          gets 99% accuracy despite failing at foreground
        - Therefore, use alongside Dice and IoU for comprehensive evaluation
    
    Formula:
        Accuracy = (Correct pixels) / (Total pixels)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    pred_binary = pred.flatten().astype(bool)
    target_binary = target.flatten().astype(bool)
    
    correct = (pred_binary == target_binary).sum()
    total = pred_binary.size
    
    accuracy = float(correct) / float(total)
    
    return accuracy


def classification_metrics(
    preds: List[int],
    targets: List[int],
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate classification metrics per class.
    
    Computes precision, recall, F1 score, and accuracy for each class in a
    multi-class classification task. Returns detailed per-class metrics along
    with macro and weighted averages. Useful for understanding model
    performance across different disease classes.
    
    Args:
        preds (List[int]): Predicted class indices for each sample.
            Values should be integers in range [0, num_classes).
            Length: N samples.
        
        targets (List[int]): Ground truth class indices for each sample.
            Values should be integers in range [0, num_classes).
            Length must match preds: N samples.
        
        class_names (List[str]): Human-readable names for each class.
            Used as keys in returned dictionary.
            Length should be num_classes.
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with metrics per class.
            - Top-level keys: class names from class_names parameter
            - Second-level keys: 'precision', 'recall', 'f1'
            - Additionally includes 'accuracy' (global), 'macro_f1', 'weighted_f1'
            - All values are floats in range [0, 1]
    
    Raises:
        ValueError: If preds and targets lengths don't match
        ValueError: If class_names length doesn't match number of classes
    
    Example:
        >>> from app.utils.metrics import classification_metrics
        >>> 
        >>> preds = [0, 1, 1, 0, 2, 2, 1, 0]
        >>> targets = [0, 1, 0, 0, 2, 1, 1, 0]
        >>> class_names = ['Normal', 'Pneumonia', 'COVID-19']
        >>> 
        >>> metrics = classification_metrics(preds, targets, class_names)
        >>> 
        >>> print(f"Normal Precision: {metrics['Normal']['precision']:.4f}")
        >>> print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    Notes:
        - Uses sklearn for robust metric computation
        - Handles imbalanced datasets properly
        - Precision: TP / (TP + FP) - correctness of positive predictions
        - Recall: TP / (TP + FN) - coverage of actual positives
        - F1: Harmonic mean of precision and recall
        - Macro F1: Unweighted average across classes
        - Weighted F1: Weighted by support (number of true instances)
    
    Medical Imaging Application:
        For disease classification (Normal, Pneumonia, COVID-19, TB):
        - High recall for each disease is critical (catch all cases)
        - High precision reduces false alarms (unnecessary treatment)
        - F1 balances both concerns
    
    Returns Dictionary Structure:
        {
            'Normal': {'precision': 0.9, 'recall': 0.85, 'f1': 0.87},
            'Pneumonia': {'precision': 0.88, 'recall': 0.91, 'f1': 0.89},
            'COVID-19': {'precision': 0.95, 'recall': 0.92, 'f1': 0.93},
            'accuracy': 0.89,
            'macro_f1': 0.90,
            'weighted_f1': 0.90,
            'confusion_matrix': [[...], [...], [...]]
        }
    """
    preds = np.array(preds, dtype=int)
    targets = np.array(targets, dtype=int)
    
    if len(preds) != len(targets):
        raise ValueError(f"Length mismatch: preds {len(preds)} vs targets {len(targets)}")
    
    num_classes = len(class_names)
    
    if preds.max() >= num_classes or targets.max() >= num_classes:
        raise ValueError(f"Class indices out of range for {num_classes} classes")
    
    metrics_dict = {}
    
    for i, class_name in enumerate(class_names):
        class_preds = (preds == i).astype(int)
        class_targets = (targets == i).astype(int)
        
        prec = precision_score(class_targets, class_preds, zero_division=0)
        rec = recall_score(class_targets, class_preds, zero_division=0)
        f1 = f1_score(class_targets, class_preds, zero_division=0)
        
        metrics_dict[class_name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
    
    overall_accuracy = accuracy_score(targets, preds)
    macro_f1 = np.mean([metrics_dict[name]['f1'] for name in class_names])
    
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    
    conf_matrix = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    metrics_dict['accuracy'] = float(overall_accuracy)
    metrics_dict['macro_f1'] = float(macro_f1)
    metrics_dict['weighted_f1'] = float(weighted_f1)
    metrics_dict['confusion_matrix'] = conf_matrix.tolist()
    
    return metrics_dict
