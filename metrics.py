"""Segmentation metrics (Section 11/15), with the agreed edge-case rule:
if both prediction and ground truth are empty, Dice/IoU = 1.0 rather than
falling through to a NaN from 0/0."""

import torch


def _binarize(x, threshold=0.5):
    return (x > threshold).float()


def dice_score(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    p_sum = pred.sum(dim=(1, 2, 3, 4))
    t_sum = target.sum(dim=(1, 2, 3, 4))

    both_empty = (p_sum == 0) & (t_sum == 0)
    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice  # (B,)


def iou_score(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) - inter

    both_empty = union == 0
    iou = (inter + eps) / (union + eps)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    return iou  # (B,)


def precision_recall(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    tp = (pred * target).sum(dim=(1, 2, 3, 4))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3, 4))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3, 4))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision, recall  # (B,), (B,)

"""Segmentation metrics (Section 11/15), with the agreed edge-case rule:
if both prediction and ground truth are empty, Dice/IoU = 1.0 rather than
falling through to a NaN from 0/0."""

import torch
from monai.metrics import compute_hausdorff_distance


def _binarize(x, threshold=0.5):
    return (x > threshold).float()


def dice_score(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    p_sum = pred.sum(dim=(1, 2, 3, 4))
    t_sum = target.sum(dim=(1, 2, 3, 4))

    both_empty = (p_sum == 0) & (t_sum == 0)
    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice  # (B,)


def iou_score(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) - inter

    both_empty = union == 0
    iou = (inter + eps) / (union + eps)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    return iou  # (B,)


def precision_recall(pred_prob, target, threshold=0.5, eps=1e-6):
    pred = _binarize(pred_prob, threshold)
    tp = (pred * target).sum(dim=(1, 2, 3, 4))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3, 4))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3, 4))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision, recall  # (B,), (B,)


def hd95_score(pred_prob, target, threshold=0.5, spacing_mm=1.0):
    """95th-percentile Hausdorff Distance, in mm, computed per case (batch
    size must be 1 when calling this -- MONAI's compute_hausdorff_distance
    is used under the hood).

    IMPORTANT: spacing_mm must match the isotropic spacing actually used
    during preprocessing (Spacingd), or this will silently report the
    right voxel-distance in the wrong physical units.

    Edge cases (union of pred/target can be degenerate, unlike Dice/IoU
    which use a simple ratio):
      - both empty  -> 0.0 (trivially perfect boundary agreement)
      - one empty, other not -> returns None (undefined distance; flag
        this case in the report rather than silently averaging in a
        fabricated number)
    """
    pred = _binarize(pred_prob, threshold)
    p_sum = pred.sum().item()
    t_sum = target.sum().item()

    if p_sum == 0 and t_sum == 0:
        return 0.0
    if p_sum == 0 or t_sum == 0:
        return None  # undefined -- caller must flag, not average in silently

    spacing = (spacing_mm, spacing_mm, spacing_mm)
    hd = compute_hausdorff_distance(
        pred, target, include_background=True, percentile=95, spacing=spacing
    )
    val = hd.flatten()[0].item()
    if val != val:  # NaN guard, in case MONAI still returns NaN in some edge case
        return None
    return val