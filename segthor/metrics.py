from torch import Tensor

def iou_coef(pred: Tensor, gt: Tensor) -> Tensor:
    intersection = (pred * gt).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - intersection
    return (intersection + 1e-8) / (union + 1e-8)

def precision_coef(pred: Tensor, gt: Tensor) -> Tensor:
    intersection = (pred * gt).sum(dim=(2, 3))
    return (intersection + 1e-8) / (pred.sum(dim=(2, 3)) + 1e-8)

def recall_coef(pred: Tensor, gt: Tensor) -> Tensor:
    intersection = (pred * gt).sum(dim=(2, 3))
    return (intersection + 1e-8) / (gt.sum(dim=(2, 3)) + 1e-8)