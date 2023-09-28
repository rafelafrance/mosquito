import torch.nn as nn


class BinaryJaccardLoss(nn.Module):
    def __init__(self, eps=1e-7, threshold=0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = (y_pred + y_true).sum() - intersection
        iou = intersection / (union + self.eps)

        return 1.0 - iou

    def zeros_and_ones(self, x):
        """Convert results to a hard 0 or 1."""
        return (x > self.threshold).float()
