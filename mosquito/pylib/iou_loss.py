import torch.nn as nn


class IoULoss(nn.Module):
    """Modified from:
    https://www.kaggle.com/code/bigironsphere/
    loss-function-library-keras-pytorch/notebook
    """

    def __init__(self, smooth=1) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou
