import torch


def one_hot(y):
    pred = torch.argmax(y, dim=1)
    results = torch.zeros_like(y).scatter_(1, pred.unsqueeze(1), 1.0)
    return results
