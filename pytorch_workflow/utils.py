import torch
import numpy as np


def accuracy(y_pred_logits, y_true):
    y_pred = torch.softmax(y_pred_logits, dim = 1)
    y_pred = torch.argmax(y_pred, dim = 1)
    acc = torch.sum(y_pred == y_true).item() / len(y_true)
    return acc