import dataclasses
import os

import numpy as np
import torch

from libs.config import Config


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ModelSaver:
    def __init__(self):
        self.best_val_loss = np.Inf

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(model, path)

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)


class ModelEval:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train()

    def __call__(self, x, y=None):
        with torch.no_grad():
            return self.model(x, y)


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_acc(x, y):
    with torch.no_grad():
        N = y.shape[0]
        out = torch.argmax(x, dim=1)
        correct = torch.sum(out == y)
        return correct / N


def batch_visualize(batch):
    raise NotImplementedError


def save_checkpoint(batch):
    raise NotImplementedError
