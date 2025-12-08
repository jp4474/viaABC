import torch
import numpy as np

class AbsoluteMeanScaler:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.abs()
        mean = x.mean()
        abs_mean = mean + self.eps
        return x / abs_mean

