import torch
import numpy as np

class AbsoluteMeanScaler:
    def __init__(self, eps: float = 1e-8, keepdim=True):
        self.eps = eps
        self.keepdim = keepdim

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            
        scale = x.abs().mean(dim=0, keepdim=self.keepdim)
        scale = scale.clamp_min(self.eps)

        return x / scale

