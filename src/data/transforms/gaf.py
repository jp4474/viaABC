import numpy as np
import torch

class GramianAngularField:
    """
    Applies Gramian Angular Field (GAF) independently to each variable.
    Input: (T, D)
    Output: (D, T, T)
    """

    def __init__(self):
        pass

    def __call__(self, x) -> torch.Tensor:
        # x: (T, D)
        gaf_list = []
        for d in range(x.shape[1]):
            gaf_list.append(self._gaf_single_dim(x[:, d], method="summation"))

        for d in range(x.shape[1]):
            gaf_list.append(self._gaf_single_dim(x[:, d], method="difference"))

        gaf_np = np.stack(gaf_list)  # (2*D, T, T)
        return torch.from_numpy(gaf_np).float()

    def _minmax_scale(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def _gaf_single_dim(self, x, method="summation"):
        # Min-max normalize to [-1, 1]
        x_scaled = self._minmax_scale(x)
        x_scaled = x_scaled * 2 - 1

        # polar encoding
        phi = np.arccos(x_scaled)

        # stack summation GAF and difference GAF

        if method == "summation":
            return np.cos(phi[:, None] + phi[None, :])
        else:
            return np.sin(phi[:, None] - phi[None, :])
