import torch

from src.models.MAEST.model import MaskedAutoencoderViT3D


class Spatial2DMaskedAutoencoderViT3D(MaskedAutoencoderViT3D):
    def __init__(self, *args, in_chans: int = 6, **kwargs):
        self.in_chans = in_chans
        super().__init__(*args, in_chans=in_chans, **kwargs)

    def patchify(self, imgs):
        n, c, t_total, height, width = imgs.shape
        if c != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} Spatial2D channels, got {c}.")

        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert height == width and height % p == 0 and t_total % u == 0
        h = w = height // p
        t = t_total // u

        x = imgs.reshape(shape=(n, c, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(n, t * h * w, u * p**2 * c))
        self.patch_info = (n, c, t_total, height, width, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        n, c, t_total, height, width, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(n, t, h, w, u, p, p, c))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        return x.reshape(shape=(n, c, t_total, height, width))
