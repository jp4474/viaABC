import torch
import torch.nn as nn

class Lambda(nn.Module):
    """
    Lambda module converts encoder outputs to latent vectors.

    If cls_token=True:
      - Assumes input shape (B, T, D)
      - Token at index 0 is CLS
      - CLS token bypasses VAE sampling
      - CLS is projected to latent_length
      - Non-CLS tokens go through VAE

    Parameters
    ----------
    hidden_size : int
        Encoder hidden dimension (D)
    latent_length : int
        Latent dimension (Z)
    cls_token : bool, default=True
        Whether the first token is a CLS token
    """

    def __init__(
        self,
        hidden_size: int,
        latent_length: int
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        # VAE projections (non-CLS tokens)
        self.hidden_to_mean = nn.Linear(hidden_size, latent_length)
        self.hidden_to_logvar = nn.Linear(hidden_size, latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

        self.latent_mean = None
        self.latent_logvar = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.latent_mean = self.hidden_to_mean(x)
        self.latent_logvar = self.hidden_to_logvar(x)

        if self.training:
            std = torch.exp(self.latent_logvar)
            eps = torch.randn_like(std)
            return eps * std + self.latent_mean
        else:
            return self.latent_mean