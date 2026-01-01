"""
Models module containing neural network implementations.
"""
from functools import partial
import math

import torch
import torch.nn as nn

# Third-party imports
from timm.models.vision_transformer import Block

from src.models.components import Lambda

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len+1, d_model)
        position = torch.arange(0, max_len+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len+1, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class TiMAEEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        emb_size: int
    ):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, emb_size, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): a tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: a tensor of shape (batch_size, seq_len, hidden_size)
        """        
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

class TSMVAE(nn.Module):
    def __init__(self, seq_len: int, in_chans: int, embed_dim: int, depth: int, num_heads: int, 
                 decoder_embed_dim: int, decoder_depth: int, decoder_num_heads: int, mlp_ratio: float = 4.0, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path = 0.0, 
                 mask_ratio = 0.15, kld_weight=1.0, trainable_pos_emb = False, noise_factor = 0.0):
        super().__init__()

        # ---------------- Encoder ----------------
        self.embedder = nn.Linear(in_chans, embed_dim, bias=True)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.z_type = 'vae' if kld_weight > 0 else 'vanilla'
        self.kld_weight = kld_weight
        self.noise_factor = noise_factor
        self.trunc_init = False
        self.decoder_embed_dim = decoder_embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding as parameter (learnable if trainable_pos_emb=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim), requires_grad=trainable_pos_emb)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path, proj_drop=0.0)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ---------------- Decoder ----------------
        if self.z_type == 'vanilla':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        else:
            self.decoder_embed = nn.Sequential(
                nn.Linear(embed_dim, decoder_embed_dim),
                Lambda(decoder_embed_dim, decoder_embed_dim)
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, decoder_embed_dim), requires_grad=trainable_pos_emb)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path, proj_drop=0.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        # ---------------- Positional embeddings ----------------
        pos_enc = PositionalEncoding(self.embed_dim, max_len=self.seq_len)
        with torch.no_grad():
            self.pos_embed.copy_(pos_enc.pe)

        dec_pos_enc = PositionalEncoding(self.decoder_embed_dim, max_len=self.seq_len)
        with torch.no_grad():
            self.decoder_pos_embed.copy_(dec_pos_enc.pe)

        # ---------------- Embedder ----------------
        if isinstance(self.embedder, nn.Linear):
            w = self.embedder.weight
        else:
            w = self.embedder.conv.weight
        if self.trunc_init:
            nn.init.trunc_normal_(w)
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.normal_(self.mask_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---------------- Other modules ----------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # embed into higher dimension
        x = self.embedder(x)

        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return x

    def forward_loss(self, x, pred, mask):
        """
        x: [B, T, D]
        pred: [B, T, D]
        mask: [B, T], 0 is keep, 1 is remove,
        """
        # \sum_{t=1}^{T} \sum_{d=1}^{D} (x_{t,d} - \hat{x}_{t,d})^2 / (T * D)

        # loss = (pred - x) ** 2
        # loss = torch.mean(torch.sum(loss, dim=-1), dim = -1).mean(0) 
        # return loss

        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum() 

        return loss

    def forward(self, x, y = None, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        if self.training and self.noise_factor > 0:
            x_ = x + self.noise_factor * torch.randn_like(x, dtype=x.dtype)
        else:
            x_ = x
            
        latent, mask, ids_restore = self.forward_encoder(x_, mask_ratio)
        x_pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, x_pred, mask)

        if self.z_type == 'vae':
            mean = self.decoder_embed[-1].latent_mean          # [B, T, D]
            logvar = self.decoder_embed[-1].latent_logvar      # log(std^2)
            var = torch.exp(logvar)

            kl_loss = 0.5 * (mean.pow(2) + var - 1.0 - logvar)
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))   # sum D, mean B,T

            space_loss = kl_loss
        else:
            space_loss = 0

        return loss, self.kld_weight * space_loss, x_pred
    
    def extract_features(self, x, pooling_method = None):
        x, _, ids_restore = self.forward_encoder(x, 0.0)
        x = self.decoder_embed(x)
        
        x_ = x[:, 1:, :]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        if pooling_method == "cls":
            return x[:, 0, :]
        elif pooling_method == "all":
            return x
        elif pooling_method == "mean":
            return torch.mean(x[:, 1:, :], dim=-1)
        elif pooling_method == "no_cls":
            return x[:, 1:, :]
    