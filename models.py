"""
Models module containing neural network implementations.
"""
# Standard library imports
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from functools import partial

# Scientific computing
import numpy as np
import math

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Third-party imports
from einops import rearrange
from timm.models.vision_transformer import Block, PatchEmbed, Attention
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, SwiGLU

# Local imports
from pos_embed import get_2d_sincos_pos_embed
from scaler import DAIN_Layer

#TODO: This is doing Masked 

class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_num):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale_value = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_num)
        
        self.norm = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x):
        queries = rearrange(self.q(x), "b n (h d q) -> b n (q h) d", h=self.num_heads, q=2, d=self.head_dim)
        queries = queries * self.scale_value

        keys = rearrange(self.k(x), "b n (h d k) -> b n (k h) d", h=self.num_heads, k=2, d=self.head_dim)
        v = rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads, d=2*self.head_dim)

        attention = torch.einsum("bnqd,bnkd->bnqk", queries, keys)
        attention = torch.nan_to_num(attention)
        attention = F.softmax(attention, dim=-1, dtype=torch.float32)

        lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_value = torch.exp(lambda_1) - torch.exp(lambda_2) + self.lambda_init

        attention = rearrange(attention, "b n (q h) (k a) -> q k b n h a", q=2, k=2, h=self.num_heads, a=self.num_heads)
        attention = attention[0, 0, ...] - lambda_value * attention[1, 1, ...]

        out = torch.einsum("bnah,bhnd->bnad", attention, v)
        out = self.norm(out)
        out = out * (1 - self.lambda_init)
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.output_projection(out)

        return out
     
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len + 1, d_model)  # +1 for [CLS] token
        
        # Generate positions (0 to max_len, where 0 is for [CLS])
        pos = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        
        # Compute i values for each dimension
        i = torch.arange(0, d_model, dtype=torch.float) // 2  # Shape (d_model,)
        
        angles = torch.exp(torch.log(pos) - 2 * torch.log(torch.tensor(10000)) * i / d_model)
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        
        # Register as buffer to avoid model training
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape (1, max_len + 1, d_model)
    
class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    reference: https://github.com/tejaslodaya/timeseries-clustering-vae/blob/master/vrae/vrae.py

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length, bias=False)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length, bias=False)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training: #and not hasattr(self, 'finetuning'):
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [B x L x D] - >[BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        self.vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents  # [B x L x D

class TiMAE(nn.Module):
    def __init__(self, seq_len: int, in_chans: int, embed_dim: int, depth: int, num_heads: int, 
                 decoder_embed_dim: int, decoder_depth: int, decoder_num_heads: int, mlp_ratio: float = 4.0, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, z_type = 'vanilla',  cls_embed = True, dropout = 0.1, 
                 mask_ratio = 0.15, diagonal_attention=False, lambda_=0.00025, scale_mode = "adaptive_scale", bag_size = 1024,
                 differential_attention = False):
        super().__init__()
        # --------------------------------------------------------------------------
        # Encoder specifics
        self.embedder = nn.Linear(in_chans, embed_dim, bias=True)
        self.cls_embed = cls_embed
        self.trunc_init = False
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.z_type = z_type
        self.lambda_ = lambda_
        self.decoder_embed_dim = decoder_embed_dim
        
        # self.scaler_layer = DAIN_Layer(scale_mode, input_dim=in_chans)

        # self.src_mask = None if not diagonal_attention else torch.ones(
        #     (int(seq_len*(1 - mask_ratio) + int(cls_embed)), 
        #      int(seq_len*(1-mask_ratio)+ int(cls_embed))), dtype=torch.bool).triu(diagonal=1)

        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.pos_embed = PositionalEncoding(embed_dim, max_len=seq_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dropout)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder specifics
        if z_type == 'vanilla':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=False)
        elif z_type == 'vae':
            self.decoder_embed = Lambda(embed_dim, decoder_embed_dim)
        elif z_type == 'vq-vae':
            self.decoder_embed = nn.Sequential(torch.nn.Linear(embed_dim, decoder_embed_dim),
                                               VectorQuantizer(bag_size, decoder_embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dropout)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True) # decoder to patch

        self.linear = nn.Linear(decoder_embed_dim, 2)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = PositionalEncoding(self.embed_dim, max_len=self.seq_len)
        self.pos_embed.data.copy_(pos_embed.pe.float())

        decoder_pos_embed = PositionalEncoding(self.decoder_embed_dim, max_len=self.seq_len)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.pe.float())
        
        w = self.embedder.weight.data
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
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

        regression_task = self.linear(x[:, :1, :].squeeze(1)) # [N, 6]
        imputation_task = self.decoder_pred(x[:, 1:, :]) 

        return regression_task, imputation_task

    def forward_loss(self, x, pred, mask):
        """
        x: [N, W, L]
        pred: [N, L, W]
        mask: [N, W], 0 is keep, 1 is remove,
        """
        # # # print(x.shape, pred.shape)

        #loss = (pred - x) ** 2
        loss = torch.abs(pred - x)
        loss = torch.nan_to_num(loss,nan=10,posinf=10,neginf=10)
        loss = torch.clamp(loss,max=10)
    
        loss = loss.mean(dim=-1)  # [N, L], mean loss per timestamp
        inv_mask = (mask -1) ** 2

        if self.training:
            loss_removed = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss_removed = 0

        loss_seen = (loss * inv_mask).sum() / inv_mask.sum()
        loss = loss_removed + loss_seen
        return loss

    def forward(self, x, y = None, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        regression_task, imputation_task = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, imputation_task, mask)

        if y is not None:
            reg_loss = F.l1_loss(regression_task, y, reduction='mean')
        else:
            reg_loss = 0

        if self.z_type == 'vae':
            space_loss = torch.mean(-0.5 * torch.sum(1 + self.decoder_embed.latent_logvar \
                                                     - self.decoder_embed.latent_mean ** 2 \
                                                        - self.decoder_embed.latent_logvar.exp(), dim = -1), dim = -1).mean(0)
        elif self.z_type == 'vq-vae':
            space_loss = self.decoder_embed[1].vq_loss
        else:
            space_loss = 0

        # kld_weight = x.shape[0]/50000
        kld_weight = 1
        
        return loss, reg_loss, self.lambda_ * kld_weight * space_loss, regression_task, imputation_task
    
    def get_latent(self, x, pooling_method = None):
        x, mask, ids_restore = self.forward_encoder(x, 0)
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        #return x[:, 0, :]

        if pooling_method == "cls":
            return x[:, 0, :]
        elif pooling_method == "all":
            return x
        else:
            return x[:, 1:, :]
    
    # def mask_using_id(self, x, mask_id):
    #     """
    #     Perform masking based on the provided boolean mask_id.
    #     x: [N, L, D], sequence
    #     mask_id: [N, L], boolean array where True indicates the element should be masked.
    #     """
    #     N, L, D = x.shape
    #     assert mask_id.shape == (N, L), "mask_id must have shape (N, L)"
    #     assert mask_id.dtype == torch.bool, "mask_id must be a boolean tensor"

    #     # Calculate the number of elements to keep (assumes same for all samples in the batch)
    #     len_keep = (~mask_id).sum(dim=1)
    #     assert torch.all(len_keep == len_keep[0]), "All samples must have the same number of kept tokens"
    #     len_keep = len_keep[0].item()

    #     # Sort the mask_id to get indices where False (keep) come first, then True (mask)
    #     # Convert to int for argsort as PyTorch's argsort doesn't support boolean tensors
    #     ids_shuffle = torch.argsort(mask_id.int(), dim=1)
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # Keep the first len_keep indices
    #     ids_keep = ids_shuffle[:, :len_keep]

    #     # Gather the kept elements
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    #     # Generate the binary mask: 0 is keep, 1 is remove
    #     # Start with all 1s, set the first len_keep to 0, then unshuffle with ids_restore
    #     mask = torch.ones((N, L), device=x.device)
    #     mask[:, :len_keep] = 0
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore