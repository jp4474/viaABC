"""
Models module containing neural network implementations.
"""
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Third-party imports
from einops import rearrange
from timm.models.vision_transformer import Attention, Block
from timm.layers import Mlp, DropPath, LayerScale

# Local imports
from pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_ts
import video_vit

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len+1, d_model)  # +1 for [CLS] token

        # Generate positions (0 to max_len, where 0 is for [CLS])
        pos = torch.arange(0, max_len+1, dtype=torch.float).unsqueeze(1)

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

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

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
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), z_type = 'vanilla', dropout = 0.0, 
                 mask_ratio = 0.15, lambda_=0.00025, trainable_pos_emb = True, noise_factor = 0.5, tokenize = 'linear'):
        super().__init__()
        # --------------------------------------------------------------------------
        # Encoder specifics
        if tokenize == 'linear':
            self.embedder = nn.Linear(in_chans, embed_dim, bias=True)
        else:
            self.embedder = TiMAEEmbedding(in_chans, embed_dim)
        
        self.trunc_init = False
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.z_type = z_type
        self.lambda_ = lambda_
        self.decoder_embed_dim = decoder_embed_dim
        self.noise_factor = noise_factor
        
        # self.src_mask = None if not diagonal_attention else torch.ones(
        #     (int(seq_len*(1 - mask_ratio) + int(cls_embed)), 
        #      int(seq_len*(1-mask_ratio)+ int(cls_embed))), dtype=torch.bool).triu(diagonal=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim), requires_grad=trainable_pos_emb)  # fixed sin-cos embedding

        # self.pos_embed = PositionalEncoding(embed_dim, max_len=seq_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=0.0, proj_drop=0.0)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder specifics

        if z_type == 'vanilla':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        elif z_type == 'vae':
            self.decoder_embed = nn.Sequential(torch.nn.Linear(embed_dim, decoder_embed_dim),
                                               Lambda(decoder_embed_dim, decoder_embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
   
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, decoder_embed_dim), requires_grad=trainable_pos_emb)
        # self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=0.0, proj_drop=0.0)
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
        
        if isinstance(self.embedder, nn.Linear):
            w = self.embedder.weight.data
        else:
            w = self.embedder.conv.weight.data

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

        loss = (pred - x) ** 2
        loss = torch.mean(torch.sum(loss, dim=-1), dim = -1).mean(0) 
        return loss

    def forward(self, x, y = None, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        if self.training and self.noise_factor > 0:
            x_ = x + self.noise_factor * torch.randn_like(x, dtype=x.dtype)
        else:
            x_ = x
            
        latent, mask, ids_restore = self.forward_encoder(x_, mask_ratio)
        x_pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, x_pred, mask)

        if self.z_type == 'vae':
            space_loss = torch.mean(-0.5 * torch.sum(1 + self.decoder_embed[1].latent_logvar \
                                                     - self.decoder_embed[1].latent_mean ** 2 \
                                                        - self.decoder_embed[1].latent_logvar.exp(), dim = -1), dim = -1).mean(0)
        else:
            space_loss = 0

        kld_weight = 1

        return loss, self.lambda_ * kld_weight * space_loss, x_pred

    def get_latent(self, x, pooling_method = None):
        x, mask, ids_restore = self.forward_encoder(x, 0)
        x = self.decoder_embed(x)

        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = x[:, 1:, :]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        if pooling_method == "cls":
            return x[:, 0, :]
        elif pooling_method == "all":
            return x
        elif pooling_method == "mean":
            return torch.mean(x, dim=1)
        elif pooling_method == "no_cls":
            return x[:, 1:, :]
    
    def impute(self, x):
        # assert all(mask_index) < 8, "Mask index must be less than the total number of timestamps"
        # x[mask_index] = 0

        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        x_pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        return x_pred, mask, ids_restore

class MaskedAutoencoderViT3D(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        z_type='vanilla',
        lambda_=0.00025,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        self.lambda_ = lambda_
        self.z_type = z_type
        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if z_type == 'vanilla':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        elif z_type == 'vae':
            self.decoder_embed = nn.Sequential(torch.nn.Linear(embed_dim, decoder_embed_dim),
                                               Lambda(decoder_embed_dim, decoder_embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs
    
    def map2labels(self, imgs):
        """
        Convert binary masks into class labels.
        Assumes:
            - imgs is a torch.Tensor of shape B x C x W x H  
            - Each channel is a binary mask for class 0, 1, 2
        Returns:
            - label_map: torch.Tensor of shape (B x W x H), where each pixel is 0, 1, or 2
        """
        label_map = torch.argmax(imgs, dim=1)
        return label_map

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

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.0e-6) ** 0.5

        ################################

        pred = self.unpatchify(pred)
        target = self.map2labels(imgs)

        loss = nn.CrossEntropyLoss(reduction='mean')(pred, target)  # [N*L]

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if self.z_type == 'vae':
            space_loss = -0.5 * torch.mean(1 + self.decoder_embed[1].latent_logvar \
                                                     - self.decoder_embed[1].latent_mean ** 2 \
                                                        - self.decoder_embed[1].latent_logvar.exp())
        elif self.z_type == 'vq-vae':
            space_loss = self.decoder_embed[1].vq_loss
        elif self.z_type == 'vanilla':
            space_loss = 0

        return loss, self.lambda_ * space_loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, space_loss = self.forward_loss(imgs, pred, mask)
        return loss, space_loss, pred
    
    def get_latent(self, x, pooling_method = None):
        targets = self.patchify(x)
        x, mask, ids_restore = self.forward_encoder(x, 0)

        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if pooling_method == "cls":
            raise NotImplementedError("Pooling method 'cls' is not implemented for 3D ViT")
        elif pooling_method == "all" or pooling_method == "no_cls":
            return x
        elif pooling_method == "mean":
            return torch.mean(x, dim=1)
