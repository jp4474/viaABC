"""
Models module containing neural network implementations.
"""
# Standard library imports
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

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

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            differential_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = DifferentialAttention(
            dim, num_heads, layer_num = 4
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
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
        
        # Compute denominators using 10000^(2i/d_model)
        denominator = 10000 ** (2 * i / d_model)  # Shape (d_model,)
        
        # Compute angles for all positions and dimensions
        angles = pos / denominator  # Broadcasting to (max_len + 1, d_model)
        
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

# TODO:
# class VectorQuantizer(base.Module):
#   """Sonnet module representing the VQ-VAE layer.

#   Implements the algorithm presented in
#   'Neural Discrete Representation Learning' by van den Oord et al.
#   https://arxiv.org/abs/1711.00937

#   Input any tensor to be quantized. Last dimension will be used as space in
#   which to quantize. All other dimensions will be flattened and will be seen
#   as different examples to quantize.

#   The output tensor will have the same shape as the input.

#   For example a tensor with shape [16, 32, 32, 64] will be reshaped into
#   [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
#   independently.

#   Attributes:
#     embedding_dim: integer representing the dimensionality of the tensors in the
#       quantized space. Inputs to the modules must be in this format as well.
#     num_embeddings: integer, the number of vectors in the quantized space.
#     commitment_cost: scalar which controls the weighting of the loss terms (see
#       equation 4 in the paper - this variable is Beta).
#   """

#   def __init__(self,
#                embedding_dim: int,
#                num_embeddings: int,
#                commitment_cost: float,
#                dtype: torch.dtype = torch.float32,
#                name: str = 'vector_quantizer'):
#     """Initializes a VQ-VAE module.

#     Args:
#       embedding_dim: dimensionality of the tensors in the quantized space.
#         Inputs to the modules must be in this format as well.
#       num_embeddings: number of vectors in the quantized space.
#       commitment_cost: scalar which controls the weighting of the loss terms
#         (see equation 4 in the paper - this variable is Beta).
#       dtype: dtype for the embeddings variable, defaults to tf.float32.
#       name: name of the module.
#     """
#     super().__init__(name=name)
#     self.embedding_dim = embedding_dim
#     self.num_embeddings = num_embeddings
#     self.commitment_cost = commitment_cost

#     self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
#     nn.init.kaiming_uniform_(self.embeddings.weight)

#   def __call__(self, inputs, is_training):
#     """Connects the module to some inputs.

#     Args:
#       inputs: Tensor, final dimension must be equal to embedding_dim. All other
#         leading dimensions will be flattened and treated as a large batch.
#       is_training: boolean, whether this connection is to training data.

#     Returns:
#       dict containing the following keys and values:
#         quantize: Tensor containing the quantized version of the input.
#         loss: Tensor containing the loss to optimize.
#         perplexity: Tensor containing the perplexity of the encodings.
#         encodings: Tensor containing the discrete encodings, ie which element
#         of the quantized space each input element was mapped to.
#         encoding_indices: Tensor containing the discrete encoding indices, ie
#         which element of the quantized space each input element was mapped to.
#     """

#     flat_inputs = rearrange(inputs, 'b l d -> (b l) d')  # [B x L x D] -> [BL x D]

#     distances = ( 
#         torch.sum(flat_inputs ** 2, dim=1, keepdim=True) -
#         2 * torch.matmul(flat_inputs, self.embeddings.weight) +
#         torch.sum(self.embeddings.weight ** 2, dim=1, keepdim=True))
    
#     encoding_indices = torch.argmax(-distances, dim=1).unsqueeze(1)

#     encodings_one_hot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
#     encodings_one_hot.scatter_(1, encoding_indices, 1)  # [BL x K]

#     # NB: if your code crashes with a reshape error on the line below about a
#     # Tensor containing the wrong number of values, then the most likely cause
#     # is that the input passed in does not have a final dimension equal to
#     # self.embedding_dim. Ideally we would catch this with an Assert but that
#     # creates various other problems related to device placement / TPUs.
#     encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
#     quantized = self.quantize(encoding_indices)

#     e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
#     q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
#     loss = q_latent_loss + self.commitment_cost * e_latent_loss

#     # Straight Through Estimator
#     quantized = inputs + tf.stop_gradient(quantized - inputs)
#     avg_probs = tf.reduce_mean(encodings, 0)
#     perplexity = tf.exp(-tf.reduce_sum(avg_probs *
#                                        tf.math.log(avg_probs + 1e-10)))

#     return {
#         'quantize': quantized,
#         'loss': loss,
#         'perplexity': perplexity,
#         'encodings': encodings,
#         'encoding_indices': encoding_indices,
#         'distances': distances,
#     }

#   def quantize(self, encoding_indices):
#     """Returns embedding tensor for a batch of indices."""
#     w = tf.transpose(self.embeddings, [1, 0])
#     # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
#     # supported in V2. Are we missing anything here?
#     return tf.nn.embedding_lookup(w, encoding_indices)

# class VectorQuantizer(nn.Module):
#     def __init__(self, embedding_dim, num_embeddings, commitment_cost: float = 0.25):
#         """
#         Initializes a VQ-VAE module.
        
#         Args:
#             embedding_dim: Dimensionality of the tensors in the quantized space.
#             num_embeddings: Number of vectors in the quantized space.
#             commitment_cost: Scalar controlling the weighting of the loss terms.
#         """
#         super(VectorQuantizer, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost

#         # Embedding layer with uniform initialization
#         self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
#         self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

#     def forward(self, inputs):
#         """
#         Forward pass for vector quantization.
        
#         Args:
#             inputs: Tensor with shape [..., embedding_dim].
        
#         Returns:
#             A dictionary containing:
#                 'quantize': Quantized tensor.
#                 'loss': VQ loss.
#                 'perplexity': Perplexity of encodings.
#                 'encodings': One-hot encoded assignment matrix.
#                 'encoding_indices': Indices of closest embeddings.
#         """
#         # Flatten inputs
#         flat_inputs = inputs.view(-1, self.embedding_dim)
        
#         # Compute distances between input and embeddings
#         distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True) 
#                      - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()) 
#                      + torch.sum(self.embeddings.weight ** 2, dim=1))
        
#         # Get encoding indices and encodings
#         encoding_indices = torch.argmin(distances, dim=1)
#         encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
#         # Quantize
#         quantized = self.embeddings(encoding_indices).view(inputs.shape)
        
#         # Compute loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
#         # Straight-Through Estimator
#         quantized = inputs + (quantized - inputs).detach()
        
#         # Compute perplexity
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
#         return {
#             'quantize': quantized,
#             'loss': loss,
#             'perplexity': perplexity,
#             'encodings': encodings,
#             'encoding_indices': encoding_indices,
#             'distances': distances
#         }

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
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, z_type = 'vanila',  cls_embed = True, dropout = 0.1, 
                 mask_ratio = 0.15, diagonal_attention=False, lambda_=0.00025, scale_mode = "adaptive_scale", bag_size = 1024,
                 differential_attention = False):
        super().__init__()
        # TODO: figure out the dimensions
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.conv_emb = torch.nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.embedder = nn.Linear(in_chans, embed_dim, bias=True)
        self.cls_embed = cls_embed
        self.trunc_init = False
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.z_type = z_type
        self.lambda_ = lambda_
        self.decoder_embed_dim = decoder_embed_dim
        
        self.scaler_layer = DAIN_Layer(scale_mode, input_dim=in_chans)

        # self.src_mask = None if not diagonal_attention else torch.ones(
        #     (int(seq_len*(1 - mask_ratio) + int(cls_embed)), 
        #      int(seq_len*(1-mask_ratio)+ int(cls_embed))), dtype=torch.bool).triu(diagonal=1)

        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.pos_embed = PositionalEncoding(embed_dim, max_len=seq_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, differential_attention=differential_attention)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        if z_type == 'vanilla':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        elif z_type == 'vae':
            self.decoder_embed = nn.Sequential(torch.nn.Linear(embed_dim,decoder_embed_dim),
                                               Lambda(decoder_embed_dim, decoder_embed_dim))
        elif z_type == 'vq-vae':
            self.decoder_embed = nn.Sequential(torch.nn.Linear(embed_dim, decoder_embed_dim),
                                               VectorQuantizer(bag_size, decoder_embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, max_len=seq_len)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, differential_attention=differential_attention)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True) # decoder to patch

        self.initialize_weights()

    # def initialize_weights(self):
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #     decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
    #     self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.patch_embed.proj.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     if self.cls_embed
    #         torch.nn.init.normal_(self.cls_token, std=.02)
    #     torch.nn.init.normal_(self.mask_token, std=.02)

    #     # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)
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

        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x, mask_ratio):
        # embed into higher dimension
        x = self.embedder(x)

        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

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
        # remove cls token
        x = x[:, 1:, :]

        return x
        
    # def masking(self, x, mask_ids):
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = L

    #     ids_keep = 
    #     ids_restore = 
        
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore, ids_keep

    def forward_loss(self, x, pred, mask, latent):
        """
        x: [N, W, L]
        pred: [N, L, W]
        mask: [N, W], 0 is keep, 1 is remove,
        """
        
        # print(x.shape, pred.shape)
        if self.training:
            loss = torch.abs(pred - x)
            loss = torch.nan_to_num(loss, nan=10, posinf=10, neginf=10)
            loss = torch.clamp(loss, max=10)
            loss = loss.mean(dim=-1)  # [N, L], mean loss per timestamp

            inv_mask = (mask -1 ) ** 2
            loss_removed = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            loss_seen = (loss * inv_mask).sum() / inv_mask.sum()  # mean loss on seen patches

        else:
            loss = torch.abs(pred - x)
            loss = torch.nan_to_num(loss, nan=10, posinf=10, neginf=10)
            loss = torch.clamp(loss, max=10)
            loss = loss.mean(dim=-1)

            loss_removed = 0
            loss_seen = loss.mean()
        
        return loss_removed , loss_seen #, forecast_loss, backcast_loss


    def forward(self, x, mask_ratio = 0.75):

        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x, pred, mask, latent)

        if self.z_type == 'vae':
            space_loss = torch.mean(-0.5 * torch.sum(1 + self.decoder_embed[1].latent_logvar \
                                                     - self.decoder_embed[1].latent_mean ** 2 \
                                                        - self.decoder_embed[1].latent_logvar.exp(), dim = -1), dim = -1).mean(0)
        elif self.z_type == 'vq-vae':
            space_loss = self.decoder_embed[1].vq_loss
        else:
            space_loss = 0
        
        total_loss = loss[0] + loss[1] + self.lambda_ * space_loss
        return total_loss, pred