# Adapted from https://github.com/karpathy/nanoGPT
from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Tuple
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based models.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
    """
    vocab_size: int = 2048
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    context_length: int = 128
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 4

@dataclass
class VQConfig:
    """Configuration for Vector Quantization.
    
    Attributes:
        latent_action_dim: Dimension of the latent action space.
        num_latents: Number of discrete latent codes in the codebook.
        dropout: Dropout probability.
        entropy_weight: Weight for entropy regularization.
        vq_beta: Beta parameter for vector quantization commitment loss.
    """
    latent_action_dim: int = 64
    num_latents: int = 512
    dropout: float = 0.0
    entropy_weight: float = 0.01
    vq_beta: float = 0.25

@dataclass
class LatentActionModelConfig:
    """Combined configuration for Latent Action Models.
    
    Combines transformer and vector quantization configurations for models
    that learn discrete latent action representations.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
        context_length: Maximum sequence length.
        lm_head_out_size: Output size of the language model head.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        latent_action_dim: Dimension of the latent action space.
        num_latents: Number of discrete latent codes in the codebook.
        entropy_weight: Weight for entropy regularization.
        vq_beta: Beta parameter for vector quantization commitment loss.
    """
    vocab_size: int = 2048
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    context_length: int = 128
    lm_head_out_size: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 4
    latent_action_dim: int = 64
    num_latents: int = 512
    entropy_weight: float = 0.01
    vq_beta: float = 0.25   

class SerialisableModule(nn.Module):
    """Base class for serializable PyTorch modules.
    
    Provides functionality to save and load models with their configurations.
    """

    def save(self, path: str):
        """Save the model configuration and parameters to a file.
        
        Args:
            path: Path to save the model checkpoint.
        """
        config = vars(self)
        params = self.state_dict()
        dic = {
            "config": config,
            "params": params,
            "class": self.__class__.__name__,
        }
        torch.save(dic, path)
        logging.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> SerialisableModule:
        """Load a model from a saved checkpoint.
        
        Args:
            path: Path to the saved model checkpoint.
            device: Device to load the model on.
            
        Returns:
            Loaded model instance.
            
        Raises:
            ValueError: If the checkpoint class doesn't match the loading class.
        """
        dic = torch.load(path, map_location=device, weights_only= False)
        if dic["class"] != cls.__name__:
            raise ValueError(
                f"Trying to load a {dic['class']} model with a {cls.__name__} class"
            )
        # get the constructor parameters of the class
        constructor_params = inspect.signature(cls.__init__).parameters
        # filter the config to only include parameters that are in the constructor
        filtered_config = {
            k: v for k, v in dic["config"].items() if k in constructor_params
        }
        # create a new instance of the class with the filtered config
        model = cls(**filtered_config)
        model.load_state_dict(dic["params"])
        logging.info(f"Model {cls.__name__} successfully loaded from {path}")
        return model


class LayerNorm(SerialisableModule):
    """LayerNorm with optional bias parameter.
    
    PyTorch's native LayerNorm doesn't support simply bias=False.
    
    Args:
        ndim: Dimensionality of the input features.
        bias: Whether to use a learnable bias term.
    """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.
        
        Args:
            input: Input tensor of shape (..., ndim).
            
        Returns:
            Normalized tensor of the same shape.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(SerialisableModule):
    """Multi-head causal self-attention layer.
    
    Implements efficient causal self-attention with optional Flash Attention support.
    
    Args:
        config: Transformer configuration containing model dimensions and settings.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.context_length, config.context_length)).view(
                    1, 1, config.context_length, config.context_length
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(SerialisableModule):
    """Multi-layer perceptron with GELU activation.
    
    Implements the feedforward network used in transformer blocks.
    
    Args:
        config: Transformer configuration containing model dimensions and settings.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(SerialisableModule):
    """Single transformer block with attention and feedforward layers.
    
    Uses pre-normalization architecture (LayerNorm before attention/MLP).
    
    Args:
        config: Transformer configuration containing model dimensions and settings.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerEncoder(SerialisableModule):
    """Transformer encoder with token and position embeddings.
    
    Args:
        config: Transformer configuration containing model dimensions and settings.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode token sequences.
        
        Args:
            idx: Token indices of shape (batch_size, seq_len).
            
        Returns:
            Encoded representations of shape (batch_size, seq_len, n_embd).
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.context_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)
        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class VectorQuantizer(SerialisableModule):
    """Vector Quantization module for discrete latent representations.
    
    Implements vector quantization with straight-through estimator,
    commitment loss, and entropy regularization.
    
    Args:
        latent_action_dim: Dimension of the latent action vectors.
        num_latents: Number of discrete codes in the codebook.
        dropout: Dropout probability applied to distances.
        entropy_weight: Weight for entropy regularization loss.
        vq_beta: Weight for commitment loss.
    """
    def __init__(self, 
                 latent_action_dim: int,
                 num_latents: int,
                 dropout: float,
                 entropy_weight: float,
                 vq_beta: float):
        super().__init__()
        self.latent_action_dim = latent_action_dim
        self.num_latents = num_latents
        self.dropout = nn.Dropout(dropout)
        self.entropy_weight = entropy_weight
        self.vq_beta = vq_beta

        # Lecun's initialization for codebook
        bound = (3 / self.latent_action_dim) ** 0.5
        self.codebook = nn.Parameter(
            torch.empty(self.num_latents, self.latent_action_dim).uniform_(-bound, bound)
        )
        self._normalize_codebook()

    def _normalize_codebook(self):
        """
        Normalize the codebook embeddings to have unit norm.
        """
        with torch.no_grad():
            self.codebook.data = F.normalize(self.codebook.data, dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Quantize continuous vectors to discrete codes.
        
        Args:
            x: Input tensor of shape (batch_size, latent_action_dim).
            
        Returns:
            Tuple containing:
                - z_q: Quantized tensor with straight-through gradients.
                - vq_loss_dict: Dictionary containing individual loss components:
                    - 'vq_loss': Combined VQ loss (reconstruction + commitment + entropy).
                    - 'q_loss': Reconstruction loss.
                    - 'commit_loss': Commitment loss.
                    - 'entropy_loss': Entropy regularization loss.
                    - 'entropy': Raw entropy value.
                - indices: Action indexes in the codebook.
        """
        # --- Normalize input and codebook ---
        x_norm = F.normalize(x, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)

        # --- Compute distances ---
        distance = -torch.matmul(x_norm, codebook_norm.T)
        distance = self.dropout(distance)

        # --- Find closest codebook entries ---
        indices = torch.argmin(distance, dim=-1)
        z = self.codebook[indices]  # a in excalidraw

        # --- STE ---
        z_q = x + (z - x).detach()

        commit_loss = F.mse_loss(x, z.detach(), reduction="mean")
        q_loss = F.mse_loss(z, x.detach(), reduction="mean")

        entropy = self.entropy_from_indices(indices)
        entropy_loss = -entropy  # Negative entropy to encourage uniform usage

        loss = q_loss + self.vq_beta * commit_loss + self.entropy_weight * entropy_loss
        
        vq_loss_dict = {
            'vq_loss': loss,
            'q_loss': q_loss,
            'commit_loss': commit_loss,
            'entropy_loss': entropy_loss,
            'entropy': entropy,
        }
        return z_q, vq_loss_dict, indices

    def get_codes(self, indices: torch.Tensor):
        """Retrieve codebook vectors for given indices.
        
        Args:
            indices: Tensor of action indexes.
            
        Returns:
            Codebook vectors corresponding to the actions.
        """
        return self.codebook[indices]

    def entropy_from_indices(
        self, indices: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        """Compute entropy of the codebook usage distribution.
        
        Higher entropy indicates more uniform codebook usage, which is desirable
        to prevent codebook collapse.
        
        Args:
            indices: Tensor of action indexes.
            eps: Small constant for numerical stability.
            
        Returns:
            Entropy value as a scalar tensor (higher is better).
        """
        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_latents).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs.clamp(min=eps)).log()).sum()

        return entropy


# class PairwiseHead(SerialisableModule):
#     """
#     Input : h   decoder hidden states shape (B, T, H)
#     Output: logits  per-pair token scores  (B, T_src, T_tgt, V)
#             (T_src = T_tgt = sequence length passed in)
#     """

#     def __init__(
#         self,
#         hidden_dim: int,  # H
#         vocab_size: int,  # V
#         max_len: int = 128,
#         rel_pos_dim: int = 32,  # R
#     ):
#         super().__init__()
#         # ( −(max_len−1)  +(max_len−1) )  ->  R‑dim vector
#         self.rel_pos_emb = nn.Embedding(2 * max_len - 1, rel_pos_dim)
#         # final classifier (H+R) -> V   (shared by every (i,j) pair)
#         self.proj = nn.Linear(hidden_dim + rel_pos_dim, vocab_size, bias=False)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         """
#         h : (B, T, H)
#         returns logits : (B, T, T, V)
#         """
#         B, T, H = h.shape

#         # broadcast source vector along target axis
#         src = h.unsqueeze(2).expand(B, T, T, H)  # (B, T_src, T_tgt, H)

#         # relative‑position embedding  delta = j − i
#         # idx: [0, 1, ..., T‑1]
#         idx = torch.arange(T, device=h.device)
#         rel = idx[None, None, :] - idx[None, :, None]  # (1, T, T)   delta = j-i
#         rel += T - 1  # shift to 0, ..., 2T‑2
#         rel = self.rel_pos_emb(rel)  # (1, T, T, R)
#         rel = rel.expand(B, -1, -1, -1)  # (B, T, T, R)

#         # concat and project
#         pair = torch.cat([src, rel], dim=-1)  # (B, T, T, H+R)
#         logits = self.proj(pair)  # (B, T, T, V)
#         return logits
