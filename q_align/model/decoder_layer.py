"""A decoder layer of mPLUG-Owl2's modality-adaptive LLaMA-2, the release's computation on
transformers 5. Text and image tokens share the queries, the output projection and the MLP, and
have keys, values and norms of their own. Attention is the eager computation the release runs
under its torch 2.0.1, where transformers 4.36 never enables SDPA::

    A = softmax_fp32(Q Kᵀ / √d + M),   O = A V,

with ``M`` the additive causal mask and ``Q``, ``K`` rotated by the angles of each position.
"""

import math
from functools import partial

import torch
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .configuration_mplug_owl2 import LlamaConfig
from .multiway import MultiwayNetwork

type Rotary = tuple[Tensor, Tensor]


class LlamaAttention(nn.Module):
    """Multi-headed attention with keys and values of each modality's own."""

    def __init__(self, config: LlamaConfig) -> None:
        """Build the projections.

        Args:
            config: The language model's settings.
        """
        super().__init__()
        self._hidden_size = config.hidden_size
        self._num_heads = config.num_attention_heads
        self._head_dim = self._hidden_size // self._num_heads
        self._num_key_value_heads = config.num_key_value_heads
        self._num_key_value_groups = self._num_heads // self._num_key_value_heads
        kv_size = self._num_key_value_heads * self._head_dim
        kv_linear = partial(nn.Linear, self._hidden_size, kv_size, bias=config.attention_bias)
        self.q_proj = nn.Linear(self._hidden_size, self._hidden_size, bias=config.attention_bias)
        self.k_proj = MultiwayNetwork(module_provider=kv_linear)
        self.v_proj = MultiwayNetwork(module_provider=kv_linear)
        self.o_proj = nn.Linear(self._hidden_size, self._hidden_size, bias=config.attention_bias)

    def forward(self, states: Tensor, modality: Tensor, mask: Tensor, rotary: Rotary) -> Tensor:
        """The attention output of every token.

        Args:
            states: Normalised tokens, ``[B, L, C]``.
            modality: Modality of each token, ``[B, L]``.
            mask: Additive causal mask, ``[B, 1, L, L]``.
            rotary: Cosines and sines of each token's position, ``[1, L, head_dim]`` each.
        """
        bsz, q_len, _ = states.size()
        query = self.q_proj(states).view(bsz, q_len, self._num_heads, self._head_dim)
        key = self.k_proj(states, modality).view(
            bsz, q_len, self._num_key_value_heads, self._head_dim
        )
        value = self.v_proj(states, modality).view(
            bsz, q_len, self._num_key_value_heads, self._head_dim
        )
        query, key = apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), *rotary)
        key = repeat_kv(key, self._num_key_value_groups)
        value = repeat_kv(value.transpose(1, 2), self._num_key_value_groups)
        weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self._head_dim) + mask
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        attended = self.o_proj(output.reshape(bsz, q_len, self._hidden_size))
        return attended


class LlamaDecoderLayer(nn.Module):
    """A pre-norm decoder layer whose norms are each modality's own."""

    def __init__(self, config: LlamaConfig) -> None:
        """Build the attention, the MLP and the norms.

        Args:
            config: The language model's settings.
        """
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        norm = partial(LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = MultiwayNetwork(module_provider=norm)
        self.post_attention_layernorm = MultiwayNetwork(module_provider=norm)

    def forward(self, states: Tensor, modality: Tensor, mask: Tensor, rotary: Rotary) -> Tensor:
        """The layer's output for every token.

        Args:
            states: Tokens, ``[B, L, C]``.
            modality: Modality of each token, ``[B, L]``.
            mask: Additive causal mask, ``[B, 1, L, L]``.
            rotary: Cosines and sines of each token's position.
        """
        attended = states + self.self_attn(
            self.input_layernorm(states, modality), modality, mask, rotary
        )
        output = attended + self.mlp(self.post_attention_layernorm(attended, modality))
        return output
