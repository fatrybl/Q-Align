"""mPLUG-Owl2's modality-adaptive LLaMA-2, the language model of Q-Align, on transformers 5.

The released module patched the Llama classes of transformers 4.36 at import; this one defines
the same network (:mod:`.decoder_layer`), with the same parameter names, so the released
checkpoints load unchanged. The rotary angles are those of 4.36's table
(:func:`.rotary.rotary_table`), their cosines and sines rounded once to the activations' dtype.
"""

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaRMSNorm

from .configuration_mplug_owl2 import LlamaConfig
from .decoder_layer import LlamaDecoderLayer
from .modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .rotary import rotary_table

# The rotary type of an unscaled embedding in transformers 5's standardised rope parameters.
_UNSCALED_ROPE = "default"


class ModalityAdaptiveLlamaModel(LlamaPreTrainedModel):
    """The decoder stack: token embeddings, modality-adaptive layers and the final norm."""

    def __init__(self, config: LlamaConfig) -> None:
        """Build the stack and 4.36's rotary table.

        Args:
            config: The language model's settings.

        Raises:
            ValueError: If the settings scale the rotary embedding or split the MLP into
                tensor-parallel slices, which the port omits.
        """
        super().__init__(config)
        rope = config.rope_parameters
        if rope["rope_type"] != _UNSCALED_ROPE or config.pretraining_tp != 1:
            raise ValueError(
                f"rope {rope} and pretraining_tp {config.pretraining_tp} are not ported"
            )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        # Plain tensors on the CPU, not buffers: transformers 5 builds a model on the meta device
        # and leaves its non-persistent buffers uninitialised after loading.
        self._rotary_cos, self._rotary_sin = rotary_table(
            head_dim, config.max_position_embeddings, rope["rope_theta"]
        )
        self.post_init()

    def forward(
        self, input_ids: Tensor | None, modality: Tensor, inputs_embeds: Tensor | None
    ) -> BaseModelOutputWithPast:
        """The last hidden states of one sequence per row.

        Args:
            input_ids: Token ids, read when ``inputs_embeds`` is ``None``.
            modality: Modality of each token, ``[B, L]``.
            inputs_embeds: Token embeddings, image features in place, ``[B, L, C]``.

        Raises:
            ValueError: If the sequence is longer than the rotary table.
        """
        embeds = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch_size, seq_length, _ = embeds.shape
        if seq_length > len(self._rotary_cos):
            raise ValueError(f"{seq_length} tokens exceed the {len(self._rotary_cos)} positions")
        positions = torch.arange(seq_length, dtype=torch.long, device=embeds.device).unsqueeze(0)
        ones = torch.ones((batch_size, seq_length), dtype=torch.bool, device=embeds.device)
        mask = _prepare_4d_causal_attention_mask(
            ones, (batch_size, seq_length), embeds, past_key_values_length=0
        )
        tables = (self._rotary_cos, self._rotary_sin)
        cos, sin = (table[:seq_length].to(embeds) for table in tables)
        rotary = (cos[positions], sin[positions])
        states = embeds
        for decoder_layer in self.layers:
            states = decoder_layer(states, modality, mask, rotary)
        output = BaseModelOutputWithPast(last_hidden_state=self.norm(states))
        return output
