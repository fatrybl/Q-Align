"""mPLUG-Owl2's modality-adaptive module: one copy of a module per modality, text (0) and image
(1), each applied to the tokens of its modality, as the release defines it."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

# The modalities, text and image.
_MODALITIES = 2


class MultiwayNetwork(nn.Module):
    """One module per modality, each applied to the tokens of its modality."""

    def __init__(self, module_provider: Callable[[], nn.Module]) -> None:
        """Build the module of every modality.

        Args:
            module_provider: Builds the module of one modality.
        """
        super().__init__()
        self.multiway = nn.ModuleList([module_provider() for _ in range(_MODALITIES)])

    def forward(self, hidden_states: Tensor, multiway_indices: Tensor) -> Tensor:
        """Each token through the module of its modality.

        Args:
            hidden_states: Tokens, ``[B, L, C]``.
            multiway_indices: Modality of each token, ``[B, L]``.
        """
        output_hidden_states = torch.empty_like(hidden_states)
        for idx, subway in enumerate(self.multiway):
            local_indices = multiway_indices.eq(idx).nonzero(as_tuple=True)
            hidden = hidden_states[local_indices].unsqueeze(1).contiguous()
            if hidden.numel():
                output_hidden_states[local_indices] = subway(hidden).squeeze(1)
        return output_hidden_states.contiguous()
