"""The slow Llama tokenizer of transformers 4.36.1 on the sentencepiece backend of transformers 5.

The release loads its tokenizer with ``use_fast=False``: sentencepiece with ``legacy=False``.
Transformers 5 ignores ``use_fast`` and returns its ``tokenizers`` conversion, which drops the
lone "▁" sentencepiece writes for the space after the image token, so the prompt would lose a
token. Transformers 5 keeps the sentencepiece backend; this restores what 4.36.1's Llama
tokenizer added to it: a "▁" before the text unless ``legacy``, and the BOS and EOS ids its
``add_bos_token`` and ``add_eos_token`` ask for.
"""

from transformers.tokenization_utils_sentencepiece import SPIECE_UNDERLINE, SentencePieceBackend

# transformers 5's special-token pattern of each (add_bos_token, add_eos_token) of 4.36.1.
_SPECIAL_TOKENS_PATTERNS = {
    (True, False): "bos",
    (False, True): "eos",
    (True, True): "bos_eos",
    (False, False): None,
}


class LlamaTokenizer(SentencePieceBackend):
    """transformers 4.36.1's slow ``LlamaTokenizer`` on transformers 5's sentencepiece backend."""

    def __init__(
        self, add_bos_token: bool = True, add_eos_token: bool = False, **kwargs: object
    ) -> None:
        """Load the sentencepiece model with 4.36.1's special tokens.

        Args:
            add_bos_token: Whether a BOS id starts every sequence.
            add_eos_token: Whether an EOS id ends every sequence.
            **kwargs: The backend's settings, those of ``tokenizer_config.json`` among them.
        """
        pattern = _SPECIAL_TOKENS_PATTERNS[(add_bos_token, add_eos_token)]
        super().__init__(special_tokens_pattern=pattern, **kwargs)

    def tokenize(
        self, text: str, pair: str | None = None, add_special_tokens: bool = False, **kwargs: object
    ) -> list[str]:
        """4.36.1's tokens: a "▁" before the text unless ``legacy``, dropped before a special token.

        Args:
            text: The text.
            pair: Passed to the backend, whose tokenization reads ``text`` alone.
            add_special_tokens: Passed to the backend, whose tokenization adds none.
            **kwargs: Passed to the backend's tokenization.
        """
        options = {"pair": pair, "add_special_tokens": add_special_tokens, **kwargs}
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **options)
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **options)
        if (
            len(tokens) > 1
            and tokens[0] == SPIECE_UNDERLINE
            and tokens[1] in self.all_special_tokens
        ):
            tokens = tokens[1:]
        return tokens
