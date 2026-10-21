"""The port to transformers 5 against the released Q-Align on transformers 4.36.1: the slow Llama
tokenizer's ids, and the logits of LMM-PCQA's prompt for fixed views, as the released code
(upstream fd0c3de, eager attention as under its torch 2.0.1) computed them."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from q_align.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from q_align.conversation import conv_templates
from q_align.mm_utils import expand2square, tokenizer_image_token
from q_align.model.builder import load_pretrained_model
from q_align.model.tokenization_llama import LlamaTokenizer

_WEIGHTS = Path(
    os.environ.get(
        "QALIGN_PCQA_WEIGHTS",
        Path(__file__).resolve().parent.parent / "LMM-PCQA" / "weights" / "q-align-pcqa-wpc",
    )
)
_DEVICE = "cuda:0"
_TOKENS = {
    " ASSISTANT: The quality of the image is": [
        1,
        29871,
        319,
        1799,
        9047,
        13566,
        29901,
        450,
        11029,
        310,
        278,
        1967,
        338,
    ],
    "USER: How would you rate the quality of this image?\n": [
        1,
        3148,
        1001,
        29901,
        1128,
        723,
        366,
        6554,
        278,
        11029,
        310,
        445,
        1967,
        29973,
        13,
    ],
    "  two  spaces": [1, 259, 1023, 29871, 8162],
    "\nnew line": [1, 29871, 13, 1482, 1196],
    "▁under ▁ line": [1, 29871, 1090, 259, 1196],
    "</s> special <s> tokens <unk>": [1, 2, 4266, 29871, 1, 18897, 29871, 0],
    "naïve 中文 😀": [1, 1055, 30085, 345, 29871, 30275, 30333, 29871, 243, 162, 155, 131],
    "excellent": [1, 15129],
}
# Six closed-form views of different shapes, (x·7 + y·13 + c·29 + i·31) mod 256.
_SIZES = ((300, 200), (200, 300), (256, 256), (120, 400), (400, 120), (333, 222))
_PATTERN = (7, 13, 29, 31)
_BYTE = 256
_RGB = 3
# pcqa_eval.py's conversation template, question and answer stem, and its colour scale.
_TEMPLATE = "mplug_owl2"
_QUESTION = "How would you rate the quality of this image?"
_ANSWER = " The quality of the image is"
_COLOUR_SCALE = 255
_LEVELS = ["excellent", "good", "fair", "poor", "bad"]
_LEVEL_LOGITS = [3.4140625, 7.96484375, 13.3515625, 14.859375, 13.5390625]
_LAST_LOGITS_SHA256 = "e918cdbdc50ec10123b20a843aba966d4b46de84a33332a104661125c398187f"

needs_weights = pytest.mark.skipif(not _WEIGHTS.is_dir(), reason=f"no weights at {_WEIGHTS}")


@pytest.fixture(scope="module")
def tokenizer() -> LlamaTokenizer:
    """The restored slow tokenizer of the released weights."""
    return LlamaTokenizer.from_pretrained(_WEIGHTS)


@needs_weights
@pytest.mark.parametrize("text", list(_TOKENS))
def test_slow_tokenizer(tokenizer: LlamaTokenizer, text: str) -> None:
    """The ids 4.36.1's slow tokenizer gives, edge cases of spaces and special tokens among them."""
    assert tokenizer(text).input_ids == _TOKENS[text]


@needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU with 16 GB")
def test_released_logits() -> None:
    """The level words' logits and the whole last-position logits of LMM-PCQA's prompt."""
    tokenizer, model, processor, _ = load_pretrained_model(
        str(_WEIGHTS), model_base=None, model_name=_WEIGHTS.name, device=_DEVICE
    )
    conversation = conv_templates[_TEMPLATE].copy()
    question = f"{_QUESTION}\n{DEFAULT_IMAGE_TOKEN}"
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt() + _ANSWER
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    words = [word[1] for word in tokenizer(_LEVELS)["input_ids"]]
    background = tuple(int(x * _COLOUR_SCALE) for x in processor.image_mean)
    images = [
        expand2square(Image.fromarray(_view(i, *size)), background) for i, size in enumerate(_SIZES)
    ]
    pixels = processor.preprocess(images, return_tensors="pt")["pixel_values"].half().to(_DEVICE)
    with torch.inference_mode():
        last = model(ids.unsqueeze(0).to(_DEVICE), images=[pixels])["logits"][:, -1].mean(0)
    assert [last[i].item() for i in words] == _LEVEL_LOGITS
    assert hashlib.sha256(last.cpu().numpy().tobytes()).hexdigest() == _LAST_LOGITS_SHA256


def _view(index: int, height: int, width: int) -> np.ndarray:
    """A closed-form RGB view, the same on every machine."""
    y, x, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(_RGB), indexing="ij")
    weight_x, weight_y, weight_c, weight_i = _PATTERN
    pattern = (x * weight_x + y * weight_y + c * weight_c + index * weight_i) % _BYTE
    view = pattern.astype(np.uint8)
    return view
