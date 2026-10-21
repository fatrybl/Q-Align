# How this fork differs from the original Q-Align code

**Original:** [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align), commit `fd0c3de`, the code
of *Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels* (Wu et al., 2024).
**Fork:** [fatrybl/Q-Align](https://github.com/fatrybl/Q-Align), branch `vape-baselines`.
`git diff fd0c3de` shows every change.

The fork serves as the language model of LMM-PCQA (Zhang et al., ACM MM 2024), a baseline in the
comparison of the VAPE project; LMM-PCQA's released scoring (`pcqa_eval.py`) imports this package.
The release requires transformers 4.36.1. The fork ports the scoring path, the one LMM-PCQA runs, to
transformers 5.20.2 without changing its arithmetic: on the same inputs it computes the released
logits bit for bit (section 3). Training, serving, video and generation were not ported. The earlier
commit `d5d5ef8` (relaxed pins, one import) is superseded.

## 1. Changes

- **`model/modeling_llama2.py` rewritten, with `model/decoder_layer.py`, `model/multiway.py` and
  `model/rotary.py` (new).** The release replaced transformers 4.36's Llama attention, decoder layer
  and forward functions at import. Transformers 5 removed or changed every one of them, so the
  module now defines mPLUG-Owl2's modality-adaptive LLaMA-2 itself, with the release's parameter
  names (the checkpoints load with no missing or unexpected key): text and image tokens share
  queries, output projection and MLP, and have keys, values and norms of their own. Kept from the
  release: its eager attention, `softmax_fp32(Q Kᵀ/√d + M) V`, and its causal mask
  (`modeling_attn_mask_utils.py`, unchanged). The release pins torch 2.0.1, under which transformers
  4.36 never enables SDPA; with a newer torch, 4.36 switches to SDPA silently, which moved the
  logits by up to 0.016 and LMM-PCQA's score by up to 2.1·10⁻⁴ on the inputs of section 3. The
  rotary table is 4.36's (`model/rotary.py`): float32 on the CPU, rounded once to fp16; transformers
  5 computes its table on the GPU per forward. It holds the checkpoint's 2048 positions, and a longer
  sequence raises an error where 4.36 rebuilt its table (LMM-PCQA's sequences have 417 tokens).
  `model/multiway.py` is the release's `MultiwayNetwork`, moved out of `modeling_llama2.py` without
  its one-module and tuple-output branches, which the release's two-modality linear and norm layers
  never reach. Not ported: KV cache, FlashAttention-2 and SDPA variants, gradient checkpointing;
  rotary scaling and tensor-parallel MLP slices raise an error (the released checkpoints use
  neither).
- **`model/modeling_mplug_owl2.py`.** No monkeypatching and no `sys.path` insertion (it put
  `q_align/model` on the host's import path, where its `utils` would shadow other packages'). The
  model no longer downloads `q-future/one-align`'s tokenizer and image processor when it is built:
  only Q-Align's `score()` helper used them, removed with it. `forward(input_ids, images)` returns
  the logits; labels, loss and generation were removed. The multimodal splicing
  (`prepare_inputs_labels_for_multimodal`, `encode_images`) is unchanged.
- **`model/tokenization_llama.py` (new).** The release asks for the slow sentencepiece tokenizer
  (`use_fast=False`). Transformers 5 ignores that flag and returns its `tokenizers` conversion,
  which drops the lone "▁" sentencepiece writes for the space after the image token: LMM-PCQA's
  prompt had 27 tokens instead of 28. The module restores 4.36.1's `LlamaTokenizer` on transformers
  5's sentencepiece backend (its `legacy=False` prefix and the BOS id). `model/builder.py` uses it
  for every model.
- **`model/builder.py`.** `CLIPImageProcessorPil` instead of `CLIPImageProcessor`, which in
  transformers 5 resizes with torchvision; the PIL backend is 4.36's processor. `dtype` for the
  renamed `torch_dtype`; `low_cpu_mem_usage` dropped (transformers 5 always builds on the meta
  device and ignores it).
- **`model/visual_encoder.py`.** `invert_attention_mask`, `get_head_mask` and
  `_convert_head_mask_to_5d`, which transformers 5 removed from `PreTrainedModel`, are restored from
  4.36.1 as methods of the visual abstractor, arithmetic unchanged (an `assert` became a
  `ValueError`). The head-pruning methods were removed: they called
  `find_pruneable_heads_and_indices`, also removed from transformers, and were already broken
  (`self.output.out_proj`, `self.encoder.layer` do not exist). `use_return_dict` reads
  `return_dict`, the attribute its deprecation names.
- **`model/configuration_mplug_owl2.py`.** `mlp_bias=False`, read by transformers 5's `LlamaMLP`;
  4.36's MLP had bias-free projections.
- **`icecream`** debug imports removed (`mm_utils.py`, `builder.py`, `visual_encoder.py`).
- **`pyproject.toml`.** The dependencies are those of the ported scoring path at the versions it is
  verified on; the release's other pins moved to the `released` extra.
- **`test_transformers5_port.py` (new):** token ids and logits against values the released code
  computed (needs the weights and a 16 GB GPU).

## 2. Kept as released

The network, its weights' layout, the conversation templates, `mm_utils.py`'s
`tokenizer_image_token` and `expand2square`, the attention-mask utilities and every module of the
training, serving and evaluation code. Those other paths may not run on transformers 5. Three names
the release uses but never defines stay as released: `F` in the vision encoder's `get_abs_pos`, a
function nothing calls; `flash_attn_func` in a branch the encoder disables with `if False`; and
`logger` in the vision configurations' `from_pretrained`, which nothing calls either. The scoring
path reaches none of them.

## 3. Verification

The reference is the released code (upstream `fd0c3de`) under transformers 4.36.1 in a venv on torch
2.9.1 with SDPA reported unavailable, as transformers 4.36 decides it under the release's torch
2.0.1, and LMM-PCQA's released `pcqa_eval.py` statements on the `q-future/q-align-pcqa-wpc` weights
(revision `4d7bfe46`). Inputs, 32 sets of six projections: the authors' published projections of
WPC's `bag_level_7`, a re-render of it, and 30 renders of LiDAR clouds (ten ONCE samples of VAPE's
comparison, each at two poses, and at a larger distortion). For all 32 the port gives the same
prompt ids, preprocessed pixels, rotary table, the same fp16 logits over the whole vocabulary at the
last position, bit for bit, and the same score; a second, freshly installed transformers 4.36.1
reproduced the reference bit for bit. The ported model and the checkpoint hold the same 862
tensors with the same shapes. The restored tokenizer equals 4.36.1's slow
tokenizer on 5,031 strings (the prompt, the level words and 5,000 seeded strings of words, spaces,
newlines, "▁", special-token text and non-ASCII characters). The scores do not change under TF32
float32 matmuls, cuDNN benchmark or deterministic mode, or without fp16 reduced-precision
reductions; every evaluation of LMM-PCQA has the same tensor shapes, six views of 448 × 448 and
417 tokens (27 of the prompt, 65 per view). Not checked: the release's own
torch 2.0.1.
