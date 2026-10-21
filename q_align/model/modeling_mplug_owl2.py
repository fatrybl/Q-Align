#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from .configuration_mplug_owl2 import MPLUGOwl2Config, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel
from .modeling_llama2 import ModalityAdaptiveLlamaModel
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def expand2square(pil_img, background_color):
        from PIL import Image
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super(MPLUGOwl2MetaModel, self).__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(**config.visual_config["visual_abstractor"]), config.hidden_size
        )
    
    def get_vision_tower(self):
        vision_model = getattr(self, 'vision_model', None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_visual_abstractor(self):
        visual_abstractor = getattr(self, 'visual_abstractor', None)
        if type(visual_abstractor) is list:
            visual_abstractor = visual_abstractor[0]
        return visual_abstractor


class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = self.get_model().visual_abstractor(encoder_hidden_states=image_features).last_hidden_state
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return input_ids, multiway_indices, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                
                cur_modality_indicators = torch.zeros(len(cur_input_embeds)).long().to(self.device)
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                
                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids[:image_token_start])).long())
                cur_modality_indicators.append(torch.ones(len(cur_image_features)).long())
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            # Modality
            cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)
            
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat((cur_modality_indicator, torch.zeros(max_len - cur_modality_indicator.shape[0], dtype=cur_modality_indicator.dtype, device=cur_modality_indicator.device)), dim=0)
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)
            
            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, new_modality_indicators, attention_mask, past_key_values, new_input_embeds, new_labels



class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, ModalityAdaptiveLlamaModel):
    config_class = MPLUGOwl2Config

    def __init__(self, config: MPLUGOwl2Config):
        super(MPLUGOwl2LlamaModel, self).__init__(config)


class MPLUGOwl2LlamaForCausalLM(LlamaPreTrainedModel, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(self, input_ids: torch.LongTensor, images: list[torch.Tensor] | torch.Tensor | None = None) -> CausalLMOutputWithPast:
        input_ids, modality_indicators, _, _, inputs_embeds, _ = \
            self.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, images)
        outputs = self.model(input_ids, modality_indicators, inputs_embeds)
        logits = self.lm_head(outputs[0])
        return CausalLMOutputWithPast(logits=logits)


AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)
