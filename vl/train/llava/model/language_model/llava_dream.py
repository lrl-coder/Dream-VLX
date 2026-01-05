#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoConfig, AutoModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .dream import DreamConfig, DreamBaseModel, DreamModel
from .dream.configuration_dream import DreamConfig
from .dream.modeling_dream import DreamBaseModel, DreamModel
# from .configuration_dream import DreamConfig
# from .modeling_dream import DreamBaseModel, DreamModel

class LlavaDreamConfig(DreamConfig):
    model_type = "llava_dream"


class LlavaDreamModel(LlavaMetaModel, DreamBaseModel):
    config_class = LlavaDreamConfig

    def __init__(self, config: LlavaDreamConfig):
        super(LlavaDreamModel, self).__init__(config)


def forward_process(batch, maskable_mask, mask_id, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    mask_indices [~maskable_mask] = False
    noisy_batch = torch.where(mask_indices, mask_id, batch)
    return noisy_batch, mask_indices, p_mask

def context_adaptive_reweight(seq_len, distribution="symmetric-geometric", **kwargs):
    position_ids_l = np.arange(seq_len).reshape(-1, 1)
    position_ids_r = np.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    distance = torch.from_numpy(distance)

    # def discrete_gaussian_kernel(t, n):
    #     return math.exp(-t) * scipy.special.iv(n, t)

    def geometric_distribution(k, cart_p=0.8, **kwargs):
        if not 0 < cart_p <= 1:
            raise ValueError("p must be between 0 and 1")

        res = (math.log(cart_p) + (k.abs()-1)*math.log(1 - cart_p)).exp() * 0.5
        res.masked_fill_(k==0, 0) # ignore distance=0
        return res

    if distribution == "symmetric-geometric":
        matrix = geometric_distribution(distance, **kwargs)
    else:
        raise ValueError(f"Unknown distribution {distribution}")
    
    return matrix

class LlavaDream(DreamModel, LlavaMetaForCausalLM):
    config_class = LlavaDreamConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        DreamModel.__init__(self, config)
        config.model_type = "llava_dream"
        config.rope_scaling = None

        self.model = LlavaDreamModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_token_id = config.mask_token_id
        self.weighting = "cart"
        self.weight_matrix = context_adaptive_reweight(50000, cart_p=0.1)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("before encoding vit")
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        # print("after encoding vit")
        if input_ids is None:
            assert inputs_embeds is not None
            input_ids = torch.full_like(labels, self.mask_token_id) 
        if dpo_forward:
            raise(ValueError, "not implemented yet")
        
        bsz, seqlen = input_ids.shape
        maskable_mask = labels != -100
        noisy_input_ids, input_mask_indices, p_mask = forward_process(input_ids, maskable_mask, mask_id=self.mask_token_id)
        if inputs_embeds is not None:
            # set the embedding of input_mask_indices to be the mask embedding
            noisy_embedding = self.model.get_input_embeddings()(noisy_input_ids)
            inputs_embeds = torch.where(input_mask_indices.unsqueeze(-1), noisy_embedding, inputs_embeds)

        labels.masked_fill_(~input_mask_indices, -100)
        shifted_labels = torch.cat([labels[:,1:], labels.new_full((bsz, 1), -100)], dim=-1)

        logits = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ).logits

        loss_mask = shifted_labels != -100
        loss = F.nll_loss(
            F.log_softmax(logits[loss_mask].float(), -1),
            shifted_labels[loss_mask],
            reduction="none"
        )
        
        if self.weighting == 'original':
            loss  = loss / p_mask[loss_mask]
        elif self.weighting == 'linear':
            loss  = loss * (1 - p_mask[loss_mask])
        elif self.weighting == 'none':
            loss  = loss
        elif self.weighting == 'cart':
            _weight_matrix = self.weight_matrix[:seqlen, :seqlen].to(loss_mask.device)
            non_mask = ~loss_mask # loss_mask indicates where is mask
            weight = non_mask.type_as(_weight_matrix).matmul(_weight_matrix).masked_fill(non_mask, 0)
            loss  = loss * weight[loss_mask]
        else:
            raise ValueError(f"weighting={self.weighting} not defined!")

        # we optimize token loss as it is more stable
        loss = loss.sum() / loss_mask.sum()
    
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        output = super().diffusion_generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return output[:, inputs_embeds.shape[1]:]

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


# AutoConfig.register("llava_dream", LlavaDreamConfig)
# AutoModelForCausalLM.register(LlavaDreamConfig, LlavaDream)
# AutoModel.register(LlavaDreamConfig, LlavaDream)