import torch
import torch.nn as nn

from transformers import AutoModel, AutoImageProcessor, AutoConfig, CLIPImageProcessor
from .qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from llava.utils import rank0_print


class HFVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower.replace("hf:", "", 1)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self.image_processor.size['max_pixels'] = 4096 * 28 * 28
        self.image_processor.max_pixels = 4096 * 28 * 28
        # self.image_processor.size['max_pixels'] = 1024 * 28 * 28
        # self.image_processor.max_pixels = 1024 * 28 * 28
        self.image_processor.size['min_pixels'] = 512 * 28 * 28
        self.image_processor.min_pixels = 512 * 28 * 28
        rank0_print(f"Loaded image processor: {self.image_processor}")
        self.vision_tower = Qwen2VisionTransformerPretrainedModel.from_pretrained(self.vision_tower_name, torch_dtype=torch.bfloat16, 
                                                      trust_remote_code=True
                                                    #   , attn_implementation="flash_attention_2"
                                                      )

        if hasattr(self.vision_tower, "vision_model"):
            self.vision_tower = self.vision_tower.vision_model
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        self.config = self.vision_tower.config

    def custom_tunable_parts(self):
        tune_vit_from_layer = 16
        for n, p in self.vision_tower.named_parameters():
            if 'blocks.' in n:
                layer_id = int(
                    n.split('blocks.')[-1].split('.')[0])
                if layer_id >= tune_vit_from_layer:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            elif 'merger' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
    def forward(self, images, image_grid_thw):
        image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype), image_grid_thw)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        return _num_patches

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
