
from llava.model.language_model.llava_dream import LlavaDream, LlavaDreamConfig
from llava.model.language_model.dream.tokenization_dream import DreamTokenizer
from transformers import AutoTokenizer, AutoConfig

# path = sys.argv[1]
model_path = '/path_to/LLaVA-SFT/outputs-si/dream-vl_sft_si_lr1e-5+2ep_lr5e-6_4k'
# model_path = 'jiacheng-ye/Dream-VL'

cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = LlavaDream.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, torch_dtype="bfloat16")
print(model)

from dreamvl.configuration_dreamvl import DreamVLConfig
from dreamvl.modeling_dreamvl import DreamVLModel

# new_config.save_pretrained('dreamvl')
new_config = DreamVLConfig.from_pretrained('dreamvl')
new_model = DreamVLModel._from_config(new_config)
print(new_model)

def transform_state_dict_keys(state_dict, name_mapping):
    """
    Transform state dict keys according to a mapping dictionary
    Supports both prefix and suffix replacements with wildcards
    
    Args:
        state_dict: Original model state dictionary
        name_mapping: Dict of {old_pattern: new_pattern} replacements
                     Can use '*' as wildcard for partial matches
    
    Returns:
        New state dictionary with transformed keys
    """
    new_state_dict = {}
    
    for old_key, value in state_dict.items():
        new_key = old_key
        
        # Try exact matches first
        for old_pattern, new_pattern in name_mapping.items():
            if old_pattern in new_key:  # Simple substring match
                new_key = new_key.replace(old_pattern, new_pattern)
                break
        
        new_state_dict[new_key] = value
    
    return new_state_dict

name_mapping = {
    'model.vision_tower.vision_tower': 'visual',  # Full path replacement,
    'model.mm_projector.0.weight': 'projector.linear_1.weight',
    'model.mm_projector.0.bias': 'projector.linear_1.bias',
    'model.mm_projector.2.weight': 'projector.linear_2.weight',
    'model.mm_projector.2.bias': 'projector.linear_2.bias'
    # 'projector.0.weight': 'projector.linear_1.weight',
    # 'projector.0.bias': 'projector.linear_1.bias',
    # 'projector.2.weight': 'projector.linear_2.weight',
    # 'projector.2.bias': 'projector.linear_2.bias'
}
model_states = model.state_dict()
model_states.pop('model.image_newline')
model_states = transform_state_dict_keys(model_states, name_mapping)
# Load compatible weights
new_model.load_state_dict(model_states, strict=True)

DreamVLConfig.register_for_auto_class()
DreamVLModel.register_for_auto_class("AutoModel")

print(model)
repo_name = "Dream-dev/Dream-VL-Instruct"
new_model.push_to_hub(repo_name, private=True)

############ upload others

from dreamvl.image_processing_dreamvl import DreamVLImageProcessor
from dreamvl.processing_dreamvl import DreamVLProcessor
image_processor = DreamVLImageProcessor.from_json_file("./dreamvl/preprocessor_config.json")
DreamVLImageProcessor.register_for_auto_class("AutoImageProcessor")

from dreamvl.tokenization_dream import DreamTokenizer
tokenizer = DreamTokenizer.from_pretrained(model_path, trust_remote_code=True)
DreamTokenizer.register_for_auto_class("AutoTokenizer")

from transformers import AutoImageProcessor, AutoTokenizer
AutoImageProcessor.register("dream-vl", DreamVLImageProcessor)
AutoTokenizer.register("dream-vl", DreamTokenizer)
tokenizer.chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|image_pad|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|video_pad|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
processor = DreamVLProcessor(image_processor, tokenizer, chat_template=tokenizer.chat_template)
DreamVLProcessor.register_for_auto_class("AutoProcessor")
processor.push_to_hub(repo_name, private=True)