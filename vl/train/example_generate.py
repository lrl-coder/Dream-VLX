import torch
from transformers import AutoProcessor, AutoModel

model_name = "Dream-org/Dream-VL-7B"

# from dreamvl.modeling_dreamvl import DreamVLModel
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to('cuda')

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

####### Method 1
from PIL import Image
import requests
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
messages = [
    {
        "role": "user","content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(text)
inputs = processor(
    text=[text], images=[image], padding=True, return_tensors="pt"
)

####### Method 2: use qwen_vl_utils
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# from qwen_vl_utils import process_vision_info
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

inputs = inputs.to("cuda")
input_ids = inputs.pop("input_ids")
output = model.diffusion_generate(
    input_ids,
    max_new_tokens=128,
    output_history=True,
    return_dict_in_generate=True,
    steps=128,
    temperature=0.1,
    top_p=1,
    alg="maskgit_plus",
    alg_temp=0,
    use_cache=False,
    **inputs
)

generations = [
    processor.tokenizer.decode(g[len(p):].cpu().tolist())
    for p, g in zip(input_ids, output.sequences)
]

for j in range(len(messages)):
    print("output:", j, generations[j].split(processor.tokenizer.eos_token)[0])

# output: The image depicts a serene beach scene featuring a young woman and a golden retriever. The woman, dressed in a plaid shirt and dark pants, is seated on the sandy shore, smiling warmly at the camera. The golden retriever, adorned with a colorful harness, sits attentively beside her, its gaze fixed on the woman. The background reveals the vast expanse of the ocean, with waves gently kissing the shore. The sky above is a clear blue, suggesting a sunny day. The overall atmosphere exudes a sense of peace and companionship between the woman and her dog.