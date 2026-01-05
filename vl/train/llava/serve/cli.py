import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, multimodal=True)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif 'qwen2' in model_name.lower():
        conv_mode = "qwen_2"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    processed_res = image_processor.preprocess(image, return_tensors="pt")
    image_tensor = processed_res["pixel_values"].cuda()
    if "qwen2vit" in args.model_path:
        image_sizes = processed_res['image_grid_thw']
    else:
        image_sizes = None
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                # inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                inp = DEFAULT_IMAGE_TOKEN + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            if 'dream' in model_name:
                output_ids = model.generate(input_ids, images=image_tensor, temperature=0.1, alg="maskgit_plus", alg_temp=0, max_new_tokens=128, steps=128, image_sizes=image_sizes)
                # output_ids = model.generate(input_ids, images=image_tensor, temperature=0., alg="maskgit_plus", alg_temp=0, max_new_tokens=1, steps=1)
                print(tokenizer.decode(output_ids[0]).split('<|im_end|>\n')[0].split('<|endoftext|>')[0].strip())
            else:
                output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0]).split('<|im_end|>\n')[0].split('<|endoftext|>')[0].strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
