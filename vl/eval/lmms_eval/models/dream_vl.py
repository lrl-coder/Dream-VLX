from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("dream_vl")
class Dream_VL(lmms):
    """
    Dream-VL Model
    "https://github.com/Dream-VLX/Dream-VLX"
    """

    def __init__(
        self,
        pretrained: str = "Dream-org/Dream-VL-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = False,
        use_flash_attention_2: Optional[bool] = True,
        max_pixels: int = 12845056,
        min_pixels: int = 200704,
        max_num_frames: int = 32,
        steps: int = 128,
        temperature: float = 0,
        top_p: float = 1.0,
        alg: str = "maskgit_plus",
        alg_temp: float = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            # model_kwargs["vision_config"] = {"attn_implementation": "flash_attention_2"}

        # self._model = AutoModel.from_pretrained(pretrained, **model_kwargs).eval()
        from ..dreamvl.modeling_dreamvl import DreamVLModel
        self._model = DreamVLModel.from_pretrained(pretrained, **model_kwargs).eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        self._tokenizer = self.processor.tokenizer
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # Generation parameters
        self.steps = steps
        self.temperature = temperature
        self.top_p = top_p
        self.alg = alg
        self.alg_temp = alg_temp

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", 32768)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Dream_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            until.append('<|im_end|>')

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # Prepare messages for chat template
            messages = []
            processed_images = []
            processed_videos = []

            for i, context in enumerate(contexts):
                # Remove <image> token if present (will be handled by chat template)
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = []

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        # Load video frames
                        try:
                            vr = decord.VideoReader(visual)
                            total_frames = len(vr)
                            indices = np.linspace(0, total_frames - 1, min(self.max_num_frames, total_frames), dtype=int)
                            if total_frames - 1 not in indices:
                                indices = np.append(indices, total_frames - 1)
                            frames = [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in indices]
                            processed_videos.append(frames)  # List of PIL Images
                            processed_images.append(None)
                            # For video, we use the video path in the message
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})
                        except Exception as e:
                            eval_logger.error(f"Error loading video {visual}: {e}")
                            processed_videos.append(None)
                            processed_images.append(None)
                            message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                    elif isinstance(visual, Image.Image):  # Single image
                        processed_images.append(visual)
                        processed_videos.append(None)
                        message.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        processed_images.append(visual)
                        processed_videos.append(None)
                        image_content = [{"type": "image"} for _ in visual]
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and isinstance(visual[0], str):  # List of image paths
                        images = [Image.open(img_path).convert("RGB") for img_path in visual]
                        processed_images.append(images)
                        processed_videos.append(None)
                        image_content = [{"type": "image"} for _ in images]
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        processed_images.append(None)
                        processed_videos.append(None)
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    processed_images.append(None)
                    processed_videos.append(None)
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)

            # Apply chat template
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

            # Prepare images and videos for processing
            # Convert list of images/videos to the format expected by processor
            images_to_process = []
            videos_to_process = []
            for img, vid in zip(processed_images, processed_videos):
                if vid is not None:
                    # Convert list of PIL Images to numpy array or tensor format
                    videos_to_process.append(vid)
                    images_to_process.append(None)
                elif img is not None:
                    images_to_process.append(img)
                    videos_to_process.append(None)
                else:
                    images_to_process.append(None)
                    videos_to_process.append(None)

            # Process inputs - only pass non-None images/videos
            has_images = any(img is not None for img in images_to_process)
            has_videos = any(vid is not None for vid in videos_to_process)
            
            inputs = self.processor(
                text=texts,
                images=images_to_process if has_images else None,
                videos=videos_to_process if has_videos else None,
                padding=True,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Get generation parameters
            max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
            temperature = gen_kwargs.get("temperature", self.temperature)
            top_p = gen_kwargs.get("top_p", self.top_p)
            steps = gen_kwargs.get("steps", self.steps)
            alg = gen_kwargs.get("alg", self.alg)
            alg_temp = gen_kwargs.get("alg_temp", self.alg_temp)

            # Extract input_ids for diffusion_generate
            input_ids = inputs.pop("input_ids")

            # Generate using diffusion_generate
            output = self.model.diffusion_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                output_history=True,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                use_cache=self.use_cache,
                **inputs,
            )

            # Decode generated sequences
            generations = [
                self.processor.tokenizer.decode(g[len(p):].cpu().tolist(), skip_special_tokens=False)
                for p, g in zip(input_ids, output.sequences)
            ]

            # Apply until stopping criteria
            for i, ans in enumerate(generations):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                generations[i] = ans

            for ans, context in zip(generations, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

