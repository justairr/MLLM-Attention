from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import copy
import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag
from qwen_vl_utils import smart_resize
from transformers.cache_utils import (
    DynamicCache,
    StaticCache,
)

try:
    from .qwen import (
        Qwen2VLForAttnExtraction,
        reweighted_vision_tokens
    )
except ModuleNotFoundError as err:
    logging.critical("qwen mod forward not found.")
    raise err


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        do_sample=False,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            #top_p=top_p,
            #top_k=top_k,
            do_sample=do_sample,
            #temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2

        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        rank, world_size = get_rank_and_world_size()

        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        if '72b' in self.model_path.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='eager'
            )
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='eager'
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='eager'
            )
            self.model.cuda().eval()

        #self.model.forward = patch_forward.__get__(
        #    self.model, Qwen2VLForConditionalGeneration
        #)
        self.small_model = Qwen2VLForAttnExtraction(self.model, self.processor)
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        final_prompt = self._prepare_content(message, dataset=dataset)
        messages.append({'role': 'user', 'content': final_prompt})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        #print("=============> ", messages)
        images, videos = process_vision_info([messages])
        
        image = images[0]
        
        h, w = image.size

        input_height,input_width = smart_resize(h, w, min_pixels=messages[0]['content'][0]['min_pixels'], max_pixels=messages[0]['content'][0]['max_pixels'])

        image = image.resize((input_height, input_width))
        images[0] = image
        self.model.embed_weight = None

        attn = self.small_model.extract_attention(
            final_prompt,
            image,
            # [],
            attn_type="contrastive",
            single_token_generation=False,
            contrast_layers=(14, 6),
        )
        #bbox = bbox_from_att_image_adaptive(attn[0], image.size, bbox_size=14 * 14)
        #crop_image = image.crop(bbox)
        #images.append(crop_image)

        # messages[0]['content'].insert(1, {'type': 'image', 'image':crop_image})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #print(messages)
        #text = text[:-1]
        inputs = self.processor(text=text[:-1], images=images, videos=videos, return_tensors='pt')
        inputs = inputs.to(self.model.device)

        keep_perc = os.environ.get('KP', "0.6")
        keep_perc = float(keep_perc)
        keep_weight = os.environ.get('KW', "1.0")
        keep_weight = float(keep_weight)
        linear_start = os.environ.get('LS', "0.0")
        linear_start = float(linear_start)
        weighting_type = os.environ.get('WT', "linear")
        suppress_alpha = os.environ.get('SA', "0.5")
        suppress_alpha = float(suppress_alpha)
        #dynamic_threshold = os.environ.get('DY', "False")
        #dynamic_threshold = dynamic_threshold.lower() == "true"
        
        #if weighting_type == "suppress":
        #    neg_images = generate_images(images, mode="noise")
        #    neg_inputs = self.processor(text=text.copy(), images=neg_images, videos=videos, padding=True, return_tensors='pt')
        #    neg_inputs = neg_inputs.to('cuda')
        #    neg_output_ids = self.model.generate(
        #        **neg_inputs,
        #        return_dict_in_generate=True,
        #        output_attentions=True,
        #        **self.generate_kwargs,
        #    )

        #    neg_mean_attn_score = get_mean_attn_score(neg_output_ids)
#
        #    neg_visual_token_attn_score = get_visual_token_mean_attn_score(
        #        neg_mean_attn_score,
        #        neg_inputs,
        #        self.model.config.vision_start_token_id,
        #        self.model.config.vision_end_token_id,
        #    )

        #    vision_token_weight_per_image = [
        #        get_visual_token_weight(
        #            v, keep_perc, keep_weight, "suppress", linear_start, neg_v, suppress_alpha
        #        )
        #        for v, neg_v in zip(visual_token_attn_score, neg_visual_token_attn_score)
        #    ]

        #else:
        #    vision_token_weight_per_image = [
        #        get_visual_token_weight(
        #            v, keep_perc, keep_weight, weighting_type, linear_start
        #        )
        #        for v in visual_token_attn_score
        #    ]

        attn_flat = attn[0].flatten()
        weight = reweighted_vision_tokens(attn_flat, keep_percentage=keep_perc, weighting_type="linear").to(self.model.device)
        
        prompt_cache = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=2048,
            device=self.model.device,
            dtype=self.model.dtype,
            # layer_device_map={0: "cuda", 1: "cuda", 2: "cuda", 3: "cuda", 4: "cuda", 5: "cuda", 6: "cuda", 7: "cuda"}
        )
        with torch.no_grad():
            # prompt_cache.reset()
            prompt_cache = DynamicCache()
            initial_outputs = self.model(**inputs, past_key_values=prompt_cache)
            prompt_cache = initial_outputs.past_key_values

        vision_start_token_id = self.model.config.vision_start_token_id
        vision_end_token_id = self.model.config.vision_end_token_id
        prefill_len = inputs["input_ids"].shape[1]
        vision_start_token_indices = (
            torch.where(inputs["input_ids"][0] == vision_start_token_id)[0] + 1
        )
        vision_end_token_indices = torch.where(inputs["input_ids"][0] == vision_end_token_id)[0]
        assert len(vision_start_token_indices) == len(vision_end_token_indices) == 1, (
            "vision_start_token_indices and vision_end_token_indices should have the same length"
        )

        for i, (k, v) in enumerate(
            zip(prompt_cache.key_cache, prompt_cache.value_cache, strict=True)
        ):
            # print(f'key layer {i:2d}: {tuple(k.shape)}')
            k[:, :, vision_start_token_indices:vision_end_token_indices, :] *= weight[
                None, None, :, None
            ]
            v[:, :, vision_start_token_indices:vision_end_token_indices, :] *= weight[
                None, None, :, None
            ]
        #print(self.generate_kwargs)
        # Generate output based on modified inputs
        new_inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
            # padding=True,
        )
        new_inputs = new_inputs.to(self.model.device)

        input_cache =copy.deepcopy(prompt_cache)

        #print(input_cache.value_cache[0].shape)
        #print(input_cache.key_cache[0].shape)
        generated_ids = self.model.generate(**new_inputs, past_key_values=input_cache, **self.generate_kwargs)
    
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]
    
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
    
        return response