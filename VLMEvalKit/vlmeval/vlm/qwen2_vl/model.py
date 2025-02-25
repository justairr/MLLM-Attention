from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag


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
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            temperature=temperature,
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

        try:
            from .qwen_mod_forward import patch_forward
        except ModuleNotFoundError as err:
            logging.critical("qwen mod forward not found.")
            raise err
        self.model.forward = patch_forward.__get__(self.model, Qwen2VLForConditionalGeneration)


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
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        self.model.embed_weight = None

        output_ids = self.model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_attentions=True,
            **self.generate_kwargs,
        )

        output_attn = output_ids.attentions
        pref_len = output_attn[0][0].shape[3]
        full_len = output_attn[-1][0].shape[3]
        prefill_attn = output_attn[0]

        assert prefill_attn[0].shape[0] == 1
        full_attn = []

        for l, layer in enumerate(prefill_attn):
            layer = layer.cpu().squeeze(0).float()
            layer = torch.nn.functional.pad(layer, (0, full_len - pref_len, 0, full_len - pref_len))
            for i in range(full_len - pref_len):
                cur_attn = output_attn[i + 1][l].cpu().squeeze(0)[:, 0, :].float()
                layer[:, pref_len + i, :pref_len + i + 1] = cur_attn
            full_attn.append(layer)

        mean_attn = torch.stack(full_attn).mean(dim=(0, 1))
        vision_start_token_idx = inputs['input_ids'][0].tolist().index(self.model.config.vision_start_token_id)
        vision_end_token_idx = inputs['input_ids'][0].tolist().index(self.model.config.vision_end_token_id)

        image_output_attn = torch.mean(mean_attn[pref_len:, vision_start_token_idx + 1:vision_end_token_idx], dim=0)

        # Calculate dynamic threshold for attention map
        def calculate_dynamic_threshold(attn, percentile=95):
            hist = torch.histc(attn, bins=100)
            cdf = torch.cumsum(hist, dim=0) / torch.sum(hist)
            threshold = torch.argmax((cdf > percentile / 100).float()).item() / 100
            return threshold

        threshold = calculate_dynamic_threshold(image_output_attn)

        # Apply weighted attention to vision tokens
        def weighted_vision_attention(attn_map, keep_percentage, if_linear, linear_start):
            sorted_attention, sorted_indices = torch.sort(attn_map, descending=True)
            num_tokens_to_keep = int(len(sorted_attention) * keep_percentage)
            weight_vision_token = torch.zeros_like(attn_map, dtype=torch.float)
            weight_vision_token[sorted_indices[:num_tokens_to_keep]] = 1.0
            if (if_linear==True): 
                weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.linspace(linear_start, 1.0, len(sorted_attention) - num_tokens_to_keep)
            else:
                weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.exp(torch.linspace(0, -3, len(sorted_attention) - num_tokens_to_keep))
            return weight_vision_token
    
        keep_perc = os.environ.get('KP', "0.6")
        keep_perc = float(keep_perc)
        linear_start = os.environ.get('LS', "0.0")
        linear_start = float(linear_start)
        if_linear = os.environ.get('IL', "True")
        if_linear = if_linear.lower() == "true"
        weight_vision_token = weighted_vision_attention(image_output_attn, keep_percentage=keep_perc, if_linear=if_linear ,linear_start=linear_start).cuda()
    
        self.model.embed_weight = weight_vision_token

        # Update image embeddings with the weighted attention
    #    input_ids = inputs["input_ids"]
    #    attention_mask = inputs["attention_mask"]
    #    pixel_values = inputs["pixel_values"]
    #    image_grid_thw = inputs["image_grid_thw"]
    #
    #    inputs_embeds = self.model.model.embed_tokens(input_ids)
    #    if pixel_values is not None:
    #        pixel_values = pixel_values.type(self.model.visual.get_dtype())
    #        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
    #        n_image_tokens = (input_ids == self.model.config.image_token_id).sum().item()
    #        n_image_features = image_embeds.shape[0]
    #        if n_image_tokens != n_image_features:
    #            raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
    #        
    #        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    #        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    #        image_embeds *= weight_vision_token[:, None]
    #        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    #
    #    if attention_mask is not None:
    #        attention_mask = attention_mask.to(inputs_embeds.device)
    
        # Generate output based on modified inputs
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
    
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