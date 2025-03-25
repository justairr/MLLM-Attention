from typing import List, Optional, Tuple, Union, Literal
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

import numpy as np
from PIL import Image
from IPython.display import display

# Calculate dynamic threshold for attention map
def calculate_dynamic_threshold(visual_token_attn_score, percentile=95):
    hist = torch.histc(visual_token_attn_score, bins=100)
    cdf = torch.cumsum(hist, dim=0) / torch.sum(hist)
    threshold = torch.argmax((cdf > percentile / 100).float()).item() / 100
    return threshold

def generate_images(image_list, mode="noise", color=(255, 255, 255)):
    """
    根据 `mode` 生成与 `image_list` 中图像等大的随机噪声或纯色图像。

    参数：
    - image_list: list[PIL.Image]，输入的图像列表。
    - mode: str，"noise" 生成随机噪声图像，"blank" 生成纯色图像。
    - color: tuple，生成纯色图像时的颜色，默认为白色 (255, 255, 255)。

    返回：
    - list[PIL.Image]，生成的图像列表。
    """
    generated_images = []
    
    for img in image_list:
        width, height = img.size
        
        if mode == "noise":
            noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            generated_image = Image.fromarray(noise_array)
        elif mode == "blank":
            generated_image = Image.new("RGB", (width, height), color)
        else:
            raise ValueError("mode could only be 'noise' or 'blank'")
        
        generated_images.append(generated_image)
    
    return generated_images

def get_mean_attn_score(output_ids) -> torch.Tensor:
    r"""
    get the mean attention weights of the prefilling and full attention
    Args:
        output_ids: the output ids of the model
    Returns:
        mean_attn: the mean attention weights of the prefilling and full attention, shape: (L, L)
    """
    output_attn = output_ids.attentions
    pref_len = output_attn[0][0].shape[3]
    full_len = output_attn[-1][0].shape[3]
    prefill_attn = output_attn[0]
    assert prefill_attn[0].shape[0] == 1, "batch size should be 1"
    full_attn = []

    for l, layer in enumerate(prefill_attn):
        layer = layer.cpu().squeeze(0).float()
        layer = torch.nn.functional.pad(
            layer, (0, full_len - pref_len, 0, full_len - pref_len)
        )
        for i in range(full_len - pref_len):
            cur_attn = output_attn[i + 1][l].cpu().squeeze(0)[:, 0, :].float()
            layer[:, pref_len + i, : pref_len + i + 1] = cur_attn
        full_attn.append(layer)
    mean_attn = torch.stack(full_attn).mean(dim=(0, 1))
    return mean_attn


def get_visual_token_mean_attn_score(
    mean_attn, inputs, vision_start_token_id, vision_end_token_id
) -> Tuple[torch.Tensor, ...]:
    r"""
    Get the attention weights of the visual tokens
    Args:
        mean_attn: the mean attention weights of the prefilling and full attention, shape: (L, L)
        inputs: the inputs of the model
    Returns:
        visual_token_attn_weights: the tuple of the attention weights of the visual tokens, each element shape: (V, V)
    """
    assert inputs["input_ids"].shape[0] == 1, "batch size should be 1"
    pref_len = len(inputs["input_ids"][0])
    vision_start_token_indices = torch.where(
        inputs["input_ids"][0] == vision_start_token_id
    )[0]
    vision_end_token_indices = torch.where(
        inputs["input_ids"][0] == vision_end_token_id
    )[0]
    # assert len(vision_start_token_indices) == len(vision_end_token_indices), "vision start and end token idx should be the same"
    # print(vision_start_token_indices)
    # print(vision_end_token_indices)
    # iterate over multiple images
    visual_token_attn_weights = tuple(
        torch.mean(mean_attn[pref_len:, s + 1 : e], dim=0)
        for s, e in zip(
            vision_start_token_indices, vision_end_token_indices, strict=True
        )
    )
    return visual_token_attn_weights


def get_visual_token_weight(
    visual_token_attn_score,
    threshold,
    keep_weight,
    weighting_type: Literal["linear", "exp", "uniform", "suppress"] | str = "linear",
    lowest_weight=0.0,
    neg_attn_weight=None,
    suppress_alpha=0.5,
):
    if weighting_type == "suppress":
        if neg_attn_weight is None:
            raise ValueError("neg_attn_weight must be provided for suppress mode")
        # 使用负样例注意力权重进行抑制
        weight_vision_token = 1 - suppress_alpha * neg_attn_weight
        return weight_vision_token

    sorted_indices = torch.argsort(visual_token_attn_score, descending=True)
    num_tokens_to_keep = int(len(visual_token_attn_score) * threshold)
    weight_vision_token = torch.zeros_like(visual_token_attn_score, dtype=torch.float)
    weight_vision_token[sorted_indices[:num_tokens_to_keep]] = keep_weight
    if weighting_type == "linear":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.linspace(
            1.0, lowest_weight, len(visual_token_attn_score) - num_tokens_to_keep
        )
    elif weighting_type == "exp":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.exp(
            torch.linspace(0, lowest_weight, len(sorted_indices) - num_tokens_to_keep)
        )
    elif weighting_type == "uniform":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = lowest_weight
    else:
        raise ValueError(f"Invalid weighting type: {weighting_type}")
    return weight_vision_token


def patch_forward(
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
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if self.embed_weight is not None:
                image_embeds *= self.embed_weight[:, None]
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                cache_position[0] + self.rope_deltas
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                delta = delta.to(position_ids.device)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
