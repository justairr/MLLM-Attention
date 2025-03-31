import re
import requests
from PIL import Image
from collections import OrderedDict
from typing import Callable, Dict, Optional, Sequence
import torch
from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import transformers
import transformers.modeling_attn_mask_utils
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    StaticCache,
    DynamicCache,
)
from copy import deepcopy
from logging import config
import warnings
import cv2
from typing import Literal

# Colorize print with ANSI escape codes on special tokens
PATTERN = re.compile(r"<\|\w+?\|>")
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "black": "\033[30m",
}
COLOR_END = "\033[0m"


def cprint(text: str) -> None:
    for token in re.findall(PATTERN, text):
        text = text.replace(token, COLORS["red"] + token + COLOR_END)
    text = text.replace("\n", COLORS["yellow"] + r"\n" + COLOR_END)
    print(text)


def make_attn_hooks(
    attn: nn.Module,
    attn_outputs: list[torch.Tensor],
    *,
    clear_hooks: bool = False,
    clear_attn_output: bool = False,
) -> tuple[RemovableHandle, RemovableHandle]:
    # Clear Attention Output List if necessary
    if clear_attn_output:
        attn_outputs.clear()

    # Patch the attention mask if using SDPA originally
    # This is a hack to enforce the attention mask work with SDPA
    if (
        attn.config._attn_implementation == "sdpa"
        or attn.config._attn_implementation_internal == "sdpa"
    ):
        AttentionMaskConverter._ignore_causal_mask_sdpa = lambda *_, **__: False

    # Set attention implementation to eager if necessary
    # if (
    #     attn.config._attn_implementation_autoset
    #     or attn.config._attn_implementation != "eager"
    #     or attn.config._attn_implementation_internal != "eager"
    # ):
    attn.config = deepcopy(attn.config)
    attn.config._attn_implementation = "eager"
    attn.config._attn_implementation_internal = "eager"
    attn.config._attn_implementation_autoset = False

    # Patch the forward method to use the super class's forward method
    # aka. the original eager forward method
    attn.forward = super(type(attn), attn).forward

    # Clear hooks if requested
    if clear_hooks and (len(attn._forward_pre_hooks) or len(attn._forward_hooks)):
        warnings.warn(
            "Clearing ALL forward hooks, this is not recommended as it also clears hooks like accelerator hooks"
        )
        attn._forward_pre_hooks = OrderedDict()
        attn._forward_hooks = OrderedDict()
    # Create pre-forward hook, for modifying kwargs
    pre_fwd_hook = attn.register_forward_pre_hook(
        lambda module, args, kwargs: kwargs.update({"output_attentions": True}),
        with_kwargs=True,
    )
    post_fwd_hook = attn.register_forward_hook(
        lambda module, args, output: attn_outputs.append(output[1].cpu())
    )
    return pre_fwd_hook, post_fwd_hook


def make_layer_attn_hooks(
    model_list: nn.ModuleList,
    layers: list[int],
    attn_outputs: dict[int, list[torch.Tensor]],
    *,
    clear_hooks: bool = False,
    clear_attn_output: bool = False,
) -> list[tuple[RemovableHandle, RemovableHandle]]:
    if clear_attn_output:
        attn_outputs.clear()

    fwd_hooks = []
    for layer_idx in layers:
        attn_outputs[layer_idx] = []
        attn = model_list[layer_idx].self_attn

        pre_fwd_hook, post_fwd_hook = make_attn_hooks(
            attn,
            attn_outputs[layer_idx],
            clear_hooks=clear_hooks,
            clear_attn_output=clear_attn_output,
        )

        fwd_hooks.append((pre_fwd_hook, post_fwd_hook))

    return fwd_hooks


def get_mean_attn_score(outputs: GenerateDecoderOnlyOutput) -> torch.Tensor:
    r"""
    get the mean attention weights of the prefilling and full attention
    Args:
        output_ids: the output ids of the model
    Returns:
        mean_attn: the mean attention weights of the prefilling and full attention, shape: (L, L)
    """
    output_attn = outputs.attentions
    assert output_attn is not None, "output_attn should not be None"

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


def merge_decoding_attn_maps(attn_maps: Sequence[torch.Tensor]) -> torch.Tensor:
    # attn_maps[0]: prefilling proc. attn maps, (batch_size, n_heads, n_prompt_tokens, n_prompt_tokens)
    # attn_maps[1:]: decoding proc. attn maps, (batch_size, n_heads, 1, current_token_index)
    # merge the decoding proc. attn maps to the prefilling proc. attn maps
    # return: (batch_size, n_heads, n_total_tokens, n_total_tokens)
    # Check if there are any attention maps to merge
    if not attn_maps:
        return torch.empty(0)

    # Get the prefill attention map (first element)
    prefill_map = attn_maps[0]

    # If there's only the prefill map, return it directly
    if len(attn_maps) == 1:
        return prefill_map

    # Extract dimensions
    batch_size, n_heads, n_prompt_tokens, _ = prefill_map.shape

    # Initialize the output tensor with zeros only where needed to optimize memory usage
    n_total_tokens = n_prompt_tokens + len(attn_maps) - 1
    merged_attn = torch.zeros(
        (batch_size, n_heads, n_total_tokens, n_total_tokens),
        dtype=prefill_map.dtype,
        device=prefill_map.device,
        # Use sparse initialization to save memory for large attention maps
        layout=prefill_map.layout,
    )

    # Use in-place operation for copying the prefill attention map to avoid creating temporary tensors
    merged_attn[:, :, :n_prompt_tokens, :n_prompt_tokens].copy_(src=prefill_map)

    # Pre-calculate indices to avoid redundant computations in the loop
    prompt_end = n_prompt_tokens

    # Process decoding attention maps in a single batch where possible
    for i, attn_map in enumerate(attn_maps[1:], 1):
        pos = prompt_end + i - 1
        # Use narrow/slice operations instead of creating new tensors
        merged_attn[:, :, pos, : pos + 1] = attn_map[:, :, 0, : pos + 1]

    return merged_attn

    # def resize_attn_map(attn_map: torch.Tensor, image_size: tuple[int, int]):

    # def
    attn_map_np = attn_map.detach().cpu().numpy()
    attn_map_np = cv2.resize(attn_map_np, image_size, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(attn_map_np)


def get_visual_token_weight(
    vision_attn_weight,
    keep_percentage,
    weighting_type: Literal["linear", "uniform", "suppress"] = "linear",
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

    sorted_indices = torch.argsort(vision_attn_weight, descending=True)
    num_tokens_to_keep = int(len(vision_attn_weight) * keep_percentage)
    weight_vision_token = torch.zeros_like(vision_attn_weight, dtype=torch.float)
    weight_vision_token[sorted_indices[:num_tokens_to_keep]] = 1.0
    if weighting_type == "linear":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.linspace(
            lowest_weight, 1.0, len(vision_attn_weight) - num_tokens_to_keep
        )
    else:
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = lowest_weight
    return weight_vision_token
