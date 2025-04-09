import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoProcessor,
)
from typing import Tuple, Union, List, Dict, Any, Literal

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateDecoderOnlyOutput
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

# from deepseek_vl.utils.io import load_pil_images
from PIL import Image

DEEPSEEK_1_3B_MODEL_ID = "deepseek-ai/deepseek-vl-1.3b-chat"
DEEPSEEK_7B_MODEL_ID = "deepseek-ai/deepseek-vl-7b-chat"


def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.

    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.

    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)

    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = (
            min(int(bbox_size * ratio / block_size[0]), att_map.shape[1]),
            min(int(bbox_size * ratio / block_size[1]), att_map.shape[0]),
        )
        if att_map.shape[1] - block_num[0] < 1 and att_map.shape[0] - block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))

        # attention aggregation map
        sliding_att = np.zeros(
            (att_map.shape[0] - block_num[1] + 1, att_map.shape[1] - block_num[0] + 1)
        )
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1] - block_num[0] + 1):
            for y in range(att_map.shape[0] - block_num[1] + 1):
                att = att_map[y : y + block_num[1], x : x + block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)

        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] - 1])
        if max_att_pos[0] < sliding_att.shape[1] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0] + 1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1] - 1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0] - 1:
            adjcent_atts.append(sliding_att[max_att_pos[1] + 1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]

    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)

    x_center = (
        selected_bbox_size // 2 if x_center < selected_bbox_size // 2 else x_center
    )
    y_center = (
        selected_bbox_size // 2 if y_center < selected_bbox_size // 2 else y_center
    )
    x_center = (
        image_size[0] - selected_bbox_size // 2
        if x_center > image_size[0] - selected_bbox_size // 2
        else x_center
    )
    y_center = (
        image_size[1] - selected_bbox_size // 2
        if y_center > image_size[1] - selected_bbox_size // 2
        else y_center
    )

    x1 = max(0, x_center - selected_bbox_size // 2)
    y1 = max(0, y_center - selected_bbox_size // 2)
    x2 = min(image_size[0], x_center + selected_bbox_size // 2)
    y2 = min(image_size[1], y_center + selected_bbox_size // 2)

    return x1, y1, x2, y2


def reweighted_vision_tokens(
    vision_attn_weight,
    keep_percentage,
    keep_weight,
    weighting_type: Literal["linear", "exp", "uniform"] = "linear",
    lowest_weight=0.6,
):
    sorted_indices = torch.argsort(vision_attn_weight, descending=True)
    num_tokens_to_keep = int(len(vision_attn_weight) * keep_percentage)
    weight_vision_token = torch.zeros_like(vision_attn_weight, dtype=torch.float)
    weight_vision_token[sorted_indices[:num_tokens_to_keep]] = keep_weight
    if weighting_type == "linear":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.linspace(
            lowest_weight, 1.0, len(vision_attn_weight) - num_tokens_to_keep
        )
    elif weighting_type == "uniform":
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = lowest_weight
    else:
        weight_vision_token[sorted_indices[num_tokens_to_keep:]] = torch.exp(
            torch.linspace(
                0, math.log(lowest_weight), len(sorted_indices) - num_tokens_to_keep
            )
        )
    return weight_vision_token


class DeepseekVLForAttnExtraction(nn.Module):
    def __init__(self, model: MultiModalityCausalLM = None):
        super().__init__()
        self.model = (
            model
            if model is not None
            else MultiModalityCausalLM.from_pretrained(
                DEEPSEEK_1_3B_MODEL_ID, torch_dtype="auto"
            )
        )
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            DEEPSEEK_1_3B_MODEL_ID, use_fast=True
        )  # type: ignore
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    def get_mean_attn(self, attn_map: torch.Tensor):
        """
        Get mean attention across all layers

        Args:
            attn_map: attention map, shape: [layer, n_visual_tokens_per_image]

        Returns:
            Mean attention tensor, shape: [n_visual_tokens_per_image]
        """
        return torch.mean(attn_map, dim=0)

    def get_contrastive_attn(
        self, attn_map: torch.Tensor, target_layers=(14, 8), rectify: bool = True
    ):
        """
        Get contrastive attention by subtracting one layer from another

        Args:
            attn_map: attention map, shape: [layer, n_visual_tokens_per_image]
            target_layers: Tuple of two layers to contrast (default: first and last)

        Returns:
            Contrastive attention scores, shape: [n_visual_tokens_per_image]
        """
        layer_h = target_layers[0]
        layer_l = target_layers[1]

        attn_map = attn_map[layer_h] - attn_map[layer_l]

        if rectify:
            attn_map = torch.clamp(attn_map, min=0)

        return attn_map

    def resize_attn_map_to_image(
        self, attn_map: torch.Tensor, image_size: Tuple[int, int]
    ):
        """
        Resize the attention map to the image size
        Args:
            attn_map: attention map, shape: [n_visual_tokens_per_image]
            image_size: size of the image, tuple of (height, width)
        Returns:
            resized attention map, shape: [height, width]
        """
        return attn_map.reshape(image_size)

    @staticmethod
    def merge_layer_attn_map(attn_outputs: tuple[torch.Tensor]) -> torch.Tensor:
        """
        Merge the attention maps of the layer method
        Args:
            attn_outputs: tuple of attention maps, shape: layer x (batch, head, prefill_len, prefill_len)
        Returns:
            merged attention map, shape: (layer, batch, head, prefill_len, prefill_len)
        """
        return torch.stack(attn_outputs, dim=0)

    @staticmethod
    def merge_generate_attn_map(
        attn_outputs: tuple[tuple[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Merge the attention maps of the generate method
        Args:
            attn_outputs: tuple of tuple of attention maps, shape: decode_len x layer x (batch, head, query_len, key_len)
        Returns:
            merged attention map, shape: (layer, batch, head, n_total_tokens - 1, n_total_tokens - 1)
        """
        dtype = attn_outputs[0][0].dtype
        device = attn_outputs[0][0].device
        layout = attn_outputs[0][0].layout

        batch_size, n_heads, prefill_len, _ = attn_outputs[0][0].shape
        decode_len = len(attn_outputs)
        layers = len(attn_outputs[0])

        total_tokens = prefill_len + decode_len

        merged_attn = torch.zeros(
            (layers, batch_size, n_heads, total_tokens - 1, total_tokens - 1),
            dtype=dtype,
            device=device,
            layout=layout,
        )
        # Prefill attention special handling
        prefill_map = DeepseekVLForAttnExtraction.merge_layer_attn_map(attn_outputs[0])
        merged_attn[:, :, :, :prefill_len, :prefill_len] = prefill_map

        # Process decoding attention maps in a single batch where possible
        for i, attn_map in enumerate(attn_outputs[1:]):
            pos = prefill_len + i
            attn_map = DeepseekVLForAttnExtraction.merge_layer_attn_map(attn_map)
            # Use narrow/slice operations instead of creating new tensors
            merged_attn[:, :, :, pos, : pos + 1] = attn_map[:, :, :, 0, : pos + 1]

        return merged_attn

    @staticmethod
    def get_visual_attn_map(
        full_attn_map: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """
        Get the visual attention map from the full attention map
        Args:
            full_attn_map: the full attention map, shape: (layer, batch, head, n_total_tokens, n_total_tokens)
            vision_start_token_id: the start token id of the visual tokens
            vision_end_token_id: the end token id of the visual tokens
        Returns:
            the visual attention map, shape: image x (layer, head, n_total_tokens, n_visual_tokens_per_image)
        """
        return (full_attn_map[:, 0, :, :, inputs["images_seq_mask"][0].cpu()],)

    @torch.no_grad()
    def extract_attention(
        self,
        text_prompt: str,
        images: Union[List[Image.Image], Image.Image],
        attn_type: str = "mean",
        single_token_generation: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Single function interface to extract attention maps from the model

        Args:
            text_prompt: input text prompt
            images: one or more PIL images
            attn_type: type of attention to extract - "mean" or "contrastive"
            single_token_generation: whether to generate just a single token or full response
            **kwargs: additional arguments including:
                - contrast_layers: tuple of (layer1, layer2) for contrastive attention

        Returns:
            attention_maps: extracted attention maps for the visual tokens
        """
        # Handle single image input
        if not isinstance(images, list):
            images = [images]

        # Process inputs
        # print(text_prompt)
        conversation = [
            {
                "role": "User",
                "content": "".join(["<image_placeholder>"] * len(images)) + text_prompt,
            }
        ]

        prepare_inputs = self.processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(self.model.device, self.model.dtype)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate outputs with attention
        if single_token_generation:
            outputs: CausalLMOutputWithPast = self.model.language_model(  # type: ignore
                inputs_embeds=inputs_embeds,
                output_attentions=True,
            )  # type: ignore
            forward_attn_outputs: tuple[torch.Tensor] | None = outputs.attentions  # type: ignore
            if forward_attn_outputs is None:
                raise ValueError("No attention outputs found")
            # Return: layer x [b h n n]
            attn_outputs: torch.Tensor = (
                DeepseekVLForAttnExtraction.merge_layer_attn_map(forward_attn_outputs)
                .detach()
                .float()
                .cpu()
            )
        else:
            outputs: GenerateDecoderOnlyOutput = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=30,
                output_attentions=True,
                return_dict_in_generate=True,
                num_beams=1,
                use_cache=True,
            )  # type: ignore
            generate_attn_outputs: tuple[tuple[torch.Tensor]] | None = (
                outputs.attentions
            )
            if generate_attn_outputs is None:
                raise ValueError("No attention outputs found")
            attn_outputs: torch.Tensor = (
                DeepseekVLForAttnExtraction.merge_generate_attn_map(
                    generate_attn_outputs
                )
                .detach()
                .float()
                .cpu()
            )

        # return attn_outputs

        # Extract the visual embedding attention
        # Get vision token IDs
        prefill_len = prepare_inputs["input_ids"].shape[1]

        visual_attn = self.get_visual_attn_map(
            attn_outputs, prepare_inputs
        )  # image x [layer, head, n_total_tokens, n_visual_tokens_per_image]

        result_attn = []

        for i, image_attn in enumerate(visual_attn):
            # slice the decoding part
            image_attn = image_attn[:, :, prefill_len - 1 :, :]
            # average over the head and n_total_tokens
            image_attn = image_attn.mean(dim=(1, 2))
            # [layer, n_visual_tokens_per_image]
            h, w = prepare_inputs["pixel_values"].shape[-2:]
            h = h // 16
            w = w // 16
            # Extract the requested attention type
            if attn_type == "mean":
                result_attn.append(
                    self.resize_attn_map_to_image(
                        self.get_mean_attn(image_attn), (h, w)
                    )
                )
            elif attn_type == "contrastive":
                contrast_layers = kwargs.get("contrast_layers", (14, 4))
                result_attn.append(
                    self.resize_attn_map_to_image(
                        self.get_contrastive_attn(image_attn, contrast_layers), (h, w)
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported attention type: {attn_type}. Choose 'mean' or 'contrastive'"
                )

        return result_attn

    def forward(self):
        raise NotImplementedError(
            "Qwen2VLForAttnExtraction does not support forward pass"
        )


if __name__ == "__main__":
    model = DeepseekVLForAttnExtraction().to("cuda")
    attn = model.extract_attention(
        "A photo of a cat",
        Image.open("img1.jpg"),
        attn_type="mean",
        single_token_generation=True,
    )
    print(attn[0].shape)
