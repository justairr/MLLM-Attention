import sys
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.utils import DynamicCache
import warnings
from .deepseek_vl_attn import DeepseekVLForAttnExtraction, reweighted_vision_tokens
from .base import BaseModel
from ..smp import *


class DeepSeekVL(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl
        except Exception as e:
            logging.critical(
                'Please first install deepseek_vl from source codes in: https://github.com/deepseek-ai/DeepSeek-VL')
            raise e

    def __init__(self, model_path='deepseek-ai/deepseek-vl-1.3b-chat', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl.models import VLChatProcessor

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        self.model = model.cuda().eval()
        self.attn_extractor = DeepseekVLForAttnExtraction().cuda(1)
        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images = '', []
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    content += s['value']
            return content, images
        conversation = []
        if 'role' not in message[0]:
            content, images = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))
        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message)
        from deepseek_vl.utils.io import load_pil_images

        keep_perc = os.environ.get('KP', "0.4")
        keep_perc = float(keep_perc)
        keep_weight = os.environ.get('KW', "1.0")
        keep_weight = float(keep_weight)
        linear_start = os.environ.get('LS', "0.6")
        linear_start = float(linear_start)
        weighting_type = os.environ.get('WT', "linear")

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        attn_map = self.attn_extractor.extract_attention(
            message[-1]['value'],
            pil_images,
            attn_type="contrastive",
            single_token_generation=False,
            contrast_layers=(14, 8),
        )
        attn_flat = attn_map[0].flatten()
        weight = reweighted_vision_tokens(attn_flat, keep_percentage=keep_perc, keep_weight=keep_weight, weighting_type=weighting_type, lowest_weight=linear_start).to(self.model.device)

        with torch.no_grad():
            prompt_cache = DynamicCache()
            initial_output = self.model.language_model(inputs_embeds=inputs_embeds[:, :-1], past_key_values=prompt_cache)
            prompt_cache = initial_output.past_key_values

        for i, (k, v) in enumerate(zip(prompt_cache.key_cache, prompt_cache.value_cache, strict=True)):
            k[:, :, prepare_inputs['images_seq_mask'][0, :-1].cpu()] *= weight[None, None, :, None]
            v[:, :, prepare_inputs['images_seq_mask'][0, :-1].cpu()] *= weight[None, None, :, None]

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=prompt_cache,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs)
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)
