import torch
from torch import nn

from .llava_arch import get_attn_2dPool, encode_images
from .mm_encoder_builder import build_vision_abstract
from .siglip_encoder import siglip_multihead_attention_pooling_head_forward
from .modeling_qwen2 import qwen2_model_forward

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipMultiheadAttentionPoolingHead
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

from .utils import llava_qwen_generate

def fastvid(model: nn.Module, dyseg_c: int = 8, dyseg_tau: float = 0.9, retention_ratio: float = 0.1, stprune_d: float = 0.4, dtm_alpha: float = 0.6, dtm_p: int = 4, k: int = 4):
    # Set FastVID parameters in the model
    fastvid_info = {
        "dyseg_c": dyseg_c,
        "dyseg_tau": dyseg_tau,
        "retention_ratio": retention_ratio,
        "stprune_d": stprune_d,
        "dtm_alpha": dtm_alpha,
        "dtm_p": dtm_p,
        "k": k,
    }
    print(f"FastVID parameters: {fastvid_info}")
    model.fastvid_info = fastvid_info
    model.model.fastvid_info = fastvid_info
    # TODO: Support more MLLMs.
    if isinstance(model, LlavaQwenForCausalLM):
        SigLipMultiheadAttentionPoolingHead.forward = siglip_multihead_attention_pooling_head_forward
        LlavaMetaForCausalLM.get_attn_2dPool = get_attn_2dPool
        LlavaMetaForCausalLM.encode_images = encode_images
        LlavaQwenForCausalLM.generate = llava_qwen_generate
        Qwen2Model.forward = qwen2_model_forward
        # load Siglip Head.
        vision_tower_abstract = build_vision_abstract(model.config, delay_load=getattr(model.config, "delay_load", False))
        vision_tower_abstract.to(device="cuda", dtype=torch.float16) # Move to the same device as the model.
        model.vision_tower_abstract = vision_tower_abstract
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}.")
    return model
