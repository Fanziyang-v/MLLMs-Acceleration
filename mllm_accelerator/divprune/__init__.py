import torch
from torch import nn

from .llava_arch import encode_images

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def divprune(model: nn.Module, num_retained_tokens: int) -> nn.Module:
    """Apply diversity-based pruning to the model"""
    if not isinstance(model, LlavaLlamaForCausalLM):
        raise NotImplementedError(f"Model type {type(model)} is not supported for diversity-based pruning.")

    # Replace the `encode_images` method in LlavaMetaForCausalLM
    LlavaMetaForCausalLM.encode_images = encode_images

    # Set the divprune info
    model.divprune_info = {
        "num_retained_tokens": num_retained_tokens,
    }

    return model
