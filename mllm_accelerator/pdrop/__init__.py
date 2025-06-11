from typing import List

import torch
from torch import nn

from .llava_llama import llava_llama_generate
from .modeling_llama import llama_model_forward

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

from transformers.models.llama.modeling_llama import LlamaModel


def pdrop(model: nn.Module, pruning_layers: List[int], retention_ratio: float = 0.5):
    """Apply PyramidDrop to the model.

    Args:
        model (nn.Module): The model to apply PyramidDrop.
        pruning_layers (List[int]): Pruning layer indices.
        retention_ratio (float): Retention ratio after pruning.

    Returns:
        nn.Module: The modified model with PruneVid applied.
    """
    # TODO: Support more MLLMs
    if isinstance(model, LlavaLlamaForCausalLM):
        LlavaLlamaForCausalLM.generate = llava_llama_generate
        LlamaModel.forward = llama_model_forward
    else:
        raise NotImplementedError(f"PyramidDrop is not implemented for {type(model)}")

    pdrop_info = {
        "pruning_layers": pruning_layers,
        "retention_ratio": retention_ratio,
    }
    # Store pdrop_info in MLLM
    model.pdrop_info = pdrop_info
    model.model.pdrop_info = pdrop_info

    return model
