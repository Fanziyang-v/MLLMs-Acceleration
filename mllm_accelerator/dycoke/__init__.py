import torch
from torch import nn

from .utils import llava_qwen_generate
from .llava_arch import prepare_inputs_labels_for_multimodal
from .modeling_qwen2 import qwen2_model_forward
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

def dycoke(model: nn.Module, merging_ratio: float = 0.3, pruning_ratio: float = 0.3, pruning_layer: int = 3, threshold: float = 0.9) -> nn.Module:
    """Apply DyCoke to the model.

    Args:
        model (nn.Module): The model to apply DyCoke.
        merging_ratio (float): Ratio of tokens to merge in TTM.
        pruning_ratio (float): Ratio of tokens to prune in DP.
        pruning_layer (int): Layer index for pruning.
        threhold (float): Threshold for dynamic pruning.

    Returns:
        nn.Module: The modified model with DyCoke applied.
    """

    dycoke_info = {
        "merging_ratio": merging_ratio,
        "pruning_ratio": pruning_ratio,
        "pruning_layer": pruning_layer,
        "threshold": threshold
    }

    if isinstance(model, LlavaQwenForCausalLM):
        model.dycoke_info = dycoke_info # Set DyCoke info in LlavaQwen
        model.model.dycoke_info = dycoke_info # Set Dycoke info in Qwen2
        LlavaQwenForCausalLM.generate = llava_qwen_generate
        Qwen2Model.forward = qwen2_model_forward
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
    else:
        raise NotImplementedError(f"DyCoke is not implemented for {type(model)}")

    return model
