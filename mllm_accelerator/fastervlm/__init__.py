import torch
from torch import nn

from .clip_encoder import clip_vision_tower_feature_select, clip_vision_tower_forward 

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

def fastervlm(model: nn.Module, retention_ratio: float = 0.5) -> nn.Module:
    """
    Apply FasterVLM to the model(e.g. LlavaLlama).

    Args:
        model (nn.Module): The MLLM to be modified.
        retention_ratio (float): The retention ratio of visual tokens.

    Returns:
        nn.Module: The modified model with FasterVLM applied.
    """
    # TODO: Support more MLLMs
    if isinstance(model, LlavaLlamaForCausalLM):
        CLIPVisionTower.forward = clip_vision_tower_forward
        CLIPVisionTower.feature_select = clip_vision_tower_feature_select
        model.model.vision_tower.retention_ratio = retention_ratio
    else:
        raise NotImplementedError(f"FasterVLM is not implemented for {type(model)}")
    return model
