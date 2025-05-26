import torch
from torch import nn

from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

from .modeling_qwen2 import qwen2_model_forward
from .llava_arch import prepare_inputs_labels_for_multimodal
from .utils import llava_qwen_generate


# TODO: implement prunevid wrapper function.
def prunevid(model: nn.Module, temporal_segment_ratio: float = 0.25, k: int = 7, threshold: float = 0.8, cluster_ratio: float = 0.5, retention_ratio: float = 0.4, selected_layer: int = 10):
    """
    Apply PruneVid to the model.

    Args:
        model (nn.Module): The model to apply PruneVid.
        temporal_segment_ratio (float): Ratio of temporal segments.
        k (int): Number of neighbours in DPC-kNN.
        threshold (float): Similarity threshold to distinguish static and dynamic tokens.
        cluster_ratio (float): Ratio of the number of clusters in each segment.
        selected_layer (int): Layer index for pruning.

    Returns:
        nn.Module: The modified model with PruneVid applied.
    """
    prunevid_info = {
        "temporal_segment_ratio": temporal_segment_ratio,
        "k": k,
        "threshold": threshold,
        "cluster_ratio": cluster_ratio,
        "selected_layer": selected_layer,
        "rentention_ratio": retention_ratio,
    }
    model.prunevid_info = prunevid_info  # Set PruneVid info in the model
    model.model.prunevid_info = prunevid_info  # Set PruneVid info in the underlying model

    # TODO: Support more MLLMs
    if isinstance(model, LlavaQwenForCausalLM):
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
        LlavaQwenForCausalLM.generate = llava_qwen_generate
    else:
        raise NotImplementedError(f"PruneVid is not implemented for {type(model)}")
    return model
