from torch import nn

from .utils import llava_generate
from .modeling_llama import llama_model_forward

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

def fastv(model: nn.Module, fastv_k: int = 2, fastv_r: float = 0.5) -> nn.Module:
    """
    Apply FastV to the model(e.g. LlavaLlama).

    Args:
        model (nn.Module): The model to be modified.
        fastv_k (int): The number of layers to apply FastV.
        fastv_r (float): The retention ratio of visual tokens.

    Returns:
        nn.Module: The modified model with FastV applied.
    """
    # TODO: Support more MLLMs
    if isinstance(model, LlavaLlamaForCausalLM):
        LlamaModel.forward = llama_model_forward
        LlavaLlamaForCausalLM.generate = llava_generate
    else:
        raise NotImplementedError(f"FastV is not implemented for {type(model)}")

    # Initialize FastV parameters
    fastv_info = {
        "fastv_k": fastv_k,
        "fastv_r": fastv_r,
    }

    # Store FastV info in LLM
    model.model.fastv_info = fastv_info
    return model
