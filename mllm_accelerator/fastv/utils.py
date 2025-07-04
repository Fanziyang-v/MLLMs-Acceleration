from typing import Optional, Union, List

import torch

from transformers.generation.utils import GenerateOutput

from llava.constants import IMAGE_TOKEN_INDEX


@torch.no_grad()
def llava_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        if not hasattr(self.model, "fastv_info"):
            raise ValueError("`fastv_info` is not set in LLaVAModel while FastV is activated.")
        visual_token_start_index = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        visual_token_end_index = -(inputs.shape[1] - visual_token_start_index)
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        visual_token_length = (inputs_embeds.shape[1] + visual_token_end_index) - visual_token_start_index + 1
        # ! Store FastV info in LLM.
        self.model.fastv_info["visual_token_start_index"] = visual_token_start_index
        self.model.fastv_info["visual_token_length"] = visual_token_length
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
