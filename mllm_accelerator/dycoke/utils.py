# Based on https://github.com/KD-TAO/DyCoke/blob/main/llava/model/llava_arch.py
from typing import Optional, List, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.generation.utils import GenerateOutput

from .cache_utils import DyCokeCache

from llava.constants import IMAGE_TOKEN_INDEX


def dycoke_ttm(image_features: torch.Tensor, merging_ratio: float, num_tokens_per_frame: int = 196) -> torch.Tensor:
    """Token Temporal Merging(TTM) in DyCoke.

    Args:
        image_features (torch.Tensor): Image features of shape (num_frames * num_tokens_per_frame, feature_dim).
        merging_ratio (float): Ratio of tokens to merge.
        num_tokens_per_frame (int): Number of tokens per frame.

    Returns:
        torch.Tensor: Modified image features by TTM.
    """
    # Split frames into tokens
    num_frames = image_features.shape[0] // num_tokens_per_frame
    keeping_ratio = 1 - merging_ratio
    modified_image_features = []
    # Calculate similarities between adjacent even frames
    similarities = []
    for i in range(0, num_frames - 1, 2):
        # Get tokens for adjacent frames
        frame1_tokens = image_features[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_features[(i + 1) * num_tokens_per_frame : (i + 2) * num_tokens_per_frame]

        # Calculate cosine similarity between normalized tokens
        frame1_norm = F.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = F.normalize(frame2_tokens, p=2, dim=1)
        similarity = F.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)

    similarities = torch.stack([torch.tensor(similarity) for similarity in similarities])

    # Process even frames
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_features[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_features[(i + 1) * num_tokens_per_frame : (i + 2) * num_tokens_per_frame]

        avg_similarity = similarities[i // 2]
        num_retained_tokens = int(keeping_ratio * num_tokens_per_frame)
        num_retained_toknes = avg_similarity.topk(num_retained_tokens, largest=False).indices

        modified_image_features.append(frame1_tokens)
        modified_image_features.append(frame2_tokens[num_retained_toknes])

    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 3, 4):
        frame1_tokens = image_features[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_features[(i + 2) * num_tokens_per_frame : (i + 3) * num_tokens_per_frame]

        similarity = F.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    odd_similarities = torch.stack([torch.tensor(similarity) for similarity in odd_similarities])

    for i in range(0, num_frames - 3, 4):
        frame1_tokens = image_features[i * num_tokens_per_frame : (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_features[(i + 2) * num_tokens_per_frame : (i + 3) * num_tokens_per_frame]

        avg_similarity = odd_similarities[i // 4]
        num_retained_tokens = int(keeping_ratio * num_tokens_per_frame)
        num_retained_toknes = avg_similarity.topk(num_retained_tokens, largest=False).indices

        modified_image_features[i] = frame1_tokens
        modified_image_features[i + 2] = frame2_tokens[num_retained_toknes]
    # Testing tokens length of each frame
    # img_feat_lengths = [img_feat.shape[0] for img_feat in modified_image_features]
    # print("Tokens length of each frame:", img_feat_lengths)

    combined_tokens = torch.cat(modified_image_features, dim=0)
    return combined_tokens


def dycoke_dp(cache: DyCokeCache, img_attn: torch.Tensor) -> None:
    """Dynamic Pruning(DP) in DyCoke.

    Note: DyCoke only prunes visual tokens in decoding stage.

    Args:
        cache (DyCokeCache): Cache object containing the last image attention and configuration.
        img_attn (torch.Tensor): Image attention tensor of shape (visual_token_length,).
    """
    last_img_attn = cache.last_img_attn
    if last_img_attn is not None:
        # Calculate cosine similarity
        similarity = F.cosine_similarity(img_attn, last_img_attn, dim=0)
        if similarity < cache.threshold:
            cache.update_cache(img_attn)
    else:
        # Always prune the visual tokens in the first decoding stage.
        cache.update_cache(img_attn)
    # update last_img_attn
    cache.update_img_attn(img_attn)


@torch.no_grad()
def llava_qwen_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        visual_token_start_index = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        visual_token_end_index = -(inputs.shape[1] - visual_token_start_index)
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        visual_token_length = (inputs_embeds.shape[1] + visual_token_end_index) - visual_token_start_index
        # update dycoke_info
        self.dycoke_info.update({
            "visual_token_start_index": visual_token_start_index,
            "visual_token_length": visual_token_length
        })
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
