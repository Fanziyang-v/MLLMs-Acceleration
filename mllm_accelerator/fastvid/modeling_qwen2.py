from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

import math

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import logger

from .utils import dyseg, stprune


def qwen2_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    seq_length = hidden_states.shape[1]
    is_prefill = seq_length > 1 # seg_length > 1 indicates the prefilling stage.
    if is_prefill:
        # ! Apply FastVID here.
        if not hasattr(self, "fastvid_info"):
            raise ValueError("fastvid_info is not set while FastVID is activated.")

        # Obtain FastVID parameters
        DYSEG_C = self.fastvid_info["dyseg_c"]
        DYSEG_TAU = self.fastvid_info["dyseg_tau"]
        RETANTION_RATIO = self.fastvid_info["retention_ratio"]
        STPRUNE_D = self.fastvid_info["stprune_d"]
        DTM_ALPHA = self.fastvid_info["dtm_alpha"]
        DTM_P = self.fastvid_info["dtm_p"]

        global_features = self.fastvid_info["global_features"]
        frame_attn_weights = self.fastvid_info["frame_attn_weights"]
        visual_token_start_index = self.fastvid_info["visual_token_start_index"]
        visual_token_length = self.fastvid_info["visual_token_length"]
        num_frames = self.fastvid_info["num_frames"]
        num_tokens_per_frame = self.fastvid_info["num_tokens_per_frame"]
        if num_frames * num_tokens_per_frame != visual_token_length:
            raise AssertionError(f"Expected num_frames * num_tokens_per_frame ({num_frames} * {num_tokens_per_frame}) to equal visual_token_length ({visual_token_length}).")

        visual_token_end_index = visual_token_start_index + visual_token_length

        final_tokens = ()
        all_keep_indices = ()
        pre_visual_features = hidden_states[:, :visual_token_start_index, :].squeeze(0)
        post_visual_features = hidden_states[:, visual_token_end_index:, :].squeeze(0)
        pre_visual_indices = torch.arange(visual_token_start_index, device=hidden_states.device, dtype=torch.long)
        post_visual_indices = torch.arange(visual_token_end_index, seq_length, device=hidden_states.device, dtype=torch.long)
        final_tokens += (pre_visual_features, post_visual_features)
        all_keep_indices += (pre_visual_indices, post_visual_indices)

        # (num_frames, num_tokens_per_frame, hidden_size)
        video_features = hidden_states[:, visual_token_start_index:visual_token_end_index, :][0].view(num_frames, num_tokens_per_frame, -1)
        # 1. Apply DySeg
        segment_lengths = dyseg(global_features, DYSEG_C, DYSEG_TAU)

        # 2. Apply STPrune
        global_indices = torch.arange(visual_token_start_index, visual_token_end_index, device=hidden_states.device, dtype=torch.long).view(num_frames, num_tokens_per_frame)
        compressed_video_features, keep_global_indices = stprune(video_features, frame_attn_weights, segment_lengths, global_indices, RETANTION_RATIO, STPRUNE_D, DTM_ALPHA, DTM_P)

        final_tokens += (compressed_video_features,)
        all_keep_indices += (keep_global_indices,)

        hidden_states = torch.cat(final_tokens, dim=0)
        keep_indices = torch.cat(all_keep_indices, dim=0)
        
        sorted_indices = torch.argsort(keep_indices)
        hidden_states = hidden_states[sorted_indices].unsqueeze(0)
        keep_indices = keep_indices[sorted_indices]

        new_seq_length = hidden_states.shape[1]
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :new_seq_length, :new_seq_length]
        position_ids = keep_indices.unsqueeze(0)
        position_embeddings = (position_embeddings[0][:, keep_indices, :], position_embeddings[1][:, keep_indices, :])
        cache_position = keep_indices


    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
