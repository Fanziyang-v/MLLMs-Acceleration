from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

import math

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import logger


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

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # Store output_attentions
    _output_attentions = output_attentions
    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        seq_length = hidden_states.shape[1]
        is_prefill = seq_length > 1  # seq_length > 1 denotes it is in prefilling stage.
        if is_prefill:
            if not hasattr(self, "prunevid_info"):
                raise ValueError("`prunevid_info` is not set while PruneVid is activated.")
            # Obtain PruneVid parameters
            pruning_layer = self.prunevid_info["pruning_layer"]
            pruning_ratio = self.prunvid_info["pruning_ratio"]
            visual_token_start_index = self.prunevid_info["visual_token_start_index"]
            visual_token_length = self.prunevid_info["visual_token_length"]
            visual_token_end_index = visual_token_start_index + visual_token_length
            num_retained_tokens = math.ceil(pruning_ratio * visual_token_length)
            if layer_idx == pruning_layer - 1:
                output_attentions = True
            elif layer_idx == pruning_layer:
                output_attentions = _output_attentions
                attn = layer_outputs[1]
                attn = attn[:, :, visual_token_end_index, visual_token_start_index:visual_token_end_index]  # (bsz, n_heads, n_text_tokens, visual_token_length)
                # Average across all attention heads
                attn = torch.mean(attn, dim=1)  # (bsz, n_text_tokens, visual_token_length)
                # Apply max pooling
                attn = torch.max(attn, dim=1).values[0]  # (visual_token_length)
                keep_indices = torch.topk(attn, k=num_retained_tokens, dim=-1).indices.sort().values + visual_token_start_index
                all_indices = torch.arange(seq_length, device=keep_indices.device, dtype=keep_indices.dtype)
                keep_indices = torch.cat([all_indices[:visual_token_start_index], keep_indices, all_indices[visual_token_end_index:]], dim=-1)
                hidden_states = hidden_states[:, keep_indices, :]
                new_seq_length = hidden_states.shape[1]
                position_ids = keep_indices.unsqueeze(0)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, new_seq_length]
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

        if _output_attentions:
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
