from typing import Optional

import math

import torch

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import logger

def llama_model_forward(
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

    _output_attentions = output_attentions
    seq_length = hidden_states.shape[1]
    is_prefill = seq_length > 1
    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if is_prefill: # PyramidDrop only applies visual token pruning in the prefilling stage.
            if not hasattr(self, "pdrop_info"):
                raise ValueError("`pdrop_info` is not set while FastV is activated.")
            # Obtain PyramidDrop parameters
            pruning_layers = self.pdrop_info["pruning_layers"]
            retention_ratio = self.pdrop_info["retention_ratio"]
            visual_token_start_index = self.pdrop_info["visual_token_start_index"]
            visual_token_length = self.pdrop_info["visual_token_length"]
            visual_token_end_index = visual_token_start_index + visual_token_length
            num_retained_tokens = math.ceil(retention_ratio * visual_token_length)
            if layer_idx + 1 in pruning_layers:
                output_attentions = True
            elif layer_idx in pruning_layers:
                output_attentions = _output_attentions
                attns = layer_outputs[1] # (bsz, n_heads, seq_len, seq_len)
                attns = attns[:, :, -1, visual_token_start_index:visual_token_end_index] # (bsz, n_heads, visual_token_length)
                # ! Assume that batch_size is 1.
                attns = torch.mean(attns, dim=1)[0] # (visual_token_length,)
                topk_indices = visual_token_start_index + attns.topk(k=num_retained_tokens, dim=-1).indices.sort().values
                # Update position_ids, causal_mask, hidden_states and cache_position.
                all_indices = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                keep_indices = torch.cat([all_indices[:visual_token_start_index], topk_indices, all_indices[visual_token_end_index:]], dim=-1)
                hidden_states = hidden_states[:, keep_indices, :]
                position_ids = position_ids[:, keep_indices]
                position_embeddings = (position_embeddings[0][:, keep_indices, :], position_embeddings[1][:, keep_indices, :])
                new_seq_length = hidden_states.shape[1]
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, :new_seq_length, :new_seq_length]
                cache_position = position_ids.squeeze(0)
                self.pdrop_info["visual_token_length"] = num_retained_tokens
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
