# Based on https://github.com/KD-TAO/DyCoke/blob/main/llava/model/language_model/modeling_qwen2.py
from typing import Tuple, Optional

import torch
from transformers.cache_utils import DynamicCache


class DyCokeCache(DynamicCache):
    def __init__(self, pruning_ratio: float, visual_token_start_index: int, visual_token_length: int, threshold: float = 0.9) -> None:
        """Initialize DyCokeCache.

        Args:
            pruning_ratio (float): The pruning ratio of Dynamic Pruning(DP) in DyCoke.
            visual_token_start_index (int): The start index of visual tokens.
            visual_token_length (int): The length of visual tokens.
            threshold (float): The threshold for dynamic pruning. Default is 0.9.
        """
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.visual_token_start_index = visual_token_start_index
        self.visual_token_length = visual_token_length
        self.threshold = threshold
        self.last_img_attn = None
        self.kv_cache = None

    def filter_cache(self, layer_idx: int, indices: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if indices is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        key_cache, value_cache = self.key_cache[layer_idx], self.value_cache[layer_idx]
        bsz, num_heads, seq_len, head_dim = key_cache.size()
        # Filter the cache based on the indices
        key_cache = torch.gather(key_cache, dim=-2, index=indices.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim))
        value_cache = torch.gather(value_cache, dim=-2, index=indices.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim))
        return key_cache, value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().update(key_states, value_states, layer_idx, cache_kwargs)
        return self.filter_cache(layer_idx, self.kv_cache)

    def update_cache(self, img_attn: torch.Tensor) -> None:
        """Update DyCoke cache when the cosine similarity of `img_attn` and `last_img_attn` is less than a threshold.

        Args:
            img_attn (torch.Tensor): Image attention tensor of shape (visual_token_length,)
        """
        total_length = self.get_seq_length()
        num_retained_tokens = int(self.visual_token_length * (1 - self.pruning_ratio))
        keep_indices = self.visual_token_start_index + torch.topk(img_attn, k=num_retained_tokens, dim=-1).indices.sort().values
        all_indices = torch.arange(total_length, device=keep_indices.device, dtype=keep_indices.dtype)
        keep_indices = torch.cat([all_indices[: self.visual_token_start_index], keep_indices, all_indices[self.visual_token_start_index + self.visual_token_length :]])
        # Record the keeping indices
        self.kv_cache = keep_indices

    def reset(self) -> None:
        self.kv_cache = None

    def update_img_attn(self, img_attn: torch.Tensor) -> None:
        """Update the last image attention.

        Args:
            img_attn (torch.Tensor): Image attention tensor of shape (visual_token_length,)
        """
        self.last_img_attn = img_attn

    @classmethod
    def from_dynamic_cache(cls, cache: DynamicCache, pruning_ratio: float, visual_token_start_index: int, visual_token_length: int, threshold: float = 0.9) -> "DyCokeCache":
        # Create DyCokeCache instance
        dycoke_cache = cls(pruning_ratio, visual_token_start_index, visual_token_length, threshold)
        # Copy properties from dynamic cache.
        dycoke_cache._seen_tokens = cache._seen_tokens
        dycoke_cache.key_cache = [key_states.clone() for key_states in cache.key_cache]
        dycoke_cache.value_cache = [value_states.clone() for value_states in cache.value_cache]
        return dycoke_cache
