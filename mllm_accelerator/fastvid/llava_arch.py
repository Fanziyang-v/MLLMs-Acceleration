import torch
from torch.nn import functional as F

import math

def get_attn_2dPool(self, cls_attn_weights: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """Apply 2D pooling to the [cls] attention weights.

    Args:
        cls_attn_weights (torch.Tensor): The attention weights for the [cls] token, shape (num_frames, 1, num_tokens).
        stride (int): The stride for pooling. Default is 2.

    Returns:
        torch.Tensor: The pooled attention weights, shape (num_frames, num_tokens_per_frame).
    """
    height = width = self.get_vision_tower().num_patches_per_side
    num_frames = cls_attn_weights.shape[0]
    cls_attn_weights = cls_attn_weights.view(num_frames, 1, height, width).contiguous()
    if self.config.mm_spatial_pool_mode == "bilinear":
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        cls_attn_weights = F.interpolate(cls_attn_weights, size=scaled_shape, mode="bilinear")
    else:
        raise NotImplementedError(f"Unsupported mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
    return cls_attn_weights.view(num_frames, -1)
