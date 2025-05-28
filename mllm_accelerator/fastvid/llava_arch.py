#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ----------------------------------------------------------------------------
# Modified by Fanziyang-v
# ----------------------------------------------------------------------------
import torch
from torch.nn import functional as F

import math

import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print
from llava.model.llava_arch import unpad_image

import re
import random


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


def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)
    num_frames, num_tokens_per_frame, _ = image_features.shape
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    # ! FastVID: obtain global frame features and frame attention weights
    global_features, frame_attn_weights = self.vision_tower_abstract(image_features)
    frame_attn_weights = self.get_attn_2dPool(frame_attn_weights)
    num_frames, num_tokens_per_frame = frame_attn_weights.shape # (num_frames, 1, num_tokens_per_frame)

    image_features = self.get_model().mm_projector(image_features)

    if not hasattr(self, "fastvid_info"):
        raise ValueError("fastvid_info is not set while FastVID is activated.")
    # Store FastVID information
    self.fastvid_info.update(
        {
            "num_frames": num_frames,
            "num_tokens_per_frame": num_tokens_per_frame,
            "global_features": global_features,
            "frame_attn_weights": frame_attn_weights,
        }
    )

    return image_features
