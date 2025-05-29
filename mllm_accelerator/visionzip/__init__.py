"""
Copy from https://github.com/dvlab-research/VisionZip/blob/main/visionzip/main.py

Modified by Fanziyang-v
"""

from torch import nn

from .utils import clip_encoder_layer_forward, clip_attn_forward, apply_info
from .clip_encoder import clip_vision_tower_feature_select, clip_vision_tower_forward
# from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi, restore_image_features_sorted
from .llava_arch import encode_images

from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.model.llava_arch import LlavaMetaForCausalLM

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def visionzip(model: nn.Module, dominant: int = 108, contextual: int = 20):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant, contextual_num=contextual)

    # TODO: Support more MLLMs
    if isinstance(model, LlavaLlamaForCausalLM):
        LlavaMetaForCausalLM.encode_images = encode_images

        # Replace CLIP encoder layer and attention forward methods for obtaining key metrics.
        CLIPEncoderLayer.forward = clip_encoder_layer_forward
        CLIPAttention.forward = clip_attn_forward

        # Replace CLIPVisionTower methods for visual features compression.
        CLIPVisionTower.feature_select = clip_vision_tower_feature_select
        CLIPVisionTower.forward = clip_vision_tower_forward

    else:
        raise ValueError(f"VisionZip is not implemented for {type(model)}.")

    return model
