import torch
from torch import nn

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel, SigLipImageProcessor, SigLipVisionConfig

from llava.utils import rank0_print

# def load_model(self, device_map=None):
#     if self.is_loaded:
#         rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
#         return

#     self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

#     # ! FastVID: Store SigLip head for global frame features extraction.
#     self.siglip_head = self.vision_tower.vision_model.head

#     del self.vision_tower.vision_model.encoder.layers[-1:]    
#     self.vision_tower.vision_model.head = nn.Identity()
#     self.vision_tower.requires_grad_(False)

#     self.is_loaded = True


# def restore_siglip_head(self, device_map=None)


def siglip_multihead_attention_pooling_head_forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_state.shape[0]
    probe = self.probe.repeat(batch_size, 1, 1)

    hidden_state, attn_weights = self.attention(probe, hidden_state, hidden_state)

    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)

    # ! FastVID: Return [CLS] tokens (global frame features) and [CLS] attentions
    return hidden_state[:, 0], attn_weights


class SigLipVisionTowerAbstract(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        if not delay_load:
            rank0_print(f"Loading vision tower abstract: {vision_tower}")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower_abstract = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        del self.vision_tower_abstract.vision_model.embeddings
        del self.vision_tower_abstract.vision_model.encoder
        self.vision_tower_abstract.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        last_hidden_state = self.vision_tower_abstract.vision_model.post_layernorm(images)
        pooled_output, attn_weights = self.vision_tower_abstract.vision_model.head(last_hidden_state)
        return pooled_output, attn_weights

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
