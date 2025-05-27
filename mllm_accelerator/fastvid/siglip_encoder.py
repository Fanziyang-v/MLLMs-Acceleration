import torch
from torch import nn

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel

from llava.utils import rank0_print

def load_model(self, device_map=None):
    if self.is_loaded:
        rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
        return

    self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

    # ! FastVID: Store SigLip head for global frame features extraction.
    self.siglip_head = self.vision_tower.vision_model.head

    del self.vision_tower.vision_model.encoder.layers[-1:]    
    self.vision_tower.vision_model.head = nn.Identity()
    self.vision_tower.requires_grad_(False)

    self.is_loaded = True


def siglip_multihead_attention_pooling_head_forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_state.shape[0]
    probe = self.probe.repeat(batch_size, 1, 1)

    hidden_state, attn_weights = self.attention(probe, hidden_state, hidden_state)

    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)

    # ! FastVID: Return [CLS] tokens (global frame features) and [CLS] attentions
    return hidden_state[:, 0], attn_weights
