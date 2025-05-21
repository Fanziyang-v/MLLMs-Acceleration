import torch

def clip_vision_tower_feature_select(self, image_forward_outs):
    if not hasattr(self, "retention_ratio"):
        raise ValueError("Retention ratio is not set while FasterVLM is applied.")

    image_features = image_forward_outs.hidden_states[self.select_layer]
    image_attentions = image_forward_outs.attentions[self.select_layer]
    num_visual_tokens = image_features.shape[1] - 1
    num_retained_tokens = int(num_visual_tokens * self.retention_ratio)

    cls_attn = image_attentions[:, :, 0, 1:] # [bsz, n_head, seq_len]
    cls_attn = torch.mean(cls_attn, dim=1)[0] # [seq_len]
    topk_indices = cls_attn.topk(k=num_retained_tokens, dim=-1).indices.sort().values # [num_retained_tokens]
    image_features = image_features[:, topk_indices + 1, :] # [bsz, num_retained_tokens, hidden_size]

    return image_features

def clip_vision_tower_forward(self, images):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_attentions=True, output_hidden_states=True)
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            image_features.append(image_feature)
    else:
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_attentions=True, output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

    return image_features
