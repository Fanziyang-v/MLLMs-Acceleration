"""
Copy from https://github.com/dvlab-research/VisionZip/blob/main/visionzip/clip_encoder.py

Modified by Fanziyang-v

1. Remove unnecessary code
2. Refactor the core code of VisionZip
3. Add detail comments
"""

import torch

from torch.nn import functional as F


def clip_vision_tower_feature_select(self, image_forward_outs):
    # ! 0. Obtain image features and attentions.
    image_features = image_forward_outs.hidden_states[self.select_layer]  # [bsz, seq_len + 1, hidden_size]
    #####################################################################
    #                           VisionZip -- START HERE                 #
    #####################################################################
    if not hasattr(self.vision_tower.vision_model.encoder.layers[self.select_layer], "metric"):
        raise AssertionError(f"Metric for token merging in VisionZip doesn't exist in `{self.select_layer}`th encoder layer.")
    if not hasattr(self.vision_tower.vision_model.encoder.layers[self.select_layer], "attentions"):
        raise AssertionError(f"Attentions for token merging in VisionZip doesn't exist in `{self.select_layer}`th encoder layer.")
    metric = self.vision_tower.vision_model.encoder.layers[self.select_layer].metric
    image_attentions = self.vision_tower.vision_model.encoder.layers[self.select_layer].attentions  # [bsz, num_heads, seq_len + 1, seq_len + 1]
    num_dominant_tokens = self.vision_tower._info["dominant"]
    num_contextual_tokens = self.vision_tower._info["contextual"]

    #!  1. Dominant Token Selection
    hidden_size = image_features.shape[-1]
    # (1) Sum attention over all heads
    image_attentions = torch.sum(image_attentions, dim=1)  # [bsz, seq_len + 1, seq_len + 1]

    # (2) Obtain [cls] attention
    cls_attention = image_attentions[:, 0, 1:]  # [bsz, seq_len]

    # (3) Obtain the indices with Top-K [cls] attention values
    device = cls_attention.device
    bsz, seq_len = cls_attention.shape
    dominant_token_indices = torch.topk(cls_attention, k=num_dominant_tokens - 1, dim=-1).indices + 1
    dominant_token_indices = torch.cat([torch.zeros((bsz, 1), dtype=torch.long, device=device), dominant_token_indices], dim=1)  # [bsz, num_dominant_tokens, hidden_size]
    mask = torch.ones_like(image_features[:, :, 0], dtype=torch.bool, device=device).scatter_(1, dominant_token_indices, src=torch.zeros_like(image_features[:, :, 0], dtype=torch.bool, device=device))  # always select [CLS] token

    # (4) Select dominant tokens
    dominant_tokens = torch.masked_select(image_features, ~mask.unsqueeze(-1)).view(bsz, num_dominant_tokens, hidden_size)  # [bsz, num_dominant_tokens, hidden_size]

    #! 2. Contextual Token Merging
    num_remaining_tokens = seq_len + 1 - num_dominant_tokens
    num_target_tokens, num_other_tokens = num_contextual_tokens, num_remaining_tokens - num_contextual_tokens
    # (1) Obtain L2 normed metrics(key states) of the remaining tokens.
    filtered_metric = metric[mask].view(bsz, num_remaining_tokens, metric.shape[2])  # [bsz, num_remaining_tokens, head_dim]
    filtered_hidden_states = torch.masked_select(image_features, mask.unsqueeze(-1)).view(bsz, num_remaining_tokens, hidden_size)  # [bsz, num_remaining_tokens, hidden_size]
    normed_metric = filtered_metric / torch.norm(filtered_metric, p=2, dim=-1, keepdim=True)  # [bsz, num_remaining_tokens, head_dim]

    # (2) Divide remaining tokens in two groups: target-tokens and tokens-to-merge
    step = max(1, num_remaining_tokens // num_contextual_tokens)
    target_indices = torch.arange(0, num_remaining_tokens, step, device=normed_metric.device)[:num_contextual_tokens]
    # metrics
    target_tokens_metric = normed_metric[:, target_indices, :]
    tokens_to_merge_metric = normed_metric[:, ~torch.isin(torch.arange(num_remaining_tokens, device=device), target_indices), :]
    # hidden states
    target_tokens = filtered_hidden_states[:, target_indices, :]
    tokens_to_merge = filtered_hidden_states[:, ~torch.isin(torch.arange(num_remaining_tokens, device=device), target_indices), :]

    # (3) Cosine similarities
    similarities = torch.bmm(tokens_to_merge_metric, target_tokens_metric.transpose(1, 2))  # [bsz, num_other_tokens, num_target_tokens]

    # # (4) Obtain assigned matrix
    cluster_indices = similarities.argmax(dim=-1)  # [bsz, num_target_tokens]
    assigned_one_hot = F.one_hot(cluster_indices, num_classes=num_target_tokens).to(tokens_to_merge.dtype)  # [bsz, num_other_tokens, num_target_tokens]

    counts = torch.sum(assigned_one_hot, dim=1).clamp(min=1).unsqueeze(-1)  # [bsz, num_target_tokens, 1]

    # (5) Aggregate tokens by averaging each group's features.
    aggregated_tokens = torch.bmm(assigned_one_hot.transpose(1, 2), tokens_to_merge) / counts
    contextual_tokens = target_tokens + aggregated_tokens  # [bsz, num_contextual_tokens, hidden_size]

    #! 3. Combine dominant and contextual tokens
    image_features = torch.cat([dominant_tokens, contextual_tokens], dim=1)
    return image_features, dominant_token_indices
    #####################################################################
    #                           VisionZip -- END HERE                   #
    #####################################################################


@torch.no_grad()
def clip_vision_tower_forward(self, images):
    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
    image_features, keep_indices = self.feature_select(image_forward_outs)
    return image_features, keep_indices
