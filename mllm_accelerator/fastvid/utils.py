from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def dyseg(global_features: torch.Tensor, dyseg_c: int, dyseg_tau: float) -> torch.Tensor:
    """Apply Dynamic Temporal Segmentation (DySeg) to partition video frames into segments.

    1. Calculate transition similarities (cosine similarity) between adjacent frames
    2. Select the indices of c-1 frames with the lowest transition similarities
    3. Select the indices of frames with transition similarities below the similarity threshold

    Args:
        global_features (torch.Tensor): Global features of shape (num_frames, feat_dim).
        dyseg_c (int): The number of frames in frame selection with lowest transition similarities.
        dyseg_tau (float): Similarity threshold for segmenting frames based on similarity.

    Returns:
        torch.Tensor: Segment lengths of shape (num_segments,).
    """
    num_frames = global_features.shape[0]
    normed_global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.sum(normed_global_features[:-1, :] * normed_global_features[1:, :], dim=-1)  # (num_frames - 1,)

    cut_indices_topk = torch.topk(similarities, k=dyseg_c - 1, largest=False).indices
    cut_indices_sim = torch.where(similarities < dyseg_tau)[0]
    cut_indices = torch.cat([cut_indices_topk, cut_indices_sim], dim=0).sort().values
    cut_indices = torch.unique(cut_indices)

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    segment_lengths = torch.diff(padded_cut_indices, n=1, dim=0)
    return segment_lengths


def select_tokens_by_attn(image_features: torch.Tensor, frame_attn_weights: torch.Tensor, num_retained_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select salient tokens based on attention weights (a.k.a. Attention Token Selection).
    
    Args:
        image_features (torch.Tensor): Image features of shape (num_frames, num_tokens, feat_dim).
        frame_attn_weights (torch.Tensor): Attention weights of shape (num_frames, num_tokens).
        num_retained_tokens (int): Number of tokens for each frame to retain based on attention weights.
    
    Returns:
        Tuple: A tuple containing:
            - salient_features (torch.Tensor): Selected tokens of shape (num_frames, num_retained_tokens, feat_dim).
            - keep_indices (torch.Tensor): Indices of the selected tokens of shape (num_frames, num_retained_tokens).
    """
    # Obtain the number of frames and tokens per frame
    num_frames, num_tokens, feat_dim = image_features.shape
    keep_indices = torch.topk(frame_attn_weights, k=num_retained_tokens, dim=-1).indices
    salient_features = torch.gather(image_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return salient_features, keep_indices


def dtm():
    pass
