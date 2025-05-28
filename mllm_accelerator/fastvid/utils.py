from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

import math


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


# TODO: Implement STPrune (Spatial-Temporal Pruning)
def stprune(features: torch.Tensor, global_features: torch.Tensor, frame_attn_weights: torch.Tensor, segment_lengths: torch.Tensor, global_indices: torch.Tensor, dyseg_c: int, dyseg_tau: float, dtm_alpha: int, dtm_p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pass



def select_tokens_by_attn(image_features: torch.Tensor, frame_attn_weights: torch.Tensor, num_salient_tokens: int, global_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select salient tokens based on attention weights (a.k.a. Attention Token Selection).

    Args:
        image_features (torch.Tensor): Image features of shape (num_frames, num_tokens, feat_dim).
        frame_attn_weights (torch.Tensor): Attention weights of shape (num_frames, num_tokens).
        num_salient_tokens (int): Number of tokens for each frame to retain based on attention weights.
        global_indices (torch.Tensor): Global indices of the tokens of shape (num_frames, num_tokens).

    Returns:
        Tuple: A tuple containing:
            - salient_features (torch.Tensor): Selected tokens of shape (num_frames * num_salient_tokens, feat_dim).
            - keep_indices (torch.Tensor): Global indices of the selected tokens of shape (num_frames * num_salient_tokens,).
    """
    num_frames, num_tokens, feat_dim = image_features.shape
    keep_indices = torch.topk(frame_attn_weights, k=num_salient_tokens, dim=-1).indices  # (num_frames, num_salient_tokens)
    salient_features = torch.gather(image_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    salient_global_indices = torch.gather(global_indices, dim=1, index=keep_indices)  # (num_frames, num_salient_tokens)
    return salient_features.view(-1, feat_dim), salient_global_indices.view(-1)


@torch.no_grad()
def dtm(
    features: torch.Tensor, segment_lengths: torch.Tensor, num_contextual_tokens: int, global_indices: torch.Tensor, dtm_alpha: float, k: int = 4, step: int = 4, mask: Optional[torch.Tensor] = None
) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    """Apply Density-based Token Merging (DTM) to video frames.

    Args:
        features (torch.Tensor): Segment features of shape (num_frames, num_tokens, feat_dim).
        segment_lengths (torch.Tensor): Segment lengths of shape (num_segments,).
        num_contextual_tokens (int): Number of contextual tokens to select for each frame.
        global_indices (torch.Tensor): Global indices of the tokens of shape (num_frames, num_tokens).
        k (int): Number of nearest neighbors to consider for clustering. Default is 4.
        step (int): Step size for anchor frame selection. Default is 4.
        mask (Optional[torch.Tensor]): Optional mask to apply on features, preventing same token selection with Attention Token Selection.

    Returns:
        Tuple: A tuple containing:
            - all_tokens (Tuple[torch.Tensor]): Contextual tokens for each segment of shape (num_segments, num_contextual_tokens, feat_dim).
            - all_keep_indices (Tuple[torch.Tensor]): Indices of the selected contextual tokens of shape (num_segments, num_contextual_tokens).
    """
    bsz, _, feat_dim = features.shape
    device = features.device
    # validate inputs
    if segment_lengths.sum() != bsz:
        raise AssertionError(f"Expected segment lengths to sum to {bsz}, but got {segment_lengths.sum()}.")
    if mask is not None and mask.shape != features.shape[:2]:
        raise AssertionError(f"Expected mask shape to match features shape {features.shape[:2]}, but got {mask.shape}.")
    if global_indices.shape != features.shape[:2]:
        raise AssertionError(f"Expected global indices shape to match features shape {features.shape[:2]}, but got {global_indices.shape}.")

    all_keep_indices = ()
    all_tokens = ()
    # DO NOT consider the tokens selected by Attention Token Selection (ATS).
    if mask is not None:
        features = torch.masked_select(features, mask.unsqueeze(-1).expand(-1, -1, feat_dim)).view(bsz, -1, feat_dim)
        global_indices = torch.masked_select(global_indices, mask).view(bsz, -1)
    seq_len = features.shape[1]  # number of tokens in a frame after ATS.

    # 1. Calculate density score for each token.
    density_scores = calc_density_score(features, k=k)  # (bsz, seq_len)

    # 2. Process each segments.
    offset = 0
    for segment_length in segment_lengths:
        segment_start_index, segment_end_index = offset, offset + segment_length
        anchor_frame_indices = torch.arange(segment_length, step=step, device=device, dtype=torch.int64)

        # (1). Slice segment features, density scores, and global indices.
        segment_features = features[segment_start_index:segment_end_index, :, :]
        segment_density_scores = density_scores[segment_start_index:segment_end_index, :]
        segment_global_indices = global_indices[segment_start_index:segment_end_index, :]

        # (2). Allocate contextual tokens for anchor frames.
        num_segment_contextual_tokens = segment_length * num_contextual_tokens
        num_anchor_frames = (segment_length + step - 1) // step
        new_num_contextual_tokens = num_segment_contextual_tokens // num_anchor_frames

        # (3). Select anchor frames and their contextual tokens based on density scores.
        anchor_token_indices = torch.topk(segment_density_scores[anchor_frame_indices], k=new_num_contextual_tokens, dim=-1).indices  # (num_anchor_frames, new_num_contextual_tokens)
        keep_indices = segment_global_indices[anchor_frame_indices][anchor_token_indices]  # (num_anchor_frames, new_num_contextual_tokens)

        # (4). Merge features between anchor tokens and to-be-merged tokens.
        anchor_frame_features = segment_features[anchor_frame_indices]  # (num_anchor_frames, seq_len, feat_dim)
        to_be_merged_tokens = segment_features.view(-1, feat_dim)  # (segment_length * seq_len, feat_dim)
        anchor_tokens = anchor_frame_features[anchor_token_indices].view(-1, feat_dim)  # (num_anchor_frames * new_num_contextual_tokens, feat_dim)
        # Normalize features
        to_be_merged_tokens = to_be_merged_tokens / to_be_merged_tokens.norm(p=2, dim=-1, keepdim=True)
        anchor_tokens = anchor_tokens / anchor_tokens.norm(p=2, dim=-1, keepdim=True)
        # Calculate cosine similarities between anchor tokens and to-be-merged tokens.
        similarities = torch.bmm(to_be_merged_tokens, anchor_tokens.transpose(1, 2))  # (num_to_be_merged_tokens, num_anchor_tokens)

        cluster_indices = similarities.argmax(dim=-1)  # (num_to_be_merged_tokens,)
        assigned_one_hot = F.one_hot(cluster_indices, num_classes=anchor_tokens.shape[0]).to(features.dtype)  # (num_to_be_merged_tokens, num_anchor_tokens)

        counts = assigned_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)  # (num_anchor_tokens,)
        aggregated_tokens = torch.matmul(assigned_one_hot.transpose(0, 1), to_be_merged_tokens) / counts  # (num_anchor_tokens, feat_dim)

        # (5). Calculate contextual tokens by merging anchor tokens and aggregated tokens.
        alpha = (1 / (counts + 1)).clamp(min=dtm_alpha).unsqueeze(-1)  # (num_anchor_tokens,)
        contextual_tokens = alpha * anchor_tokens + (1 - alpha) * aggregated_tokens  # (num_anchor_tokens, feat_dim)

        all_tokens += (contextual_tokens,)
        all_keep_indices += (keep_indices.view(-1),)
        offset += segment_length

    return all_tokens, all_keep_indices


def calc_density_score(features: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Calculate density score for each token, based on DPC-kNN algorithm.

    Args:
        features (torch.Tensor): Segment features of shape (num_frames, num_tokens, feat_dim).
        num_clusters (int): Number of clusters to form.
        k (int): Number of nearest neighbors to consider for clustering. Default is 4.
        mask (Optional[torch.Tensor]): Optional mask to apply on features, same shape as features.

    Returns:
        torch.Tensor: Density scores of shape (num_frames, num_tokens).
    """
    bsz, seq_len, feat_dim = features.shape
    # Calculate pairwise distances between features
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)  # (bsz, seq_len, seq_len)
    nearest_dist = torch.topk(dists, k=k, dim=-1).values  # (bsz, seq_len, k)
    density = torch.mean(-(nearest_dist**2), dim=-1)  # (bsz, seq_len)

    # Add little noise to ensure no tokens have the same density.
    density = density + torch.rand_like(density) * 1e-6

    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.reshape(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)  # (bsz, seq_len, seq_len)
    dist = torch.min(modified_dists, dim=-1).values  # (bsz, seq_len)

    density_scores = dist * density  # (bsz, seq_len)
    return density_scores
