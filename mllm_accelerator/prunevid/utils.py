"""
Based on https://github.com/Visual-AI/PruneVid/blob/main/models/pllava/modeling_pllava.py

Modified by: Fanziyang-v
"""

from typing import Optional, List, Tuple, Union

import torch
from torch.nn import functional as F
import math


from llava.constants import IMAGE_TOKEN_INDEX
from transformers.generation.utils import GenerateOutput


def merge_tokens(image_features: torch.Tensor, temporal_segment_ratio: float = 0.25, k: int = 7, threshold: float = 0.8, cluster_ratio: float = 0.5, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply token merging in PruneVid.

    Args:
        image_features (torch.Tensor): The Image features, of shape (num_frames, H, W, feat_dim).
        temporal_segment_ratio (float): Ratio of the segment length to the total sequence length.
        k (int): The number of nearest neighbors to consider for local density.
        threshold (float): Similarity threshold to distinguish static and dynamic tokens.
        cluster_ratio (float): Ratio of the number of clusters to the number of tokens in each segment.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (H * W,).
    """
    num_frames, H, W, feat_dim = image_features.shape
    # Obtain temporal features by applying global average pooling to the image features
    temporal_features = F.adaptive_avg_pool2d(image_features.permute(0, 3, 1, 2), (1, 1)).reshape(1, num_frames, feat_dim)  # (1, num_frames, feat_dim)

    # Cluster frames by applying DPC-kNN clustering algorithm
    num_clusters = int(num_frames * temporal_segment_ratio)
    cluster_indices = dpc_knn(temporal_features, num_clusters=num_clusters, k=k, valid_token_mask=valid_token_mask)  # (num_frames,)
    refined_cluster_indices = refine_clusters(cluster_indices)[0]  # (num_frames,)

    start_indices, end_indices, cluster_ids = extract_continuous_clusters(refined_cluster_indices)  # (num_continuous_clusters,)
    segment_image_features = [image_features[start_idx : end_idx + 1, :, :, :] for start_idx, end_idx in zip(start_indices, end_indices)]
    segment_similarities = [calc_segment_similarities(segment) for segment in segment_image_features]
    masks = [partition(similarity, threshold) for similarity in segment_similarities]
    static_masks = [mask[0] for mask in masks]  # (H, W)
    dynamic_masks = [mask[1] for mask in masks]  # (H, W)

    merged_segments = [spatial_merge(segment, static_mask, dynamic_mask, cluster_ratio, k) for segment, static_mask, dynamic_mask in zip(segment_image_features, static_masks, dynamic_masks)]
    return torch.cat(merged_segments, dim=0)  # (num_frames, num_static_clusters + num_dynamic_clusters, feat_dim)


@torch.no_grad()
def dpc_knn(features: torch.Tensor, num_clusters: int, k: int, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply DPC-kNN clustering algorithm to the pooled image features, generating preliminary clustering result.

    Args:
        features (torch.Tensor): Pooled image features (temporal features), of shape (batch_size, seq_len, feat_dim).
        num_clusters (int): The number of clusters.
        k (int): The number of nearest neighbors to consider for local density.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (batch_size, seq_len).

    Returns:
        torch.Tensor: Cluster indices of shape (batch_size, seq_len).
    """
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape

    # Calculate euclidean distance and local density
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)  # (batch_size, seq_len, seq_len)
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), float("+inf"))
    nearest_dist = torch.topk(dists, k=k, dim=-1, largest=False).values  # (batch_size, seq_len, k)
    density = torch.mean(-(nearest_dist**2), dim=-1).exp()  # (batch_size, seq_len)

    # Add little random noise to ensure no tokens have the same density.
    density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6

    # Ensure the density of the empty token be 0
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)

    # Obtain the minimum distance to the point with higher density.
    mask = density[:, None, :] <= density[:, :, None]
    modified_dists = torch.masked_fill(dists, mask, float("+inf"))  # (batch_size, seq_len, seq_len)
    dist, _ = torch.min(modified_dists, dim=-1)  # (batch_size, seq_len)

    # Calculate clustering score (clustering centers have the highest score)
    score = dist * density  # (batch_size, seq_len)
    cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices  # (batch_size, num_clusters)

    # Obtain the distance to cluster centers (batch_size, seq_len, num_clusters)
    dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
    cluster_indices = torch.argmin(dists, dim=-1)  # (batch_size, seq_len)
    # Ensure each cluster center to merge with itself
    cluster_indices.scatter_(dim=-1, index=cluster_center_indices, src=torch.arange(num_clusters, device=cluster_indices.device, dtype=cluster_indices.dtype).unsqueeze(0).expand(bsz, -1))
    return cluster_indices


def refine_clusters(cluster_indices: torch.Tensor) -> torch.Tensor:
    """Refine the DPC-kNN clustering results to a set of segments so that each segment is continuous in time

    Args:
        cluster_indices (torch.Tensor): Cluster indices, of shape (batch_size, num_frames).

    Returns:
        torch.Tensor: Refined cluster indices, of shape (batch_size, num_frames).
    """
    bsz = cluster_indices.shape[0]
    refined_cluster_indices = cluster_indices.clone()
    for i in range(bsz):
        start_indices, end_indices, cluster_ids = extract_continuous_clusters(cluster_indices[i])
        lengths = end_indices - start_indices + 1  # Length of each continuous segment
        indices_to_be_refined = get_indices_to_be_refined(start_indices, end_indices, cluster_ids)
        for idx, cluster_id in enumerate(cluster_ids):
            if idx in indices_to_be_refined:
                start_idx, end_idx = start_indices[idx], end_indices[idx]
                # Obtain the lengths of left and right neighbour segments
                left_length = lengths[idx - 1] if idx > 0 else 0
                right_length = lengths[idx + 1] if idx < len(lengths) - 1 else 0
                # Obtain the cluster IDs of left and right neighbour segments
                left_cluster_id = cluster_ids[idx - 1] if idx > 0 else None
                right_cluster_id = cluster_ids[idx + 1] if idx < len(lengths) - 1 else None
                # Reassign the cluster ID.
                new_cluster_id = 0
                # ! TODO: FIX ME (This is not aligned with the original implementation.)
                if left_length == 0 and right_length == 0:
                    # If both left and right segments are empty, assign cluster ID to 0.
                    cluster_ids[idx] = new_cluster_id = 0
                elif left_length >= right_length:
                    # If the left segment is longer, merge with the left segment
                    cluster_ids[idx] = new_cluster_id = left_cluster_id
                    lengths[idx] = left_length + 1
                elif right_length > left_length:
                    # If the right segment is longer, merge with the right segment
                    cluster_ids[idx] = new_cluster_id = right_cluster_id
                    lengths[idx] = right_length + 1
                refined_cluster_indices[i, start_idx : end_idx + 1] = new_cluster_id
    return refined_cluster_indices


def calc_segment_similarities(segment: torch.Tensor) -> torch.Tensor:
    """Calculate average similarities of the image features along the temporal dimension at each spatial location.

    Args:
        segment (torch.Tensor): The image features in a segment, of shape (seg_length, H, W, feat_dim).

    Returns:
        torch.Tensor: Average similarities of the image features along the temporal dimension at each spatial location, of shape (H, W).
    """
    seg_length, H, W, feat_dim = segment.shape
    normed_segment = F.normalize(segment, p=2, dim=-1)  # L2 normalization
    similarities = torch.einsum("s h w c, t h w c -> h w s t", normed_segment, normed_segment)  # (H, W, seg_length, seg_length)
    # DO NOT count the similarity between the same frame
    similarities = similarities - torch.eye(seg_length, device=similarities.device, dtype=similarities.dtype).unsqueeze(0).unsqueeze(0)  # (H, W, seg_length, seg_length)
    avg_similarites = torch.einsum("h w s t -> h w", similarities) / ((seg_length) * (seg_length - 1))  # (H, W)
    return avg_similarites


def partition(segment_similarities: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partition visual tokens into static and dynamic tokens based on the similarity threshold in the given segment.

    Args:
        segment_similarities (torch.Tensor): Average similarities of the image features along the temporal dimension at each spatial location, of shape (H, W).
        threshold (float): Similarity threshold to distinguish static and dynamic tokens.

    Returns:
        Tuple: A tuple containing:
            - static_mask (torch.Tensor): Boolean mask indicating static tokens, of shape (H, W).
            - dynamic_mask (torch.Tensor): Boolean mask indicating dynamic tokens, of shape (H, W).
    """
    static_mask = segment_similarities > threshold
    dynamic_mask = ~static_mask
    return static_mask, dynamic_mask


def spatial_merge(segment: torch.Tensor, static_mask: torch.Tensor, dynamic_mask: torch.Tensor, cluster_ratio: float, k: int) -> torch.Tensor:
    """Merge static and dynamic tokens in the image feature.

    Args:
        segment (torch.Tensor): The image feature, of shape (seg_length, H, W, feat_dim).
        static_mask (torch.Tensor): Boolean mask indicating static tokens, of shape (H, W).
        dynamic_mask (torch.Tensor): Boolean mask indicating dynamic tokens, of shape (H, W).

    Returns:
        torch.Tensor: Merged image feature, of shape (seg_length, num_static_clusters + num_dynamic_clusters, feat_dim).
    """
    # 0. Obtain static and dynamic tokens
    static_tokens = segment[:, static_mask, :]  # (seg_length, num_static_tokens, feat_dim)
    dynamic_tokens = segment[:, dynamic_mask, :]  # (seg_length, num_dynamic_tokens, feat_dim)

    num_static_tokens, num_dynamic_tokens = static_tokens.shape[1], dynamic_tokens.shape[1]

    # 1. Apply DPC-kNN clustering to the static and dynamic tokens respectively
    num_static_clusters, num_dynamic_clusters = int(num_static_tokens * cluster_ratio), int(num_dynamic_tokens * cluster_ratio)
    if num_static_clusters + num_dynamic_clusters < int(cluster_ratio * (num_static_tokens + num_dynamic_tokens)):
        num_static_clusters += 1
    static_cluster_indices = dpc_knn(static_tokens, num_clusters=num_static_clusters, k=k)  # (seg_length, num_static_tokens)
    dynamic_cluster_indices = dpc_knn(dynamic_tokens, num_clusters=num_dynamic_clusters, k=k)  # (seg_length, num_dynamic_tokens)

    # 2. Average each cluster's features in static and dynamic tokens
    assigned_static_one_hot = F.one_hot(static_cluster_indices, num_classes=num_static_clusters).to(static_tokens.dtype)  # (seg_length, num_static_tokens, num_static_clusters)
    static_features = torch.einsum("s n c, s n f -> s c f", assigned_static_one_hot, static_tokens)  # (seg_length, num_clusters, feat_dim)
    assigned_dynamic_one_hot = F.one_hot(dynamic_cluster_indices, num_classes=num_dynamic_clusters).to(dynamic_tokens.dtype)  # (seg_length, num_dynamic_tokens, num_dynamic_clusters)
    dynamic_features = torch.einsum("s n c, s n f -> s c f", assigned_dynamic_one_hot, dynamic_tokens)  # (seg_length, num_clusters, feat_dim)

    # ! ATTENTION: Ensure that each cluster has at least one token.
    static_cluster_counts = assigned_static_one_hot.sum(dim=1).unsqueeze(-1)  # (seg_length, num_static_clusters, 1)
    dynamic_cluster_counts = assigned_dynamic_one_hot.sum(dim=1).unsqueeze(-1)  # (seg_length, num_dynamic_clusters, 1)

    static_features = static_features / static_cluster_counts  # (seg_length, num_static_clusters, feat_dim)
    dynamic_features = dynamic_features / dynamic_cluster_counts  # (seg_length, num_dynamic_clusters, feat_dim)

    # 3. Concatenate static and dynamic features
    return torch.cat([static_features, dynamic_features], dim=1)


def extract_continuous_clusters(cluster_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract continuous clusters from the cluster indices.

    Args:
        cluster_indices (torch.Tensor): Cluster indices, of shape (num_frames,).

    Returns:
        Tuple: A tuple containing:
            - start_indices (torch.Tensor): Start indices of each continuous cluster, of shape (num_continuous_clusters,).
            - end_indices (torch.Tensor): End indices of each continuous cluster (Inclusive), of shape (num_continuous_clusters,).
            - cluster_ids (torch.Tensor): Cluster IDs for each continuous segment, of shape (num_continuous_clusters,).
    """
    device = cluster_indices.device
    num_frames = cluster_indices.shape[0]
    diff_mask = cluster_indices[1:] != cluster_indices[:-1]  # (num_frames - 1,)
    diff_indices = torch.where(diff_mask)[0] + 1  # Get the indices where the cluster changes
    start_indices = torch.cat([torch.tensor([0], device=device, dtype=torch.long), diff_indices])  # Start of each continuous cluster
    end_indices = torch.cat([diff_indices - 1, torch.tensor([num_frames - 1], device=device, dtype=torch.long)])  # End of each continuous cluster
    cluster_ids = cluster_indices[start_indices]  # Get the cluster indices for each continuous segment
    return start_indices, end_indices, cluster_ids


# get indices to be refined which are not the longest continuous segment.
def get_indices_to_be_refined(start_indices: torch.Tensor, end_indices: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
    """Get indices to be refined which are not the longest continuous segment or are single frame segments.

    Args:
        start_indices (torch.Tensor): Start indices of each continuous cluster, of shape (num_continuous_clusters,).
        end_indices (torch.Tensor): End indices of each continuous cluster (Inclusive), of shape (num_continuous_clusters,).
        cluster_ids (torch.Tensor): Cluster IDs for each continuous segment, of shape (num_continuous_clusters,).

    Returns:
        torch.Tensor: Indices to be refined, of shape (num_indices_to_be_refined,).
    """
    lengths = end_indices - start_indices + 1  # Length of each continuous segment
    mask = torch.zeros_like(cluster_ids, dtype=torch.bool, device=cluster_ids.device)
    for cluster_id in torch.unique(cluster_ids):
        # all indices of the current cluster
        indices = torch.where(cluster_ids == cluster_id)[0]
        cluster_lengths = lengths[indices]
        max_length = cluster_lengths.max()
        # Mark indices that are not the longest segment or are single frame segments
        mask[indices[cluster_lengths < max_length]] = True
        mask[indices[cluster_lengths == 1]] = True
    return torch.where(mask)[0]  # Return the indices to be refined


@torch.no_grad()
def llava_qwen_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    modalities: Optional[List[str]] = ["image"],
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        visual_token_start_index = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        visual_token_end_index = -(inputs.shape[1] - visual_token_start_index)
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        visual_token_length = (inputs_embeds.shape[1] + visual_token_end_index) - visual_token_start_index
        # update dycoke_info
        self.prunevid_info.update({"visual_token_start_index": visual_token_start_index, "visual_token_length": visual_token_length})
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    return super(type(self), self).generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
