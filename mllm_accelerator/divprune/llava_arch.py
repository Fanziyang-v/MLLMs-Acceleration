import torch

def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)

    # ! Diversity-based Pruning
    bsz, num_visual_tokens, feat_dim = image_features.shape
    if bsz != 1:
        raise NotImplementedError(f"Currently Diversity-based Pruning only supports batch size 1, got batch size : {bsz}.")
    num_retained_tokens = self.divprune_info["num_retained_tokens"]
    selected_features, keep_indices = diversity_based_pruning(image_features.squeeze(0), num_retained_tokens)
    self.divprune_info["keep_indices"] = keep_indices
    image_features = selected_features.unsqueeze(0)
    return image_features

def calc_pairwise_cosine_distances(features: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise cosine distances for a batch of feature vectors.

    Args:
        features (torch.Tensor): Visual features, of shape (num_visual_tokens, feat_dim)

    Returns:
        torch.Tensor: Pairwise cosine distances, of shape (num_visual_tokens, num_visual_tokens)
    """
    normed_features = features / features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.mm(normed_features, normed_features.t())
    return 1.0 - similarities


def diversity_based_pruning(features: torch.Tensor, num_retained_tokens: int) -> torch.Tensor:
    """Prune visual tokens based on diversity.

    Args:
        features (torch.Tensor): Visual featsures, of shape (num_visual_tokens, feat_dim)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (num_retained_tokens, feat_dim)
    """
    # Cosine distances
    dist_matrix = calc_pairwise_cosine_distances(features)

    # Initialize keeping indices.
    keep_indices = torch.zeros(num_retained_tokens, dtype=torch.long, device=features.device)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=0, largest=False).values[1, :]  # (num_visual_tokens,)
    keep_indices[0] = torch.argmax(min_dist)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        # Get the distances to the already selected tokens.
        dist_sub_matrix = dist_matrix[keep_indices[:i], :]  # (num_visual_tokens,)
        min_dist = torch.min(dist_sub_matrix, dim=0).values
        keep_indices[i] = torch.argmax(min_dist)

    keep_indices = keep_indices.sort().values
    selected_features = features[keep_indices]
    return selected_features, keep_indices
