"""Inference-time state update gates for the released baselines.

The TTSA3R rule follows the implementation in the source TTT3R project.
The MeMix patch selector follows https://github.com/dongjiacheng06/MeMix
(CC BY-NC-SA 4.0); see THIRD_PARTY.md.
"""

import torch


def ttsa3r_gate(cross_attn_state, feat_i, new_state_feat,
                prev_feat_i, prev_new_state_feat):
    """Temporal state-change gate times spatial attention/change gate."""
    if prev_feat_i is None or prev_new_state_feat is None:
        return new_state_feat.new_ones((*new_state_feat.shape[:2], 1))

    state_change = (new_state_feat - prev_new_state_feat).norm(dim=-1)
    state_change_normalized = state_change / (state_change.mean(dim=1, keepdim=True) + 1e-8)
    temporal_mask = torch.sigmoid(state_change_normalized - 1.5)

    feat_i_norm = feat_i / (feat_i.norm(dim=-1, keepdim=True) + 1e-8)
    prev_feat_norm = prev_feat_i / (prev_feat_i.norm(dim=-1, keepdim=True) + 1e-8)
    feat_dissimilarity = 1.0 - (feat_i_norm * prev_feat_norm).sum(dim=-1)
    # [layers, batch, heads, state tokens, pose token + image tokens].
    # The source code concatenates layers with batch because evaluation uses B=1.
    attention_magnitude = torch.stack(cross_attn_state, dim=0)[..., 1:].mean(dim=(0, 2)).abs()
    spatial_signal = (attention_magnitude * feat_dissimilarity.unsqueeze(1)).max(dim=-1).values
    spatial_mask = torch.sigmoid(spatial_signal)
    return (temporal_mask * spatial_mask).unsqueeze(-1)


def memix_patch_gate(new_state_feat, feat_i, decoder_embed,
                     num_patches=768, topk=708):
    """MeMix's default smallest-similarity patch selection (708 of 768)."""
    num_tokens = new_state_feat.shape[1]
    num_patches = min(num_patches, num_tokens)
    if topk <= 0:
        return new_state_feat.new_zeros((*new_state_feat.shape[:2], 1))
    if num_patches <= 1 or topk >= num_patches:
        return new_state_feat.new_ones((*new_state_feat.shape[:2], 1))

    img_feat_mean = decoder_embed(feat_i).mean(dim=1, keepdim=True)
    score_token = (new_state_feat * img_feat_mean).sum(dim=-1)
    chunks = torch.chunk(score_token, num_patches, dim=1)
    scores = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
    topk_idx = torch.topk(scores, k=min(int(topk), scores.shape[1]),
                          dim=1, largest=False).indices
    gate = new_state_feat.new_zeros((*new_state_feat.shape[:2], 1))
    start = 0
    for patch, chunk in enumerate(chunks):
        end = start + chunk.shape[1]
        selected = (topk_idx == patch).any(dim=1)
        gate[selected, start:end, 0] = 1.0
        start = end
    return gate
