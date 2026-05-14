"""Empirical conditional entropy H(A_t | S_{0:t}) from a labelled action dataset.

For each position t, molecules are grouped by their causal prefix s_{0:t}.
Within each group the empirical action distribution is computed and its
Shannon entropy (in nats) is averaged over molecules and positions.

This gives the hard lower bound on any causal model's cross-entropy loss on
this dataset: no matter how well trained, the policy cannot go below this value.

Early stopping: once every prefix is unique the per-position entropy is zero
for all remaining positions, so we break out of the loop early.  In practice
this happens after ~10-20 token positions for SMILES/SELFIES datasets.
"""
from __future__ import annotations

import numpy as np
import torch


def empirical_conditional_entropy(
    tokens: torch.Tensor,   # [N, T]   — full token sequences (including BOS/EOS)
    actions: torch.Tensor,  # [N, T-1] — action index at each position
    num_latents: int,
    pad_token_id: int = 0,
) -> float:
    """Return H(A_t | S_{0:t}) in nats, averaged over valid (t, molecule) pairs."""
    x = tokens.numpy().astype(np.int32)   # [N, T]
    y = actions.numpy().astype(np.int64)  # [N, T-1]
    N, T = x.shape

    total_H = 0.0
    total_n = 0

    for t in range(T - 1):
        actions_t = y[:, t]

        # Exclude positions where the action is padding
        valid = actions_t != pad_token_id
        if not valid.any():
            continue
        prefix_v  = np.ascontiguousarray(x[valid, :t + 1], dtype=np.int32)
        actions_v = actions_t[valid]
        n_valid   = int(valid.sum())

        # Represent each prefix row as a raw-bytes key for np.unique
        row_bytes = np.dtype((np.void, prefix_v.dtype.itemsize * (t + 1)))
        prefix_void = prefix_v.view(row_bytes).ravel()

        _, inverse, group_sizes = np.unique(
            prefix_void, return_inverse=True, return_counts=True
        )
        n_groups = len(group_sizes)

        # Sort molecules by group so we can slice contiguous runs
        order          = np.argsort(inverse, kind="stable")
        actions_sorted = actions_v[order]
        boundaries     = np.concatenate([[0], np.cumsum(group_sizes)])

        for g in range(n_groups):
            acts  = actions_sorted[boundaries[g] : boundaries[g + 1]]
            n_g   = len(acts)
            if n_g == 1:           # entropy = 0, skip
                total_n += 1
                continue
            counts = np.bincount(acts, minlength=num_latents)
            probs  = counts[counts > 0].astype(np.float64) / n_g
            H      = float(-np.sum(probs * np.log(probs)))
            total_H += H * n_g
            total_n += n_g

        # Once every prefix is unique, all later positions are too → H = 0
        if n_groups == n_valid:
            # Account for all remaining positions contributing zero entropy
            remaining_positions = (T - 1) - (t + 1)
            total_n += n_valid * remaining_positions
            break

    return total_H / max(1, total_n)
