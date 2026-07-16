"""Self-supervised losses: NT-Xent (GraphCL, multi-view contrastive) and STRL MSE (temporal).

See claude.md for the derivations. NT-Xent trains on scene-embedding pairs
from two views of the same moment (north/south, or H5 roadside/vehicle);
STRL MSE trains on node-embedding pairs from consecutive frames of the same
tracklet (H2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy over a batch of (z1_i, z2_i) positive pairs.

    z1, z2: (N, d) projected embeddings for view A / view B of the same N scenes.
    Requires N >= 2 so each positive pair has in-batch negatives.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, d)
    sim = torch.mm(z, z.T) / tau  # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, labels)


def strl_mse(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Temporal self-supervised MSE between normalized embeddings of a tracklet pair (H2)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2) ** 2).sum(dim=1).mean()


def combined_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float = 0.5,
    lambda_strl: float = 0.5,
    strl_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """L = L_nt_xent(z1, z2) + lambda * L_strl(strl_pairs).

    strl_pairs carries its own (H2 tracklet) embeddings separately from
    z1/z2, since the two losses run over different pairings of the batch —
    pass None to fall back to NT-Xent alone.
    """
    loss = nt_xent(z1, z2, tau=tau)
    if strl_pairs is not None:
        s1, s2 = strl_pairs
        loss = loss + lambda_strl * strl_mse(s1, s2)
    return loss
