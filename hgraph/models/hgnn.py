"""HGNN encoder (Feng et al., AAAI 2019) + projection head.

Hypergraph convolution: X' = Dv^-1/2 H De^-1 H^T Dv^-1/2 X Theta
See claude.md for the derivation and the shared-weight two-view pipeline
(encoder -> mean pool -> projector -> NT-Xent).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hypergraph_laplacian(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Precompute Dv^-1/2 H De^-1 H^T Dv^-1/2, the normalized hypergraph propagation matrix.

    H: (N, E) incidence matrix (hard 0/1 membership, or soft weights in [0, 1]).
    Degrees are clamped away from zero so isolated nodes / empty hyperedges
    (which shouldn't occur given H1's singleton fallback, but cost nothing
    to guard) don't produce inf/nan.
    """
    dv = H.sum(dim=1)  # (N,)
    de = H.sum(dim=0)  # (E,)
    dv_inv_sqrt = torch.pow(dv.clamp(min=eps), -0.5)
    de_inv = torch.pow(de.clamp(min=eps), -1.0)
    Dv_inv_sqrt = torch.diag(dv_inv_sqrt)
    De_inv = torch.diag(de_inv)
    return Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt


class HGNNConv(nn.Module):
    """One hypergraph convolution layer: X' = L @ (X @ Theta)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return L @ self.theta(x)


class HGNNEncoder(nn.Module):
    """2-layer HGNN encoder, ReLU + dropout between layers."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv1 = HGNNConv(in_dim, hidden_dim)
        self.conv2 = HGNNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """x: (N, F_in) node features, H: (N, E) incidence matrix -> (N, hidden_dim) node embeddings."""
        L = hypergraph_laplacian(H)
        h = F.relu(self.conv1(x, L))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, L)
        return h

    @staticmethod
    def pool(node_embeds: torch.Tensor) -> torch.Tensor:
        """Mean pool over nodes -> scene embedding z. (N, d) -> (d,)"""
        return node_embeds.mean(dim=0)


class ProjectionHead(nn.Module):
    """2-layer MLP projector g: encoder_dim -> encoder_dim -> projector_dim (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None, out_dim: int = 128):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HGNNModel(nn.Module):
    """Full pipeline for one graph: HGNN encoder -> mean pool -> projector."""

    def __init__(self, in_dim: int, encoder_dim: int = 256, projector_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = HGNNEncoder(in_dim, encoder_dim, dropout=dropout)
        self.projector = ProjectionHead(encoder_dim, encoder_dim, projector_dim)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scene embedding z, projected embedding p) for one graph."""
        node_embeds = self.encoder(x, H)
        z = self.encoder.pool(node_embeds)
        p = self.projector(z)
        return z, p

    def embed_nodes(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Node-level embeddings only, e.g. for the linear-probe eval (no pooling/projection)."""
        return self.encoder(x, H)
