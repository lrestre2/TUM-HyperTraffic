"""Vanilla 2-layer GCN baseline (Kipf & Welling normalization) for the ablation.

Uses only pairwise BEV proximity edges (H1's underlying graph, before the
connected-components step folds it into hyperedges) — no hypergraph
structure — to isolate what the hyperedge formulation buys over a standard
GCN. See claude.md ablation protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hgnn import ProjectionHead


def gcn_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    n = A.size(0)
    A_hat = A + torch.eye(n, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=eps), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return L @ self.theta(x)


class GCNEncoder(nn.Module):
    """2-layer GCN, ReLU + dropout — same shape contract as HGNNEncoder for a like-for-like ablation."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x: (N, F_in) node features, A: (N, N) adjacency -> (N, hidden_dim) node embeddings."""
        L = gcn_laplacian(A)
        h = F.relu(self.conv1(x, L))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, L)
        return h

    @staticmethod
    def pool(node_embeds: torch.Tensor) -> torch.Tensor:
        return node_embeds.mean(dim=0)


class GCNModel(nn.Module):
    """Same encoder -> pool -> projector pipeline as HGNNModel, for a controlled ablation."""

    def __init__(self, in_dim: int, encoder_dim: int = 256, projector_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, encoder_dim, dropout=dropout)
        self.projector = ProjectionHead(encoder_dim, encoder_dim, projector_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_embeds = self.encoder(x, A)
        z = self.encoder.pool(node_embeds)
        p = self.projector(z)
        return z, p

    def embed_nodes(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, A)
