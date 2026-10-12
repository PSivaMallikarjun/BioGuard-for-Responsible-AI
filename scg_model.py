"""
BioGuard - Sparse Connectome Guard (SCG)
=========================================
Bio-inspired minimal neural architecture for LLM jailbreak detection.

Inspired by:
  - Megaphragma amalphitanum: ~400-neuron anucleate nervous system (Polilov 2012)
  - MaleCNS v1.0 Drosophila connectome: 166,700 neurons, sparse local connectivity
    (Berg et al., Cell 2026; HHMI Janelia / Cambridge / Google Research)
  - Connectome-constrained networks: Lappalainen et al., Nature 2024

Architecture:
  Input tokens → Embedding projection → SCG sparse layers → Binary classification
  Adjacency matrix is FIXED after init; only edge weights are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal


# ─────────────────────────────────────────────
# Graph Topology Builders
# ─────────────────────────────────────────────

def build_random_sparse_graph(n_nodes: int, avg_degree: int = 5, seed: int = 42) -> torch.Tensor:
    """
    Erdős–Rényi sparse directed graph.
    p = avg_degree / (n_nodes - 1)
    Returns adjacency as sparse COO tensor.
    """
    rng = np.random.default_rng(seed)
    p = avg_degree / max(n_nodes - 1, 1)
    adj = rng.random((n_nodes, n_nodes)) < p
    np.fill_diagonal(adj, 0)               # no self-loops
    rows, cols = np.where(adj)
    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values  = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


def build_scale_free_graph(n_nodes: int, m: int = 3, seed: int = 42) -> torch.Tensor:
    """
    Barabási–Albert scale-free graph (preferential attachment).
    Mimics hub-spoke wiring seen in biological connectomes.
    m: edges added per new node.
    """
    rng = np.random.default_rng(seed)
    edges = []
    degrees = np.zeros(n_nodes, dtype=float)

    # Seed with a small clique
    seed_size = min(m + 1, n_nodes)
    for i in range(seed_size):
        for j in range(seed_size):
            if i != j:
                edges.append((i, j))
                degrees[i] += 1

    for new_node in range(seed_size, n_nodes):
        probs = degrees / degrees.sum()
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs[:new_node] / probs[:new_node].sum())
        for t in targets:
            edges.append((new_node, t))
            edges.append((t, new_node))
            degrees[new_node] += 1
            degrees[t] += 1

    rows, cols = zip(*edges)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values  = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


def build_bio_layered_graph(n_nodes: int, n_layers: int = 4, seed: int = 42) -> torch.Tensor:
    """
    Layered feedforward graph with sparse long-range connections (2% of edges).
    Mimics the layered + long-range 'highway neuron' structure of the Drosophila CNS
    (Berg et al., Cell 2026).
    """
    rng = np.random.default_rng(seed)
    layer_size = n_nodes // n_layers
    edges = []

    # Short-range: adjacent layer connections
    for layer in range(n_layers - 1):
        src_start = layer * layer_size
        dst_start = (layer + 1) * layer_size
        src_end   = src_start + layer_size
        dst_end   = min(dst_start + layer_size, n_nodes)

        for s in range(src_start, src_end):
            # Each node connects to avg 4 nodes in next layer
            n_dst = min(4, dst_end - dst_start)
            targets = rng.choice(range(dst_start, dst_end), size=n_dst, replace=False)
            for t in targets:
                edges.append((s, int(t)))

    # Long-range: 2% sparse "highway" connections skipping layers
    n_long = max(1, int(0.02 * n_nodes * 4))
    for _ in range(n_long):
        s = int(rng.integers(0, n_nodes // 2))
        t = int(rng.integers(n_nodes // 2, n_nodes))
        if s != t:
            edges.append((s, t))

    rows, cols = zip(*edges)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values  = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


TOPOLOGY_BUILDERS = {
    "random_sparse": build_random_sparse_graph,
    "scale_free":    build_scale_free_graph,
    "bio_layered":   build_bio_layered_graph,
}


# ─────────────────────────────────────────────
# SCG Layer: Sparse Message Passing
# ─────────────────────────────────────────────

class SCGLayer(nn.Module):
    """
    Single sparse connectome guard layer.

    Forward pass:
      1. Gather source node features for each active edge.
      2. Apply learnable per-edge weight (one scalar per edge).
      3. Aggregate (sum) into destination nodes.
      4. Apply threshold activation: spike if x > theta (SNN-inspired).

    Only |E| scalar parameters — dramatically fewer than a dense layer.
    """

    def __init__(self, edge_index: torch.Tensor, n_nodes: int, feat_dim: int):
        super().__init__()
        self.n_nodes   = n_nodes
        self.feat_dim  = feat_dim
        self.register_buffer("edge_src", edge_index[0])   # shape (E,)
        self.register_buffer("edge_dst", edge_index[1])   # shape (E,)
        n_edges = edge_index.shape[1]

        # Per-edge learnable scalar weights
        self.edge_weight = nn.Parameter(torch.randn(n_edges) * 0.1)

        # Per-node learnable spike threshold (SNN-inspired)
        self.theta = nn.Parameter(torch.zeros(n_nodes))

        # Node-level transform (shared across nodes to keep params small)
        self.node_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm      = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_nodes, feat_dim)
        returns: (batch, n_nodes, feat_dim)
        """
        B, N, D = x.shape
        assert N == self.n_nodes

        # --- Message passing ---
        src_feats = x[:, self.edge_src, :]             # (B, E, D)
        weighted  = src_feats * self.edge_weight.view(1, -1, 1)  # (B, E, D)

        # Scatter-add into destination nodes
        agg = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        dst = self.edge_dst.view(1, -1, 1).expand(B, -1, D)
        agg.scatter_add_(1, dst, weighted)

        # --- Node update ---
        out = self.node_proj(agg) + x                  # residual
        out = self.norm(out)

        # --- Spike threshold (SNN-like binary gate) ---
        # Hard threshold during inference; straight-through estimator for grad
        spike_mask = (out > self.theta.view(1, -1, 1)).float()
        # Straight-through: gradient flows through as if identity
        spike_mask = out + (spike_mask - out).detach()
        return spike_mask


# ─────────────────────────────────────────────
# Full SCG Model
# ─────────────────────────────────────────────

class SparseConnectomeGuard(nn.Module):
    """
    BioGuard: Sparse Connectome Guard for LLM jailbreak detection.

    Parameters
    ----------
    vocab_size    : size of tokenizer vocabulary
    embed_dim     : embedding dimension per token
    n_nodes       : number of SCG graph nodes (bio analogy: ~400-2000)
    n_layers      : number of SCG message-passing layers
    topology      : 'random_sparse' | 'scale_free' | 'bio_layered'
    avg_degree    : average node degree (only for random_sparse)
    seed          : random seed for reproducible topology
    """

    def __init__(
        self,
        vocab_size:  int = 32000,
        embed_dim:   int = 64,
        n_nodes:     int = 512,
        n_layers:    int = 3,
        topology:    Literal["random_sparse", "scale_free", "bio_layered"] = "bio_layered",
        avg_degree:  int = 5,
        seed:        int = 42,
    ):
        super().__init__()
        self.n_nodes  = n_nodes
        self.embed_dim = embed_dim

        # ── Token embedding (frozen after init if desired) ──────────────
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ── Project sequence → fixed n_nodes representation ─────────────
        # Each "node" gets a projection of the full sequence via attention pooling
        self.seq_to_nodes = nn.Linear(embed_dim, n_nodes * embed_dim)

        # ── Build and freeze graph topology ─────────────────────────────
        builder  = TOPOLOGY_BUILDERS[topology]
        kwargs   = {"avg_degree": avg_degree} if topology == "random_sparse" else {}
        adj_sp   = builder(n_nodes, seed=seed, **kwargs)
        edge_idx = adj_sp.coalesce().indices()          # (2, E)
        self.register_buffer("edge_index", edge_idx)

        n_edges = edge_idx.shape[1]
        param_count = n_edges + n_nodes + n_layers * (embed_dim * embed_dim + embed_dim)
        print(f"[BioGuard] Topology: {topology} | Nodes: {n_nodes} | Edges: {n_edges:,} | "
              f"Trainable params: ~{param_count:,}")

        # ── SCG layers ───────────────────────────────────────────────────
        self.scg_layers = nn.ModuleList([
            SCGLayer(edge_idx, n_nodes, embed_dim) for _ in range(n_layers)
        ])

        # ── Output head: global mean pool → MLP → binary logit ──────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids      : (B, seq_len) — tokenized prompt
        attention_mask : (B, seq_len) — optional padding mask
        returns        : (B,) logits — positive = adversarial
        """
        B, L = input_ids.shape

        # 1. Embed tokens → mean pool over sequence
        tok_emb = self.token_embed(input_ids)           # (B, L, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            seq_repr = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            seq_repr = tok_emb.mean(1)                  # (B, D)

        # 2. Project to node space
        node_feats = self.seq_to_nodes(seq_repr)        # (B, N*D)
        node_feats = node_feats.view(B, self.n_nodes, self.embed_dim)  # (B, N, D)

        # 3. SCG message passing (sparse, fixed topology)
        h = node_feats
        for layer in self.scg_layers:
            h = layer(h)

        # 4. Global mean pool over nodes → classify
        pooled = h.mean(dim=1)                          # (B, D)
        logit  = self.classifier(pooled).squeeze(-1)   # (B,)
        return logit

    def count_parameters(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
