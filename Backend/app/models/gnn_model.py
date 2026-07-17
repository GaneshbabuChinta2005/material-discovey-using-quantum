"""
GNN Model — MEGNet-Inspired Graph Neural Network for Material Property Prediction.

Architecture:
  Input: Atomic graph with 9D node features + 5D edge features
  → 3 × EdgeConv message-passing layers (with skip connections)
  → Global pooling (mean + max concatenation)
  → Readout MLP → Scalar property prediction

Designed for predicting:
  - Formation energy (eV/atom)
  - Band gap (eV)

References:
  - MEGNet: Chen et al., "Graph Networks as a Universal Machine Learning
    Framework for Molecules and Crystals" (2019)
  - Roost: Goodall & Lee, "Predicting Materials Properties without Crystal
    Structure" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data

from app.utils.graph_constructor import NODE_FEAT_DIM, EDGE_FEAT_DIM

SIZE_NORMALIZER = 20.0


class EdgeConvLayer(MessagePassing):
    """
    Edge-conditioned graph convolution layer.

    Updates node features using edge features and neighbor information:
      m_ij = MLP_edge([x_i || x_j || e_ij])
      x_i' = MLP_node(x_i || Σ_j m_ij)

    Where || denotes concatenation and Σ is sum aggregation.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add")

        # Edge message MLP: takes [x_i, x_j, e_ij] → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update MLP: takes [x_i, aggregated_messages] → new x_i
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Skip connection projection (if dimensions don't match)
        self.skip = (
            nn.Linear(node_dim, hidden_dim)
            if node_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr, edge_gate):
        # Message passing + aggregation
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_gate=edge_gate)
        # Node update with skip connection
        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        out = out + self.skip(x)
        return F.silu(out)

    def message(self, x_i, x_j, edge_attr, edge_gate):
        # Construct edge messages from source, target, and edge features
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.edge_mlp(inputs)
        return msg * edge_gate


class MaterialGNN(nn.Module):
    """
    Graph Neural Network for material property prediction.

    Architecture:
      - Input embedding: project raw features to hidden_dim
      - 3 EdgeConv message-passing layers with skip connections
      - Global graph pooling (mean + max)
      - Readout MLP with dropout → scalar output

    Args:
        node_feat_dim: dimension of input node features (default 9)
        edge_feat_dim: dimension of input edge features (default 5)
        hidden_dim: hidden layer dimension (default 128)
        n_conv_layers: number of message-passing layers (default 3)
        dropout: dropout rate for readout MLP (default 0.15)
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 256,
        n_conv_layers: int = 6,
        dropout: float = 0.15,
        enable_zero_gap_gate: bool = False,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.enable_zero_gap_gate = enable_zero_gap_gate

        # Input projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
        )
        gate_input_dim = min(4, edge_feat_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._gate_input_dim = gate_input_dim

        # Message-passing layers
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(
                EdgeConvLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )
            )

        # Post-convolution transform
        self.post_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Readout MLP: mean+max pooling plus size feature → scalar
        readout_in = hidden_dim * 2 + 1
        mid_dim = max(hidden_dim // 2, 64)
        low_dim = max(hidden_dim // 4, 32)
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, low_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(low_dim, 1),
        )
        self.global_feat_dim = readout_in

        if enable_zero_gap_gate:
            gate_hidden = max(hidden_dim // 2, 64)
            self.metal_gate = nn.Sequential(
                nn.Linear(readout_in, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.metal_gate = None

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch

        Returns:
            Tensor of shape (batch_size, 1) — predicted property
        """
        x = self.node_embed(data.x)
        raw_edge_attr = data.edge_attr
        edge_attr = self.edge_embed(raw_edge_attr)
        gate_input = raw_edge_attr[:, : self._gate_input_dim]
        edge_gate = self.edge_gate(gate_input)

        # Message passing with skip connections
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_attr, edge_gate)

        # Post-convolution normalization
        x = self.post_conv(x)

        # Global graph pooling: concatenate mean and max
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        graph_sizes = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
        graph_sizes = graph_sizes.to(x.device)
        size_feat = torch.clamp(graph_sizes / SIZE_NORMALIZER, max=1.0)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max, size_feat], dim=-1)

        raw_output = self.readout(x_global)
        if self.metal_gate is not None:
            metal_prob = self.metal_gate(x_global)
            raw_output = raw_output * (1.0 - metal_prob)
        return raw_output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_energy_model(
    hidden_dim: int = 256,
    n_conv_layers: int = 6,
) -> MaterialGNN:
    """Create a GNN model for formation energy prediction."""
    return MaterialGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=hidden_dim,
        n_conv_layers=n_conv_layers,
        enable_zero_gap_gate=False,
    )


def create_bandgap_model(
    hidden_dim: int = 256,
    n_conv_layers: int = 6,
) -> MaterialGNN:
    """Create a GNN model for band gap prediction."""
    return MaterialGNN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=hidden_dim,
        n_conv_layers=n_conv_layers,
        enable_zero_gap_gate=True,
    )


# -----------------------------------------------------------------------
if __name__ == "__main__":
    from torch_geometric.data import Batch

    print("=" * 60)
    print("GNN Model Architecture Test")
    print("=" * 60)

    model = MaterialGNN()
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"\nArchitecture:\n{model}")

    # Create dummy batch of 4 graphs
    graphs = []
    for n in [3, 5, 4, 2]:
        x = torch.randn(n, 9)
        edge_index = torch.stack([
            torch.arange(n).repeat_interleave(n),
            torch.arange(n).repeat(n),
        ])
        edge_attr = torch.randn(n * n, 5)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    batch = Batch.from_data_list(graphs)
    print(f"\nBatch: {batch}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print(f"\nOutput shape: {out.shape}  (expected: [4, 1])")
    print(f"Output values: {out.squeeze().tolist()}")
    print("\n✓ Forward pass successful!")
