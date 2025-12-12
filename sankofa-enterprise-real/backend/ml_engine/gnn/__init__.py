"""
Sankofa Enterprise Pro - Advanced GNN Module
Graph Neural Networks para detecção de fraude

Baseado em:
- ATLAS (Capital One) - Spatio-temporal directed graphs
- HOGRL (IJCAI 2024) - High-order graph representation
- FraudGNN-RL - Reinforcement learning sampling
- PyTorch Geometric
"""

from .fraud_gnn import (
    FraudGNN,
    FraudGNNConfig,
    create_fraud_gnn
)

from .temporal_gnn import (
    TemporalGraphNetwork,
    TGNConfig,
    create_tgn
)

from .graph_builder import (
    TransactionGraphBuilder,
    HeteroGraphBuilder,
    create_graph_builder
)

from .gnn_trainer import (
    GNNTrainer,
    TrainerConfig,
    create_trainer
)

__all__ = [
    # Main GNN
    "FraudGNN",
    "FraudGNNConfig",
    "create_fraud_gnn",

    # Temporal GNN
    "TemporalGraphNetwork",
    "TGNConfig",
    "create_tgn",

    # Graph builders
    "TransactionGraphBuilder",
    "HeteroGraphBuilder",
    "create_graph_builder",

    # Trainer
    "GNNTrainer",
    "TrainerConfig",
    "create_trainer",
]

__version__ = "1.0.0"
