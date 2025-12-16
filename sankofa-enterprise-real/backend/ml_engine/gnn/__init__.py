"""
Sankofa Enterprise Pro - Advanced GNN Module
Graph Neural Networks para detecção de fraude

Baseado em:
- ATLAS (Capital One) - Spatio-temporal directed graphs
- HOGRL (IJCAI 2024) - High-order graph representation
- FraudGNN-RL - Reinforcement learning sampling
- PyTorch Geometric
"""

import logging

logger = logging.getLogger(__name__)

# Safe imports - catch any torch-related errors
try:
    from .fraud_gnn import (
        FraudGNN,
        FraudGNNConfig,
        create_fraud_gnn,
        HAS_TORCH,
        HAS_PYG
    )
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not import fraud_gnn: {e}")
    FraudGNN = None
    FraudGNNConfig = None
    create_fraud_gnn = None
    HAS_TORCH = False
    HAS_PYG = False

try:
    from .temporal_gnn import (
        TemporalGraphNetwork,
        TGNConfig,
        create_tgn
    )
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not import temporal_gnn: {e}")
    TemporalGraphNetwork = None
    TGNConfig = None
    create_tgn = None

try:
    from .graph_builder import (
        TransactionGraphBuilder,
        HeteroGraphBuilder,
        create_graph_builder
    )
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not import graph_builder: {e}")
    TransactionGraphBuilder = None
    HeteroGraphBuilder = None
    create_graph_builder = None

try:
    from .gnn_trainer import (
        GNNTrainer,
        TrainerConfig,
        create_trainer
    )
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not import gnn_trainer: {e}")
    GNNTrainer = None
    TrainerConfig = None
    create_trainer = None

__all__ = [
    # Main GNN
    "FraudGNN",
    "FraudGNNConfig",
    "create_fraud_gnn",
    "HAS_TORCH",
    "HAS_PYG",

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
