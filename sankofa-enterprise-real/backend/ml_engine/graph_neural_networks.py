"""
Graph Neural Networks (GNN) - Deep learning on transaction graphs
Advanced fraud ring detection using PyTorch Geometric
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from collections import defaultdict
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. Install with: pip install torch torch-geometric")

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in transaction graph"""
    node_id: str
    node_type: str  # 'customer', 'device', 'merchant', 'account'
    features: np.ndarray
    label: Optional[int] = None  # 0=legitimate, 1=fraud


@dataclass
class GraphEdge:
    """Edge in transaction graph"""
    source_id: str
    target_id: str
    edge_type: str  # 'transacts_with', 'uses_device', 'same_ip', etc.
    weight: float = 1.0
    timestamp: Optional[datetime] = None


class FraudGNNModel(nn.Module):
    """
    Graph Neural Network for fraud detection

    Architecture:
    - Input: Node features + graph structure
    - GNN Layers: 3x GraphSAGE (aggregates neighbor information)
    - Attention: GAT layer for important neighbor weighting
    - Output: Node-level fraud probability

    Features:
    - Heterogeneous graphs (customers, devices, merchants)
    - Temporal edges (transaction sequences)
    - Inductive learning (works on new nodes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(FraudGNNModel, self).__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric not installed")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # GraphSAGE layers (good for inductive learning)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Attention layer (GAT) for final aggregation
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        logger.info(
            f"GNN Model initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}"
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node predictions (num_nodes, output_dim)
        """
        # GraphSAGE layers with ReLU and dropout
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT attention layer
        x = self.gat(x, edge_index)
        x = F.relu(x)

        # Classification
        out = self.classifier(x)

        return out

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Fraud probabilities (num_nodes,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1]  # Fraud probability


class TransactionGraphBuilder:
    """
    Builds transaction graphs from historical data

    Graph structure:
    - Nodes: Customers, Devices, Merchants, Accounts
    - Edges: Transactions, Device usage, IP sharing, etc.

    Features per node:
    - Customer: transaction history, account age, behavior
    - Device: device fingerprint, usage patterns
    - Merchant: merchant category, risk score
    """

    def __init__(self):
        """Initialize graph builder"""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_index_map: Dict[str, int] = {}

        logger.info("Transaction Graph Builder initialized")

    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Add transaction to graph

        Creates:
        - Customer node
        - Merchant node
        - Device node
        - Edges between them
        """
        customer_id = transaction.get('customer_id', '')
        merchant_id = transaction.get('merchant_id', '')
        device_id = transaction.get('device_id', '')

        # Add customer node
        if customer_id and customer_id not in self.nodes:
            self.nodes[customer_id] = GraphNode(
                node_id=customer_id,
                node_type='customer',
                features=self._extract_customer_features(transaction)
            )

        # Add merchant node
        if merchant_id and merchant_id not in self.nodes:
            self.nodes[merchant_id] = GraphNode(
                node_id=merchant_id,
                node_type='merchant',
                features=self._extract_merchant_features(transaction)
            )

        # Add device node
        if device_id and device_id not in self.nodes:
            self.nodes[device_id] = GraphNode(
                node_id=device_id,
                node_type='device',
                features=self._extract_device_features(transaction)
            )

        # Add edges
        if customer_id and merchant_id:
            self.edges.append(GraphEdge(
                source_id=customer_id,
                target_id=merchant_id,
                edge_type='transacts_with',
                weight=float(transaction.get('amount', 0)) / 1000.0,
                timestamp=datetime.fromisoformat(transaction.get('created_at', datetime.now(timezone.utc).isoformat()))
            ))

        if customer_id and device_id:
            self.edges.append(GraphEdge(
                source_id=customer_id,
                target_id=device_id,
                edge_type='uses_device',
                weight=1.0
            ))

    def _extract_customer_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Extract customer node features"""
        # Simplified - in production, aggregate from history
        features = np.array([
            float(transaction.get('amount', 0)) / 1000.0,  # Normalized amount
            1.0,  # Transaction count (placeholder)
            0.5,  # Average risk score (placeholder)
            1.0 if transaction.get('channel') == 'PIX' else 0.0,
            datetime.now(timezone.utc).hour / 24.0,  # Time of day
        ], dtype=np.float32)

        return features

    def _extract_merchant_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Extract merchant node features"""
        features = np.array([
            0.5,  # Merchant risk score (placeholder)
            1.0,  # Transaction volume (placeholder)
            0.1,  # Fraud rate (placeholder)
            1.0,  # Active status
        ], dtype=np.float32)

        return features

    def _extract_device_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Extract device node features"""
        features = np.array([
            1.0,  # Device trust score (placeholder)
            1.0,  # Device age (placeholder)
            0.5,  # Usage frequency (placeholder)
        ], dtype=np.float32)

        return features

    def build_pytorch_geometric_data(self) -> Data:
        """
        Convert graph to PyTorch Geometric Data object

        Returns:
            Data object with x (node features) and edge_index
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric not installed")

        # Create node index mapping
        self.node_index_map = {node_id: idx for idx, node_id in enumerate(self.nodes.keys())}

        # Stack node features
        node_features_list = []
        for node_id in self.node_index_map.keys():
            node_features_list.append(self.nodes[node_id].features)

        x = torch.tensor(np.vstack(node_features_list), dtype=torch.float)

        # Build edge index
        edge_sources = []
        edge_targets = []
        for edge in self.edges:
            if edge.source_id in self.node_index_map and edge.target_id in self.node_index_map:
                source_idx = self.node_index_map[edge.source_id]
                target_idx = self.node_index_map[edge.target_id]

                edge_sources.append(source_idx)
                edge_targets.append(target_idx)

                # Add reverse edge (undirected graph)
                edge_sources.append(target_idx)
                edge_targets.append(source_idx)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        logger.info(
            f"PyTorch Geometric Data created: "
            f"nodes={data.num_nodes}, edges={data.num_edges}"
        )

        return data

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type] += 1

        edge_types = defaultdict(int)
        for edge in self.edges:
            edge_types[edge.edge_type] += 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types)
        }


class GNNFraudDetector:
    """
    Production GNN fraud detector

    Workflow:
    1. Build transaction graph from recent transactions
    2. Run GNN inference to get node-level fraud scores
    3. Aggregate scores for new transaction
    4. Combine with other fraud signals
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lookback_days: int = 30
    ):
        """
        Args:
            model_path: Path to trained GNN model
            lookback_days: Days of transaction history to include in graph
        """
        self.lookback_days = lookback_days
        self.model: Optional[FraudGNNModel] = None

        # Load model if path provided
        if model_path and TORCH_GEOMETRIC_AVAILABLE:
            self.model = self._load_model(model_path)

        # Graph builder
        self.graph_builder = TransactionGraphBuilder()

        logger.info(
            f"GNN Fraud Detector initialized: "
            f"lookback_days={lookback_days}, model_loaded={self.model is not None}"
        )

    def _load_model(self, model_path: str) -> FraudGNNModel:
        """Load trained GNN model"""
        try:
            checkpoint = torch.load(model_path)
            model = FraudGNNModel(
                input_dim=checkpoint['input_dim'],
                hidden_dim=checkpoint.get('hidden_dim', 64),
                num_layers=checkpoint.get('num_layers', 3)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            logger.info(f"GNN model loaded from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            return None

    async def build_graph_from_history(
        self,
        transactions: List[Dict[str, Any]]
    ) -> None:
        """
        Build graph from transaction history

        Args:
            transactions: Historical transactions
        """
        logger.info(f"Building graph from {len(transactions)} transactions...")

        for txn in transactions:
            self.graph_builder.add_transaction(txn)

        stats = self.graph_builder.get_graph_statistics()
        logger.info(f"Graph built: {stats}")

    async def predict_fraud_gnn(
        self,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict fraud using GNN

        Args:
            transaction: Transaction to score

        Returns:
            GNN prediction result
        """
        if self.model is None or not TORCH_GEOMETRIC_AVAILABLE:
            return {
                'gnn_available': False,
                'gnn_score': 0.5,
                'message': 'GNN model not available'
            }

        try:
            # Add current transaction to graph
            self.graph_builder.add_transaction(transaction)

            # Build PyTorch Geometric data
            data = self.graph_builder.build_pytorch_geometric_data()

            # Get customer node index
            customer_id = transaction.get('customer_id', '')
            if customer_id not in self.graph_builder.node_index_map:
                return {
                    'gnn_available': True,
                    'gnn_score': 0.5,
                    'message': 'Customer not in graph'
                }

            customer_idx = self.graph_builder.node_index_map[customer_id]

            # Predict
            fraud_probs = self.model.predict_proba(data.x, data.edge_index)
            customer_fraud_prob = float(fraud_probs[customer_idx])

            # Get neighbor scores (guilt by association)
            neighbor_indices = self._get_neighbors(customer_id, data.edge_index)
            neighbor_fraud_scores = [float(fraud_probs[idx]) for idx in neighbor_indices]

            avg_neighbor_score = (
                np.mean(neighbor_fraud_scores)
                if neighbor_fraud_scores
                else 0.5
            )

            # Combined score
            combined_score = 0.7 * customer_fraud_prob + 0.3 * avg_neighbor_score

            return {
                'gnn_available': True,
                'gnn_score': combined_score,
                'customer_score': customer_fraud_prob,
                'neighbor_avg_score': avg_neighbor_score,
                'num_neighbors': len(neighbor_indices),
                'message': 'GNN prediction successful'
            }

        except Exception as e:
            logger.error(f"GNN prediction error: {e}")
            return {
                'gnn_available': False,
                'gnn_score': 0.5,
                'error': str(e)
            }

    def _get_neighbors(
        self,
        node_id: str,
        edge_index: torch.Tensor
    ) -> List[int]:
        """Get neighbor node indices"""
        if node_id not in self.graph_builder.node_index_map:
            return []

        node_idx = self.graph_builder.node_index_map[node_id]

        # Find edges where source is node_idx
        mask = edge_index[0] == node_idx
        neighbor_indices = edge_index[1][mask].tolist()

        return neighbor_indices

    def save_model(self, model_path: str) -> None:
        """Save GNN model"""
        if self.model is None:
            logger.warning("No model to save")
            return

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers
        }

        torch.save(checkpoint, model_path)
        logger.info(f"GNN model saved to {model_path}")


# Singleton instance
_gnn_detector_instance: Optional[GNNFraudDetector] = None


def get_gnn_detector() -> GNNFraudDetector:
    """
    Get singleton GNN detector

    Returns:
        GNNFraudDetector instance
    """
    global _gnn_detector_instance

    if _gnn_detector_instance is None:
        _gnn_detector_instance = GNNFraudDetector(
            lookback_days=30
        )

    return _gnn_detector_instance


# Example usage
async def example_gnn_fraud_detection():
    """Example: GNN fraud detection"""

    if not TORCH_GEOMETRIC_AVAILABLE:
        print("PyTorch Geometric not installed. Install with: pip install torch torch-geometric")
        return

    detector = get_gnn_detector()

    # Simulate historical transactions
    historical_txns = [
        {
            'id': f'TXN_{i}',
            'customer_id': f'CUST_{i % 10}',
            'merchant_id': f'MERCH_{i % 5}',
            'device_id': f'DEV_{i % 8}',
            'amount': np.random.uniform(100, 5000),
            'channel': 'PIX',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        for i in range(100)
    ]

    # Build graph
    await detector.build_graph_from_history(historical_txns)

    # New transaction
    new_txn = {
        'id': 'TXN_NEW',
        'customer_id': 'CUST_5',
        'merchant_id': 'MERCH_2',
        'device_id': 'DEV_3',
        'amount': 10000,
        'channel': 'PIX',
        'created_at': datetime.now(timezone.utc).isoformat()
    }

    # Predict
    result = await detector.predict_fraud_gnn(new_txn)

    print(f"GNN Prediction Result:")
    print(f"  GNN Score: {result.get('gnn_score', 0):.4f}")
    print(f"  Customer Score: {result.get('customer_score', 0):.4f}")
    print(f"  Neighbor Avg: {result.get('neighbor_avg_score', 0):.4f}")
    print(f"  Neighbors: {result.get('num_neighbors', 0)}")

    # Graph stats
    stats = detector.graph_builder.get_graph_statistics()
    print(f"\nGraph Statistics: {stats}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_gnn_fraud_detection())
