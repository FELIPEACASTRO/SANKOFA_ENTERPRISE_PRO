"""
Sankofa Enterprise Pro - Graph Builder
Construtor de grafos para GNN

Baseado em:
- PyTorch Geometric data structures
- Heterogeneous graphs
- Temporal graphs
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Verificar se PyTorch Geometric está disponível
try:
    from torch_geometric.data import Data, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Criar classe Data fallback
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class HeteroData:
        def __init__(self):
            self._store = {}

        def __setitem__(self, key, value):
            self._store[key] = value

        def __getitem__(self, key):
            return self._store.get(key)


class TransactionGraphBuilder:
    """
    Construtor de grafos de transações

    Converte DataFrame de transações em formato PyG Data
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Mapeamento de IDs
        self.node_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        self.node_count = 0

        logger.info(f"TransactionGraphBuilder v{self.VERSION} initialized")

    def build(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_id",
        receiver_col: str = "receiver_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        label_col: str = "is_fraud",
        feature_cols: Optional[List[str]] = None
    ) -> Data:
        """
        Construir grafo de transações

        Args:
            df: DataFrame com transações
            sender_col: Coluna de remetente
            receiver_col: Coluna de destinatário
            amount_col: Coluna de valor
            timestamp_col: Coluna de timestamp
            label_col: Coluna de label (fraude)
            feature_cols: Colunas de features

        Returns:
            Data object do PyG (ou equivalente)
        """
        logger.info(f"Building graph from {len(df)} transactions...")

        # Reset mappings
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.node_count = 0

        # 1. Mapear nós
        all_nodes = set(df[sender_col].dropna()) | set(df[receiver_col].dropna())
        for node in all_nodes:
            self._add_node(node)

        logger.info(f"Total nodes: {self.node_count}")

        # 2. Construir edge_index
        edge_index = self._build_edge_index(df, sender_col, receiver_col)

        # 3. Construir edge features
        edge_attr = self._build_edge_features(df, amount_col, timestamp_col)

        # 4. Construir node features
        x = self._build_node_features(df, sender_col, receiver_col, feature_cols)

        # 5. Labels (por aresta, não por nó)
        y = None
        if label_col in df.columns:
            y = torch.tensor(df[label_col].values, dtype=torch.float)

        # 6. Timestamps
        timestamps = None
        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col])
            timestamps = torch.tensor(
                (ts - ts.min()).dt.total_seconds().values,
                dtype=torch.float
            )

        # Criar Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            timestamps=timestamps,
            num_nodes=self.node_count
        )

        logger.info(f"Graph built: {data.num_nodes} nodes, {edge_index.size(1)} edges")

        return data

    def _add_node(self, node_id: str) -> int:
        """Adicionar nó ao mapeamento"""
        if node_id not in self.node_mapping:
            self.node_mapping[node_id] = self.node_count
            self.reverse_mapping[self.node_count] = node_id
            self.node_count += 1
        return self.node_mapping[node_id]

    def _build_edge_index(
        self,
        df: pd.DataFrame,
        sender_col: str,
        receiver_col: str
    ) -> torch.Tensor:
        """Construir tensor de conectividade"""

        sources = []
        targets = []

        for _, row in df.iterrows():
            sender = row[sender_col]
            receiver = row[receiver_col]

            if pd.isna(sender) or pd.isna(receiver):
                continue

            src_idx = self.node_mapping.get(sender)
            tgt_idx = self.node_mapping.get(receiver)

            if src_idx is not None and tgt_idx is not None:
                sources.append(src_idx)
                targets.append(tgt_idx)

        return torch.tensor([sources, targets], dtype=torch.long)

    def _build_edge_features(
        self,
        df: pd.DataFrame,
        amount_col: str,
        timestamp_col: str
    ) -> torch.Tensor:
        """Construir features das arestas"""

        features = []

        for _, row in df.iterrows():
            edge_feat = []

            # Amount
            if amount_col in df.columns:
                amount = float(row.get(amount_col, 0))
                edge_feat.extend([
                    amount,
                    np.log1p(amount),
                    amount % 100,  # Centavos
                    int(amount % 100 == 0),  # É valor redondo?
                ])
            else:
                edge_feat.extend([0, 0, 0, 0])

            # Timestamp features
            if timestamp_col in df.columns:
                ts = pd.to_datetime(row.get(timestamp_col))
                if pd.notna(ts):
                    edge_feat.extend([
                        ts.hour / 24,
                        ts.dayofweek / 7,
                        int(ts.hour >= 22 or ts.hour <= 6),  # É noite?
                        int(ts.dayofweek >= 5),  # É fim de semana?
                    ])
                else:
                    edge_feat.extend([0, 0, 0, 0])
            else:
                edge_feat.extend([0, 0, 0, 0])

            features.append(edge_feat)

        return torch.tensor(features, dtype=torch.float)

    def _build_node_features(
        self,
        df: pd.DataFrame,
        sender_col: str,
        receiver_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Construir features dos nós"""

        # Calcular estatísticas por nó
        node_stats = defaultdict(lambda: {
            "out_count": 0,
            "in_count": 0,
            "out_amount": 0,
            "in_amount": 0,
            "unique_out": set(),
            "unique_in": set()
        })

        amount_col = "amount" if "amount" in df.columns else "value" if "value" in df.columns else None

        for _, row in df.iterrows():
            sender = row.get(sender_col)
            receiver = row.get(receiver_col)
            amount = float(row.get(amount_col, 0)) if amount_col else 0

            if pd.notna(sender):
                node_stats[sender]["out_count"] += 1
                node_stats[sender]["out_amount"] += amount
                if pd.notna(receiver):
                    node_stats[sender]["unique_out"].add(receiver)

            if pd.notna(receiver):
                node_stats[receiver]["in_count"] += 1
                node_stats[receiver]["in_amount"] += amount
                if pd.notna(sender):
                    node_stats[receiver]["unique_in"].add(sender)

        # Construir tensor de features
        features = []

        for node_idx in range(self.node_count):
            node_id = self.reverse_mapping[node_idx]
            stats = node_stats[node_id]

            feat = [
                stats["out_count"],
                stats["in_count"],
                stats["out_amount"],
                stats["in_amount"],
                len(stats["unique_out"]),
                len(stats["unique_in"]),
                stats["out_amount"] / (stats["out_count"] + 1),  # Avg out
                stats["in_amount"] / (stats["in_count"] + 1),  # Avg in
            ]

            features.append(feat)

        return torch.tensor(features, dtype=torch.float)

    def get_node_id(self, index: int) -> str:
        """Obter ID original do nó pelo índice"""
        return self.reverse_mapping.get(index)

    def get_node_index(self, node_id: str) -> int:
        """Obter índice do nó pelo ID original"""
        return self.node_mapping.get(node_id)


class HeteroGraphBuilder:
    """
    Construtor de grafos heterogêneos

    Suporta múltiplos tipos de nós e arestas:
    - Nós: customer, device, IP, merchant, account
    - Arestas: transaction, uses_device, has_ip, transacts_with
    """

    VERSION = "1.0.0"

    NODE_TYPES = ["customer", "device", "ip", "merchant", "account"]
    EDGE_TYPES = [
        ("customer", "sends_to", "customer"),
        ("customer", "uses", "device"),
        ("customer", "connects_from", "ip"),
        ("customer", "transacts_with", "merchant")
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Mapeamentos por tipo
        self.node_mappings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_counts: Dict[str, int] = defaultdict(int)

        logger.info(f"HeteroGraphBuilder v{self.VERSION} initialized")

    def build(
        self,
        df: pd.DataFrame,
        col_mapping: Optional[Dict[str, str]] = None
    ) -> HeteroData:
        """
        Construir grafo heterogêneo

        Args:
            df: DataFrame com transações
            col_mapping: Mapeamento de colunas para tipos de nós

        Returns:
            HeteroData object
        """
        # Mapeamento padrão
        default_mapping = {
            "customer": "customer_id",
            "device": "device_id",
            "ip": "ip_address",
            "merchant": "merchant_id",
            "receiver": "receiver_id"
        }

        mapping = col_mapping or default_mapping

        # Reset
        self.node_mappings = defaultdict(dict)
        self.node_counts = defaultdict(int)

        data = HeteroData()

        # 1. Mapear todos os nós
        for node_type, col_name in mapping.items():
            if col_name in df.columns:
                unique_values = df[col_name].dropna().unique()
                for val in unique_values:
                    self._add_node(node_type, val)

        # 2. Construir arestas de transação
        if mapping.get("customer") in df.columns and mapping.get("receiver") in df.columns:
            sender_col = mapping["customer"]
            receiver_col = mapping["receiver"]

            edge_index = self._build_edge_index(
                df, sender_col, receiver_col,
                "customer", "customer"  # Ambos são customers
            )

            data[("customer", "sends_to", "customer")] = {
                "edge_index": edge_index
            }

        # 3. Construir arestas customer -> device
        if mapping.get("customer") in df.columns and mapping.get("device") in df.columns:
            edge_index = self._build_edge_index(
                df, mapping["customer"], mapping["device"],
                "customer", "device"
            )

            data[("customer", "uses", "device")] = {
                "edge_index": edge_index
            }

        # 4. Construir features de nós
        for node_type in self.node_counts.keys():
            num_nodes = self.node_counts[node_type]
            # Features simples: embedding aleatório (em produção, usar features reais)
            x = torch.randn(num_nodes, 32)
            data[node_type] = {"x": x, "num_nodes": num_nodes}

        return data

    def _add_node(self, node_type: str, node_id: str) -> int:
        """Adicionar nó de tipo específico"""
        if node_id not in self.node_mappings[node_type]:
            idx = self.node_counts[node_type]
            self.node_mappings[node_type][node_id] = idx
            self.node_counts[node_type] += 1
        return self.node_mappings[node_type][node_id]

    def _build_edge_index(
        self,
        df: pd.DataFrame,
        src_col: str,
        dst_col: str,
        src_type: str,
        dst_type: str
    ) -> torch.Tensor:
        """Construir edge_index para tipo específico de aresta"""

        sources = []
        targets = []

        for _, row in df.iterrows():
            src = row.get(src_col)
            dst = row.get(dst_col)

            if pd.isna(src) or pd.isna(dst):
                continue

            src_idx = self.node_mappings[src_type].get(src)
            dst_idx = self.node_mappings[dst_type].get(dst)

            if src_idx is not None and dst_idx is not None:
                sources.append(src_idx)
                targets.append(dst_idx)

        return torch.tensor([sources, targets], dtype=torch.long)


def create_graph_builder(
    hetero: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Union[TransactionGraphBuilder, HeteroGraphBuilder]:
    """Factory function para criar graph builder"""
    if hetero:
        return HeteroGraphBuilder(config)
    return TransactionGraphBuilder(config)
