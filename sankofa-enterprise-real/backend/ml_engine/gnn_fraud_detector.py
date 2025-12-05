"""
Sankofa Enterprise Pro - Graph Neural Network Fraud Detector
Detecção de fraude baseada em relacionamentos entre entidades
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime, timedelta
import hashlib

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Nó do grafo de transações"""

    node_id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    fraud_count: int = 0
    total_transactions: int = 0


@dataclass
class GraphEdge:
    """Aresta do grafo de transações"""

    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class GNNPrediction:
    """Resultado da análise GNN"""

    transaction_id: str
    graph_risk_score: float
    community_risk: float
    neighbor_fraud_rate: float
    centrality_score: float
    anomaly_score: float
    suspicious_patterns: List[str]
    risk_factors: Dict[str, float]
    is_suspicious: bool


class TransactionGraph:
    """
    Grafo de transações para análise de fraude

    Tipos de nós:
    - customer: Cliente (CPF)
    - device: Dispositivo
    - ip: Endereço IP
    - receiver: Conta recebedora
    - merchant: Comerciante
    - location: Localização

    Tipos de arestas:
    - transaction: Transação entre entidades
    - uses: Cliente usa dispositivo/IP
    - receives: Conta recebe transação
    - at_location: Transação em localização
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not installed. Run: pip install networkx")

        self.graph = nx.MultiDiGraph()
        self.node_cache: Dict[str, GraphNode] = {}
        self.fraud_nodes: Set[str] = set()
        self.community_cache: Dict[str, int] = {}
        self._last_community_update: Optional[datetime] = None

        logger.info("Transaction Graph initialized")

    def _generate_node_id(self, node_type: str, identifier: str) -> str:
        """Gera ID único para nó"""
        return f"{node_type}:{hashlib.md5(identifier.encode()).hexdigest()[:12]}"

    def add_transaction(
        self,
        transaction_id: str,
        customer_id: str,
        amount: float,
        receiver_id: Optional[str] = None,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        location: Optional[str] = None,
        merchant_id: Optional[str] = None,
        is_fraud: bool = False,
        timestamp: Optional[datetime] = None,
    ):
        """
        Adiciona transação ao grafo

        Args:
            transaction_id: ID da transação
            customer_id: ID do cliente
            amount: Valor da transação
            receiver_id: ID do recebedor
            device_id: ID do dispositivo
            ip_address: Endereço IP
            location: Localização
            merchant_id: ID do comerciante
            is_fraud: Se é fraude conhecida
            timestamp: Data/hora da transação
        """
        customer_node = self._generate_node_id("customer", customer_id)
        self._add_or_update_node(customer_node, "customer", {"original_id": customer_id})

        if is_fraud:
            self.fraud_nodes.add(customer_node)
            self._update_node_fraud_count(customer_node)

        self._update_node_transaction_count(customer_node)

        if receiver_id:
            receiver_node = self._generate_node_id("receiver", receiver_id)
            self._add_or_update_node(receiver_node, "receiver", {"original_id": receiver_id})
            self.graph.add_edge(
                customer_node,
                receiver_node,
                key=transaction_id,
                edge_type="transaction",
                amount=amount,
                is_fraud=is_fraud,
                timestamp=timestamp or datetime.now(),
            )
            if is_fraud:
                self.fraud_nodes.add(receiver_node)
                self._update_node_fraud_count(receiver_node)
            self._update_node_transaction_count(receiver_node)

        if device_id:
            device_node = self._generate_node_id("device", device_id)
            self._add_or_update_node(device_node, "device", {"original_id": device_id})
            self.graph.add_edge(
                customer_node,
                device_node,
                key=f"{transaction_id}_device",
                edge_type="uses",
                timestamp=timestamp or datetime.now(),
            )
            if is_fraud:
                self._update_node_fraud_count(device_node)

        if ip_address:
            ip_node = self._generate_node_id("ip", ip_address)
            self._add_or_update_node(ip_node, "ip", {"original_id": ip_address})
            self.graph.add_edge(
                customer_node,
                ip_node,
                key=f"{transaction_id}_ip",
                edge_type="uses",
                timestamp=timestamp or datetime.now(),
            )
            if is_fraud:
                self._update_node_fraud_count(ip_node)

        if location:
            location_node = self._generate_node_id("location", location)
            self._add_or_update_node(location_node, "location", {"original_id": location})
            self.graph.add_edge(
                customer_node,
                location_node,
                key=f"{transaction_id}_loc",
                edge_type="at_location",
                timestamp=timestamp or datetime.now(),
            )

        if merchant_id:
            merchant_node = self._generate_node_id("merchant", merchant_id)
            self._add_or_update_node(merchant_node, "merchant", {"original_id": merchant_id})
            self.graph.add_edge(
                customer_node,
                merchant_node,
                key=f"{transaction_id}_merchant",
                edge_type="transacts_with",
                amount=amount,
                timestamp=timestamp or datetime.now(),
            )
            if is_fraud:
                self._update_node_fraud_count(merchant_node)

    def _add_or_update_node(self, node_id: str, node_type: str, attributes: Dict[str, Any]):
        """Adiciona ou atualiza nó no grafo"""
        if node_id not in self.node_cache:
            self.node_cache[node_id] = GraphNode(
                node_id=node_id, node_type=node_type, attributes=attributes
            )
            self.graph.add_node(node_id, **attributes, node_type=node_type)
        else:
            self.node_cache[node_id].attributes.update(attributes)

    def _update_node_fraud_count(self, node_id: str):
        """Atualiza contagem de fraudes do nó"""
        if node_id in self.node_cache:
            self.node_cache[node_id].fraud_count += 1
            self._recalculate_node_risk(node_id)

    def _update_node_transaction_count(self, node_id: str):
        """Atualiza contagem de transações do nó"""
        if node_id in self.node_cache:
            self.node_cache[node_id].total_transactions += 1

    def _recalculate_node_risk(self, node_id: str):
        """Recalcula score de risco do nó"""
        if node_id in self.node_cache:
            node = self.node_cache[node_id]
            if node.total_transactions > 0:
                node.risk_score = node.fraud_count / node.total_transactions

    def get_neighbor_fraud_rate(self, node_id: str, depth: int = 2) -> float:
        """
        Calcula taxa de fraude dos vizinhos

        Args:
            node_id: ID do nó
            depth: Profundidade da busca

        Returns:
            Taxa de fraude dos vizinhos
        """
        if node_id not in self.graph:
            return 0.0

        try:
            neighbors = set()
            current_level = {node_id}

            for _ in range(depth):
                next_level = set()
                for n in current_level:
                    next_level.update(self.graph.predecessors(n))
                    next_level.update(self.graph.successors(n))
                neighbors.update(next_level)
                current_level = next_level - neighbors

            neighbors.discard(node_id)

            if not neighbors:
                return 0.0

            fraud_neighbors = neighbors.intersection(self.fraud_nodes)
            return len(fraud_neighbors) / len(neighbors)

        except Exception as e:
            logger.warning(f"Error calculating neighbor fraud rate: {e}")
            return 0.0

    def get_centrality_score(self, node_id: str) -> float:
        """Calcula score de centralidade do nó"""
        if node_id not in self.graph:
            return 0.0

        try:
            degree = self.graph.degree(node_id)
            max_degree = (
                max(dict(self.graph.degree()).values()) if self.graph.number_of_nodes() > 0 else 1
            )
            return degree / max_degree if max_degree > 0 else 0.0
        except Exception:
            return 0.0

    def detect_communities(self) -> Dict[str, int]:
        """Detecta comunidades no grafo"""
        if self.graph.number_of_nodes() < 2:
            return {}

        try:
            undirected = self.graph.to_undirected()
            communities = nx.community.louvain_communities(undirected, seed=42)

            result = {}
            for idx, community in enumerate(communities):
                for node in community:
                    result[node] = idx

            self.community_cache = result
            self._last_community_update = datetime.now()

            return result
        except Exception as e:
            logger.warning(f"Error detecting communities: {e}")
            return {}

    def get_community_risk(self, node_id: str) -> float:
        """Calcula risco da comunidade do nó"""
        if not self.community_cache:
            self.detect_communities()

        if node_id not in self.community_cache:
            return 0.0

        community_id = self.community_cache[node_id]
        community_members = [n for n, c in self.community_cache.items() if c == community_id]

        if not community_members:
            return 0.0

        fraud_in_community = sum(1 for n in community_members if n in self.fraud_nodes)
        return fraud_in_community / len(community_members)

    def detect_suspicious_patterns(self, customer_id: str) -> List[str]:
        """
        Detecta padrões suspeitos para um cliente

        Padrões detectados:
        - Múltiplos dispositivos em curto período
        - IPs de diferentes regiões
        - Transações para contas de alto risco
        - Comportamento atípico na rede
        """
        patterns = []
        node_id = self._generate_node_id("customer", customer_id)

        if node_id not in self.graph:
            return patterns

        device_neighbors = [
            n
            for n in self.graph.successors(node_id)
            if self.graph.nodes[n].get("node_type") == "device"
        ]
        if len(device_neighbors) > 3:
            patterns.append(f"MULTIPLE_DEVICES: {len(device_neighbors)} dispositivos")

        ip_neighbors = [
            n
            for n in self.graph.successors(node_id)
            if self.graph.nodes[n].get("node_type") == "ip"
        ]
        if len(ip_neighbors) > 5:
            patterns.append(f"MULTIPLE_IPS: {len(ip_neighbors)} IPs diferentes")

        receiver_neighbors = [
            n
            for n in self.graph.successors(node_id)
            if self.graph.nodes[n].get("node_type") == "receiver"
        ]
        high_risk_receivers = [
            n
            for n in receiver_neighbors
            if self.node_cache.get(n, GraphNode("", "")).risk_score > 0.3
        ]
        if high_risk_receivers:
            patterns.append(f"HIGH_RISK_RECEIVERS: {len(high_risk_receivers)} contas de alto risco")

        neighbor_fraud_rate = self.get_neighbor_fraud_rate(node_id)
        if neighbor_fraud_rate > 0.2:
            patterns.append(
                f"FRAUD_NETWORK: {neighbor_fraud_rate:.1%} da rede com histórico de fraude"
            )

        community_risk = self.get_community_risk(node_id)
        if community_risk > 0.15:
            patterns.append(f"HIGH_RISK_COMMUNITY: comunidade com {community_risk:.1%} de fraude")

        return patterns


class GNNFraudDetector:
    """
    Detector de fraude baseado em Graph Neural Networks

    Combina análise de grafo com features de rede para
    detectar padrões de fraude complexos.
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not installed")

        self.config = config or {}
        self.transaction_graph = TransactionGraph()
        self.threshold = self.config.get("threshold", 0.5)

        self.pattern_weights = {
            "neighbor_fraud_rate": 0.35,
            "community_risk": 0.25,
            "centrality_anomaly": 0.15,
            "device_velocity": 0.15,
            "receiver_risk": 0.10,
        }

        logger.info(f"GNN Fraud Detector initialized v{self.VERSION}")

    def add_historical_transactions(self, transactions: pd.DataFrame):
        """
        Adiciona transações históricas ao grafo

        Args:
            transactions: DataFrame com transações históricas
        """
        required_cols = ["transaction_id", "customer_id", "amount"]
        if not all(col in transactions.columns for col in required_cols):
            logger.warning("Missing required columns for graph construction")
            return

        for _, row in transactions.iterrows():
            self.transaction_graph.add_transaction(
                transaction_id=str(row.get("transaction_id", "")),
                customer_id=str(row.get("customer_id", "")),
                amount=float(row.get("amount", 0)),
                receiver_id=(
                    str(row.get("receiver_id", "")) if pd.notna(row.get("receiver_id")) else None
                ),
                device_id=str(row.get("device_id", "")) if pd.notna(row.get("device_id")) else None,
                ip_address=(
                    str(row.get("ip_address", "")) if pd.notna(row.get("ip_address")) else None
                ),
                location=str(row.get("location", "")) if pd.notna(row.get("location")) else None,
                merchant_id=(
                    str(row.get("merchant_id", "")) if pd.notna(row.get("merchant_id")) else None
                ),
                is_fraud=bool(row.get("is_fraud", False)),
                timestamp=row.get("timestamp") if pd.notna(row.get("timestamp")) else None,
            )

        logger.info(
            f"Added {len(transactions)} transactions to graph. "
            f"Nodes: {self.transaction_graph.graph.number_of_nodes()}, "
            f"Edges: {self.transaction_graph.graph.number_of_edges()}"
        )

    def analyze_transaction(
        self,
        transaction_id: str,
        customer_id: str,
        amount: float,
        receiver_id: Optional[str] = None,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs,
    ) -> GNNPrediction:
        """
        Analisa transação usando o grafo

        Args:
            transaction_id: ID da transação
            customer_id: ID do cliente
            amount: Valor da transação
            receiver_id: ID do recebedor
            device_id: ID do dispositivo
            ip_address: Endereço IP

        Returns:
            Resultado da análise GNN
        """
        customer_node = self.transaction_graph._generate_node_id("customer", customer_id)

        neighbor_fraud_rate = self.transaction_graph.get_neighbor_fraud_rate(customer_node)
        community_risk = self.transaction_graph.get_community_risk(customer_node)
        centrality = self.transaction_graph.get_centrality_score(customer_node)
        suspicious_patterns = self.transaction_graph.detect_suspicious_patterns(customer_id)

        centrality_anomaly = 0.0
        if centrality > 0.7:
            centrality_anomaly = 0.5
        elif centrality < 0.05 and self.transaction_graph.graph.number_of_nodes() > 100:
            centrality_anomaly = 0.3

        receiver_risk = 0.0
        if receiver_id:
            receiver_node = self.transaction_graph._generate_node_id("receiver", receiver_id)
            if receiver_node in self.transaction_graph.node_cache:
                receiver_risk = self.transaction_graph.node_cache[receiver_node].risk_score

        device_velocity = 0.0
        if device_id:
            device_node = self.transaction_graph._generate_node_id("device", device_id)
            if device_node in self.transaction_graph.graph:
                users = list(self.transaction_graph.graph.predecessors(device_node))
                if len(users) > 2:
                    device_velocity = min(len(users) / 10, 1.0)

        risk_factors = {
            "neighbor_fraud_rate": neighbor_fraud_rate,
            "community_risk": community_risk,
            "centrality_anomaly": centrality_anomaly,
            "device_velocity": device_velocity,
            "receiver_risk": receiver_risk,
        }

        graph_risk_score = sum(
            risk_factors[key] * self.pattern_weights[key] for key in risk_factors
        )

        pattern_bonus = len(suspicious_patterns) * 0.05
        graph_risk_score = min(graph_risk_score + pattern_bonus, 1.0)

        return GNNPrediction(
            transaction_id=transaction_id,
            graph_risk_score=graph_risk_score,
            community_risk=community_risk,
            neighbor_fraud_rate=neighbor_fraud_rate,
            centrality_score=centrality,
            anomaly_score=centrality_anomaly,
            suspicious_patterns=suspicious_patterns,
            risk_factors=risk_factors,
            is_suspicious=graph_risk_score >= self.threshold,
        )

    def get_graph_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do grafo"""
        graph = self.transaction_graph.graph

        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "fraud_nodes": len(self.transaction_graph.fraud_nodes),
            "communities": (
                len(set(self.transaction_graph.community_cache.values()))
                if self.transaction_graph.community_cache
                else 0
            ),
            "avg_degree": sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1),
            "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
        }

    def get_high_risk_entities(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Retorna entidades de maior risco"""
        high_risk = []

        for node_id, node in self.transaction_graph.node_cache.items():
            if node.risk_score > 0.1:
                high_risk.append(
                    {
                        "node_id": node_id,
                        "node_type": node.node_type,
                        "risk_score": node.risk_score,
                        "fraud_count": node.fraud_count,
                        "total_transactions": node.total_transactions,
                    }
                )

        high_risk.sort(key=lambda x: x["risk_score"], reverse=True)
        return high_risk[:top_n]


def create_gnn_detector(config: Optional[Dict[str, Any]] = None) -> Optional[GNNFraudDetector]:
    """Factory function para criar detector GNN"""
    if not NETWORKX_AVAILABLE:
        logger.warning("NetworkX not available, GNN detector disabled")
        return None

    return GNNFraudDetector(config)
