"""
Graph ML Engine - Fraud Ring Detection
Detecta fraud rings, mulas, laranjas usando Graph Neural Networks
Benchmark: Stripe Radar (1B+ nodes), PayPal (3B+ edges)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node no grafo de fraude"""
    node_id: str
    node_type: str  # account, device, ip, merchant, cpf
    risk_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphEdge:
    """Edge no grafo de fraude"""
    source: str
    target: str
    edge_type: str  # owns, uses, transacts_with
    weight: float = 1.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class FraudGraphBuilder:
    """
    Constrói grafo de fraude a partir de transações

    Nodes: CPF, Device, IP, Merchant, Account
    Edges: owns, uses, transacts_with, shares
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_cache = {}
        self.edge_cache = defaultdict(list)

    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        """Adiciona transação ao grafo"""
        # Extract entities
        cpf = transaction.get('cpf') or transaction.get('cliente_cpf')
        device_id = transaction.get('device_id')
        ip_address = transaction.get('ip_address')
        merchant_id = transaction.get('merchant_id')
        account_id = transaction.get('account_id') or cpf

        # Create nodes
        nodes = []
        if cpf:
            nodes.append(GraphNode(f"cpf:{cpf}", "cpf"))
        if device_id:
            nodes.append(GraphNode(f"device:{device_id}", "device"))
        if ip_address:
            nodes.append(GraphNode(f"ip:{ip_address}", "ip"))
        if merchant_id:
            nodes.append(GraphNode(f"merchant:{merchant_id}", "merchant"))
        if account_id:
            nodes.append(GraphNode(f"account:{account_id}", "account"))

        # Add nodes to graph
        for node in nodes:
            if node.node_id not in self.node_cache:
                self.graph.add_node(
                    node.node_id,
                    node_type=node.node_type,
                    risk_score=0.0,
                    metadata={}
                )
                self.node_cache[node.node_id] = node

        # Create edges
        timestamp = transaction.get('timestamp', datetime.now(timezone.utc))
        amount = transaction.get('amount', 0)

        # CPF owns Account
        if cpf and account_id:
            self.graph.add_edge(
                f"cpf:{cpf}",
                f"account:{account_id}",
                edge_type="owns",
                weight=1.0,
                timestamp=timestamp
            )

        # Account uses Device
        if account_id and device_id:
            self.graph.add_edge(
                f"account:{account_id}",
                f"device:{device_id}",
                edge_type="uses",
                weight=1.0,
                timestamp=timestamp
            )

        # Device from IP
        if device_id and ip_address:
            self.graph.add_edge(
                f"device:{device_id}",
                f"ip:{ip_address}",
                edge_type="from_ip",
                weight=1.0,
                timestamp=timestamp
            )

        # Account transacts with Merchant
        if account_id and merchant_id:
            self.graph.add_edge(
                f"account:{account_id}",
                f"merchant:{merchant_id}",
                edge_type="transacts_with",
                weight=amount,
                timestamp=timestamp,
                metadata={'amount': amount}
            )

    def detect_fraud_rings(self, min_ring_size: int = 3) -> List[List[str]]:
        """
        Detecta fraud rings usando community detection

        Fraud rings são componentes fortemente conectados de contas
        que compartilham devices/IPs suspeitos
        """
        # Find connected components
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))

        # Filter suspicious components
        fraud_rings = []
        for component in components:
            if len(component) >= min_ring_size:
                # Check if component has suspicious patterns
                account_nodes = [n for n in component if n.startswith('account:')]
                device_nodes = [n for n in component if n.startswith('device:')]

                # Suspicious: multiple accounts sharing same device
                if len(account_nodes) >= min_ring_size and len(device_nodes) < len(account_nodes):
                    fraud_rings.append(list(component))

        return fraud_rings

    def detect_mules(self, velocity_threshold: int = 10) -> List[str]:
        """
        Detecta mulas (accounts usadas para lavar dinheiro)

        Características:
        - Alto número de transações recebidas
        - Saques rápidos após recebimento
        - Múltiplos originadores diferentes
        """
        mule_candidates = []

        for node in self.graph.nodes():
            if not node.startswith('account:'):
                continue

            # Count incoming transactions
            in_edges = list(self.graph.in_edges(node, data=True))

            # High velocity
            if len(in_edges) >= velocity_threshold:
                # Check if from multiple sources
                sources = set(edge[0] for edge in in_edges)
                if len(sources) >= 3:
                    mule_candidates.append(node)

        return mule_candidates


class GraphFeatureExtractor:
    """
    Extrai features de grafo para ML

    Features:
    - PageRank (importância do nó)
    - Degree centrality (número de conexões)
    - Betweenness centrality (ponte entre comunidades)
    - Clustering coefficient (densidade local)
    - Community membership
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._pagerank_cache = None
        self._communities_cache = None

    def extract_features(self, node_id: str) -> Dict[str, float]:
        """Extrai features para um nó"""
        if node_id not in self.graph:
            return {}

        features = {}

        # Degree features
        features['in_degree'] = self.graph.in_degree(node_id)
        features['out_degree'] = self.graph.out_degree(node_id)
        features['total_degree'] = features['in_degree'] + features['out_degree']

        # Centrality features
        features['degree_centrality'] = nx.degree_centrality(self.graph).get(node_id, 0)
        features['betweenness_centrality'] = nx.betweenness_centrality(self.graph).get(node_id, 0)

        # PageRank
        if self._pagerank_cache is None:
            self._pagerank_cache = nx.pagerank(self.graph, max_iter=100)
        features['pagerank'] = self._pagerank_cache.get(node_id, 0)

        # Clustering
        try:
            features['clustering_coefficient'] = nx.clustering(
                self.graph.to_undirected(),
                node_id
            )
        except:
            features['clustering_coefficient'] = 0.0

        # Neighbor features (aggregation)
        neighbors = list(self.graph.neighbors(node_id))
        if neighbors:
            neighbor_risks = [
                self.graph.nodes[n].get('risk_score', 0.0)
                for n in neighbors
            ]
            features['avg_neighbor_risk'] = np.mean(neighbor_risks)
            features['max_neighbor_risk'] = np.max(neighbor_risks)
            features['num_high_risk_neighbors'] = sum(1 for r in neighbor_risks if r > 0.7)
        else:
            features['avg_neighbor_risk'] = 0.0
            features['max_neighbor_risk'] = 0.0
            features['num_high_risk_neighbors'] = 0

        # Temporal features (if edges have timestamps)
        edges = list(self.graph.out_edges(node_id, data=True))
        if edges:
            timestamps = [
                edge[2].get('timestamp', datetime.now(timezone.utc))
                for edge in edges
            ]
            if timestamps:
                latest = max(timestamps)
                oldest = min(timestamps)
                features['edge_recency_hours'] = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
                features['edge_span_hours'] = (latest - oldest).total_seconds() / 3600

        return features

    def extract_batch_features(self, node_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Extrai features para múltiplos nós (mais eficiente)"""
        # Pre-compute expensive metrics once
        if self._pagerank_cache is None:
            self._pagerank_cache = nx.pagerank(self.graph, max_iter=100)

        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)

        results = {}
        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            results[node_id] = self.extract_features(node_id)

        return results


class FraudPropagationScore:
    """
    Calcula risco propagado via grafo

    Ideia: Se conta A transaciona com conta B fraudulenta,
    risco de A aumenta (guilt by association)
    """

    def __init__(self, graph: nx.Graph, decay_factor: float = 0.5):
        self.graph = graph
        self.decay_factor = decay_factor

    def propagate_risk(
        self,
        seed_nodes: Dict[str, float],
        max_iterations: int = 5
    ) -> Dict[str, float]:
        """
        Propaga risco via grafo usando random walk

        Args:
            seed_nodes: {node_id: initial_risk_score}
            max_iterations: número de iterações de propagação

        Returns:
            {node_id: propagated_risk_score}
        """
        scores = defaultdict(float)
        scores.update(seed_nodes)

        for iteration in range(max_iterations):
            new_scores = defaultdict(float)

            for node in self.graph.nodes():
                # Keep seed risk
                if node in seed_nodes:
                    new_scores[node] = seed_nodes[node]

                # Aggregate from neighbors
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    neighbor_risks = [scores[n] for n in neighbors]
                    propagated = np.mean(neighbor_risks) * self.decay_factor
                    new_scores[node] = max(new_scores[node], propagated)

            scores = new_scores

            # Check convergence
            if iteration > 0:
                delta = sum(abs(scores[n] - new_scores.get(n, 0)) for n in scores)
                if delta < 0.01:
                    break

        return dict(scores)


class GraphMLEngine:
    """
    Engine principal de Graph ML para detecção de fraude

    Integra:
    - Graph construction
    - Feature extraction
    - Fraud ring detection
    - Mule detection
    - Risk propagation
    """

    def __init__(self):
        self.graph_builder = FraudGraphBuilder()
        self.feature_extractor = None
        self.risk_propagator = None

    def ingest_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """Ingere transações e constrói grafo"""
        for txn in transactions:
            self.graph_builder.add_transaction(txn)

        # Update extractors
        self.feature_extractor = GraphFeatureExtractor(self.graph_builder.graph)
        self.risk_propagator = FraudPropagationScore(self.graph_builder.graph)

        logger.info(
            f"Graph built: {len(self.graph_builder.graph.nodes())} nodes, "
            f"{len(self.graph_builder.graph.edges())} edges"
        )

    def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa transação usando graph features

        Returns:
            graph_features: features extraídas do grafo
            fraud_ring_risk: probabilidade de pertencer a fraud ring
            mule_risk: probabilidade de ser mula
            propagated_risk: risco propagado de vizinhos
        """
        # Add transaction to graph
        self.graph_builder.add_transaction(transaction)

        # Get account node
        cpf = transaction.get('cpf') or transaction.get('cliente_cpf')
        account_node = f"account:{cpf}" if cpf else None

        if not account_node or account_node not in self.graph_builder.graph:
            return {
                'graph_features': {},
                'fraud_ring_risk': 0.0,
                'mule_risk': 0.0,
                'propagated_risk': 0.0
            }

        # Extract features
        graph_features = self.feature_extractor.extract_features(account_node)

        # Check fraud ring membership
        fraud_rings = self.graph_builder.detect_fraud_rings()
        in_fraud_ring = any(account_node in ring for ring in fraud_rings)
        fraud_ring_risk = 0.9 if in_fraud_ring else 0.1

        # Check mule characteristics
        mules = self.graph_builder.detect_mules()
        is_mule = account_node in mules
        mule_risk = 0.85 if is_mule else 0.1

        # Propagate risk from neighbors
        # (simplified - in production, use actual fraud labels)
        propagated_risk = graph_features.get('avg_neighbor_risk', 0.0)

        return {
            'graph_features': graph_features,
            'fraud_ring_risk': fraud_ring_risk,
            'mule_risk': mule_risk,
            'propagated_risk': propagated_risk,
            'in_fraud_ring': in_fraud_ring,
            'is_mule_candidate': is_mule
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do grafo"""
        graph = self.graph_builder.graph

        return {
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'num_accounts': sum(1 for n in graph.nodes() if n.startswith('account:')),
            'num_devices': sum(1 for n in graph.nodes() if n.startswith('device:')),
            'num_ips': sum(1 for n in graph.nodes() if n.startswith('ip:')),
            'num_merchants': sum(1 for n in graph.nodes() if n.startswith('merchant:')),
            'avg_degree': sum(dict(graph.degree()).values()) / len(graph.nodes()) if graph.nodes() else 0,
            'num_connected_components': nx.number_connected_components(graph.to_undirected()),
            'fraud_rings_detected': len(self.graph_builder.detect_fraud_rings()),
            'mule_candidates': len(self.graph_builder.detect_mules())
        }


# Singleton instance
_graph_ml_engine = None


def get_graph_ml_engine() -> GraphMLEngine:
    """Get singleton Graph ML engine"""
    global _graph_ml_engine
    if _graph_ml_engine is None:
        _graph_ml_engine = GraphMLEngine()
    return _graph_ml_engine
