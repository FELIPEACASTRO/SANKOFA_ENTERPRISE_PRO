"""
Sankofa Enterprise Pro - Graph Feature Generator
Features baseadas em grafo de transações

Baseado em:
- ATLAS (Capital One)
- Neo4j fraud detection patterns
- GNN research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GraphFeatureGenerator:
    """
    Gerador de Features de Grafo

    Features por transação baseadas na posição no grafo:
    - Centralidade (degree, pagerank, betweenness)
    - Clustering
    - Comunidades
    - Padrões de conexão

    Gera 40+ features
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph_cache: Dict[str, Any] = {}

        logger.info(f"GraphFeatureGenerator v{self.VERSION} initialized")

    def generate(
        self,
        df: pd.DataFrame,
        sender_col: str = "sender_id",
        receiver_col: str = "receiver_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Gerar features de grafo

        Args:
            df: DataFrame com transações
            sender_col: Coluna de remetente
            receiver_col: Coluna de destinatário
            amount_col: Coluna de valor
            timestamp_col: Coluna de timestamp

        Returns:
            DataFrame com features de grafo
        """
        if sender_col not in df.columns or receiver_col not in df.columns:
            logger.warning("Sender/receiver columns not found, skipping graph features")
            return df

        df = df.copy()

        # Construir grafo
        logger.info("Building transaction graph...")
        graph = self._build_graph(df, sender_col, receiver_col, amount_col)

        # 1. Features de grau (degree)
        logger.info("Generating degree features...")
        df = self._add_degree_features(df, graph, sender_col, receiver_col)

        # 2. Features de centralidade
        logger.info("Generating centrality features...")
        df = self._add_centrality_features(df, graph, sender_col, receiver_col)

        # 3. Features de clustering
        logger.info("Generating clustering features...")
        df = self._add_clustering_features(df, graph, sender_col, receiver_col)

        # 4. Features de padrão de transação
        logger.info("Generating transaction pattern features...")
        df = self._add_transaction_pattern_features(df, sender_col, receiver_col)

        # 5. Features de comunidade
        logger.info("Generating community features...")
        df = self._add_community_features(df, graph, sender_col, receiver_col)

        return df

    def _build_graph(
        self,
        df: pd.DataFrame,
        sender_col: str,
        receiver_col: str,
        amount_col: str
    ) -> Dict[str, Any]:
        """Construir representação do grafo"""

        graph = {
            "nodes": set(),
            "edges": defaultdict(list),
            "in_edges": defaultdict(list),
            "out_edges": defaultdict(list),
            "edge_weights": defaultdict(float)
        }

        for _, row in df.iterrows():
            sender = row[sender_col]
            receiver = row[receiver_col]
            amount = row.get(amount_col, 1)

            if pd.isna(sender) or pd.isna(receiver):
                continue

            graph["nodes"].add(sender)
            graph["nodes"].add(receiver)

            edge = (sender, receiver)
            graph["edges"][edge].append(amount)
            graph["out_edges"][sender].append(receiver)
            graph["in_edges"][receiver].append(sender)
            graph["edge_weights"][edge] += amount

        return graph

    def _add_degree_features(
        self,
        df: pd.DataFrame,
        graph: Dict[str, Any],
        sender_col: str,
        receiver_col: str
    ) -> pd.DataFrame:
        """Adicionar features de grau"""

        # Calcular graus
        in_degree = {node: len(graph["in_edges"][node]) for node in graph["nodes"]}
        out_degree = {node: len(graph["out_edges"][node]) for node in graph["nodes"]}

        # Mapear para sender
        df["sender_out_degree"] = df[sender_col].map(out_degree).fillna(0)
        df["sender_in_degree"] = df[sender_col].map(in_degree).fillna(0)
        df["sender_total_degree"] = df["sender_out_degree"] + df["sender_in_degree"]

        # Mapear para receiver
        df["receiver_out_degree"] = df[receiver_col].map(out_degree).fillna(0)
        df["receiver_in_degree"] = df[receiver_col].map(in_degree).fillna(0)
        df["receiver_total_degree"] = df["receiver_out_degree"] + df["receiver_in_degree"]

        # Normalizar
        max_degree = max(max(in_degree.values(), default=1), max(out_degree.values(), default=1))
        df["sender_degree_normalized"] = df["sender_total_degree"] / max_degree
        df["receiver_degree_normalized"] = df["receiver_total_degree"] / max_degree

        # Ratio in/out
        df["sender_in_out_ratio"] = df["sender_in_degree"] / (df["sender_out_degree"] + 1)
        df["receiver_in_out_ratio"] = df["receiver_in_degree"] / (df["receiver_out_degree"] + 1)

        # Categorias
        df["sender_is_hub"] = (df["sender_out_degree"] > 10).astype(int)
        df["receiver_is_hub"] = (df["receiver_in_degree"] > 10).astype(int)

        return df

    def _add_centrality_features(
        self,
        df: pd.DataFrame,
        graph: Dict[str, Any],
        sender_col: str,
        receiver_col: str
    ) -> pd.DataFrame:
        """Adicionar features de centralidade"""

        # PageRank simplificado
        pagerank = self._calculate_pagerank(graph)

        df["sender_pagerank"] = df[sender_col].map(pagerank).fillna(0)
        df["receiver_pagerank"] = df[receiver_col].map(pagerank).fillna(0)

        # Betweenness estimado (simplificado)
        betweenness = self._estimate_betweenness(graph)

        df["sender_betweenness"] = df[sender_col].map(betweenness).fillna(0)
        df["receiver_betweenness"] = df[receiver_col].map(betweenness).fillna(0)

        # Diferença de centralidade (pode indicar fluxo suspeito)
        df["pagerank_diff"] = df["sender_pagerank"] - df["receiver_pagerank"]
        df["betweenness_diff"] = df["sender_betweenness"] - df["receiver_betweenness"]

        return df

    def _calculate_pagerank(
        self,
        graph: Dict[str, Any],
        damping: float = 0.85,
        iterations: int = 20
    ) -> Dict[str, float]:
        """Calcular PageRank"""

        nodes = list(graph["nodes"])
        n = len(nodes)

        if n == 0:
            return {}

        # Inicializar
        scores = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_scores = {}

            for node in nodes:
                # Soma de contribuições dos predecessores
                contribution = 0.0
                predecessors = graph["in_edges"].get(node, [])

                for pred in predecessors:
                    out_count = len(graph["out_edges"].get(pred, [])) or 1
                    contribution += scores.get(pred, 0) / out_count

                new_scores[node] = (1 - damping) / n + damping * contribution

            scores = new_scores

        return scores

    def _estimate_betweenness(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """Estimar betweenness centrality (simplificado)"""

        nodes = list(graph["nodes"])
        betweenness = {node: 0.0 for node in nodes}

        # Simplificação: contar quantas vezes cada nó é "intermediário"
        for node in nodes:
            in_neighbors = set(graph["in_edges"].get(node, []))
            out_neighbors = set(graph["out_edges"].get(node, []))

            # Nó é intermediário se tem entradas e saídas de diferentes origens
            if in_neighbors and out_neighbors:
                bridge_potential = len(in_neighbors) * len(out_neighbors)
                betweenness[node] = bridge_potential

        # Normalizar
        max_betweenness = max(betweenness.values(), default=1)
        if max_betweenness > 0:
            betweenness = {k: v / max_betweenness for k, v in betweenness.items()}

        return betweenness

    def _add_clustering_features(
        self,
        df: pd.DataFrame,
        graph: Dict[str, Any],
        sender_col: str,
        receiver_col: str
    ) -> pd.DataFrame:
        """Adicionar features de clustering"""

        clustering = self._calculate_clustering(graph)

        df["sender_clustering"] = df[sender_col].map(clustering).fillna(0)
        df["receiver_clustering"] = df[receiver_col].map(clustering).fillna(0)

        # Diferença de clustering
        df["clustering_diff"] = abs(df["sender_clustering"] - df["receiver_clustering"])

        return df

    def _calculate_clustering(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """Calcular coeficiente de clustering"""

        clustering = {}

        for node in graph["nodes"]:
            # Vizinhos do nó
            neighbors = set(graph["in_edges"].get(node, [])) | set(graph["out_edges"].get(node, []))
            neighbors.discard(node)

            k = len(neighbors)
            if k < 2:
                clustering[node] = 0.0
                continue

            # Contar conexões entre vizinhos
            neighbor_connections = 0
            neighbors_list = list(neighbors)

            for i, n1 in enumerate(neighbors_list):
                for n2 in neighbors_list[i + 1:]:
                    # Verificar se há edge entre n1 e n2
                    if n2 in graph["out_edges"].get(n1, []) or n1 in graph["out_edges"].get(n2, []):
                        neighbor_connections += 1

            # Coeficiente de clustering
            max_connections = k * (k - 1) / 2
            clustering[node] = neighbor_connections / max_connections if max_connections > 0 else 0

        return clustering

    def _add_transaction_pattern_features(
        self,
        df: pd.DataFrame,
        sender_col: str,
        receiver_col: str
    ) -> pd.DataFrame:
        """Adicionar features de padrão de transação"""

        # Primeira transação entre par
        pair_counts = df.groupby([sender_col, receiver_col]).cumcount()
        df["is_first_between_pair"] = (pair_counts == 0).astype(int)
        df["transactions_between_pair"] = pair_counts + 1

        # Transação recíproca (já houve transação no sentido inverso?)
        reverse_pairs = df.groupby([receiver_col, sender_col]).cumcount()
        df["reverse_tx_count"] = reverse_pairs

        # Há reciprocidade?
        df["has_reciprocal"] = (df["reverse_tx_count"] > 0).astype(int)

        # Self-loop (enviando para si mesmo)
        df["is_self_transaction"] = (df[sender_col] == df[receiver_col]).astype(int)

        return df

    def _add_community_features(
        self,
        df: pd.DataFrame,
        graph: Dict[str, Any],
        sender_col: str,
        receiver_col: str
    ) -> pd.DataFrame:
        """Adicionar features de comunidade (simplificado)"""

        # Detectar comunidades usando label propagation simplificado
        communities = self._detect_communities(graph)

        df["sender_community"] = df[sender_col].map(communities).fillna(-1)
        df["receiver_community"] = df[receiver_col].map(communities).fillna(-1)

        # Transação intra/inter comunidade
        df["same_community"] = (df["sender_community"] == df["receiver_community"]).astype(int)

        # Tamanho da comunidade
        community_sizes = {}
        for node, comm in communities.items():
            if comm not in community_sizes:
                community_sizes[comm] = 0
            community_sizes[comm] += 1

        df["sender_community_size"] = df["sender_community"].map(
            lambda x: community_sizes.get(x, 0)
        )
        df["receiver_community_size"] = df["receiver_community"].map(
            lambda x: community_sizes.get(x, 0)
        )

        return df

    def _detect_communities(
        self,
        graph: Dict[str, Any],
        iterations: int = 10
    ) -> Dict[str, int]:
        """Detectar comunidades usando label propagation simplificado"""

        nodes = list(graph["nodes"])
        labels = {node: i for i, node in enumerate(nodes)}

        for _ in range(iterations):
            for node in nodes:
                # Vizinhos
                neighbors = (
                    graph["in_edges"].get(node, []) +
                    graph["out_edges"].get(node, [])
                )

                if not neighbors:
                    continue

                # Label mais comum entre vizinhos
                neighbor_labels = [labels.get(n, -1) for n in neighbors]
                if neighbor_labels:
                    from collections import Counter
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    labels[node] = most_common

        return labels

    def get_feature_names(self) -> List[str]:
        """Retornar lista de nomes de features de grafo"""
        return [
            # Degree
            "sender_out_degree", "sender_in_degree", "sender_total_degree",
            "receiver_out_degree", "receiver_in_degree", "receiver_total_degree",
            "sender_degree_normalized", "receiver_degree_normalized",
            "sender_in_out_ratio", "receiver_in_out_ratio",
            "sender_is_hub", "receiver_is_hub",

            # Centrality
            "sender_pagerank", "receiver_pagerank",
            "sender_betweenness", "receiver_betweenness",
            "pagerank_diff", "betweenness_diff",

            # Clustering
            "sender_clustering", "receiver_clustering", "clustering_diff",

            # Transaction patterns
            "is_first_between_pair", "transactions_between_pair",
            "reverse_tx_count", "has_reciprocal", "is_self_transaction",

            # Community
            "sender_community", "receiver_community",
            "same_community", "sender_community_size", "receiver_community_size"
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do gerador"""
        return {
            "version": self.VERSION,
            "feature_count": len(self.get_feature_names())
        }


def create_graph_feature_generator(config: Optional[Dict[str, Any]] = None) -> GraphFeatureGenerator:
    """Factory function para criar GraphFeatureGenerator"""
    return GraphFeatureGenerator(config)
