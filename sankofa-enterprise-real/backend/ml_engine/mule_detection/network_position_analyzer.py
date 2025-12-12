"""
Sankofa Enterprise Pro - Network Position Analyzer
Analisador de posição na rede de transações

Baseado em:
- FICO Falcon Network Intelligence
- Feedzai IQ TrustScore
- Graph-based fraud detection research
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkPosition:
    """Posição na rede de transações"""
    account_id: str
    is_central_node: bool
    network_risk_score: float
    degree_centrality: float
    in_degree: int
    out_degree: int
    betweenness_estimate: float
    pagerank_estimate: float
    clustering_coefficient: float
    unique_counterparties: int
    in_suspicious_cluster: bool
    connected_to_known_mules: int
    connected_to_known_fraudsters: int
    network_patterns: List[str]
    community_id: Optional[str]
    risk_propagated_from: List[str]
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class NetworkPositionAnalyzer:
    """
    Analisador de Posição na Rede

    Analisa:
    - Centralidade (nó importante na rede)
    - Clustering (agrupamento com outros suspeitos)
    - Propagação de risco (vizinhos suspeitos)
    - Padrões de rede (fan-in, fan-out, ciclos)
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.known_mules: Set[str] = set()
        self.known_fraudsters: Set[str] = set()
        self.account_risks: Dict[str, float] = {}

        logger.info(f"NetworkPositionAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        account_id: str,
        transaction_history: List[Dict[str, Any]],
        network_data: Optional[Dict[str, Any]] = None
    ) -> NetworkPosition:
        """
        Analisar posição na rede de uma conta

        Args:
            account_id: ID da conta
            transaction_history: Histórico de transações
            network_data: Dados de rede pré-calculados (opcional)

        Returns:
            NetworkPosition com análise completa
        """
        # Construir grafo local
        local_graph = self._build_local_graph(account_id, transaction_history)

        # Calcular métricas de centralidade
        degree_metrics = self._calculate_degree_metrics(account_id, local_graph)

        # Estimar PageRank local
        pagerank = self._estimate_local_pagerank(account_id, local_graph)

        # Estimar betweenness (simplificado)
        betweenness = self._estimate_betweenness(account_id, local_graph)

        # Calcular clustering coefficient
        clustering = self._calculate_clustering(account_id, local_graph)

        # Contar contrapartes únicas
        unique_counterparties = len(local_graph.get("counterparties", set()))

        # Verificar conexões suspeitas
        mule_connections = self._count_mule_connections(account_id, local_graph)
        fraudster_connections = self._count_fraudster_connections(account_id, local_graph)

        # Detectar se está em cluster suspeito
        in_suspicious = self._detect_suspicious_cluster(account_id, local_graph)

        # Identificar padrões de rede
        patterns = self._identify_network_patterns(account_id, local_graph, transaction_history)

        # Propagar risco de vizinhos
        risk_sources = self._propagate_risk(account_id, local_graph)

        # Usar dados de rede externos se disponíveis
        if network_data:
            degree_metrics["in_degree"] = network_data.get("in_degree", degree_metrics["in_degree"])
            degree_metrics["out_degree"] = network_data.get("out_degree", degree_metrics["out_degree"])
            pagerank = network_data.get("pagerank", pagerank)
            betweenness = network_data.get("betweenness_centrality", betweenness)
            clustering = network_data.get("clustering_coefficient", clustering)
            mule_connections = network_data.get("connections_to_known_mules", mule_connections)

        # Calcular score de risco de rede
        network_score = self._calculate_network_score(
            degree_metrics, pagerank, betweenness,
            mule_connections, fraudster_connections,
            in_suspicious, patterns
        )

        # Determinar se é nó central
        is_central = self._is_central_node(
            degree_metrics, pagerank, betweenness, network_score
        )

        # Gerar explicação
        explanation = self._generate_explanation(
            is_central, network_score, degree_metrics,
            mule_connections, patterns
        )

        return NetworkPosition(
            account_id=account_id,
            is_central_node=is_central,
            network_risk_score=network_score,
            degree_centrality=degree_metrics["degree_centrality"],
            in_degree=degree_metrics["in_degree"],
            out_degree=degree_metrics["out_degree"],
            betweenness_estimate=betweenness,
            pagerank_estimate=pagerank,
            clustering_coefficient=clustering,
            unique_counterparties=unique_counterparties,
            in_suspicious_cluster=in_suspicious,
            connected_to_known_mules=mule_connections,
            connected_to_known_fraudsters=fraudster_connections,
            network_patterns=patterns,
            community_id=None,
            risk_propagated_from=risk_sources,
            explanation=explanation
        )

    def _build_local_graph(
        self,
        account_id: str,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construir grafo local a partir das transações"""

        graph = {
            "nodes": set(),
            "edges": [],
            "counterparties": set(),
            "incoming": defaultdict(list),
            "outgoing": defaultdict(list)
        }

        for tx in transactions:
            sender = tx.get("sender_id", tx.get("conta_origem", ""))
            receiver = tx.get("receiver_id", tx.get("conta_recebedor", ""))
            amount = float(tx.get("amount", tx.get("value", 0)))
            timestamp = tx.get("timestamp", tx.get("created_at"))

            if not sender or not receiver:
                continue

            graph["nodes"].add(sender)
            graph["nodes"].add(receiver)

            edge = {
                "sender": sender,
                "receiver": receiver,
                "amount": amount,
                "timestamp": timestamp
            }

            graph["edges"].append(edge)
            graph["outgoing"][sender].append(edge)
            graph["incoming"][receiver].append(edge)

            # Identificar contrapartes
            if sender == account_id:
                graph["counterparties"].add(receiver)
            elif receiver == account_id:
                graph["counterparties"].add(sender)

        return graph

    def _calculate_degree_metrics(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcular métricas de grau do nó"""

        in_degree = len(graph["incoming"].get(account_id, []))
        out_degree = len(graph["outgoing"].get(account_id, []))
        total_degree = in_degree + out_degree

        # Normalizar pelo número total de nós
        num_nodes = len(graph["nodes"])
        degree_centrality = total_degree / (2 * (num_nodes - 1)) if num_nodes > 1 else 0

        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": total_degree,
            "degree_centrality": min(degree_centrality, 1.0)
        }

    def _estimate_local_pagerank(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> float:
        """Estimar PageRank localmente (simplificado)"""

        if not graph["nodes"]:
            return 0.0

        num_nodes = len(graph["nodes"])
        damping = 0.85
        iterations = 10

        # Inicializar scores
        scores = {node: 1.0 / num_nodes for node in graph["nodes"]}

        for _ in range(iterations):
            new_scores = {}

            for node in graph["nodes"]:
                # Soma de contribuições dos predecessores
                contribution = 0.0
                incoming = graph["incoming"].get(node, [])

                for edge in incoming:
                    sender = edge["sender"]
                    sender_outgoing = len(graph["outgoing"].get(sender, [])) or 1
                    contribution += scores.get(sender, 0) / sender_outgoing

                new_scores[node] = (1 - damping) / num_nodes + damping * contribution

            scores = new_scores

        return scores.get(account_id, 0.0)

    def _estimate_betweenness(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> float:
        """Estimar betweenness centrality (simplificado)"""

        # Simplificação: contar quantos caminhos passam pelo nó
        # Em produção, usar algoritmo completo ou dados do Neo4j

        if len(graph["nodes"]) < 3:
            return 0.0

        # Contar pares de nós que se conectam através desta conta
        in_nodes = set(e["sender"] for e in graph["incoming"].get(account_id, []))
        out_nodes = set(e["receiver"] for e in graph["outgoing"].get(account_id, []))

        # Nós que podem usar esta conta como intermediário
        bridges = len(in_nodes) * len(out_nodes)

        # Total de possíveis pares
        n = len(graph["nodes"])
        total_pairs = n * (n - 1) if n > 1 else 1

        return min(bridges / total_pairs, 1.0)

    def _calculate_clustering(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> float:
        """Calcular coeficiente de clustering"""

        counterparties = list(graph["counterparties"])

        if len(counterparties) < 2:
            return 0.0

        # Contar conexões entre vizinhos
        neighbor_connections = 0
        possible_connections = len(counterparties) * (len(counterparties) - 1) / 2

        for i, cp1 in enumerate(counterparties):
            for cp2 in counterparties[i + 1:]:
                # Verificar se cp1 e cp2 têm conexão
                for edge in graph["edges"]:
                    if (edge["sender"] == cp1 and edge["receiver"] == cp2) or \
                       (edge["sender"] == cp2 and edge["receiver"] == cp1):
                        neighbor_connections += 1
                        break

        return neighbor_connections / possible_connections if possible_connections > 0 else 0

    def _count_mule_connections(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> int:
        """Contar conexões com mulas conhecidas"""

        counterparties = graph["counterparties"]
        return len(counterparties.intersection(self.known_mules))

    def _count_fraudster_connections(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> int:
        """Contar conexões com fraudadores conhecidos"""

        counterparties = graph["counterparties"]
        return len(counterparties.intersection(self.known_fraudsters))

    def _detect_suspicious_cluster(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> bool:
        """Detectar se está em cluster suspeito"""

        counterparties = graph["counterparties"]

        # Verificar se muitas contrapartes são suspeitas
        suspicious_count = 0
        for cp in counterparties:
            if cp in self.known_mules or cp in self.known_fraudsters:
                suspicious_count += 1
            elif cp in self.account_risks and self.account_risks[cp] > 0.6:
                suspicious_count += 1

        # Se > 30% das contrapartes são suspeitas
        if len(counterparties) > 0:
            if suspicious_count / len(counterparties) > 0.3:
                return True

        # Verificar alto clustering com suspeitos
        # (múltiplas contrapartes conectadas entre si)
        if len(counterparties) >= 5:
            # Contar quantas contrapartes transacionam entre si
            interconnections = 0
            counterparty_list = list(counterparties)

            for i, cp1 in enumerate(counterparty_list):
                for cp2 in counterparty_list[i + 1:]:
                    for edge in graph["edges"]:
                        if (edge["sender"] == cp1 and edge["receiver"] == cp2) or \
                           (edge["sender"] == cp2 and edge["receiver"] == cp1):
                            interconnections += 1
                            break

            # Alto grau de interconexão pode indicar cluster organizado
            if interconnections > len(counterparties):
                return True

        return False

    def _identify_network_patterns(
        self,
        account_id: str,
        graph: Dict[str, Any],
        transactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identificar padrões de rede suspeitos"""

        patterns = []

        in_degree = len(graph["incoming"].get(account_id, []))
        out_degree = len(graph["outgoing"].get(account_id, []))

        # 1. Fan-in: muitas entradas, poucas saídas (coletor)
        if in_degree > 10 and out_degree <= 3:
            patterns.append("FAN_IN: Possível conta coletora")

        # 2. Fan-out: poucas entradas, muitas saídas (distribuidor)
        if in_degree <= 3 and out_degree > 10:
            patterns.append("FAN_OUT: Possível conta distribuidora")

        # 3. Pass-through: in ~ out (mula típica)
        if in_degree > 5 and out_degree > 5:
            if 0.7 <= in_degree / out_degree <= 1.3:
                patterns.append("PASS_THROUGH: Possível conta de passagem")

        # 4. Ciclos detectados
        cycles = self._detect_cycles(account_id, graph)
        if cycles > 0:
            patterns.append(f"CYCLES: {cycles} ciclos de transações detectados")

        # 5. Star pattern (hub central)
        unique_in = len(set(e["sender"] for e in graph["incoming"].get(account_id, [])))
        unique_out = len(set(e["receiver"] for e in graph["outgoing"].get(account_id, [])))

        if unique_in > 15 or unique_out > 15:
            patterns.append("STAR: Padrão estrela (hub central)")

        # 6. Transações recíprocas (A→B, B→A)
        reciprocal = self._count_reciprocal(account_id, graph)
        if reciprocal > 3:
            patterns.append(f"RECIPROCAL: {reciprocal} pares de transações recíprocas")

        # 7. Chain pattern (A→B→C→D)
        chain_length = self._detect_chain(account_id, graph)
        if chain_length >= 4:
            patterns.append(f"CHAIN: Cadeia de {chain_length} contas detectada")

        return patterns

    def _detect_cycles(
        self,
        account_id: str,
        graph: Dict[str, Any],
        max_depth: int = 4
    ) -> int:
        """Detectar ciclos envolvendo a conta"""

        cycles_found = 0
        visited = set()

        def dfs(current: str, path: List[str], depth: int):
            nonlocal cycles_found

            if depth > max_depth:
                return

            if current == account_id and len(path) > 2:
                cycles_found += 1
                return

            if current in visited:
                return

            visited.add(current)

            for edge in graph["outgoing"].get(current, []):
                next_node = edge["receiver"]
                dfs(next_node, path + [next_node], depth + 1)

            visited.discard(current)

        # Iniciar busca
        for edge in graph["outgoing"].get(account_id, []):
            dfs(edge["receiver"], [account_id, edge["receiver"]], 1)

        return cycles_found

    def _count_reciprocal(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> int:
        """Contar transações recíprocas"""

        incoming_senders = set(e["sender"] for e in graph["incoming"].get(account_id, []))
        outgoing_receivers = set(e["receiver"] for e in graph["outgoing"].get(account_id, []))

        # Interseção = contrapartes com transações em ambas direções
        return len(incoming_senders.intersection(outgoing_receivers))

    def _detect_chain(
        self,
        account_id: str,
        graph: Dict[str, Any],
        max_length: int = 6
    ) -> int:
        """Detectar cadeia de transações"""

        max_chain = 0

        def follow_chain(current: str, length: int, visited: Set[str]):
            nonlocal max_chain

            if length > max_length:
                return

            max_chain = max(max_chain, length)

            for edge in graph["outgoing"].get(current, []):
                next_node = edge["receiver"]
                if next_node not in visited:
                    visited.add(next_node)
                    follow_chain(next_node, length + 1, visited)
                    visited.discard(next_node)

        follow_chain(account_id, 1, {account_id})

        return max_chain

    def _propagate_risk(
        self,
        account_id: str,
        graph: Dict[str, Any]
    ) -> List[str]:
        """Identificar fontes de risco propagado"""

        risk_sources = []

        for cp in graph["counterparties"]:
            risk = self.account_risks.get(cp, 0)

            if cp in self.known_mules:
                risk_sources.append(f"{cp} (mula conhecida)")
            elif cp in self.known_fraudsters:
                risk_sources.append(f"{cp} (fraudador conhecido)")
            elif risk > 0.7:
                risk_sources.append(f"{cp} (risco: {risk:.0%})")

        return risk_sources[:5]  # Limitar a 5

    def _calculate_network_score(
        self,
        degree_metrics: Dict[str, Any],
        pagerank: float,
        betweenness: float,
        mule_connections: int,
        fraudster_connections: int,
        in_suspicious: bool,
        patterns: List[str]
    ) -> float:
        """Calcular score de risco de rede"""

        score = 0.0

        # Score de centralidade
        if degree_metrics["degree_centrality"] > 0.3:
            score += 0.15
        elif degree_metrics["degree_centrality"] > 0.1:
            score += 0.08

        # Score de PageRank
        if pagerank > 0.1:
            score += 0.15
        elif pagerank > 0.05:
            score += 0.08

        # Score de betweenness
        if betweenness > 0.2:
            score += 0.15
        elif betweenness > 0.1:
            score += 0.08

        # Conexões com mulas
        if mule_connections > 0:
            score += min(mule_connections * 0.15, 0.3)

        # Conexões com fraudadores
        if fraudster_connections > 0:
            score += min(fraudster_connections * 0.2, 0.4)

        # Em cluster suspeito
        if in_suspicious:
            score += 0.2

        # Padrões suspeitos
        pattern_score = 0
        for pattern in patterns:
            if "FAN_IN" in pattern or "FAN_OUT" in pattern:
                pattern_score += 0.1
            elif "PASS_THROUGH" in pattern:
                pattern_score += 0.15
            elif "CYCLES" in pattern:
                pattern_score += 0.15
            elif "CHAIN" in pattern:
                pattern_score += 0.1

        score += min(pattern_score, 0.25)

        return min(score, 1.0)

    def _is_central_node(
        self,
        degree_metrics: Dict[str, Any],
        pagerank: float,
        betweenness: float,
        network_score: float
    ) -> bool:
        """Determinar se é nó central na rede"""

        # Central se tem alta centralidade em qualquer métrica
        if degree_metrics["degree_centrality"] > 0.2:
            return True

        if pagerank > 0.1:
            return True

        if betweenness > 0.15:
            return True

        if network_score > 0.5:
            return True

        return False

    def _generate_explanation(
        self,
        is_central: bool,
        score: float,
        degree_metrics: Dict[str, Any],
        mule_connections: int,
        patterns: List[str]
    ) -> str:
        """Gerar explicação da análise"""

        if not is_central and score < 0.3:
            return "Conta com posição normal na rede, sem indicadores significativos."

        parts = []

        if is_central:
            parts.append(
                f"ALERTA: Conta identificada como nó central na rede "
                f"(score de risco: {score:.1%})."
            )
        else:
            parts.append(
                f"Conta apresenta alguns indicadores de rede suspeitos "
                f"(score: {score:.1%})."
            )

        parts.append(
            f"Grau: {degree_metrics['in_degree']} entradas, "
            f"{degree_metrics['out_degree']} saídas."
        )

        if mule_connections > 0:
            parts.append(f"Conectada a {mule_connections} mulas conhecidas.")

        if patterns:
            parts.append("Padrões: " + "; ".join(patterns[:2]))

        return " ".join(parts)

    def add_known_mule(self, account_id: str):
        """Adicionar mula conhecida"""
        self.known_mules.add(account_id)

    def add_known_fraudster(self, account_id: str):
        """Adicionar fraudador conhecido"""
        self.known_fraudsters.add(account_id)

    def set_account_risk(self, account_id: str, risk: float):
        """Definir risco de uma conta"""
        self.account_risks[account_id] = risk

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "known_mules_count": len(self.known_mules),
            "known_fraudsters_count": len(self.known_fraudsters),
            "tracked_accounts": len(self.account_risks)
        }


def create_network_analyzer(config: Optional[Dict[str, Any]] = None) -> NetworkPositionAnalyzer:
    """Factory function para criar NetworkPositionAnalyzer"""
    return NetworkPositionAnalyzer(config)
