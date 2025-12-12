"""
Sankofa Enterprise Pro - Mule Account Detector
Detector principal de contas mula

Baseado em:
- BioCatch Mule Detection
- MuleHunter AI (RBI India)
- Metro Bank UK implementation
- Academic papers on money mule detection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MuleType(Enum):
    """Tipos de mulas identificadas"""
    UNKNOWN = "unknown"
    WITTING = "witting"  # Mula consciente
    UNWITTING = "unwitting"  # Mula inconsciente (vítima)
    MONEY_MULE = "money_mule"  # Mula tradicional
    ACCOUNT_TAKEOVER = "account_takeover"  # Conta roubada usada como mula
    SYNTHETIC_ID = "synthetic_id"  # Identidade sintética
    DORMANT_ACTIVATED = "dormant_activated"  # Conta dormante ativada


class RiskLevel(Enum):
    """Níveis de risco de mula"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MuleIndicator:
    """Indicador individual de comportamento de mula"""
    name: str
    score: float  # 0-1
    weight: float
    description: str
    evidence: Dict[str, Any]
    confidence: float


@dataclass
class MuleScore:
    """Resultado da análise de mula"""
    account_id: str
    is_mule: bool
    mule_probability: float
    mule_type: MuleType
    risk_level: RiskLevel
    indicators: List[MuleIndicator]
    total_score: float
    confidence: float
    explanation: str
    recommended_actions: List[str]
    network_connections: int
    suspicious_patterns: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class MuleDetector:
    """
    Detector de Contas Mula

    Indicadores analisados:
    - Dormancy → Sudden Activation
    - Rapid Turnover (alto volume in/out)
    - Circular Flows (padrões circulares)
    - Behavioral Change (mudança súbita)
    - Network Position (centralidade no grafo)
    - Recruitment Patterns (conexão com mulas conhecidas)
    - Age-Activity Mismatch
    - Geographic Anomalies
    """

    VERSION = "1.0.0"

    # Configuração de indicadores
    INDICATOR_CONFIG = {
        "dormancy_activation": {
            "weight": 0.15,
            "threshold": 0.6,
            "description": "Conta inativa que súbitamente se torna muito ativa"
        },
        "rapid_turnover": {
            "weight": 0.20,
            "threshold": 0.7,
            "description": "Alto volume de entrada/saída em curto período"
        },
        "circular_flows": {
            "weight": 0.18,
            "threshold": 0.75,
            "description": "Padrões circulares de transações entre contas"
        },
        "behavioral_change": {
            "weight": 0.12,
            "threshold": 0.65,
            "description": "Mudança súbita no padrão de comportamento"
        },
        "network_centrality": {
            "weight": 0.15,
            "threshold": 0.7,
            "description": "Posição central em rede de transações suspeitas"
        },
        "recruitment_connection": {
            "weight": 0.10,
            "threshold": 0.8,
            "description": "Conexão com mulas conhecidas ou suspeitas"
        },
        "age_activity_mismatch": {
            "weight": 0.05,
            "threshold": 0.6,
            "description": "Incompatibilidade entre idade da conta e atividade"
        },
        "geographic_anomaly": {
            "weight": 0.05,
            "threshold": 0.6,
            "description": "Anomalias geográficas nas transações"
        }
    }

    # Thresholds de classificação
    CLASSIFICATION_THRESHOLDS = {
        "critical": 0.85,
        "high": 0.70,
        "medium": 0.50,
        "low": 0.30
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.known_mules: set = set()
        self.suspicious_accounts: Dict[str, float] = {}

        # Sub-analisadores (serão injetados)
        self.dormancy_analyzer = None
        self.turnover_analyzer = None
        self.network_analyzer = None

        logger.info(f"MuleDetector v{self.VERSION} initialized")

    def set_analyzers(
        self,
        dormancy_analyzer=None,
        turnover_analyzer=None,
        network_analyzer=None
    ):
        """Injetar sub-analisadores"""
        self.dormancy_analyzer = dormancy_analyzer
        self.turnover_analyzer = turnover_analyzer
        self.network_analyzer = network_analyzer

    def detect(
        self,
        account_id: str,
        account_data: Dict[str, Any],
        transaction_history: List[Dict[str, Any]],
        network_data: Optional[Dict[str, Any]] = None
    ) -> MuleScore:
        """
        Detectar se conta é uma mula

        Args:
            account_id: ID da conta
            account_data: Dados cadastrais da conta
            transaction_history: Histórico de transações
            network_data: Dados de rede/grafo (opcional)

        Returns:
            MuleScore com resultado da análise
        """
        indicators = []

        # 1. Análise de Dormência/Ativação
        dormancy_indicator = self._analyze_dormancy(
            account_id, account_data, transaction_history
        )
        if dormancy_indicator:
            indicators.append(dormancy_indicator)

        # 2. Análise de Turnover
        turnover_indicator = self._analyze_turnover(
            account_id, transaction_history
        )
        if turnover_indicator:
            indicators.append(turnover_indicator)

        # 3. Análise de Fluxos Circulares
        circular_indicator = self._analyze_circular_flows(
            account_id, transaction_history, network_data
        )
        if circular_indicator:
            indicators.append(circular_indicator)

        # 4. Análise de Mudança Comportamental
        behavioral_indicator = self._analyze_behavioral_change(
            account_id, account_data, transaction_history
        )
        if behavioral_indicator:
            indicators.append(behavioral_indicator)

        # 5. Análise de Posição na Rede
        network_indicator = self._analyze_network_position(
            account_id, network_data
        )
        if network_indicator:
            indicators.append(network_indicator)

        # 6. Análise de Conexão com Mulas Conhecidas
        recruitment_indicator = self._analyze_recruitment_connection(
            account_id, transaction_history, network_data
        )
        if recruitment_indicator:
            indicators.append(recruitment_indicator)

        # 7. Análise de Age-Activity Mismatch
        age_indicator = self._analyze_age_activity_mismatch(
            account_data, transaction_history
        )
        if age_indicator:
            indicators.append(age_indicator)

        # 8. Análise de Anomalias Geográficas
        geo_indicator = self._analyze_geographic_anomalies(
            transaction_history
        )
        if geo_indicator:
            indicators.append(geo_indicator)

        # Calcular score total
        total_score = self._calculate_total_score(indicators)

        # Determinar tipo de mula
        mule_type = self._determine_mule_type(indicators, account_data)

        # Determinar nível de risco
        risk_level = self._determine_risk_level(total_score)

        # Determinar se é mula
        is_mule = total_score >= self.CLASSIFICATION_THRESHOLDS["medium"]

        # Calcular confiança
        confidence = self._calculate_confidence(indicators, transaction_history)

        # Gerar explicação
        explanation = self._generate_explanation(
            is_mule, total_score, indicators, mule_type
        )

        # Determinar ações recomendadas
        recommended_actions = self._get_recommended_actions(
            is_mule, risk_level, mule_type
        )

        # Identificar padrões suspeitos
        suspicious_patterns = self._identify_suspicious_patterns(
            indicators, transaction_history
        )

        # Contar conexões de rede
        network_connections = self._count_network_connections(
            account_id, network_data
        )

        return MuleScore(
            account_id=account_id,
            is_mule=is_mule,
            mule_probability=total_score,
            mule_type=mule_type,
            risk_level=risk_level,
            indicators=indicators,
            total_score=total_score,
            confidence=confidence,
            explanation=explanation,
            recommended_actions=recommended_actions,
            network_connections=network_connections,
            suspicious_patterns=suspicious_patterns
        )

    def _analyze_dormancy(
        self,
        account_id: str,
        account_data: Dict[str, Any],
        transaction_history: List[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar padrão de dormência → ativação"""

        if not transaction_history:
            return None

        config = self.INDICATOR_CONFIG["dormancy_activation"]

        # Calcular dias desde criação da conta
        account_created = account_data.get("created_at")
        if not account_created:
            return None

        if isinstance(account_created, str):
            account_created = datetime.fromisoformat(account_created.replace("Z", "+00:00"))

        account_age_days = (datetime.now() - account_created.replace(tzinfo=None)).days

        # Analisar atividade
        if len(transaction_history) < 5:
            return None

        # Ordenar por data
        sorted_txs = sorted(
            transaction_history,
            key=lambda x: x.get("timestamp", x.get("created_at", ""))
        )

        # Identificar períodos de dormência
        dormancy_periods = []
        last_tx_date = None

        for tx in sorted_txs:
            tx_date = tx.get("timestamp", tx.get("created_at"))
            if isinstance(tx_date, str):
                tx_date = datetime.fromisoformat(tx_date.replace("Z", "+00:00"))

            if last_tx_date:
                days_gap = (tx_date.replace(tzinfo=None) - last_tx_date.replace(tzinfo=None)).days
                if days_gap > 30:  # Dormência > 30 dias
                    dormancy_periods.append(days_gap)

            last_tx_date = tx_date

        # Calcular atividade recente vs histórica
        recent_txs = [
            tx for tx in sorted_txs
            if self._is_recent(tx, days=7)
        ]

        historical_avg = len(sorted_txs) / max(account_age_days, 1) * 7

        # Score de dormência
        dormancy_score = 0.0
        evidence = {}

        # Se houve período de dormência e depois ativação intensa
        if dormancy_periods and len(recent_txs) > historical_avg * 3:
            dormancy_score = min(0.5 + (len(recent_txs) / (historical_avg + 1)) * 0.1, 1.0)
            evidence["dormancy_days"] = max(dormancy_periods)
            evidence["activation_multiplier"] = len(recent_txs) / (historical_avg + 0.1)

        # Se conta nova com atividade intensa imediata
        if account_age_days < 30 and len(transaction_history) > 50:
            dormancy_score = max(dormancy_score, 0.7)
            evidence["new_account_high_activity"] = True
            evidence["tx_count_first_month"] = len(transaction_history)

        if dormancy_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="dormancy_activation",
            score=dormancy_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.7 if dormancy_periods else 0.5
        )

    def _analyze_turnover(
        self,
        account_id: str,
        transaction_history: List[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar rapidez do turnover (entrada/saída)"""

        if len(transaction_history) < 10:
            return None

        config = self.INDICATOR_CONFIG["rapid_turnover"]

        # Separar transações de entrada e saída
        inflows = []
        outflows = []

        for tx in transaction_history:
            amount = float(tx.get("amount", tx.get("value", 0)))
            tx_type = tx.get("type", tx.get("transaction_type", ""))

            if tx_type in ["credit", "deposit", "receive", "PIX_IN"]:
                inflows.append(tx)
            elif tx_type in ["debit", "withdrawal", "send", "PIX_OUT"]:
                outflows.append(tx)
            else:
                # Inferir pelo sinal ou campo
                if tx.get("receiver_id") == account_id or tx.get("conta_recebedor") == account_id:
                    inflows.append(tx)
                else:
                    outflows.append(tx)

        if not inflows or not outflows:
            return None

        # Calcular métricas de turnover
        total_inflow = sum(float(tx.get("amount", tx.get("value", 0))) for tx in inflows)
        total_outflow = sum(float(tx.get("amount", tx.get("value", 0))) for tx in outflows)

        # Calcular velocidade de turnover (tempo médio entre entrada e saída)
        turnover_times = []
        for inflow in inflows[-20:]:  # Últimas 20 entradas
            in_time = inflow.get("timestamp", inflow.get("created_at"))
            if isinstance(in_time, str):
                in_time = datetime.fromisoformat(in_time.replace("Z", "+00:00"))

            # Encontrar saída mais próxima após a entrada
            for outflow in outflows:
                out_time = outflow.get("timestamp", outflow.get("created_at"))
                if isinstance(out_time, str):
                    out_time = datetime.fromisoformat(out_time.replace("Z", "+00:00"))

                time_diff = (out_time.replace(tzinfo=None) - in_time.replace(tzinfo=None)).total_seconds()
                if 0 < time_diff < 86400:  # Dentro de 24h
                    turnover_times.append(time_diff)
                    break

        # Calcular score
        turnover_score = 0.0
        evidence = {}

        # Ratio de turnover alto (quase todo dinheiro que entra, sai rapidamente)
        if total_inflow > 0:
            turnover_ratio = total_outflow / total_inflow
            if 0.8 <= turnover_ratio <= 1.2:  # Quase 1:1
                turnover_score += 0.4
                evidence["turnover_ratio"] = turnover_ratio

        # Velocidade de turnover
        if turnover_times:
            avg_turnover_time = np.mean(turnover_times)
            if avg_turnover_time < 3600:  # Menos de 1 hora
                turnover_score += 0.4
                evidence["avg_turnover_hours"] = avg_turnover_time / 3600
            elif avg_turnover_time < 14400:  # Menos de 4 horas
                turnover_score += 0.2
                evidence["avg_turnover_hours"] = avg_turnover_time / 3600

        # Volume absoluto alto
        if total_inflow > 50000 and len(inflows) > 10:
            turnover_score += 0.2
            evidence["high_volume"] = True
            evidence["total_inflow"] = total_inflow

        turnover_score = min(turnover_score, 1.0)

        if turnover_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="rapid_turnover",
            score=turnover_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.8 if turnover_times else 0.6
        )

    def _analyze_circular_flows(
        self,
        account_id: str,
        transaction_history: List[Dict[str, Any]],
        network_data: Optional[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar fluxos circulares de dinheiro"""

        config = self.INDICATOR_CONFIG["circular_flows"]

        # Construir grafo simples de transações
        connections = {}
        for tx in transaction_history:
            sender = tx.get("sender_id", tx.get("conta_origem", ""))
            receiver = tx.get("receiver_id", tx.get("conta_recebedor", ""))

            if sender and receiver:
                if sender not in connections:
                    connections[sender] = set()
                connections[sender].add(receiver)

        # Detectar ciclos simples
        cycles_found = []

        def find_cycles(start, current, path, visited):
            if len(path) > 5:  # Limitar profundidade
                return
            if current == start and len(path) > 2:
                cycles_found.append(path.copy())
                return
            if current in visited:
                return

            visited.add(current)
            for neighbor in connections.get(current, []):
                path.append(neighbor)
                find_cycles(start, neighbor, path, visited)
                path.pop()
            visited.discard(current)

        # Buscar ciclos a partir da conta alvo
        find_cycles(account_id, account_id, [account_id], set())

        # Usar dados de rede se disponíveis
        network_cycles = 0
        if network_data:
            network_cycles = network_data.get("cycles_involving_account", 0)

        # Calcular score
        circular_score = 0.0
        evidence = {}

        if cycles_found:
            circular_score = min(len(cycles_found) * 0.2, 0.8)
            evidence["cycles_found"] = len(cycles_found)
            evidence["shortest_cycle"] = min(len(c) for c in cycles_found) if cycles_found else 0

        if network_cycles > 0:
            circular_score = max(circular_score, min(network_cycles * 0.15, 0.9))
            evidence["network_cycles"] = network_cycles

        if circular_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="circular_flows",
            score=circular_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.85 if network_data else 0.6
        )

    def _analyze_behavioral_change(
        self,
        account_id: str,
        account_data: Dict[str, Any],
        transaction_history: List[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar mudança súbita de comportamento"""

        if len(transaction_history) < 30:
            return None

        config = self.INDICATOR_CONFIG["behavioral_change"]

        # Dividir em período histórico e recente
        sorted_txs = sorted(
            transaction_history,
            key=lambda x: x.get("timestamp", x.get("created_at", ""))
        )

        split_point = len(sorted_txs) * 2 // 3
        historical = sorted_txs[:split_point]
        recent = sorted_txs[split_point:]

        if len(historical) < 10 or len(recent) < 5:
            return None

        # Calcular métricas para cada período
        def calc_metrics(txs):
            amounts = [float(tx.get("amount", tx.get("value", 0))) for tx in txs]
            return {
                "avg_amount": np.mean(amounts) if amounts else 0,
                "std_amount": np.std(amounts) if amounts else 0,
                "tx_count": len(txs),
                "unique_counterparties": len(set(
                    tx.get("receiver_id", tx.get("conta_recebedor", "")) or
                    tx.get("sender_id", tx.get("conta_origem", ""))
                    for tx in txs
                ))
            }

        hist_metrics = calc_metrics(historical)
        recent_metrics = calc_metrics(recent)

        # Calcular mudanças
        changes = {}
        behavioral_score = 0.0

        # Mudança no valor médio
        if hist_metrics["avg_amount"] > 0:
            amount_change = abs(recent_metrics["avg_amount"] - hist_metrics["avg_amount"]) / hist_metrics["avg_amount"]
            if amount_change > 2:  # Mais que 200% de mudança
                behavioral_score += 0.3
                changes["amount_change_pct"] = amount_change * 100

        # Mudança na frequência
        hist_days = 30  # Aproximação
        recent_days = 10
        hist_freq = hist_metrics["tx_count"] / hist_days
        recent_freq = recent_metrics["tx_count"] / recent_days

        if hist_freq > 0:
            freq_change = recent_freq / hist_freq
            if freq_change > 3:  # 3x mais frequente
                behavioral_score += 0.3
                changes["frequency_multiplier"] = freq_change

        # Mudança no número de contrapartes
        if hist_metrics["unique_counterparties"] > 0:
            counterparty_change = recent_metrics["unique_counterparties"] / hist_metrics["unique_counterparties"]
            if counterparty_change > 2:
                behavioral_score += 0.2
                changes["counterparty_multiplier"] = counterparty_change

        behavioral_score = min(behavioral_score, 1.0)

        if behavioral_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="behavioral_change",
            score=behavioral_score,
            weight=config["weight"],
            description=config["description"],
            evidence=changes,
            confidence=0.7
        )

    def _analyze_network_position(
        self,
        account_id: str,
        network_data: Optional[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar posição na rede de transações"""

        if not network_data:
            return None

        config = self.INDICATOR_CONFIG["network_centrality"]

        # Extrair métricas de rede
        pagerank = network_data.get("pagerank", 0)
        betweenness = network_data.get("betweenness_centrality", 0)
        degree = network_data.get("degree_centrality", 0)
        in_suspicious_community = network_data.get("in_suspicious_community", False)

        # Calcular score
        network_score = 0.0
        evidence = {}

        # Alto PageRank em rede de fraude
        if pagerank > 0.01:  # Top percentile
            network_score += 0.3
            evidence["high_pagerank"] = pagerank

        # Alta centralidade de intermediação (ponte entre grupos)
        if betweenness > 0.05:
            network_score += 0.3
            evidence["high_betweenness"] = betweenness

        # Alto grau (muitas conexões)
        if degree > 0.1:
            network_score += 0.2
            evidence["high_degree"] = degree

        # Em comunidade suspeita
        if in_suspicious_community:
            network_score += 0.3
            evidence["in_suspicious_community"] = True

        network_score = min(network_score, 1.0)

        if network_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="network_centrality",
            score=network_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.85
        )

    def _analyze_recruitment_connection(
        self,
        account_id: str,
        transaction_history: List[Dict[str, Any]],
        network_data: Optional[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar conexões com mulas conhecidas"""

        config = self.INDICATOR_CONFIG["recruitment_connection"]

        # Identificar contrapartes
        counterparties = set()
        for tx in transaction_history:
            sender = tx.get("sender_id", tx.get("conta_origem", ""))
            receiver = tx.get("receiver_id", tx.get("conta_recebedor", ""))

            if sender and sender != account_id:
                counterparties.add(sender)
            if receiver and receiver != account_id:
                counterparties.add(receiver)

        # Verificar conexões com mulas conhecidas
        mule_connections = counterparties.intersection(self.known_mules)

        # Verificar conexões com contas suspeitas
        suspicious_connections = [
            cp for cp in counterparties
            if cp in self.suspicious_accounts and self.suspicious_accounts[cp] > 0.6
        ]

        # Usar dados de rede
        network_mule_connections = 0
        if network_data:
            network_mule_connections = network_data.get("connections_to_known_mules", 0)

        # Calcular score
        recruitment_score = 0.0
        evidence = {}

        if mule_connections:
            recruitment_score = min(len(mule_connections) * 0.3, 0.9)
            evidence["known_mule_connections"] = len(mule_connections)

        if suspicious_connections:
            recruitment_score = max(recruitment_score, min(len(suspicious_connections) * 0.15, 0.7))
            evidence["suspicious_connections"] = len(suspicious_connections)

        if network_mule_connections > 0:
            recruitment_score = max(recruitment_score, min(network_mule_connections * 0.2, 0.85))
            evidence["network_mule_connections"] = network_mule_connections

        if recruitment_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="recruitment_connection",
            score=recruitment_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.9 if mule_connections else 0.7
        )

    def _analyze_age_activity_mismatch(
        self,
        account_data: Dict[str, Any],
        transaction_history: List[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar incompatibilidade entre idade da conta e atividade"""

        config = self.INDICATOR_CONFIG["age_activity_mismatch"]

        account_created = account_data.get("created_at")
        if not account_created:
            return None

        if isinstance(account_created, str):
            account_created = datetime.fromisoformat(account_created.replace("Z", "+00:00"))

        account_age_days = (datetime.now() - account_created.replace(tzinfo=None)).days

        # Conta muito nova com muita atividade
        mismatch_score = 0.0
        evidence = {}

        if account_age_days < 30:
            if len(transaction_history) > 100:
                mismatch_score = 0.8
                evidence["new_account"] = True
                evidence["tx_count"] = len(transaction_history)
            elif len(transaction_history) > 50:
                mismatch_score = 0.5
                evidence["new_account"] = True
                evidence["tx_count"] = len(transaction_history)

        # Conta antiga que era dormant e agora tem muita atividade (capturado em dormancy)
        elif account_age_days > 365:
            recent_txs = [tx for tx in transaction_history if self._is_recent(tx, days=30)]
            if len(recent_txs) > len(transaction_history) * 0.8:
                mismatch_score = 0.6
                evidence["old_account_sudden_activity"] = True
                evidence["recent_tx_ratio"] = len(recent_txs) / len(transaction_history)

        if mismatch_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="age_activity_mismatch",
            score=mismatch_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.7
        )

    def _analyze_geographic_anomalies(
        self,
        transaction_history: List[Dict[str, Any]]
    ) -> Optional[MuleIndicator]:
        """Analisar anomalias geográficas"""

        config = self.INDICATOR_CONFIG["geographic_anomaly"]

        # Extrair localizações
        locations = []
        for tx in transaction_history:
            loc = tx.get("location", tx.get("ip_location", {}))
            if loc:
                locations.append(loc)

        if len(locations) < 5:
            return None

        # Analisar dispersão geográfica
        geo_score = 0.0
        evidence = {}

        # Contar estados/países únicos
        states = set()
        countries = set()
        for loc in locations:
            if isinstance(loc, dict):
                if loc.get("state"):
                    states.add(loc["state"])
                if loc.get("country"):
                    countries.add(loc["country"])

        # Muitos estados diferentes em pouco tempo
        if len(states) > 5:
            geo_score += 0.4
            evidence["unique_states"] = len(states)

        # Múltiplos países
        if len(countries) > 2:
            geo_score += 0.4
            evidence["unique_countries"] = len(countries)

        geo_score = min(geo_score, 1.0)

        if geo_score < config["threshold"] * 0.5:
            return None

        return MuleIndicator(
            name="geographic_anomaly",
            score=geo_score,
            weight=config["weight"],
            description=config["description"],
            evidence=evidence,
            confidence=0.6
        )

    def _calculate_total_score(self, indicators: List[MuleIndicator]) -> float:
        """Calcular score total ponderado"""

        if not indicators:
            return 0.0

        weighted_sum = sum(ind.score * ind.weight for ind in indicators)
        total_weight = sum(ind.weight for ind in indicators)

        if total_weight == 0:
            return 0.0

        # Normalizar pelo peso total possível
        max_weight = sum(c["weight"] for c in self.INDICATOR_CONFIG.values())
        normalized_score = weighted_sum / max_weight

        # Aplicar boost se múltiplos indicadores fortes
        strong_indicators = sum(1 for ind in indicators if ind.score > 0.7)
        if strong_indicators >= 3:
            normalized_score *= 1.15

        return min(normalized_score, 1.0)

    def _determine_mule_type(
        self,
        indicators: List[MuleIndicator],
        account_data: Dict[str, Any]
    ) -> MuleType:
        """Determinar tipo de mula baseado nos indicadores"""

        indicator_names = {ind.name for ind in indicators}

        # Conta dormant ativada
        if "dormancy_activation" in indicator_names:
            dormancy_ind = next(i for i in indicators if i.name == "dormancy_activation")
            if dormancy_ind.score > 0.7:
                return MuleType.DORMANT_ACTIVATED

        # Money mule tradicional (alto turnover + network)
        if "rapid_turnover" in indicator_names and "network_centrality" in indicator_names:
            return MuleType.MONEY_MULE

        # Possível account takeover
        if "behavioral_change" in indicator_names:
            behavioral_ind = next(i for i in indicators if i.name == "behavioral_change")
            if behavioral_ind.score > 0.8:
                return MuleType.ACCOUNT_TAKEOVER

        # Conexão com recrutamento
        if "recruitment_connection" in indicator_names:
            return MuleType.WITTING

        # Conta nova suspeita (possível synthetic ID)
        if "age_activity_mismatch" in indicator_names:
            age_ind = next(i for i in indicators if i.name == "age_activity_mismatch")
            if age_ind.evidence.get("new_account"):
                return MuleType.SYNTHETIC_ID

        return MuleType.UNKNOWN

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determinar nível de risco"""

        if score >= self.CLASSIFICATION_THRESHOLDS["critical"]:
            return RiskLevel.CRITICAL
        elif score >= self.CLASSIFICATION_THRESHOLDS["high"]:
            return RiskLevel.HIGH
        elif score >= self.CLASSIFICATION_THRESHOLDS["medium"]:
            return RiskLevel.MEDIUM
        elif score >= self.CLASSIFICATION_THRESHOLDS["low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _calculate_confidence(
        self,
        indicators: List[MuleIndicator],
        transaction_history: List[Dict[str, Any]]
    ) -> float:
        """Calcular confiança na análise"""

        base_confidence = 0.3

        # Mais indicadores = mais confiança
        base_confidence += min(len(indicators) * 0.1, 0.3)

        # Mais histórico = mais confiança
        base_confidence += min(len(transaction_history) / 100 * 0.2, 0.2)

        # Média de confiança dos indicadores
        if indicators:
            avg_indicator_confidence = np.mean([ind.confidence for ind in indicators])
            base_confidence += avg_indicator_confidence * 0.2

        return min(base_confidence, 0.95)

    def _generate_explanation(
        self,
        is_mule: bool,
        score: float,
        indicators: List[MuleIndicator],
        mule_type: MuleType
    ) -> str:
        """Gerar explicação da análise"""

        if not is_mule:
            return f"Conta não apresenta características significativas de mula (score: {score:.1%})"

        explanation_parts = [
            f"Conta identificada como possível mula do tipo {mule_type.value} "
            f"(probabilidade: {score:.1%})."
        ]

        # Top indicadores
        sorted_indicators = sorted(indicators, key=lambda x: x.score * x.weight, reverse=True)

        explanation_parts.append("\nPrincipais indicadores:")
        for ind in sorted_indicators[:3]:
            explanation_parts.append(f"- {ind.description} (score: {ind.score:.1%})")

        return "\n".join(explanation_parts)

    def _get_recommended_actions(
        self,
        is_mule: bool,
        risk_level: RiskLevel,
        mule_type: MuleType
    ) -> List[str]:
        """Determinar ações recomendadas"""

        if not is_mule:
            return ["Monitoramento padrão"]

        actions = []

        if risk_level == RiskLevel.CRITICAL:
            actions.extend([
                "BLOQUEIO IMEDIATO da conta",
                "Reportar ao COAF",
                "Investigação prioritária",
                "Congelar todos os fundos"
            ])
        elif risk_level == RiskLevel.HIGH:
            actions.extend([
                "Suspender transações de saída",
                "Solicitar verificação de identidade",
                "Escalar para equipe de compliance",
                "Monitoramento intensificado"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            actions.extend([
                "Limite de transações reduzido",
                "Verificação adicional para grandes valores",
                "Monitoramento aumentado",
                "Revisão manual de transações suspeitas"
            ])
        else:
            actions.extend([
                "Monitoramento aumentado",
                "Flag para revisão periódica"
            ])

        # Ações específicas por tipo
        if mule_type == MuleType.ACCOUNT_TAKEOVER:
            actions.append("Verificar legitimidade do titular atual")
        elif mule_type == MuleType.SYNTHETIC_ID:
            actions.append("Verificação aprofundada de documentos")

        return actions

    def _identify_suspicious_patterns(
        self,
        indicators: List[MuleIndicator],
        transaction_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Identificar padrões suspeitos específicos"""

        patterns = []

        for ind in indicators:
            if ind.score > 0.6:
                if ind.name == "rapid_turnover" and ind.evidence.get("avg_turnover_hours", 99) < 2:
                    patterns.append("Turnover ultra-rápido (<2h)")

                if ind.name == "circular_flows" and ind.evidence.get("cycles_found", 0) > 2:
                    patterns.append("Múltiplos ciclos de transações detectados")

                if ind.name == "dormancy_activation" and ind.evidence.get("activation_multiplier", 0) > 5:
                    patterns.append("Ativação explosiva após dormência")

                if ind.name == "recruitment_connection" and ind.evidence.get("known_mule_connections", 0) > 0:
                    patterns.append("Conexão direta com mulas conhecidas")

        return patterns

    def _count_network_connections(
        self,
        account_id: str,
        network_data: Optional[Dict[str, Any]]
    ) -> int:
        """Contar conexões de rede"""

        if network_data:
            return network_data.get("total_connections", 0)
        return 0

    def _is_recent(self, tx: Dict[str, Any], days: int = 7) -> bool:
        """Verificar se transação é recente"""

        tx_date = tx.get("timestamp", tx.get("created_at"))
        if not tx_date:
            return False

        if isinstance(tx_date, str):
            tx_date = datetime.fromisoformat(tx_date.replace("Z", "+00:00"))

        cutoff = datetime.now() - timedelta(days=days)
        return tx_date.replace(tzinfo=None) > cutoff

    def add_known_mule(self, account_id: str):
        """Adicionar conta à lista de mulas conhecidas"""
        self.known_mules.add(account_id)

    def add_suspicious_account(self, account_id: str, score: float):
        """Adicionar conta suspeita com score"""
        self.suspicious_accounts[account_id] = score

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do detector"""
        return {
            "version": self.VERSION,
            "known_mules_count": len(self.known_mules),
            "suspicious_accounts_count": len(self.suspicious_accounts),
            "indicator_count": len(self.INDICATOR_CONFIG)
        }


def create_mule_detector(config: Optional[Dict[str, Any]] = None) -> MuleDetector:
    """Factory function para criar MuleDetector"""
    return MuleDetector(config)
