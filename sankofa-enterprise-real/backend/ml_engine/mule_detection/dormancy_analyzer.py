"""
Sankofa Enterprise Pro - Dormancy Analyzer
Analisador de padrões de dormência e ativação de contas

Baseado em:
- BioCatch dormancy detection
- Academic research on mule account patterns
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class DormancyPeriod:
    """Período de dormência identificado"""
    start_date: datetime
    end_date: datetime
    duration_days: int
    preceding_activity: float  # Transações/dia antes
    following_activity: float  # Transações/dia depois


@dataclass
class DormancyPattern:
    """Padrão de dormência da conta"""
    account_id: str
    is_dormant_activated: bool
    dormancy_score: float
    dormancy_periods: List[DormancyPeriod]
    current_state: str  # "active", "dormant", "activating"
    activation_intensity: float
    days_since_last_activity: int
    activity_trend: str  # "increasing", "decreasing", "stable", "spike"
    anomaly_detected: bool
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class DormancyAnalyzer:
    """
    Analisador de Dormência de Contas

    Detecta padrões suspeitos de:
    - Contas dormentes que subitamente se ativam
    - Mudanças bruscas no padrão de atividade
    - Contas criadas e deixadas inativas para posterior uso
    """

    VERSION = "1.0.0"

    # Thresholds de configuração
    DORMANCY_THRESHOLD_DAYS = 30  # Dias sem atividade = dormência
    ACTIVATION_THRESHOLD = 3.0  # Multiplicador de atividade para detectar ativação
    MIN_HISTORY_DAYS = 60  # Mínimo de histórico para análise

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dormancy_days = self.config.get("dormancy_days", self.DORMANCY_THRESHOLD_DAYS)
        self.activation_threshold = self.config.get("activation_threshold", self.ACTIVATION_THRESHOLD)

        logger.info(f"DormancyAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        account_id: str,
        account_data: Dict[str, Any],
        transaction_history: List[Dict[str, Any]]
    ) -> DormancyPattern:
        """
        Analisar padrões de dormência de uma conta

        Args:
            account_id: ID da conta
            account_data: Dados cadastrais da conta
            transaction_history: Histórico de transações

        Returns:
            DormancyPattern com análise completa
        """
        # Ordenar transações por data
        sorted_txs = self._sort_transactions(transaction_history)

        # Identificar períodos de dormência
        dormancy_periods = self._identify_dormancy_periods(sorted_txs)

        # Calcular atividade atual vs histórica
        current_activity = self._calculate_recent_activity(sorted_txs, days=7)
        historical_activity = self._calculate_historical_activity(sorted_txs)

        # Determinar estado atual
        current_state = self._determine_current_state(sorted_txs, current_activity)

        # Calcular intensidade de ativação
        activation_intensity = self._calculate_activation_intensity(
            current_activity, historical_activity
        )

        # Determinar tendência
        activity_trend = self._determine_activity_trend(sorted_txs)

        # Calcular dias desde última atividade
        days_since_last = self._days_since_last_activity(sorted_txs)

        # Calcular score de dormência
        dormancy_score = self._calculate_dormancy_score(
            dormancy_periods,
            activation_intensity,
            current_state,
            account_data
        )

        # Determinar se é dormant activated
        is_dormant_activated = self._is_dormant_activated(
            dormancy_periods,
            activation_intensity,
            current_state
        )

        # Detectar anomalia
        anomaly_detected = dormancy_score > 0.6 or (
            is_dormant_activated and activation_intensity > 5
        )

        # Gerar explicação
        explanation = self._generate_explanation(
            is_dormant_activated,
            dormancy_score,
            dormancy_periods,
            activation_intensity
        )

        return DormancyPattern(
            account_id=account_id,
            is_dormant_activated=is_dormant_activated,
            dormancy_score=dormancy_score,
            dormancy_periods=dormancy_periods,
            current_state=current_state,
            activation_intensity=activation_intensity,
            days_since_last_activity=days_since_last,
            activity_trend=activity_trend,
            anomaly_detected=anomaly_detected,
            explanation=explanation
        )

    def _sort_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ordenar transações por data"""

        def get_date(tx):
            date_str = tx.get("timestamp", tx.get("created_at", ""))
            if isinstance(date_str, str) and date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            elif isinstance(date_str, datetime):
                return date_str
            return datetime.min

        return sorted(transactions, key=get_date)

    def _identify_dormancy_periods(
        self,
        sorted_txs: List[Dict[str, Any]]
    ) -> List[DormancyPeriod]:
        """Identificar períodos de dormência"""

        if len(sorted_txs) < 2:
            return []

        dormancy_periods = []
        last_date = None

        for i, tx in enumerate(sorted_txs):
            tx_date = self._get_tx_date(tx)

            if last_date:
                gap_days = (tx_date - last_date).days

                if gap_days >= self.dormancy_days:
                    # Calcular atividade antes e depois
                    preceding = self._calculate_activity_around_date(
                        sorted_txs, last_date, before=True
                    )
                    following = self._calculate_activity_around_date(
                        sorted_txs, tx_date, before=False
                    )

                    dormancy_periods.append(DormancyPeriod(
                        start_date=last_date,
                        end_date=tx_date,
                        duration_days=gap_days,
                        preceding_activity=preceding,
                        following_activity=following
                    ))

            last_date = tx_date

        return dormancy_periods

    def _calculate_activity_around_date(
        self,
        sorted_txs: List[Dict[str, Any]],
        reference_date: datetime,
        before: bool,
        window_days: int = 14
    ) -> float:
        """Calcular taxa de atividade antes ou depois de uma data"""

        if before:
            start = reference_date - timedelta(days=window_days)
            end = reference_date
        else:
            start = reference_date
            end = reference_date + timedelta(days=window_days)

        relevant_txs = [
            tx for tx in sorted_txs
            if start <= self._get_tx_date(tx) <= end
        ]

        return len(relevant_txs) / window_days if window_days > 0 else 0

    def _calculate_recent_activity(
        self,
        sorted_txs: List[Dict[str, Any]],
        days: int = 7
    ) -> float:
        """Calcular atividade recente (transações/dia)"""

        cutoff = datetime.now() - timedelta(days=days)
        recent_txs = [
            tx for tx in sorted_txs
            if self._get_tx_date(tx) > cutoff
        ]

        return len(recent_txs) / days if days > 0 else 0

    def _calculate_historical_activity(
        self,
        sorted_txs: List[Dict[str, Any]]
    ) -> float:
        """Calcular atividade histórica média"""

        if len(sorted_txs) < 2:
            return 0

        first_date = self._get_tx_date(sorted_txs[0])
        last_date = self._get_tx_date(sorted_txs[-1])

        total_days = (last_date - first_date).days

        if total_days < 1:
            return 0

        return len(sorted_txs) / total_days

    def _determine_current_state(
        self,
        sorted_txs: List[Dict[str, Any]],
        current_activity: float
    ) -> str:
        """Determinar estado atual da conta"""

        if not sorted_txs:
            return "inactive"

        last_tx_date = self._get_tx_date(sorted_txs[-1])
        days_since_last = (datetime.now() - last_tx_date).days

        if days_since_last > self.dormancy_days:
            return "dormant"
        elif current_activity > 0 and days_since_last < 7:
            # Verificar se está saindo de dormência
            historical = self._calculate_historical_activity(sorted_txs)
            if current_activity > historical * self.activation_threshold:
                return "activating"
            return "active"
        else:
            return "low_activity"

    def _calculate_activation_intensity(
        self,
        current_activity: float,
        historical_activity: float
    ) -> float:
        """Calcular intensidade da ativação"""

        if historical_activity <= 0:
            return current_activity * 10 if current_activity > 0 else 0

        return current_activity / historical_activity

    def _determine_activity_trend(
        self,
        sorted_txs: List[Dict[str, Any]]
    ) -> str:
        """Determinar tendência de atividade"""

        if len(sorted_txs) < 10:
            return "insufficient_data"

        # Comparar últimos 7 dias vs 7 dias anteriores
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        two_weeks_ago = now - timedelta(days=14)

        recent_count = sum(1 for tx in sorted_txs if self._get_tx_date(tx) > week_ago)
        previous_count = sum(
            1 for tx in sorted_txs
            if two_weeks_ago < self._get_tx_date(tx) <= week_ago
        )

        if previous_count == 0:
            if recent_count > 5:
                return "spike"
            return "stable"

        ratio = recent_count / previous_count

        if ratio > 3:
            return "spike"
        elif ratio > 1.5:
            return "increasing"
        elif ratio < 0.5:
            return "decreasing"
        else:
            return "stable"

    def _days_since_last_activity(
        self,
        sorted_txs: List[Dict[str, Any]]
    ) -> int:
        """Calcular dias desde última atividade"""

        if not sorted_txs:
            return 9999

        last_date = self._get_tx_date(sorted_txs[-1])
        return (datetime.now() - last_date).days

    def _calculate_dormancy_score(
        self,
        dormancy_periods: List[DormancyPeriod],
        activation_intensity: float,
        current_state: str,
        account_data: Dict[str, Any]
    ) -> float:
        """Calcular score de risco de dormência"""

        score = 0.0

        # Score baseado em períodos de dormência
        if dormancy_periods:
            # Período de dormência longo seguido de ativação
            for period in dormancy_periods:
                if period.duration_days > 60 and period.following_activity > 1:
                    score += 0.3
                elif period.duration_days > 30 and period.following_activity > 2:
                    score += 0.2

        # Score baseado na intensidade de ativação
        if activation_intensity > 10:
            score += 0.4
        elif activation_intensity > 5:
            score += 0.2
        elif activation_intensity > 3:
            score += 0.1

        # Score baseado no estado atual
        if current_state == "activating":
            score += 0.2

        # Conta nova com muita atividade
        account_created = account_data.get("created_at")
        if account_created:
            if isinstance(account_created, str):
                account_created = datetime.fromisoformat(account_created.replace("Z", "+00:00"))
            account_age = (datetime.now() - account_created.replace(tzinfo=None)).days

            if account_age < 30 and activation_intensity > 0:
                score += 0.2

        return min(score, 1.0)

    def _is_dormant_activated(
        self,
        dormancy_periods: List[DormancyPeriod],
        activation_intensity: float,
        current_state: str
    ) -> bool:
        """Determinar se conta é dormant-activated"""

        # Teve período de dormência E está se ativando
        if dormancy_periods and current_state == "activating":
            return True

        # Teve período de dormência longo E atividade intensa recente
        for period in dormancy_periods:
            if period.duration_days > 45:
                if period.following_activity > period.preceding_activity * 3:
                    return True

        # Ativação muito intensa
        if activation_intensity > 5 and dormancy_periods:
            return True

        return False

    def _generate_explanation(
        self,
        is_dormant_activated: bool,
        dormancy_score: float,
        dormancy_periods: List[DormancyPeriod],
        activation_intensity: float
    ) -> str:
        """Gerar explicação da análise"""

        if not is_dormant_activated and dormancy_score < 0.3:
            return "Conta com padrão de atividade normal, sem sinais de dormência suspeita."

        parts = []

        if is_dormant_activated:
            parts.append(
                f"ALERTA: Conta identificada como dormant-activated "
                f"(score de risco: {dormancy_score:.1%})."
            )
        else:
            parts.append(
                f"Conta apresenta alguns sinais de dormência "
                f"(score: {dormancy_score:.1%})."
            )

        if dormancy_periods:
            longest = max(dormancy_periods, key=lambda p: p.duration_days)
            parts.append(
                f"Maior período de dormência: {longest.duration_days} dias."
            )

        if activation_intensity > 3:
            parts.append(
                f"Intensidade de ativação: {activation_intensity:.1f}x o normal."
            )

        return " ".join(parts)

    def _get_tx_date(self, tx: Dict[str, Any]) -> datetime:
        """Extrair data da transação"""

        date_str = tx.get("timestamp", tx.get("created_at", ""))

        if isinstance(date_str, datetime):
            return date_str.replace(tzinfo=None)

        if isinstance(date_str, str) and date_str:
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass

        return datetime.min

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "dormancy_threshold_days": self.dormancy_days,
            "activation_threshold": self.activation_threshold
        }


def create_dormancy_analyzer(config: Optional[Dict[str, Any]] = None) -> DormancyAnalyzer:
    """Factory function para criar DormancyAnalyzer"""
    return DormancyAnalyzer(config)
