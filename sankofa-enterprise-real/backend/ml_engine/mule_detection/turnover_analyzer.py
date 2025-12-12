"""
Sankofa Enterprise Pro - Turnover Analyzer
Analisador de padrões de turnover (entrada/saída rápida de fundos)

Baseado em:
- BioCatch mule detection research
- FATF money mule indicators
- Financial crime typologies
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TurnoverEvent:
    """Evento individual de turnover"""
    inflow_time: datetime
    outflow_time: datetime
    inflow_amount: float
    outflow_amount: float
    turnover_seconds: int
    retention_ratio: float  # Quanto ficou na conta


@dataclass
class TurnoverPattern:
    """Padrão de turnover da conta"""
    account_id: str
    is_rapid_turnover: bool
    turnover_score: float
    avg_turnover_time_hours: float
    min_turnover_time_hours: float
    total_inflow: float
    total_outflow: float
    turnover_ratio: float
    turnover_events: List[TurnoverEvent]
    daily_patterns: Dict[str, float]
    hourly_patterns: Dict[int, float]
    suspicious_behaviors: List[str]
    risk_level: str
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class TurnoverAnalyzer:
    """
    Analisador de Turnover de Fundos

    Detecta padrões de:
    - Dinheiro que entra e sai rapidamente
    - Múltiplas transferências em curto período
    - Padrões de "pass-through" (conta apenas de passagem)
    - Retenção mínima de fundos
    """

    VERSION = "1.0.0"

    # Thresholds de configuração
    RAPID_TURNOVER_HOURS = 4  # Turnover < 4h é suspeito
    HIGH_TURNOVER_RATIO = 0.9  # Se > 90% sai rapidamente
    MIN_TRANSACTIONS = 5  # Mínimo para análise

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rapid_hours = self.config.get("rapid_turnover_hours", self.RAPID_TURNOVER_HOURS)
        self.high_ratio = self.config.get("high_turnover_ratio", self.HIGH_TURNOVER_RATIO)

        logger.info(f"TurnoverAnalyzer v{self.VERSION} initialized")

    def analyze(
        self,
        account_id: str,
        transaction_history: List[Dict[str, Any]]
    ) -> TurnoverPattern:
        """
        Analisar padrões de turnover de uma conta

        Args:
            account_id: ID da conta
            transaction_history: Histórico de transações

        Returns:
            TurnoverPattern com análise completa
        """
        # Separar inflows e outflows
        inflows, outflows = self._separate_flows(account_id, transaction_history)

        # Calcular totais
        total_inflow = sum(f["amount"] for f in inflows)
        total_outflow = sum(f["amount"] for f in outflows)

        # Identificar eventos de turnover
        turnover_events = self._identify_turnover_events(inflows, outflows)

        # Calcular métricas de turnover
        avg_turnover_time = self._calculate_avg_turnover_time(turnover_events)
        min_turnover_time = self._calculate_min_turnover_time(turnover_events)

        # Calcular ratio de turnover
        turnover_ratio = total_outflow / total_inflow if total_inflow > 0 else 0

        # Analisar padrões temporais
        daily_patterns = self._analyze_daily_patterns(inflows, outflows)
        hourly_patterns = self._analyze_hourly_patterns(inflows, outflows)

        # Identificar comportamentos suspeitos
        suspicious_behaviors = self._identify_suspicious_behaviors(
            turnover_events, turnover_ratio, avg_turnover_time,
            inflows, outflows
        )

        # Calcular score de turnover
        turnover_score = self._calculate_turnover_score(
            avg_turnover_time, turnover_ratio,
            turnover_events, suspicious_behaviors
        )

        # Determinar se é rapid turnover
        is_rapid = self._is_rapid_turnover(
            avg_turnover_time, turnover_ratio, turnover_score
        )

        # Determinar nível de risco
        risk_level = self._determine_risk_level(turnover_score)

        # Gerar explicação
        explanation = self._generate_explanation(
            is_rapid, turnover_score, avg_turnover_time,
            turnover_ratio, suspicious_behaviors
        )

        return TurnoverPattern(
            account_id=account_id,
            is_rapid_turnover=is_rapid,
            turnover_score=turnover_score,
            avg_turnover_time_hours=avg_turnover_time,
            min_turnover_time_hours=min_turnover_time,
            total_inflow=total_inflow,
            total_outflow=total_outflow,
            turnover_ratio=turnover_ratio,
            turnover_events=turnover_events[:20],  # Limitar para não sobrecarregar
            daily_patterns=daily_patterns,
            hourly_patterns=hourly_patterns,
            suspicious_behaviors=suspicious_behaviors,
            risk_level=risk_level,
            explanation=explanation
        )

    def _separate_flows(
        self,
        account_id: str,
        transactions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Separar transações em inflows e outflows"""

        inflows = []
        outflows = []

        for tx in transactions:
            amount = float(tx.get("amount", tx.get("value", 0)))
            tx_type = tx.get("type", tx.get("transaction_type", "")).lower()
            timestamp = self._get_timestamp(tx)

            # Determinar direção
            is_inflow = False

            if tx_type in ["credit", "deposit", "receive", "pix_in", "ted_in", "doc_in"]:
                is_inflow = True
            elif tx_type in ["debit", "withdrawal", "send", "pix_out", "ted_out", "doc_out"]:
                is_inflow = False
            else:
                # Inferir pela estrutura
                receiver = tx.get("receiver_id", tx.get("conta_recebedor", ""))
                sender = tx.get("sender_id", tx.get("conta_origem", ""))

                if receiver == account_id:
                    is_inflow = True
                elif sender == account_id:
                    is_inflow = False
                else:
                    # Usar sinal do amount se disponível
                    is_inflow = amount > 0

            flow = {
                "amount": abs(amount),
                "timestamp": timestamp,
                "counterparty": tx.get("receiver_id") if not is_inflow else tx.get("sender_id"),
                "original_tx": tx
            }

            if is_inflow:
                inflows.append(flow)
            else:
                outflows.append(flow)

        # Ordenar por timestamp
        inflows.sort(key=lambda x: x["timestamp"])
        outflows.sort(key=lambda x: x["timestamp"])

        return inflows, outflows

    def _identify_turnover_events(
        self,
        inflows: List[Dict],
        outflows: List[Dict]
    ) -> List[TurnoverEvent]:
        """Identificar eventos de turnover (inflow seguido de outflow próximo)"""

        events = []

        for inflow in inflows:
            in_time = inflow["timestamp"]
            in_amount = inflow["amount"]

            # Buscar outflows que ocorrem após este inflow
            for outflow in outflows:
                out_time = outflow["timestamp"]
                out_amount = outflow["amount"]

                # Outflow deve ser depois do inflow
                if out_time <= in_time:
                    continue

                # Calcular tempo entre eventos
                time_diff = (out_time - in_time).total_seconds()

                # Se está dentro de 24h, considerar como evento de turnover
                if time_diff < 86400:  # 24 horas
                    # Verificar se amounts são similares (possível mesmo dinheiro)
                    amount_ratio = min(out_amount, in_amount) / max(out_amount, in_amount) if max(out_amount, in_amount) > 0 else 0

                    # Se amounts são similares (>50%), é provável turnover do mesmo valor
                    if amount_ratio > 0.5:
                        retention = 1 - (out_amount / in_amount) if in_amount > 0 else 0

                        events.append(TurnoverEvent(
                            inflow_time=in_time,
                            outflow_time=out_time,
                            inflow_amount=in_amount,
                            outflow_amount=out_amount,
                            turnover_seconds=int(time_diff),
                            retention_ratio=max(0, retention)
                        ))
                        break  # Um outflow por inflow

        return events

    def _calculate_avg_turnover_time(self, events: List[TurnoverEvent]) -> float:
        """Calcular tempo médio de turnover em horas"""

        if not events:
            return float('inf')

        times = [e.turnover_seconds / 3600 for e in events]
        return np.mean(times)

    def _calculate_min_turnover_time(self, events: List[TurnoverEvent]) -> float:
        """Calcular tempo mínimo de turnover em horas"""

        if not events:
            return float('inf')

        times = [e.turnover_seconds / 3600 for e in events]
        return min(times)

    def _analyze_daily_patterns(
        self,
        inflows: List[Dict],
        outflows: List[Dict]
    ) -> Dict[str, float]:
        """Analisar padrões por dia da semana"""

        patterns = defaultdict(float)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for flow in inflows + outflows:
            day_idx = flow["timestamp"].weekday()
            patterns[days[day_idx]] += flow["amount"]

        return dict(patterns)

    def _analyze_hourly_patterns(
        self,
        inflows: List[Dict],
        outflows: List[Dict]
    ) -> Dict[int, float]:
        """Analisar padrões por hora do dia"""

        patterns = defaultdict(float)

        for flow in inflows + outflows:
            hour = flow["timestamp"].hour
            patterns[hour] += flow["amount"]

        return dict(patterns)

    def _identify_suspicious_behaviors(
        self,
        turnover_events: List[TurnoverEvent],
        turnover_ratio: float,
        avg_turnover_time: float,
        inflows: List[Dict],
        outflows: List[Dict]
    ) -> List[str]:
        """Identificar comportamentos suspeitos"""

        suspicious = []

        # 1. Turnover muito rápido
        if avg_turnover_time < 1:  # Menos de 1 hora
            suspicious.append("Turnover ultra-rápido (< 1 hora)")
        elif avg_turnover_time < self.rapid_hours:
            suspicious.append(f"Turnover rápido (< {self.rapid_hours}h)")

        # 2. Ratio de saída muito alto
        if turnover_ratio > 0.95:
            suspicious.append("Quase 100% dos fundos saem rapidamente")
        elif turnover_ratio > self.high_ratio:
            suspicious.append(f"Alto ratio de saída ({turnover_ratio:.1%})")

        # 3. Múltiplos outflows após único inflow
        inflow_dates = defaultdict(int)
        for inflow in inflows:
            date_key = inflow["timestamp"].date()
            inflow_dates[date_key] += 1

        outflow_dates = defaultdict(int)
        for outflow in outflows:
            date_key = outflow["timestamp"].date()
            outflow_dates[date_key] += 1

        for date, in_count in inflow_dates.items():
            out_count = outflow_dates.get(date, 0)
            if in_count <= 2 and out_count >= 5:
                suspicious.append("Fan-out: poucos inflows, muitos outflows no mesmo dia")
                break

        # 4. Valores redondos frequentes
        round_amounts = 0
        for flow in inflows + outflows:
            if flow["amount"] % 100 == 0:
                round_amounts += 1

        if round_amounts > len(inflows + outflows) * 0.7:
            suspicious.append("Alta frequência de valores redondos")

        # 5. Atividade fora do horário comercial
        off_hours = 0
        for flow in inflows + outflows:
            hour = flow["timestamp"].hour
            if hour < 6 or hour > 22:
                off_hours += 1

        if off_hours > len(inflows + outflows) * 0.3:
            suspicious.append("Muitas transações fora do horário comercial")

        # 6. Turnover consistente (muito regular)
        if len(turnover_events) >= 5:
            times = [e.turnover_seconds for e in turnover_events]
            cv = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            if cv < 0.2:  # Muito regular
                suspicious.append("Padrão de turnover muito regular (possível automação)")

        # 7. Split transactions (valores divididos)
        amounts = [f["amount"] for f in outflows]
        if len(amounts) >= 3:
            # Verificar se há valores que somam um valor redondo
            for i in range(len(amounts) - 2):
                for j in range(i + 1, len(amounts) - 1):
                    for k in range(j + 1, len(amounts)):
                        total = amounts[i] + amounts[j] + amounts[k]
                        if total % 1000 == 0 and 5000 <= total <= 50000:
                            suspicious.append("Possível divisão de valores para evitar limites")
                            break

        return suspicious

    def _calculate_turnover_score(
        self,
        avg_turnover_time: float,
        turnover_ratio: float,
        turnover_events: List[TurnoverEvent],
        suspicious_behaviors: List[str]
    ) -> float:
        """Calcular score de risco de turnover"""

        score = 0.0

        # Score baseado no tempo de turnover
        if avg_turnover_time < 1:
            score += 0.4
        elif avg_turnover_time < 2:
            score += 0.3
        elif avg_turnover_time < self.rapid_hours:
            score += 0.2

        # Score baseado no ratio
        if turnover_ratio > 0.95:
            score += 0.3
        elif turnover_ratio > 0.9:
            score += 0.2
        elif turnover_ratio > 0.8:
            score += 0.1

        # Score baseado na quantidade de eventos
        if len(turnover_events) >= 10:
            score += 0.2
        elif len(turnover_events) >= 5:
            score += 0.1

        # Score baseado em comportamentos suspeitos
        score += min(len(suspicious_behaviors) * 0.05, 0.2)

        # Baixa retenção
        if turnover_events:
            avg_retention = np.mean([e.retention_ratio for e in turnover_events])
            if avg_retention < 0.05:  # Menos de 5% fica na conta
                score += 0.15

        return min(score, 1.0)

    def _is_rapid_turnover(
        self,
        avg_turnover_time: float,
        turnover_ratio: float,
        turnover_score: float
    ) -> bool:
        """Determinar se é rapid turnover"""

        # É rapid se:
        # - Tempo médio < threshold E ratio alto
        # - OU score muito alto
        if avg_turnover_time < self.rapid_hours and turnover_ratio > 0.8:
            return True

        if turnover_score > 0.6:
            return True

        return False

    def _determine_risk_level(self, score: float) -> str:
        """Determinar nível de risco"""

        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"

    def _generate_explanation(
        self,
        is_rapid: bool,
        score: float,
        avg_time: float,
        ratio: float,
        suspicious: List[str]
    ) -> str:
        """Gerar explicação da análise"""

        if not is_rapid and score < 0.3:
            return "Padrão de turnover normal, sem indicadores significativos de mula."

        parts = []

        if is_rapid:
            parts.append(
                f"ALERTA: Padrão de rapid turnover detectado (score: {score:.1%})."
            )
        else:
            parts.append(
                f"Conta apresenta alguns indicadores de turnover suspeito (score: {score:.1%})."
            )

        if avg_time < float('inf'):
            parts.append(f"Tempo médio de turnover: {avg_time:.1f} horas.")

        parts.append(f"Ratio de saída: {ratio:.1%} dos fundos que entram.")

        if suspicious:
            parts.append("Comportamentos suspeitos: " + "; ".join(suspicious[:3]))

        return " ".join(parts)

    def _get_timestamp(self, tx: Dict[str, Any]) -> datetime:
        """Extrair timestamp da transação"""

        ts = tx.get("timestamp", tx.get("created_at", ""))

        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)

        if isinstance(ts, str) and ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass

        return datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Retornar estatísticas do analisador"""
        return {
            "version": self.VERSION,
            "rapid_turnover_hours": self.rapid_hours,
            "high_turnover_ratio": self.high_ratio
        }


def create_turnover_analyzer(config: Optional[Dict[str, Any]] = None) -> TurnoverAnalyzer:
    """Factory function para criar TurnoverAnalyzer"""
    return TurnoverAnalyzer(config)
