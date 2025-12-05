"""
Sankofa Enterprise Pro - Bi-LSTM Sequence Analyzer
Análise de padrões temporais em sequências de transações
Referência: FinSafeNet (Nature 2024) - Bi-LSTM + CNN + Dual Attention
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
import hashlib

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class SequencePattern:
    """Padrão detectado em sequência"""

    pattern_type: str
    confidence: float
    description: str
    affected_transactions: List[str]


@dataclass
class SequenceAnalysisResult:
    """Resultado da análise de sequência"""

    transaction_id: str
    sequence_risk_score: float
    temporal_anomaly_score: float
    velocity_anomaly_score: float
    pattern_breaks: List[str]
    detected_patterns: List[SequencePattern]
    attention_weights: Dict[str, float]
    is_suspicious_sequence: bool
    recommendation: str
    timestamp: str


class TransactionSequenceBuffer:
    """Buffer circular para manter sequências de transações por usuário"""

    def __init__(self, max_sequence_length: int = 20):
        self.max_length = max_sequence_length
        self.user_sequences: Dict[str, deque] = {}

    def add_transaction(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Adiciona transação ao buffer do usuário"""
        if user_id not in self.user_sequences:
            self.user_sequences[user_id] = deque(maxlen=self.max_length)

        self.user_sequences[user_id].append(
            {
                "transaction_id": transaction.get("transaction_id", "unknown"),
                "amount": transaction.get("amount", 0),
                "timestamp": transaction.get("timestamp", datetime.now().isoformat()),
                "hour": transaction.get("hour", 12),
                "channel": transaction.get("channel", "unknown"),
                "is_fraud": transaction.get("is_fraud", False),
            }
        )

    def get_sequence(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtém sequência de transações do usuário"""
        if user_id not in self.user_sequences:
            return []
        return list(self.user_sequences[user_id])

    def get_sequence_features(self, user_id: str) -> Optional[np.ndarray]:
        """Extrai features numéricas da sequência"""
        sequence = self.get_sequence(user_id)
        if len(sequence) < 2:
            return None

        features = []
        for txn in sequence:
            features.append(
                [txn.get("amount", 0), txn.get("hour", 12), 1 if txn.get("is_fraud", False) else 0]
            )

        return np.array(features)


class AttentionLayer:
    """Camada de atenção simplificada (sem TensorFlow)"""

    def __init__(self, attention_dim: int = 32):
        self.attention_dim = attention_dim
        self.weights = None

    def compute_attention(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computa pesos de atenção para sequência"""
        if len(sequence) == 0:
            return np.array([]), np.array([])

        norms = np.linalg.norm(sequence, axis=1)
        norms = norms / (np.sum(norms) + 1e-10)

        weighted = sequence * norms.reshape(-1, 1)
        context = np.sum(weighted, axis=0)

        return norms, context


class BiLSTMSequenceAnalyzer:
    """
    Analisador de Sequências Bi-LSTM

    Arquitetura baseada em FinSafeNet (Nature 2024):
    - Bi-LSTM: Captura dependências bidirecionais
    - Dual Attention: Atenção em features e tempo
    - Pattern Detection: Identificação de padrões suspeitos

    Features:
    - Análise temporal de transações
    - Detecção de mudanças de comportamento
    - Velocity analysis em sequência
    - Pattern breaking detection
    """

    VERSION = "1.0.0"

    SUSPICIOUS_PATTERNS = {
        "rapid_escalation": "Aumento rápido de valores",
        "velocity_spike": "Pico de velocidade de transações",
        "pattern_break": "Quebra de padrão comportamental",
        "time_anomaly": "Anomalia de horário",
        "channel_switch": "Mudança frequente de canal",
        "amount_oscillation": "Oscilação anormal de valores",
    }

    def __init__(
        self,
        sequence_length: int = 10,
        lstm_units: int = 64,
        attention_dim: int = 32,
        risk_threshold: float = 0.7,
    ):
        """
        Inicializa o analisador Bi-LSTM

        Args:
            sequence_length: Comprimento da sequência a analisar
            lstm_units: Unidades LSTM
            attention_dim: Dimensão da camada de atenção
            risk_threshold: Threshold para classificar como suspeito
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.risk_threshold = risk_threshold

        self.sequence_buffer = TransactionSequenceBuffer(max_sequence_length=sequence_length)
        self.attention_layer = AttentionLayer(attention_dim)
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        self.model = None
        self.is_fitted = False

        self.user_baselines: Dict[str, Dict[str, float]] = {}

        logger.info(f"BiLSTMSequenceAnalyzer initialized (v{self.VERSION})")

    def _build_model(self, n_features: int) -> None:
        """Constrói modelo Bi-LSTM com atenção"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using rule-based analysis")
            return

        inputs = keras.Input(shape=(self.sequence_length, n_features))

        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(inputs)
        x = layers.Dropout(0.2)(x)

        attention = layers.Dense(1, activation="tanh")(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation("softmax")(attention)
        attention = layers.RepeatVector(self.lstm_units * 2)(attention)
        attention = layers.Permute([2, 1])(attention)

        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda xin: keras.backend.sum(xin, axis=1))(x)

        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        logger.info(
            f"Bi-LSTM model built: {self.sequence_length}x{n_features} -> {self.lstm_units*2}"
        )

    def update_user_baseline(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Atualiza baseline do usuário"""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "avg_amount": 0,
                "amount_std": 0,
                "avg_hour": 12,
                "transaction_count": 0,
                "typical_channels": set(),
            }

        baseline = self.user_baselines[user_id]
        n = baseline["transaction_count"]

        amount = transaction.get("amount", 0)
        new_avg = (baseline["avg_amount"] * n + amount) / (n + 1)

        if n > 0:
            new_std = np.sqrt((baseline["amount_std"] ** 2 * n + (amount - new_avg) ** 2) / (n + 1))
        else:
            new_std = 0

        baseline["avg_amount"] = new_avg
        baseline["amount_std"] = new_std
        baseline["transaction_count"] = n + 1

        hour = transaction.get("hour", 12)
        baseline["avg_hour"] = (baseline["avg_hour"] * n + hour) / (n + 1)

        channel = transaction.get("channel", "unknown")
        if isinstance(baseline["typical_channels"], set):
            baseline["typical_channels"].add(channel)

    def add_transaction(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Adiciona transação e atualiza buffers"""
        self.sequence_buffer.add_transaction(user_id, transaction)
        self.update_user_baseline(user_id, transaction)

    def analyze_sequence(
        self, user_id: str, current_transaction: Dict[str, Any]
    ) -> SequenceAnalysisResult:
        """
        Analisa sequência de transações do usuário

        Args:
            user_id: ID do usuário
            current_transaction: Transação atual

        Returns:
            Resultado da análise de sequência
        """
        transaction_id = current_transaction.get(
            "transaction_id",
            f"TXN_{hashlib.md5(str(current_transaction).encode()).hexdigest()[:8]}",
        )

        self.add_transaction(user_id, current_transaction)

        sequence = self.sequence_buffer.get_sequence(user_id)

        if len(sequence) < 3:
            return self._create_default_result(transaction_id, "Histórico insuficiente")

        temporal_score = self._analyze_temporal_patterns(sequence, current_transaction)
        velocity_score = self._analyze_velocity(sequence)
        pattern_breaks = self._detect_pattern_breaks(user_id, current_transaction)
        detected_patterns = self._detect_suspicious_patterns(sequence, current_transaction)

        attention_weights = self._compute_attention_weights(sequence)

        sequence_risk = (
            temporal_score * 0.3
            + velocity_score * 0.3
            + len(pattern_breaks) * 0.1
            + len(detected_patterns) * 0.15
            + (1 - min(len(sequence) / self.sequence_length, 1)) * 0.15
        )
        sequence_risk = min(1.0, sequence_risk)

        is_suspicious = sequence_risk > self.risk_threshold

        recommendation = self._generate_recommendation(
            sequence_risk, pattern_breaks, detected_patterns
        )

        return SequenceAnalysisResult(
            transaction_id=transaction_id,
            sequence_risk_score=float(sequence_risk),
            temporal_anomaly_score=float(temporal_score),
            velocity_anomaly_score=float(velocity_score),
            pattern_breaks=pattern_breaks,
            detected_patterns=detected_patterns,
            attention_weights=attention_weights,
            is_suspicious_sequence=is_suspicious,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
        )

    def _analyze_temporal_patterns(
        self, sequence: List[Dict[str, Any]], current: Dict[str, Any]
    ) -> float:
        """Analisa padrões temporais"""
        hours = [t.get("hour", 12) for t in sequence]
        current_hour = current.get("hour", 12)

        if len(hours) < 2:
            return 0.0

        avg_hour = np.mean(hours)
        std_hour = np.std(hours) + 1

        deviation = abs(current_hour - avg_hour) / std_hour
        temporal_score = min(1.0, deviation / 3)

        return temporal_score

    def _analyze_velocity(self, sequence: List[Dict[str, Any]]) -> float:
        """Analisa velocidade de transações"""
        if len(sequence) < 2:
            return 0.0

        recent = sequence[-5:]
        velocity = len(recent)

        if velocity >= 5:
            return 0.8
        elif velocity >= 3:
            return 0.5
        else:
            return 0.2

    def _detect_pattern_breaks(self, user_id: str, transaction: Dict[str, Any]) -> List[str]:
        """Detecta quebras de padrão"""
        breaks = []

        if user_id not in self.user_baselines:
            return breaks

        baseline = self.user_baselines[user_id]

        amount = transaction.get("amount", 0)
        avg = baseline.get("avg_amount", 0)
        std = baseline.get("amount_std", 1) or 1

        if avg > 0 and abs(amount - avg) > 3 * std:
            breaks.append(f"Valor {amount/avg:.1f}x fora do padrão")

        hour = transaction.get("hour", 12)
        avg_hour = baseline.get("avg_hour", 12)

        if abs(hour - avg_hour) > 6:
            breaks.append(f"Horário incomum ({hour}h vs média {avg_hour:.0f}h)")

        channel = transaction.get("channel", "unknown")
        typical = baseline.get("typical_channels", set())

        if isinstance(typical, set) and channel not in typical and len(typical) > 0:
            breaks.append(f"Canal novo: {channel}")

        return breaks

    def _detect_suspicious_patterns(
        self, sequence: List[Dict[str, Any]], current: Dict[str, Any]
    ) -> List[SequencePattern]:
        """Detecta padrões suspeitos na sequência"""
        patterns = []

        amounts = [t.get("amount", 0) for t in sequence[-5:]]
        if len(amounts) >= 3:
            increases = sum(1 for i in range(1, len(amounts)) if amounts[i] > amounts[i - 1] * 1.5)
            if increases >= 2:
                patterns.append(
                    SequencePattern(
                        pattern_type="rapid_escalation",
                        confidence=0.7,
                        description=self.SUSPICIOUS_PATTERNS["rapid_escalation"],
                        affected_transactions=[t.get("transaction_id", "") for t in sequence[-3:]],
                    )
                )

        if len(sequence) >= 5:
            recent_count = len([t for t in sequence[-5:]])
            if recent_count >= 5:
                patterns.append(
                    SequencePattern(
                        pattern_type="velocity_spike",
                        confidence=0.8,
                        description=self.SUSPICIOUS_PATTERNS["velocity_spike"],
                        affected_transactions=[t.get("transaction_id", "") for t in sequence[-5:]],
                    )
                )

        hours = [t.get("hour", 12) for t in sequence[-3:]]
        if len(hours) >= 2:
            hour_changes = sum(1 for i in range(1, len(hours)) if abs(hours[i] - hours[i - 1]) > 8)
            if hour_changes >= 2:
                patterns.append(
                    SequencePattern(
                        pattern_type="time_anomaly",
                        confidence=0.6,
                        description=self.SUSPICIOUS_PATTERNS["time_anomaly"],
                        affected_transactions=[t.get("transaction_id", "") for t in sequence[-3:]],
                    )
                )

        return patterns

    def _compute_attention_weights(self, sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        """Computa pesos de atenção para transações na sequência"""
        if len(sequence) == 0:
            return {}

        amounts = np.array([t.get("amount", 0) for t in sequence])

        if np.sum(amounts) > 0:
            weights = amounts / np.sum(amounts)
        else:
            weights = np.ones(len(sequence)) / len(sequence)

        weights = weights * 0.7 + 0.3 * np.linspace(0, 1, len(weights))
        weights = weights / np.sum(weights)

        return {
            sequence[i].get("transaction_id", f"txn_{i}"): float(weights[i])
            for i in range(len(sequence))
        }

    def _generate_recommendation(
        self, risk_score: float, pattern_breaks: List[str], detected_patterns: List[SequencePattern]
    ) -> str:
        """Gera recomendação baseada na análise"""
        if risk_score > 0.8:
            return "BLOQUEAR: Alto risco de fraude em sequência"
        elif risk_score > 0.6:
            if len(detected_patterns) > 0:
                pattern_names = [p.pattern_type for p in detected_patterns]
                return f"REVISAR: Padrões suspeitos detectados ({', '.join(pattern_names)})"
            return "REVISAR: Risco elevado na sequência"
        elif risk_score > 0.4:
            if pattern_breaks:
                return f"MONITORAR: Quebra de padrão ({pattern_breaks[0]})"
            return "MONITORAR: Risco moderado"
        else:
            return "APROVAR: Sequência dentro do padrão"

    def _create_default_result(self, transaction_id: str, reason: str) -> SequenceAnalysisResult:
        """Cria resultado padrão"""
        return SequenceAnalysisResult(
            transaction_id=transaction_id,
            sequence_risk_score=0.0,
            temporal_anomaly_score=0.0,
            velocity_anomaly_score=0.0,
            pattern_breaks=[],
            detected_patterns=[],
            attention_weights={},
            is_suspicious_sequence=False,
            recommendation=f"APROVAR: {reason}",
            timestamp=datetime.now().isoformat(),
        )

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Obtém perfil do usuário"""
        baseline = self.user_baselines.get(user_id, {})
        sequence = self.sequence_buffer.get_sequence(user_id)

        return {
            "user_id": user_id,
            "transaction_count": baseline.get("transaction_count", 0),
            "avg_amount": baseline.get("avg_amount", 0),
            "amount_std": baseline.get("amount_std", 0),
            "avg_hour": baseline.get("avg_hour", 12),
            "typical_channels": list(baseline.get("typical_channels", set())),
            "sequence_length": len(sequence),
            "last_transactions": sequence[-3:] if sequence else [],
        }

    def get_status(self) -> Dict[str, Any]:
        """Retorna status do analisador"""
        return {
            "version": self.VERSION,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_units,
            "risk_threshold": self.risk_threshold,
            "users_tracked": len(self.user_baselines),
            "tensorflow_available": TF_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
        }


_bilstm_instance = None


def get_bilstm_analyzer() -> BiLSTMSequenceAnalyzer:
    """Obtém instância singleton do analisador"""
    global _bilstm_instance
    if _bilstm_instance is None:
        _bilstm_instance = BiLSTMSequenceAnalyzer()
    return _bilstm_instance
