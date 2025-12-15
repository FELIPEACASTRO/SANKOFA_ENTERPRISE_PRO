"""
Sankofa Enterprise Pro - Scam Detector
Motor principal de detecção de APP Fraud e Scams

Baseado em:
- FICO Scam Detection Score
- BioCatch behavioral analysis
- Academic research on APP fraud patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ScamType(Enum):
    """Tipos de scam detectados"""
    ROMANCE_SCAM = "romance_scam"
    INVESTMENT_SCAM = "investment_scam"
    IMPERSONATION = "impersonation"
    PURCHASE_SCAM = "purchase_scam"
    ADVANCE_FEE = "advance_fee"
    CEO_FRAUD = "ceo_fraud"
    TECH_SUPPORT = "tech_support"
    LOTTERY_PRIZE = "lottery_prize"
    PIX_BUG = "pix_bug"
    WHATSAPP_CLONE = "whatsapp_clone"
    UNKNOWN = "unknown"


@dataclass
class ScamAnalysis:
    """Resultado da análise de scam"""
    transaction_id: str
    scam_probability: float
    scam_type: ScamType
    risk_level: str
    indicators: List[str]
    behavioral_signals: Dict[str, float]
    recommended_action: str
    cooling_off_minutes: int
    explanation: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ScamDetector:
    """
    Detector de APP Fraud e Scams

    Combina múltiplos sinais para detectar transações onde o usuário
    está sendo manipulado por fraudadores:

    1. Behavioral signals (hesitação, mudança de padrão)
    2. Transaction patterns (valor atípico, novo beneficiário)
    3. Timing patterns (horário incomum, urgência)
    4. Social engineering indicators
    """

    VERSION = "1.0.0"

    # Indicadores de alto risco
    HIGH_RISK_INDICATORS = {
        "first_time_receiver": 0.15,
        "high_value_transfer": 0.12,
        "unusual_hour": 0.10,
        "rapid_sequence": 0.10,
        "round_amount": 0.08,
        "foreign_receiver": 0.12,
        "new_device": 0.10,
        "vpn_detected": 0.08,
        "typing_hesitation": 0.15,
        "behavioral_anomaly": 0.20
    }

    # Padrões de valor suspeitos (em R$)
    SUSPICIOUS_AMOUNTS = [
        (1000, 2000),    # Faixa comum de scams menores
        (4900, 5100),    # Abaixo do limite PIX original
        (9800, 10200),   # Próximo de R$10k
        (49000, 51000),  # Próximo de R$50k
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)
        self.cooling_off_enabled = self.config.get("cooling_off_enabled", True)

        # Modelo de anomalia
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )

        # Scaler para normalização
        self.scaler = StandardScaler()

        # Cache de padrões do usuário
        self.user_patterns: Dict[str, Dict] = {}

        # Modelo de classificação de tipo de scam
        self.scam_classifier = None

        self.is_trained = False

        logger.info(f"ScamDetector v{self.VERSION} initialized")

    def analyze_transaction(
        self,
        transaction: Dict[str, Any],
        behavioral_data: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> ScamAnalysis:
        """
        Analisa uma transação para detectar possível scam

        Args:
            transaction: Dados da transação
            behavioral_data: Dados comportamentais da sessão
            session_data: Dados da sessão do usuário

        Returns:
            ScamAnalysis com resultado da análise
        """
        transaction_id = transaction.get("id", f"TX_{hash(str(transaction)) % 10000:04d}")

        # 1. Extrair features da transação
        tx_features = self._extract_transaction_features(transaction)

        # 2. Analisar comportamento se disponível
        behavioral_signals = {}
        if behavioral_data:
            behavioral_signals = self._analyze_behavioral_signals(behavioral_data)

        # 3. Analisar padrão do usuário
        user_id = transaction.get("customer_id") or transaction.get("client_cpf")
        pattern_deviation = self._analyze_user_pattern(user_id, transaction)

        # 4. Identificar indicadores de risco
        indicators = self._identify_risk_indicators(transaction, tx_features, behavioral_signals)

        # 5. Calcular probabilidade de scam
        scam_probability = self._calculate_scam_probability(
            tx_features, behavioral_signals, pattern_deviation, indicators
        )

        # 6. Classificar tipo de scam
        scam_type = self._classify_scam_type(transaction, indicators, scam_probability)

        # 7. Determinar nível de risco e ação
        risk_level = self._get_risk_level(scam_probability)
        recommended_action = self._get_recommended_action(scam_probability, scam_type)
        cooling_off = self._get_cooling_off_period(scam_probability, scam_type)

        # 8. Gerar explicação
        explanation = self._generate_explanation(
            scam_probability, scam_type, indicators, transaction
        )

        # 9. Calcular confiança
        confidence = self._calculate_confidence(
            len(indicators), behavioral_data is not None, self.is_trained
        )

        return ScamAnalysis(
            transaction_id=transaction_id,
            scam_probability=scam_probability,
            scam_type=scam_type,
            risk_level=risk_level,
            indicators=indicators,
            behavioral_signals=behavioral_signals,
            recommended_action=recommended_action,
            cooling_off_minutes=cooling_off,
            explanation=explanation,
            confidence=confidence
        )

    def _extract_transaction_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Extrai features numéricas da transação"""
        features = {}

        # Valor
        amount = float(transaction.get("amount") or transaction.get("value") or 0)
        features["amount"] = amount
        features["amount_log"] = np.log1p(amount)
        features["is_round_amount"] = float(amount % 100 == 0)
        features["is_suspicious_amount"] = float(self._is_suspicious_amount(amount))

        # Temporal
        timestamp = transaction.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour = timestamp.hour
            features["hour"] = hour
            features["is_night"] = float(hour >= 22 or hour <= 6)
            features["is_weekend"] = float(timestamp.weekday() >= 5)
        else:
            features["hour"] = 12
            features["is_night"] = 0.0
            features["is_weekend"] = 0.0

        # Receptor
        features["is_new_receiver"] = float(transaction.get("is_new_receiver", True))
        features["receiver_account_age_days"] = float(
            transaction.get("receiver_account_age_days", 0)
        )

        # Canal
        channel = transaction.get("channel", "").upper()
        features["is_mobile"] = float(channel == "MOBILE")
        features["is_web"] = float(channel == "WEB")

        # Dispositivo
        features["is_new_device"] = float(transaction.get("is_new_device", False))
        features["device_age_days"] = float(transaction.get("device_age_days", 30))

        return features

    def _analyze_behavioral_signals(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Analisa sinais comportamentais"""
        signals = {}

        # Keystroke dynamics
        if "keystroke" in behavioral_data:
            ks = behavioral_data["keystroke"]
            signals["typing_hesitation"] = ks.get("hesitation_score", 0.0)
            signals["typing_speed_deviation"] = ks.get("speed_deviation", 0.0)
            signals["error_rate"] = ks.get("error_rate", 0.0)

        # Mouse dynamics
        if "mouse" in behavioral_data:
            ms = behavioral_data["mouse"]
            signals["mouse_jitter"] = ms.get("jitter_score", 0.0)
            signals["mouse_velocity_anomaly"] = ms.get("velocity_anomaly", 0.0)

        # Session behavior
        if "session" in behavioral_data:
            sess = behavioral_data["session"]
            signals["time_on_page"] = sess.get("time_on_page_anomaly", 0.0)
            signals["scroll_pattern_anomaly"] = sess.get("scroll_anomaly", 0.0)
            signals["hesitation_before_confirm"] = sess.get("confirm_hesitation", 0.0)

        # Duress indicators
        if "duress" in behavioral_data:
            signals["duress_score"] = behavioral_data["duress"].get("score", 0.0)

        return signals

    def _analyze_user_pattern(
        self,
        user_id: Optional[str],
        transaction: Dict[str, Any]
    ) -> float:
        """Analisa desvio do padrão do usuário"""
        if not user_id:
            return 0.5  # Sem histórico, risco médio

        # Buscar padrão do usuário no cache
        pattern = self.user_patterns.get(user_id, {})

        if not pattern:
            return 0.5  # Usuário novo, risco médio

        deviation_scores = []

        # Desvio de valor
        amount = float(transaction.get("amount") or transaction.get("value") or 0)
        avg_amount = pattern.get("avg_amount", amount)
        std_amount = pattern.get("std_amount", avg_amount * 0.5)
        if std_amount > 0:
            amount_zscore = abs(amount - avg_amount) / std_amount
            deviation_scores.append(min(amount_zscore / 3, 1.0))

        # Desvio de horário
        timestamp = transaction.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            usual_hours = pattern.get("usual_hours", list(range(8, 22)))
            if timestamp.hour not in usual_hours:
                deviation_scores.append(0.7)

        # Desvio de canal
        channel = transaction.get("channel", "").upper()
        usual_channels = pattern.get("usual_channels", ["MOBILE", "WEB"])
        if channel and channel not in usual_channels:
            deviation_scores.append(0.5)

        return np.mean(deviation_scores) if deviation_scores else 0.3

    def _identify_risk_indicators(
        self,
        transaction: Dict[str, Any],
        features: Dict[str, float],
        behavioral_signals: Dict[str, float]
    ) -> List[str]:
        """Identifica indicadores de risco presentes"""
        indicators = []

        # Transaction-based indicators
        if features.get("is_new_receiver", 0) > 0.5:
            indicators.append("first_time_receiver")

        if features.get("amount", 0) > 5000:
            indicators.append("high_value_transfer")

        if features.get("is_night", 0) > 0.5:
            indicators.append("unusual_hour")

        if features.get("is_round_amount", 0) > 0.5:
            indicators.append("round_amount")

        if features.get("is_new_device", 0) > 0.5:
            indicators.append("new_device")

        if features.get("is_suspicious_amount", 0) > 0.5:
            indicators.append("suspicious_amount_range")

        # Behavioral indicators
        if behavioral_signals.get("typing_hesitation", 0) > 0.6:
            indicators.append("typing_hesitation")

        if behavioral_signals.get("duress_score", 0) > 0.7:
            indicators.append("possible_duress")

        if behavioral_signals.get("mouse_jitter", 0) > 0.5:
            indicators.append("behavioral_anomaly")

        if behavioral_signals.get("hesitation_before_confirm", 0) > 0.7:
            indicators.append("confirm_hesitation")

        # Transaction metadata
        if transaction.get("is_urgent_request", False):
            indicators.append("urgency_flag")

        if transaction.get("is_foreign_receiver", False):
            indicators.append("foreign_receiver")

        if transaction.get("vpn_detected", False):
            indicators.append("vpn_detected")

        return indicators

    def _calculate_scam_probability(
        self,
        tx_features: Dict[str, float],
        behavioral_signals: Dict[str, float],
        pattern_deviation: float,
        indicators: List[str]
    ) -> float:
        """Calcula probabilidade de scam"""

        # Base score from indicators
        indicator_score = sum(
            self.HIGH_RISK_INDICATORS.get(ind, 0.05)
            for ind in indicators
        )

        # Behavioral component
        behavioral_score = 0.0
        if behavioral_signals:
            behavioral_scores = [
                behavioral_signals.get("typing_hesitation", 0) * 0.2,
                behavioral_signals.get("duress_score", 0) * 0.3,
                behavioral_signals.get("mouse_jitter", 0) * 0.1,
                behavioral_signals.get("hesitation_before_confirm", 0) * 0.2
            ]
            behavioral_score = sum(behavioral_scores)

        # Pattern deviation component
        deviation_score = pattern_deviation * 0.2

        # Transaction risk component
        tx_score = 0.0
        tx_score += tx_features.get("is_night", 0) * 0.05
        tx_score += tx_features.get("is_round_amount", 0) * 0.03
        tx_score += tx_features.get("is_new_receiver", 0) * 0.10
        tx_score += tx_features.get("is_suspicious_amount", 0) * 0.08

        # Combine all components
        raw_probability = indicator_score + behavioral_score + deviation_score + tx_score

        # Normalize to [0, 1]
        probability = min(raw_probability, 1.0)

        # Apply boosting for multiple high-risk indicators
        if len(indicators) >= 4:
            probability = min(probability * 1.2, 1.0)

        if len(indicators) >= 6:
            probability = min(probability * 1.3, 1.0)

        return round(probability, 4)

    def _classify_scam_type(
        self,
        transaction: Dict[str, Any],
        indicators: List[str],
        scam_probability: float
    ) -> ScamType:
        """Classifica o tipo de scam mais provável"""

        if scam_probability < 0.3:
            return ScamType.UNKNOWN

        # Heurísticas para classificação
        description = transaction.get("description", "").lower()
        receiver_name = transaction.get("receiver_name", "").lower()

        # PIX Bug scam
        if any(kw in description for kw in ["bug", "dobro", "retorno", "teste"]):
            return ScamType.PIX_BUG

        # WhatsApp clone
        if any(kw in description for kw in ["numero novo", "troquei", "me ajuda"]):
            return ScamType.WHATSAPP_CLONE

        # Investment scam
        if any(kw in description for kw in ["investimento", "rendimento", "lucro", "crypto"]):
            return ScamType.INVESTMENT_SCAM

        # Lottery/Prize
        if any(kw in description for kw in ["premio", "sorteio", "ganhou", "contemplado"]):
            return ScamType.LOTTERY_PRIZE

        # Tech support
        if any(kw in description for kw in ["suporte", "banco", "desbloqueio", "atualização"]):
            return ScamType.TECH_SUPPORT

        # CEO Fraud (high value to new receiver)
        if "high_value_transfer" in indicators and "first_time_receiver" in indicators:
            amount = float(transaction.get("amount") or transaction.get("value") or 0)
            if amount > 20000:
                return ScamType.CEO_FRAUD

        # Impersonation (general)
        if "urgency_flag" in indicators or "possible_duress" in indicators:
            return ScamType.IMPERSONATION

        return ScamType.UNKNOWN

    def _get_risk_level(self, probability: float) -> str:
        """Determina nível de risco"""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        return "VERY_LOW"

    def _get_recommended_action(self, probability: float, scam_type: ScamType) -> str:
        """Determina ação recomendada"""
        if probability >= 0.9:
            return "BLOCK"
        elif probability >= 0.7:
            return "CALL_VERIFICATION"
        elif probability >= 0.5:
            return "DELAY_WITH_WARNING"
        elif probability >= 0.3:
            return "WARN_USER"
        return "ALLOW"

    def _get_cooling_off_period(self, probability: float, scam_type: ScamType) -> int:
        """Determina período de cooling-off em minutos"""
        if not self.cooling_off_enabled:
            return 0

        # High risk scam types get longer cooling-off
        high_risk_types = [
            ScamType.ROMANCE_SCAM,
            ScamType.INVESTMENT_SCAM,
            ScamType.CEO_FRAUD
        ]

        base_minutes = 0

        if probability >= 0.8:
            base_minutes = 30
        elif probability >= 0.6:
            base_minutes = 15
        elif probability >= 0.4:
            base_minutes = 5

        # Add extra time for high-risk scam types
        if scam_type in high_risk_types:
            base_minutes = int(base_minutes * 1.5)

        return base_minutes

    def _generate_explanation(
        self,
        probability: float,
        scam_type: ScamType,
        indicators: List[str],
        transaction: Dict[str, Any]
    ) -> str:
        """Gera explicação legível da análise"""

        if probability < 0.2:
            return "Transação apresenta baixo risco de ser um golpe."

        explanation_parts = []

        # Risk level
        if probability >= 0.7:
            explanation_parts.append(
                f"ALERTA: Esta transação apresenta ALTO risco de ser um golpe "
                f"(probabilidade: {probability*100:.0f}%)."
            )
        elif probability >= 0.4:
            explanation_parts.append(
                f"ATENÇÃO: Esta transação apresenta características suspeitas "
                f"(probabilidade de golpe: {probability*100:.0f}%)."
            )
        else:
            explanation_parts.append(
                f"Esta transação apresenta alguns indicadores de risco "
                f"(probabilidade de golpe: {probability*100:.0f}%)."
            )

        # Scam type
        scam_descriptions = {
            ScamType.PIX_BUG: "Possível golpe do 'bug do PIX'",
            ScamType.WHATSAPP_CLONE: "Possível golpe de clonagem de WhatsApp",
            ScamType.INVESTMENT_SCAM: "Possível golpe de investimento",
            ScamType.IMPERSONATION: "Possível golpe de falsa identidade",
            ScamType.LOTTERY_PRIZE: "Possível golpe de falso prêmio",
            ScamType.TECH_SUPPORT: "Possível golpe de falso suporte técnico",
            ScamType.CEO_FRAUD: "Possível golpe de fraude corporativa"
        }

        if scam_type in scam_descriptions:
            explanation_parts.append(scam_descriptions[scam_type] + ".")

        # Main indicators
        indicator_descriptions = {
            "first_time_receiver": "Primeira transferência para este destinatário",
            "high_value_transfer": "Valor acima do padrão usual",
            "unusual_hour": "Horário incomum para transações",
            "typing_hesitation": "Padrão de digitação indica hesitação",
            "possible_duress": "Sinais de que usuário pode estar sob pressão",
            "urgency_flag": "Solicitação marcada como urgente",
            "behavioral_anomaly": "Comportamento diferente do padrão habitual",
            "new_device": "Dispositivo não reconhecido"
        }

        relevant_indicators = [
            indicator_descriptions.get(ind)
            for ind in indicators[:3]
            if ind in indicator_descriptions
        ]

        if relevant_indicators:
            explanation_parts.append("Principais alertas: " + "; ".join(relevant_indicators) + ".")

        return " ".join(explanation_parts)

    def _calculate_confidence(
        self,
        num_indicators: int,
        has_behavioral_data: bool,
        is_trained: bool
    ) -> float:
        """Calcula confiança na análise"""
        base_confidence = 0.5

        # Mais indicadores = mais confiança
        base_confidence += min(num_indicators * 0.05, 0.2)

        # Dados comportamentais aumentam confiança
        if has_behavioral_data:
            base_confidence += 0.15

        # Modelo treinado aumenta confiança
        if is_trained:
            base_confidence += 0.1

        return min(base_confidence, 0.95)

    def _is_suspicious_amount(self, amount: float) -> bool:
        """Verifica se valor está em faixa suspeita"""
        for low, high in self.SUSPICIOUS_AMOUNTS:
            if low <= amount <= high:
                return True
        return False

    def train(self, historical_data: pd.DataFrame, labels: pd.Series):
        """Treina o detector com dados históricos"""
        logger.info("Training ScamDetector...")

        # Extrair features
        features = []
        for _, row in historical_data.iterrows():
            tx_features = self._extract_transaction_features(row.to_dict())
            features.append(list(tx_features.values()))

        X = np.array(features)

        # Treinar anomaly detector
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.anomaly_detector.fit(X_scaled)

        self.is_trained = True
        logger.info("ScamDetector training complete")

    def update_user_pattern(self, user_id: str, transaction: Dict[str, Any]):
        """Atualiza padrão do usuário"""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                "amounts": [],
                "hours": [],
                "channels": [],
                "receivers": set()
            }

        pattern = self.user_patterns[user_id]

        # Atualizar com nova transação
        amount = float(transaction.get("amount") or transaction.get("value") or 0)
        pattern["amounts"].append(amount)

        timestamp = transaction.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            pattern["hours"].append(timestamp.hour)

        channel = transaction.get("channel", "")
        if channel:
            pattern["channels"].append(channel.upper())

        receiver = transaction.get("receiver_id") or transaction.get("conta_recebedor")
        if receiver:
            pattern["receivers"].add(receiver)

        # Calcular estatísticas
        if len(pattern["amounts"]) > 1:
            pattern["avg_amount"] = np.mean(pattern["amounts"][-50:])
            pattern["std_amount"] = np.std(pattern["amounts"][-50:])

        if len(pattern["hours"]) > 5:
            from collections import Counter
            hour_counts = Counter(pattern["hours"][-100:])
            pattern["usual_hours"] = [
                h for h, c in hour_counts.most_common(10) if c >= 2
            ]

        if len(pattern["channels"]) > 5:
            pattern["usual_channels"] = list(set(pattern["channels"][-50:]))

    def save_model(self, path: str):
        """Salva o modelo treinado"""
        model_data = {
            "version": self.VERSION,
            "anomaly_detector": self.anomaly_detector,
            "scaler": self.scaler,
            "user_patterns": self.user_patterns,
            "is_trained": self.is_trained,
            "config": self.config
        }
        joblib.dump(model_data, path)
        logger.info(f"ScamDetector saved to {path}")

    def load_model(self, path: str):
        """Carrega modelo salvo"""
        model_data = joblib.load(path)
        self.anomaly_detector = model_data["anomaly_detector"]
        self.scaler = model_data["scaler"]
        self.user_patterns = model_data["user_patterns"]
        self.is_trained = model_data["is_trained"]
        self.config = model_data.get("config", {})
        logger.info(f"ScamDetector loaded from {path}")


def create_scam_detector(config: Optional[Dict[str, Any]] = None) -> ScamDetector:
    """Factory function para criar ScamDetector"""
    return ScamDetector(config)


# Aliases para compatibilidade
APPScamDetector = ScamDetector
AppFraudScamDetector = ScamDetector
