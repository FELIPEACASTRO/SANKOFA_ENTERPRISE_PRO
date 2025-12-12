"""
Chargeback Engine - Automated Dispute Management
ML-based decision making for representment vs accept
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChargebackReason(str, Enum):
    """Chargeback reason codes (Visa/Mastercard)"""
    FRAUD = "fraud"
    AUTHORIZATION = "authorization"
    PROCESSING_ERROR = "processing_error"
    CONSUMER_DISPUTE = "consumer_dispute"
    PRODUCT_NOT_RECEIVED = "product_not_received"
    PRODUCT_DEFECTIVE = "product_defective"
    DUPLICATE_PROCESSING = "duplicate_processing"
    CREDIT_NOT_PROCESSED = "credit_not_processed"
    INCORRECT_AMOUNT = "incorrect_amount"
    CANCELLED_RECURRING = "cancelled_recurring"


class DisputeDecision(str, Enum):
    """Decisão de como tratar disputa"""
    ACCEPT = "accept"  # Aceitar chargeback (perda)
    REPRESENT = "represent"  # Contestar com evidências
    PARTIAL_REFUND = "partial_refund"  # Oferecer reembolso parcial
    ESCALATE = "escalate"  # Escalar para análise manual


@dataclass
class ChargebackCase:
    """Representa um caso de chargeback"""
    case_id: str
    transaction_id: str
    amount: float
    currency: str
    reason_code: str
    reason_category: ChargebackReason
    dispute_date: datetime
    deadline: datetime
    merchant_id: str
    customer_id: str
    acquirer: str
    card_brand: str
    evidence_required: List[str]
    metadata: Dict[str, Any]


class ChargebackEngine:
    """
    Engine de decisão automatizada para chargebacks

    Fluxo:
    1. Recebe notificação de chargeback
    2. Coleta evidências automaticamente
    3. ML scoring de probabilidade de vitória
    4. Decide: aceitar vs representar
    5. Submete resposta ao adquirente
    6. Tracking de outcome para feedback loop

    Win Rate Target: 85% (vs baseline 60%)
    """

    # Thresholds para decisão
    WIN_PROBABILITY_THRESHOLD = 0.70
    PARTIAL_REFUND_THRESHOLD = 0.50

    # Costs (em R$)
    CHARGEBACK_FEE = 50.0
    REPRESENTMENT_FEE = 100.0

    def __init__(self):
        """Initialize chargeback engine"""
        self.cases_processed = 0
        self.decisions_made = 0
        self.win_rate = 0.0

        # ML model (placeholder - em produção seria modelo treinado)
        self._ml_model = None

        logger.info("Chargeback Engine initialized")

    async def process_chargeback(
        self,
        chargeback: ChargebackCase
    ) -> Dict[str, Any]:
        """
        Processa chargeback e retorna decisão

        Args:
            chargeback: Chargeback case

        Returns:
            Dict com decisão e evidências
        """
        try:
            logger.info(f"Processing chargeback: {chargeback.case_id}")

            # Step 1: Validate chargeback
            if not self._validate_chargeback(chargeback):
                return self._create_error_response("Invalid chargeback data")

            # Step 2: Check if past deadline
            if datetime.utcnow() > chargeback.deadline:
                logger.warning(f"Chargeback past deadline: {chargeback.case_id}")
                return self._create_decision(
                    decision=DisputeDecision.ACCEPT,
                    reason="Past deadline",
                    win_probability=0.0
                )

            # Step 3: Collect evidence
            from .evidence_collector import EvidenceCollector
            evidence_collector = EvidenceCollector()

            evidence = await evidence_collector.collect_evidence(
                transaction_id=chargeback.transaction_id,
                reason_category=chargeback.reason_category
            )

            # Step 4: Calculate win probability
            win_probability = await self._calculate_win_probability(
                chargeback=chargeback,
                evidence=evidence
            )

            # Step 5: Make decision
            decision = self._make_decision(
                chargeback=chargeback,
                evidence=evidence,
                win_probability=win_probability
            )

            # Step 6: Calculate expected value
            expected_value = self._calculate_expected_value(
                amount=chargeback.amount,
                win_probability=win_probability,
                decision=decision
            )

            self.cases_processed += 1
            self.decisions_made += 1

            return {
                'case_id': chargeback.case_id,
                'decision': decision.value,
                'win_probability': win_probability,
                'expected_value': expected_value,
                'evidence': evidence,
                'reason': self._get_decision_reason(decision, win_probability),
                'recommended_actions': self._get_recommended_actions(decision),
                'deadline': chargeback.deadline.isoformat(),
                'processed_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing chargeback: {e}")
            return self._create_error_response(str(e))

    async def _calculate_win_probability(
        self,
        chargeback: ChargebackCase,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Calcula probabilidade de vencer disputa

        Usa ML model treinado em histórico de chargebacks

        Args:
            chargeback: Chargeback case
            evidence: Collected evidence

        Returns:
            Win probability (0.0 to 1.0)
        """
        # Features para o modelo
        features = {
            # Evidence strength
            'has_delivery_confirmation': evidence.get('delivery_confirmation', False),
            'has_signature': evidence.get('signature', False),
            'has_ip_match': evidence.get('ip_location_match', False),
            'has_device_match': evidence.get('device_fingerprint_match', False),
            'has_prior_successful_txns': evidence.get('prior_transactions_count', 0) > 0,

            # Transaction characteristics
            'amount': chargeback.amount,
            'time_since_transaction_days': (
                chargeback.dispute_date -
                evidence.get('transaction_date', chargeback.dispute_date)
            ).days,

            # Reason category
            'is_fraud_claim': chargeback.reason_category == ChargebackReason.FRAUD,
            'is_product_issue': chargeback.reason_category in [
                ChargebackReason.PRODUCT_NOT_RECEIVED,
                ChargebackReason.PRODUCT_DEFECTIVE
            ],

            # Historical data
            'merchant_historical_win_rate': evidence.get('merchant_win_rate', 0.6),
            'customer_chargeback_history': evidence.get('customer_cb_count', 0),
        }

        # Score baseado em regras (em produção seria ML model)
        score = self._rule_based_scoring(features, chargeback.reason_category)

        return min(max(score, 0.0), 1.0)

    def _rule_based_scoring(
        self,
        features: Dict[str, Any],
        reason: ChargebackReason
    ) -> float:
        """
        Rule-based scoring (placeholder para ML model)

        Args:
            features: Feature dict
            reason: Chargeback reason

        Returns:
            Win probability
        """
        base_score = 0.5

        # Strong evidence boosts
        if features.get('has_delivery_confirmation'):
            base_score += 0.25

        if features.get('has_signature'):
            base_score += 0.15

        if features.get('has_ip_match') and features.get('has_device_match'):
            base_score += 0.20

        if features.get('has_prior_successful_txns'):
            base_score += 0.10

        # Reason-specific adjustments
        if reason == ChargebackReason.FRAUD:
            # Fraud claims são mais difíceis de vencer
            base_score -= 0.15
        elif reason == ChargebackReason.PRODUCT_NOT_RECEIVED:
            # Delivery confirmation é crítico
            if not features.get('has_delivery_confirmation'):
                base_score -= 0.30

        # Amount factor (valores altos são mais contestados)
        if features.get('amount', 0) > 5000:
            base_score -= 0.05

        # Customer history
        if features.get('customer_chargeback_history', 0) > 2:
            base_score += 0.10  # Serial chargebacker

        return base_score

    def _make_decision(
        self,
        chargeback: ChargebackCase,
        evidence: Dict[str, Any],
        win_probability: float
    ) -> DisputeDecision:
        """
        Decide ação baseado em probabilidade e evidências

        Args:
            chargeback: Chargeback case
            evidence: Evidence collected
            win_probability: Win probability

        Returns:
            Decision
        """
        # High probability - represent
        if win_probability >= self.WIN_PROBABILITY_THRESHOLD:
            return DisputeDecision.REPRESENT

        # Medium probability - consider partial refund
        if win_probability >= self.PARTIAL_REFUND_THRESHOLD:
            # Se valor alto, vale tentar partial refund
            if chargeback.amount > 1000:
                return DisputeDecision.PARTIAL_REFUND
            else:
                return DisputeDecision.ACCEPT

        # Low probability - accept loss
        if win_probability < self.PARTIAL_REFUND_THRESHOLD:
            # Exceções: valores muito altos sempre escalar
            if chargeback.amount > 10000:
                return DisputeDecision.ESCALATE
            else:
                return DisputeDecision.ACCEPT

        return DisputeDecision.ESCALATE

    def _calculate_expected_value(
        self,
        amount: float,
        win_probability: float,
        decision: DisputeDecision
    ) -> float:
        """
        Calcula expected value da decisão

        Args:
            amount: Chargeback amount
            win_probability: Win probability
            decision: Decision made

        Returns:
            Expected value em R$
        """
        if decision == DisputeDecision.ACCEPT:
            # Loss = amount + fee
            return -(amount + self.CHARGEBACK_FEE)

        elif decision == DisputeDecision.REPRESENT:
            # EV = (win_prob * amount) - (lose_prob * amount) - representment_fee
            ev_win = win_probability * amount
            ev_lose = (1 - win_probability) * amount
            return ev_win - ev_lose - self.REPRESENTMENT_FEE

        elif decision == DisputeDecision.PARTIAL_REFUND:
            # Assume 50% refund
            return -(amount * 0.5)

        else:  # ESCALATE
            return 0.0  # Manual review

    def _validate_chargeback(self, chargeback: ChargebackCase) -> bool:
        """Valida dados do chargeback"""
        if not chargeback.case_id or not chargeback.transaction_id:
            return False

        if chargeback.amount <= 0:
            return False

        if chargeback.deadline < datetime.utcnow():
            logger.warning(f"Chargeback past deadline: {chargeback.case_id}")
            # Ainda processar, mas marcar

        return True

    def _get_decision_reason(
        self,
        decision: DisputeDecision,
        win_probability: float
    ) -> str:
        """Retorna reason humano-legível para decisão"""
        if decision == DisputeDecision.REPRESENT:
            return f"High win probability ({win_probability:.1%}), strong evidence available"

        elif decision == DisputeDecision.ACCEPT:
            return f"Low win probability ({win_probability:.1%}), insufficient evidence"

        elif decision == DisputeDecision.PARTIAL_REFUND:
            return f"Medium win probability ({win_probability:.1%}), partial refund optimal"

        else:
            return "High-value case requiring manual review"

    def _get_recommended_actions(self, decision: DisputeDecision) -> List[str]:
        """Retorna ações recomendadas"""
        if decision == DisputeDecision.REPRESENT:
            return [
                "Prepare representment package with all evidence",
                "Submit to acquirer within deadline",
                "Follow up on case status"
            ]

        elif decision == DisputeDecision.ACCEPT:
            return [
                "Accept chargeback",
                "Analyze root cause to prevent future cases",
                "Update fraud models if applicable"
            ]

        elif decision == DisputeDecision.PARTIAL_REFUND:
            return [
                "Contact customer with partial refund offer",
                "Negotiate settlement",
                "Document agreement"
            ]

        else:
            return [
                "Escalate to senior analyst",
                "Conduct detailed investigation",
                "Decision required within 48 hours"
            ]

    def _create_decision(
        self,
        decision: DisputeDecision,
        reason: str,
        win_probability: float
    ) -> Dict[str, Any]:
        """Helper to create decision response"""
        return {
            'decision': decision.value,
            'reason': reason,
            'win_probability': win_probability,
            'recommended_actions': self._get_recommended_actions(decision)
        }

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Helper to create error response"""
        return {
            'success': False,
            'error': error,
            'decision': DisputeDecision.ESCALATE.value
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do engine"""
        return {
            'cases_processed': self.cases_processed,
            'decisions_made': self.decisions_made,
            'current_win_rate': self.win_rate,
            'target_win_rate': 0.85
        }


# Singleton
_engine_instance: Optional[ChargebackEngine] = None


def get_chargeback_engine() -> ChargebackEngine:
    """Returns singleton chargeback engine"""
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = ChargebackEngine()

    return _engine_instance
