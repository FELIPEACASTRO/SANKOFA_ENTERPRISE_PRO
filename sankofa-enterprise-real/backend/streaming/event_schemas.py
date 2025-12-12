"""
Event Schemas - Avro/JSON schemas para Kafka events
Type-safe event definitions
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal


class EventType(str, Enum):
    """Tipos de eventos no sistema"""
    TRANSACTION_CREATED = "transaction.created"
    TRANSACTION_UPDATED = "transaction.updated"
    PREDICTION_COMPLETED = "prediction.completed"
    ALERT_CREATED = "alert.created"
    ALERT_RESOLVED = "alert.resolved"
    MANUAL_REVIEW_REQUESTED = "manual_review.requested"
    MANUAL_REVIEW_COMPLETED = "manual_review.completed"
    CHARGEBACK_RECEIVED = "chargeback.received"


@dataclass
class TransactionEvent:
    """
    Event de transação criada

    Published to: transactions.incoming
    """
    event_id: str
    event_type: EventType
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict (para serialização JSON)"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'data': self.data,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_transaction(cls, transaction: Dict[str, Any]) -> 'TransactionEvent':
        """
        Cria event a partir de transação

        Args:
            transaction: Transaction data

        Returns:
            TransactionEvent
        """
        import hashlib
        from datetime import datetime

        # Generate event ID
        content = str(transaction)
        event_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        return cls(
            event_id=f"txn_{event_id}",
            event_type=EventType.TRANSACTION_CREATED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=transaction,
            metadata={
                'source': 'fraud-api',
                'version': '1.0'
            }
        )


@dataclass
class FraudPredictionEvent:
    """
    Event de predição de fraude completada

    Published to: model.predictions
    """
    event_id: str
    event_type: EventType
    timestamp: str
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    model_version: str
    features_used: List[str]
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'transaction_id': self.transaction_id,
            'is_fraud': self.is_fraud,
            'fraud_probability': self.fraud_probability,
            'risk_score': self.risk_score,
            'model_version': self.model_version,
            'features_used': self.features_used,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_prediction(cls, prediction: Dict[str, Any]) -> 'FraudPredictionEvent':
        """Cria event a partir de prediction result"""
        import hashlib
        from datetime import datetime

        event_id = hashlib.sha256(
            f"{prediction.get('transaction_id')}:{datetime.now(timezone.utc)}".encode()
        ).hexdigest()[:16]

        return cls(
            event_id=f"pred_{event_id}",
            event_type=EventType.PREDICTION_COMPLETED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transaction_id=prediction.get('transaction_id', ''),
            is_fraud=prediction.get('is_fraud', False),
            fraud_probability=prediction.get('fraud_probability', 0.0),
            risk_score=prediction.get('risk_score', 0.0),
            model_version=prediction.get('model_version', 'unknown'),
            features_used=prediction.get('features_used', []),
            latency_ms=prediction.get('latency_ms', 0.0),
            metadata={
                'model_name': prediction.get('model_name', 'ensemble'),
                'explanation': prediction.get('explanation', {})
            }
        )


@dataclass
class AlertEvent:
    """
    Event de alerta de fraude

    Published to: fraud.alerts
    """
    event_id: str
    event_type: EventType
    timestamp: str
    transaction_id: str
    alert_type: str  # FRAUD_DETECTED, HIGH_RISK, MANUAL_REVIEW_REQUIRED
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float
    fraud_probability: float
    matched_rules: List[str]
    recommended_action: str  # BLOCK, MANUAL_REVIEW, ALLOW_WITH_MONITORING
    requires_manual_review: bool
    assignee: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'transaction_id': self.transaction_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'risk_score': self.risk_score,
            'fraud_probability': self.fraud_probability,
            'matched_rules': self.matched_rules,
            'recommended_action': self.recommended_action,
            'requires_manual_review': self.requires_manual_review,
            'assignee': self.assignee,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_prediction(
        cls,
        prediction: Dict[str, Any],
        alert_type: str = 'FRAUD_DETECTED'
    ) -> 'AlertEvent':
        """Cria alert event a partir de prediction"""
        import hashlib
        from datetime import datetime

        event_id = hashlib.sha256(
            f"{prediction.get('transaction_id')}:alert:{datetime.now(timezone.utc)}".encode()
        ).hexdigest()[:16]

        # Determine severity based on risk score
        risk_score = prediction.get('risk_score', 0.0)
        if risk_score >= 0.9:
            severity = 'CRITICAL'
        elif risk_score >= 0.7:
            severity = 'HIGH'
        elif risk_score >= 0.5:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        # Determine action
        if risk_score >= 0.9:
            action = 'BLOCK'
            requires_review = True
        elif risk_score >= 0.7:
            action = 'MANUAL_REVIEW'
            requires_review = True
        else:
            action = 'ALLOW_WITH_MONITORING'
            requires_review = False

        return cls(
            event_id=f"alert_{event_id}",
            event_type=EventType.ALERT_CREATED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transaction_id=prediction.get('transaction_id', ''),
            alert_type=alert_type,
            severity=severity,
            risk_score=risk_score,
            fraud_probability=prediction.get('fraud_probability', 0.0),
            matched_rules=prediction.get('matched_rules', []),
            recommended_action=action,
            requires_manual_review=requires_review,
            metadata={
                'model_version': prediction.get('model_version', 'unknown'),
                'explanation': prediction.get('explanation', {})
            }
        )


@dataclass
class ManualReviewEvent:
    """
    Event de review manual

    Published to: manual-review.events
    """
    event_id: str
    event_type: EventType
    timestamp: str
    transaction_id: str
    alert_id: str
    analyst_id: str
    decision: str  # APPROVE, REJECT, ESCALATE
    confidence: float
    notes: str
    review_duration_seconds: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict"""
        return asdict(self)


@dataclass
class ChargebackEvent:
    """
    Event de chargeback recebido

    Published to: chargebacks.incoming
    """
    event_id: str
    event_type: EventType
    timestamp: str
    transaction_id: str
    chargeback_id: str
    reason_code: str
    amount: float
    currency: str
    dispute_date: str
    acquirer: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict"""
        return asdict(self)


# Avro Schemas (para Schema Registry)
TRANSACTION_EVENT_AVRO_SCHEMA = {
    "type": "record",
    "name": "TransactionEvent",
    "namespace": "com.sankofa.fraud",
    "fields": [
        {"name": "event_id", "type": "string"},
        {"name": "event_type", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {
            "name": "data",
            "type": {
                "type": "map",
                "values": "string"
            }
        },
        {
            "name": "metadata",
            "type": ["null", {"type": "map", "values": "string"}],
            "default": None
        }
    ]
}

PREDICTION_EVENT_AVRO_SCHEMA = {
    "type": "record",
    "name": "FraudPredictionEvent",
    "namespace": "com.sankofa.fraud",
    "fields": [
        {"name": "event_id", "type": "string"},
        {"name": "event_type", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "transaction_id", "type": "string"},
        {"name": "is_fraud", "type": "boolean"},
        {"name": "fraud_probability", "type": "double"},
        {"name": "risk_score", "type": "double"},
        {"name": "model_version", "type": "string"},
        {"name": "features_used", "type": {"type": "array", "items": "string"}},
        {"name": "latency_ms", "type": "double"},
        {
            "name": "metadata",
            "type": ["null", {"type": "map", "values": "string"}],
            "default": None
        }
    ]
}

ALERT_EVENT_AVRO_SCHEMA = {
    "type": "record",
    "name": "AlertEvent",
    "namespace": "com.sankofa.fraud",
    "fields": [
        {"name": "event_id", "type": "string"},
        {"name": "event_type", "type": "string"},
        {"name": "timestamp", "type": "string"},
        {"name": "transaction_id", "type": "string"},
        {"name": "alert_type", "type": "string"},
        {"name": "severity", "type": "string"},
        {"name": "risk_score", "type": "double"},
        {"name": "fraud_probability", "type": "double"},
        {"name": "matched_rules", "type": {"type": "array", "items": "string"}},
        {"name": "recommended_action", "type": "string"},
        {"name": "requires_manual_review", "type": "boolean"},
        {"name": "assignee", "type": ["null", "string"], "default": None},
        {
            "name": "metadata",
            "type": ["null", {"type": "map", "values": "string"}],
            "default": None
        }
    ]
}
