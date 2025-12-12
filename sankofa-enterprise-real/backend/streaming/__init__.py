"""
Streaming Module - Kafka Event-Driven Architecture
Real-time fraud detection event processing
"""

from .kafka_producer import KafkaFraudProducer
from .kafka_consumer import KafkaFraudConsumer
from .event_schemas import (
    TransactionEvent,
    FraudPredictionEvent,
    AlertEvent,
    EventType
)
from .stream_processor import StreamProcessor

__all__ = [
    'KafkaFraudProducer',
    'KafkaFraudConsumer',
    'TransactionEvent',
    'FraudPredictionEvent',
    'AlertEvent',
    'EventType',
    'StreamProcessor'
]
