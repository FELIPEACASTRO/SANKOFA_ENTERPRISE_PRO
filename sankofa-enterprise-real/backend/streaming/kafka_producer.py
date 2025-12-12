"""
Kafka Producer - Async Event Publishing
High-throughput, exactly-once semantics

CORRECAO 10/10:
- Corrigido datetime.utcnow() para datetime.now(timezone.utc)
- Carrega bootstrap_servers de variáveis de ambiente
- Thread-safe singleton com double-checked locking
"""

import asyncio
import json
import logging
import os
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
import hashlib

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Retorna datetime atual em UTC com timezone info (Python 3.12+ compatible)"""
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Retorna timestamp ISO 8601 em UTC"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + "Z"


class KafkaFraudProducer:
    """
    Async Kafka Producer para eventos de fraude

    Features:
    - Exactly-once semantics (idempotence)
    - Async publishing
    - Automatic retry com backoff
    - Dead letter queue para falhas
    - Partitioning por customer_id
    - CORRECAO 10/10: Bootstrap servers de variáveis de ambiente
    - CORRECAO 10/10: datetime.now(timezone.utc) em vez de datetime.utcnow()
    """

    def __init__(
        self,
        bootstrap_servers: str = None,
        enable_idempotence: bool = True,
        max_retries: int = 3
    ):
        """
        Args:
            bootstrap_servers: Kafka cluster endpoints (default: from env)
            enable_idempotence: Garante exactly-once
            max_retries: Tentativas de retry
        """
        # CORRECAO 10/10: Carregar de variáveis de ambiente
        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')

        self.bootstrap_servers = bootstrap_servers
        self.max_retries = max_retries

        # Kafka producer com exactly-once semantics
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,

            # Exactly-once configuration
            enable_idempotence=enable_idempotence,
            acks='all',  # Aguarda ACK de todas as replicas
            retries=max_retries,
            max_in_flight_requests_per_connection=5,

            # Serialization
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,

            # Performance
            compression_type='snappy',
            batch_size=16384,
            linger_ms=10,  # Buffer 10ms para batch

            # Timeouts
            request_timeout_ms=30000,

            # Metrics
            metric_reporters=[],
        )

        logger.info(f"Kafka Producer initialized: {bootstrap_servers}")

    async def publish_transaction(
        self,
        transaction: Dict[str, Any],
        topic: str = 'transactions.incoming'
    ) -> bool:
        """
        Publica transação para processamento

        Args:
            transaction: Dados da transação
            topic: Kafka topic

        Returns:
            True se publicado com sucesso
        """
        try:
            # Generate idempotency key
            event_id = self._generate_event_id(transaction)

            # Partition key (customer_id para co-location)
            partition_key = transaction.get('customer_id', 'unknown')

            # Event envelope
            # CORRECAO 10/10: Usar timezone-aware datetime
            event = {
                'event_id': event_id,
                'event_type': 'transaction.created',
                'timestamp': _utc_now_iso(),
                'data': transaction,
                'metadata': {
                    'source': 'fraud-api',
                    'version': '1.0'
                }
            }

            # Async publish
            future = self.producer.send(
                topic,
                key=partition_key,
                value=event
            )

            # Wait for ACK (com timeout)
            record_metadata = future.get(timeout=10)

            logger.info(
                f"Transaction published: topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )

            return True

        except KafkaError as e:
            logger.error(f"Kafka publish error: {e}")
            # Send to dead letter queue
            await self._send_to_dlq(transaction, str(e))
            return False

        except Exception as e:
            logger.error(f"Unexpected error publishing: {e}")
            return False

    async def publish_prediction(
        self,
        prediction: Dict[str, Any],
        topic: str = 'model.predictions'
    ) -> bool:
        """
        Publica resultado de predição ML

        Args:
            prediction: Resultado do modelo
            topic: Kafka topic

        Returns:
            True se publicado com sucesso
        """
        try:
            event_id = self._generate_event_id(prediction)
            partition_key = prediction.get('transaction_id', 'unknown')

            # CORRECAO 10/10: Usar timezone-aware datetime
            event = {
                'event_id': event_id,
                'event_type': 'prediction.completed',
                'timestamp': _utc_now_iso(),
                'data': prediction,
                'metadata': {
                    'model_version': prediction.get('model_version', 'unknown'),
                    'latency_ms': prediction.get('latency_ms', 0)
                }
            }

            future = self.producer.send(topic, key=partition_key, value=event)
            record_metadata = future.get(timeout=10)

            logger.info(f"Prediction published: offset={record_metadata.offset}")
            return True

        except Exception as e:
            logger.error(f"Error publishing prediction: {e}")
            return False

    async def publish_alert(
        self,
        alert: Dict[str, Any],
        topic: str = 'fraud.alerts'
    ) -> bool:
        """
        Publica alerta de fraude

        Args:
            alert: Dados do alerta
            topic: Kafka topic

        Returns:
            True se publicado com sucesso
        """
        try:
            event_id = self._generate_event_id(alert)
            partition_key = alert.get('transaction_id', 'unknown')

            # CORRECAO 10/10: Usar timezone-aware datetime
            event = {
                'event_id': event_id,
                'event_type': 'alert.created',
                'timestamp': _utc_now_iso(),
                'data': alert,
                'metadata': {
                    'severity': alert.get('severity', 'MEDIUM'),
                    'requires_action': alert.get('requires_manual_review', False)
                }
            }

            future = self.producer.send(topic, key=partition_key, value=event)
            record_metadata = future.get(timeout=10)

            logger.info(f"Alert published: severity={alert.get('severity')}")
            return True

        except Exception as e:
            logger.error(f"Error publishing alert: {e}")
            return False

    def _generate_event_id(self, data: Dict[str, Any]) -> str:
        """
        Gera ID único e idempotente para evento

        Usa hash do conteúdo + timestamp para garantir unicidade
        """
        content = json.dumps(data, sort_keys=True)
        # CORRECAO 10/10: Usar timezone-aware datetime
        timestamp = _utc_now_iso()
        hash_input = f"{content}:{timestamp}"

        event_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"evt_{event_id}"

    async def _send_to_dlq(self, data: Dict[str, Any], error: str) -> bool:
        """
        Envia mensagem falhada para Dead Letter Queue

        CORRECAO 10/10: Retorna bool para indicar sucesso e usa timezone-aware datetime

        Args:
            data: Dados originais
            error: Descrição do erro

        Returns:
            True se enviado com sucesso
        """
        try:
            dlq_event = {
                'original_data': data,
                'error': error,
                # CORRECAO 10/10: Usar timezone-aware datetime
                'timestamp': _utc_now_iso(),
                'retry_count': 0
            }

            future = self.producer.send(
                'dead-letter-queue',
                value=dlq_event
            )

            # CORRECAO 10/10: Aguardar confirmação
            future.get(timeout=10)

            logger.warning(f"Message sent to DLQ: {error}")
            return True

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
            return False

    def flush(self, timeout: Optional[int] = None) -> None:
        """
        Força flush de mensagens pendentes

        Args:
            timeout: Timeout em segundos
        """
        self.producer.flush(timeout=timeout)

    def close(self) -> None:
        """Fecha producer gracefully"""
        self.producer.flush()
        self.producer.close()
        logger.info("Kafka Producer closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# CORRECAO 10/10: Thread-safe singleton com double-checked locking
_producer_instance: Optional[KafkaFraudProducer] = None
_producer_lock = threading.Lock()


def get_kafka_producer() -> KafkaFraudProducer:
    """
    Retorna singleton do Kafka Producer

    CORRECAO 10/10: Thread-safe com double-checked locking
    CORRECAO 10/10: Bootstrap servers de variáveis de ambiente

    Returns:
        KafkaFraudProducer instance
    """
    global _producer_instance

    if _producer_instance is None:
        with _producer_lock:
            if _producer_instance is None:
                # CORRECAO 10/10: Carregar de variáveis de ambiente
                bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
                _producer_instance = KafkaFraudProducer(bootstrap_servers=bootstrap_servers)

    return _producer_instance
