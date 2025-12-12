"""
Kafka Consumer - Event Processing with Consumer Groups
Exactly-once semantics, auto-commit, error handling
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from kafka.structs import TopicPartition

logger = logging.getLogger(__name__)


class KafkaFraudConsumer:
    """
    Kafka Consumer com consumer groups e exactly-once processing

    Features:
    - Consumer groups para paralelização
    - Exactly-once semantics
    - Auto-offset management
    - Error handling com retry e DLQ
    - Graceful shutdown
    - Metrics tracking
    """

    def __init__(
        self,
        group_id: str,
        topics: List[str],
        bootstrap_servers: str = 'localhost:9092',
        auto_offset_reset: str = 'earliest',
        enable_auto_commit: bool = True
    ):
        """
        Args:
            group_id: Consumer group ID (para paralelização)
            topics: Lista de topics para consumir
            bootstrap_servers: Kafka cluster
            auto_offset_reset: Comportamento quando sem offset (earliest/latest)
            enable_auto_commit: Auto-commit offsets (False para manual)
        """
        self.group_id = group_id
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers
        self.running = False
        self.message_handler: Optional[Callable] = None

        # Kafka consumer configuration
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,

            # Offset management
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            auto_commit_interval_ms=5000,

            # Exactly-once settings
            isolation_level='read_committed',  # Lê apenas msgs committed (evita duplicatas)

            # Deserialization
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,

            # Performance
            fetch_min_bytes=1024,
            fetch_max_wait_ms=500,
            max_poll_records=500,

            # Session management
            session_timeout_ms=10000,
            heartbeat_interval_ms=3000,

            # Error handling
            consumer_timeout_ms=1000,  # Poll timeout
        )

        logger.info(
            f"Kafka Consumer initialized: group={group_id}, "
            f"topics={topics}"
        )

        # Metrics
        self.messages_processed = 0
        self.messages_failed = 0

    def register_handler(self, handler: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Registra function handler para processar mensagens

        Args:
            handler: Function que processa mensagem e retorna True se sucesso

        Example:
            async def process_transaction(event):
                # Process event
                return True

            consumer.register_handler(process_transaction)
        """
        self.message_handler = handler
        logger.info(f"Message handler registered: {handler.__name__}")

    async def start(self) -> None:
        """
        Inicia consumer loop (blocking)

        Processa mensagens continuamente até stop() ser chamado
        """
        if not self.message_handler:
            raise ValueError("No message handler registered. Call register_handler() first.")

        self.running = True
        logger.info(f"Starting consumer loop for group: {self.group_id}")

        try:
            while self.running:
                # Poll messages (batch)
                message_batch = self.consumer.poll(timeout_ms=1000)

                if not message_batch:
                    await asyncio.sleep(0.1)
                    continue

                # Process batch
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_message(message)

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")

        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise

        finally:
            self.stop()

    async def _process_message(self, message) -> None:
        """
        Processa mensagem individual

        Args:
            message: Kafka message
        """
        try:
            event = message.value

            # Idempotency check (evita processar duplicadas)
            event_id = event.get('event_id')
            if self._is_already_processed(event_id):
                logger.debug(f"Skipping duplicate event: {event_id}")
                return

            # Call handler
            success = await self._safe_call_handler(event)

            if success:
                self.messages_processed += 1
                self._mark_as_processed(event_id)

                # Log progress a cada 100 mensagens
                if self.messages_processed % 100 == 0:
                    logger.info(
                        f"Processed {self.messages_processed} messages "
                        f"(failed: {self.messages_failed})"
                    )
            else:
                self.messages_failed += 1
                await self._handle_processing_failure(event, message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.messages_failed += 1
            await self._handle_processing_failure(event, message, error=str(e))

    async def _safe_call_handler(self, event: Dict[str, Any]) -> bool:
        """
        Chama handler com timeout e error handling

        Args:
            event: Event data

        Returns:
            True se processado com sucesso
        """
        try:
            # Call handler with timeout (30s)
            result = await asyncio.wait_for(
                self.message_handler(event),
                timeout=30.0
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Handler timeout for event: {event.get('event_id')}")
            return False

        except Exception as e:
            logger.error(f"Handler error: {e}")
            return False

    async def _handle_processing_failure(
        self,
        event: Dict[str, Any],
        message,
        error: Optional[str] = None
    ) -> None:
        """
        Trata falha no processamento

        - Retry logic
        - Send to DLQ após max retries

        Args:
            event: Event data
            message: Kafka message original
            error: Error description
        """
        retry_count = event.get('metadata', {}).get('retry_count', 0)

        if retry_count < 3:
            # Retry
            logger.warning(f"Retrying event {event.get('event_id')} (attempt {retry_count + 1})")

            event['metadata']['retry_count'] = retry_count + 1

            # Re-publish para retry topic
            # TODO: Implementar retry topic
            pass

        else:
            # Max retries excedido - send to DLQ
            logger.error(f"Max retries exceeded for event {event.get('event_id')}, sending to DLQ")

            await self._send_to_dlq(event, error)

    async def _send_to_dlq(self, event: Dict[str, Any], error: Optional[str] = None) -> None:
        """
        Envia mensagem para Dead Letter Queue

        Args:
            event: Event data
            error: Error description
        """
        try:
            # TODO: Use Kafka Producer to send to DLQ topic
            dlq_event = {
                'original_event': event,
                'error': error,
                'failed_at': datetime.utcnow().isoformat(),
                'consumer_group': self.group_id
            }

            logger.warning(f"Sent to DLQ: {event.get('event_id')}")

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")

    def _is_already_processed(self, event_id: str) -> bool:
        """
        Verifica se evento já foi processado (idempotency)

        Em produção, usar Redis cache para tracking

        Args:
            event_id: Event ID

        Returns:
            True se já processado
        """
        # TODO: Implementar Redis cache
        # return redis.exists(f"processed:{event_id}")
        return False

    def _mark_as_processed(self, event_id: str) -> None:
        """
        Marca evento como processado

        Args:
            event_id: Event ID
        """
        # TODO: Implementar Redis cache com TTL de 24h
        # redis.setex(f"processed:{event_id}", 86400, "1")
        pass

    def stop(self) -> None:
        """Para consumer gracefully"""
        logger.info(f"Stopping consumer group: {self.group_id}")
        self.running = False
        self.consumer.close()
        logger.info(
            f"Consumer stopped. Processed: {self.messages_processed}, "
            f"Failed: {self.messages_failed}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas do consumer

        Returns:
            Dict com métricas
        """
        return {
            'group_id': self.group_id,
            'topics': self.topics,
            'messages_processed': self.messages_processed,
            'messages_failed': self.messages_failed,
            'success_rate': (
                self.messages_processed / (self.messages_processed + self.messages_failed)
                if (self.messages_processed + self.messages_failed) > 0
                else 0
            ),
            'running': self.running
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Example usage
async def example_usage():
    """Example: Como usar o Kafka Consumer"""

    # Define handler function
    async def process_transaction_event(event: Dict[str, Any]) -> bool:
        """Process incoming transaction"""
        try:
            transaction_data = event.get('data', {})

            # Process transaction (call fraud engine, etc.)
            logger.info(f"Processing transaction: {transaction_data.get('id')}")

            # Simulate processing
            await asyncio.sleep(0.1)

            return True

        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return False

    # Create consumer
    consumer = KafkaFraudConsumer(
        group_id='fraud-detection-group',
        topics=['transactions.incoming'],
        bootstrap_servers='localhost:9092'
    )

    # Register handler
    consumer.register_handler(process_transaction_event)

    # Start processing (blocking)
    await consumer.start()


if __name__ == '__main__':
    # Run example
    asyncio.run(example_usage())
