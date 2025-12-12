"""
Stream Processor - Orchestrates end-to-end fraud detection pipeline
Coordinates Kafka consumers, ML inference, and alerting
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .kafka_producer import get_kafka_producer
from .kafka_consumer import KafkaFraudConsumer
from .event_schemas import (
    TransactionEvent,
    FraudPredictionEvent,
    AlertEvent,
    EventType
)

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processa stream de transações em tempo real

    Pipeline:
    1. Consume transaction events (Kafka)
    2. Run fraud detection (ML model)
    3. Publish prediction results
    4. Create alerts se fraude detectada
    5. Trigger manual review se necessário
    """

    def __init__(
        self,
        consumer_group_id: str = 'fraud-detection-processor',
        bootstrap_servers: str = 'localhost:9092'
    ):
        """
        Args:
            consumer_group_id: Kafka consumer group
            bootstrap_servers: Kafka cluster endpoints
        """
        self.consumer_group_id = consumer_group_id
        self.bootstrap_servers = bootstrap_servers

        # Kafka components
        self.producer = get_kafka_producer()
        self.consumer: Optional[KafkaFraudConsumer] = None

        # ML Engine (lazy load)
        self._fraud_engine = None

        # Metrics
        self.transactions_processed = 0
        self.frauds_detected = 0
        self.alerts_created = 0
        self.avg_latency_ms = 0.0

        logger.info(f"Stream Processor initialized: group={consumer_group_id}")

    @property
    def fraud_engine(self):
        """Lazy load fraud engine"""
        if self._fraud_engine is None:
            from ml_engine.production_fraud_engine import get_fraud_engine
            self._fraud_engine = get_fraud_engine()
        return self._fraud_engine

    async def start(self) -> None:
        """
        Inicia stream processing (blocking)

        Consome transações e processa em real-time
        """
        # Create consumer
        self.consumer = KafkaFraudConsumer(
            group_id=self.consumer_group_id,
            topics=['transactions.incoming'],
            bootstrap_servers=self.bootstrap_servers
        )

        # Register message handler
        self.consumer.register_handler(self.process_transaction_event)

        # Start consuming (blocking)
        logger.info("Starting stream processing...")
        await self.consumer.start()

    async def process_transaction_event(self, event: Dict[str, Any]) -> bool:
        """
        Processa evento de transação

        Args:
            event: Transaction event from Kafka

        Returns:
            True se processado com sucesso
        """
        start_time = datetime.utcnow()

        try:
            # Extract transaction data
            transaction_data = event.get('data', {})
            transaction_id = transaction_data.get('id', 'unknown')

            logger.info(f"Processing transaction: {transaction_id}")

            # Step 1: Run fraud detection
            prediction = await self._run_fraud_detection(transaction_data)

            # Step 2: Publish prediction event
            await self._publish_prediction(prediction, transaction_id)

            # Step 3: Create alert se fraud detectado
            if prediction.get('is_fraud') or prediction.get('risk_score', 0) > 0.7:
                await self._create_alert(prediction, transaction_id)

            # Step 4: Trigger manual review se necessário
            if prediction.get('risk_score', 0) > 0.8:
                await self._trigger_manual_review(prediction, transaction_id)

            # Update metrics
            self.transactions_processed += 1
            if prediction.get('is_fraud'):
                self.frauds_detected += 1

            # Calculate latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_latency(latency_ms)

            logger.info(
                f"Transaction processed: {transaction_id}, "
                f"fraud={prediction.get('is_fraud')}, "
                f"latency={latency_ms:.2f}ms"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return False

    async def _run_fraud_detection(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa detecção de fraude

        Args:
            transaction: Transaction data

        Returns:
            Prediction result
        """
        try:
            # Run ML model
            prediction = self.fraud_engine.predict(transaction)

            return {
                'transaction_id': transaction.get('id', ''),
                'is_fraud': prediction.is_fraud if hasattr(prediction, 'is_fraud') else False,
                'fraud_probability': prediction.fraud_probability if hasattr(prediction, 'fraud_probability') else 0.0,
                'risk_score': prediction.risk_score if hasattr(prediction, 'risk_score') else 0.0,
                'model_version': prediction.model_version if hasattr(prediction, 'model_version') else '1.0',
                'features_used': prediction.features_used if hasattr(prediction, 'features_used') else [],
                'explanation': prediction.explanation if hasattr(prediction, 'explanation') else {},
                'matched_rules': prediction.matched_rules if hasattr(prediction, 'matched_rules') else [],
                'latency_ms': prediction.latency_ms if hasattr(prediction, 'latency_ms') else 0.0
            }

        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            # Return safe default
            return {
                'transaction_id': transaction.get('id', ''),
                'is_fraud': False,
                'fraud_probability': 0.0,
                'risk_score': 0.0,
                'model_version': 'error',
                'error': str(e)
            }

    async def _publish_prediction(
        self,
        prediction: Dict[str, Any],
        transaction_id: str
    ) -> None:
        """
        Publica resultado da predição

        Args:
            prediction: Prediction result
            transaction_id: Transaction ID
        """
        try:
            # Create prediction event
            event = FraudPredictionEvent.from_prediction(prediction)

            # Publish to Kafka
            await self.producer.publish_prediction(
                prediction=event.to_dict(),
                topic='model.predictions'
            )

            logger.debug(f"Prediction published: {transaction_id}")

        except Exception as e:
            logger.error(f"Error publishing prediction: {e}")

    async def _create_alert(
        self,
        prediction: Dict[str, Any],
        transaction_id: str
    ) -> None:
        """
        Cria alerta de fraude

        Args:
            prediction: Prediction result
            transaction_id: Transaction ID
        """
        try:
            # Create alert event
            alert = AlertEvent.from_prediction(prediction)

            # Publish alert
            await self.producer.publish_alert(
                alert=alert.to_dict(),
                topic='fraud.alerts'
            )

            self.alerts_created += 1

            logger.info(
                f"Alert created: {transaction_id}, "
                f"severity={alert.severity}, "
                f"action={alert.recommended_action}"
            )

        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    async def _trigger_manual_review(
        self,
        prediction: Dict[str, Any],
        transaction_id: str
    ) -> None:
        """
        Aciona review manual

        Args:
            prediction: Prediction result
            transaction_id: Transaction ID
        """
        try:
            # Create manual review request
            review_request = {
                'transaction_id': transaction_id,
                'risk_score': prediction.get('risk_score', 0),
                'fraud_probability': prediction.get('fraud_probability', 0),
                'explanation': prediction.get('explanation', {}),
                'priority': 'HIGH' if prediction.get('risk_score', 0) > 0.9 else 'MEDIUM',
                'requested_at': datetime.utcnow().isoformat()
            }

            # Publish manual review request
            # TODO: Publish to manual-review.requests topic
            logger.info(f"Manual review requested: {transaction_id}")

        except Exception as e:
            logger.error(f"Error triggering manual review: {e}")

    def _update_latency(self, latency_ms: float) -> None:
        """
        Atualiza latência média (moving average)

        Args:
            latency_ms: Latency in milliseconds
        """
        alpha = 0.1  # Smoothing factor
        self.avg_latency_ms = (alpha * latency_ms) + ((1 - alpha) * self.avg_latency_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas do processor

        Returns:
            Dict com métricas
        """
        fraud_rate = (
            self.frauds_detected / self.transactions_processed
            if self.transactions_processed > 0
            else 0
        )

        return {
            'transactions_processed': self.transactions_processed,
            'frauds_detected': self.frauds_detected,
            'fraud_rate': fraud_rate,
            'alerts_created': self.alerts_created,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'consumer_metrics': self.consumer.get_metrics() if self.consumer else {}
        }

    def stop(self) -> None:
        """Para stream processor gracefully"""
        logger.info("Stopping stream processor...")

        if self.consumer:
            self.consumer.stop()

        self.producer.flush()

        logger.info(
            f"Stream processor stopped. "
            f"Processed: {self.transactions_processed}, "
            f"Frauds: {self.frauds_detected}, "
            f"Alerts: {self.alerts_created}"
        )


# CLI entry point
async def main():
    """
    Main entry point para executar stream processor

    Usage:
        python -m streaming.stream_processor
    """
    import signal

    processor = StreamProcessor(
        consumer_group_id='fraud-detection-processor',
        bootstrap_servers='localhost:9092'
    )

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        processor.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start processing
    try:
        await processor.start()
    except KeyboardInterrupt:
        processor.stop()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
