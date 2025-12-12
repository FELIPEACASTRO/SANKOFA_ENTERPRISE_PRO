"""
Feature Materializer - Materializes computed features to Redis
Bridges Flink computation and Feature Store serving
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .window_aggregator import get_window_aggregator
from .feature_store import get_feature_store

logger = logging.getLogger(__name__)


class FeatureMaterializer:
    """
    Materializa features computadas para Redis Feature Store

    Flow:
    1. Window Aggregator computa features em tempo real
    2. Materializer persiste no Redis
    3. Feature Store serve features com <5ms latency

    Materialization strategies:
    - On-demand: Computa quando solicitado
    - Periodic: Background job a cada N minutos
    - Event-driven: Materializa após cada transação
    """

    def __init__(
        self,
        materialization_strategy: str = 'event_driven',
        batch_size: int = 100
    ):
        """
        Args:
            materialization_strategy: 'on_demand', 'periodic', 'event_driven'
            batch_size: Batch size para materialization
        """
        self.strategy = materialization_strategy
        self.batch_size = batch_size

        # Components
        self.aggregator = get_window_aggregator()
        self.feature_store = get_feature_store()

        # Background task
        self.running = False
        self.materialization_task: Optional[asyncio.Task] = None

        logger.info(f"Feature Materializer initialized: strategy={materialization_strategy}")

    async def materialize_features(
        self,
        entity_id: str,
        entity_type: str = 'customer',
        windows: Optional[List[str]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Materializa features para uma entidade

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            windows: Windows to materialize (default: all)
            ttl: TTL em segundos

        Returns:
            True se sucesso
        """
        try:
            # Compute features
            features = await self.aggregator.compute_features(
                entity_id=entity_id,
                windows=windows
            )

            # Add session features
            session_features = await self.aggregator.compute_session_features(entity_id)
            features.update(session_features)

            # Persist to feature store
            success = await self.feature_store.set_features_batch(
                entity_id=entity_id,
                entity_type=entity_type,
                features=features,
                ttl=ttl
            )

            if success:
                logger.debug(
                    f"Materialized {len(features)} features for {entity_id}"
                )

            return success

        except Exception as e:
            logger.error(f"Error materializing features: {e}")
            return False

    async def materialize_transaction_features(
        self,
        transaction: Dict[str, Any]
    ) -> bool:
        """
        Materializa features após processar transação

        Args:
            transaction: Transaction data

        Returns:
            True se sucesso
        """
        try:
            customer_id = transaction.get('customer_id', '')
            if not customer_id:
                return False

            # Add transaction to aggregator
            await self.aggregator.add_event(
                entity_id=customer_id,
                event=transaction
            )

            # Materialize features
            return await self.materialize_features(
                entity_id=customer_id,
                entity_type='customer',
                ttl=3600  # 1 hour TTL
            )

        except Exception as e:
            logger.error(f"Error materializing transaction features: {e}")
            return False

    async def materialize_batch(
        self,
        entity_ids: List[str],
        entity_type: str = 'customer'
    ) -> Dict[str, bool]:
        """
        Materializa features em batch (parallel)

        Args:
            entity_ids: List of entity IDs
            entity_type: Entity type

        Returns:
            Dict mapping entity_id -> success
        """
        tasks = []
        for entity_id in entity_ids:
            task = self.materialize_features(
                entity_id=entity_id,
                entity_type=entity_type
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            entity_id: (result is True)
            for entity_id, result in zip(entity_ids, results)
        }

    async def start_periodic_materialization(
        self,
        interval_seconds: int = 300,  # 5 minutes
        active_entities_only: bool = True
    ) -> None:
        """
        Inicia materialization periódica em background

        Args:
            interval_seconds: Interval between materializations
            active_entities_only: Only materialize active entities
        """
        if self.running:
            logger.warning("Periodic materialization already running")
            return

        self.running = True

        async def periodic_task():
            while self.running:
                try:
                    logger.info("Starting periodic feature materialization...")

                    # Get active entities (simplificado - em produção usar query)
                    entity_ids = list(self.aggregator.events.keys())

                    if entity_ids:
                        # Materialize em batches
                        for i in range(0, len(entity_ids), self.batch_size):
                            batch = entity_ids[i:i + self.batch_size]
                            results = await self.materialize_batch(batch)

                            success_count = sum(1 for v in results.values() if v)
                            logger.info(
                                f"Batch materialized: {success_count}/{len(batch)} successful"
                            )

                    # Wait for next interval
                    await asyncio.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"Error in periodic materialization: {e}")
                    await asyncio.sleep(60)  # Wait 1 min on error

        self.materialization_task = asyncio.create_task(periodic_task())
        logger.info(f"Periodic materialization started: interval={interval_seconds}s")

    async def stop_periodic_materialization(self) -> None:
        """Para materialization periódica"""
        self.running = False

        if self.materialization_task:
            self.materialization_task.cancel()
            try:
                await self.materialization_task
            except asyncio.CancelledError:
                pass

        logger.info("Periodic materialization stopped")

    async def backfill_features(
        self,
        entity_id: str,
        transactions: List[Dict[str, Any]],
        entity_type: str = 'customer'
    ) -> bool:
        """
        Backfill features a partir de transações históricas

        Args:
            entity_id: Entity ID
            transactions: Historical transactions
            entity_type: Entity type

        Returns:
            True se sucesso
        """
        try:
            logger.info(f"Backfilling features for {entity_id}: {len(transactions)} transactions")

            # Add all transactions to aggregator
            for txn in sorted(transactions, key=lambda t: t.get('created_at', '')):
                timestamp = txn.get('created_at')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

                await self.aggregator.add_event(
                    entity_id=entity_id,
                    event=txn,
                    timestamp=timestamp
                )

            # Materialize computed features
            success = await self.materialize_features(
                entity_id=entity_id,
                entity_type=entity_type,
                ttl=86400  # 24 hours for backfilled data
            )

            logger.info(f"Backfill completed for {entity_id}: success={success}")
            return success

        except Exception as e:
            logger.error(f"Error backfilling features: {e}")
            return False

    async def get_materialization_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de materialization

        Returns:
            Dict com stats
        """
        # Count entities with events
        entities_with_events = len(self.aggregator.events)

        # Count total events
        total_events = sum(len(events) for events in self.aggregator.events.values())

        return {
            'strategy': self.strategy,
            'running': self.running,
            'entities_tracked': entities_with_events,
            'total_events': total_events,
            'avg_events_per_entity': (
                total_events / entities_with_events
                if entities_with_events > 0
                else 0
            ),
            'batch_size': self.batch_size
        }


# Singleton instance
_materializer_instance: Optional[FeatureMaterializer] = None


def get_feature_materializer(strategy: str = 'event_driven') -> FeatureMaterializer:
    """
    Retorna singleton do Feature Materializer

    Args:
        strategy: Materialization strategy

    Returns:
        FeatureMaterializer instance
    """
    global _materializer_instance

    if _materializer_instance is None:
        _materializer_instance = FeatureMaterializer(
            materialization_strategy=strategy,
            batch_size=100
        )

    return _materializer_instance


# Example usage
async def example_usage():
    """Example: Como usar Feature Materializer"""

    materializer = get_feature_materializer(strategy='event_driven')

    # Simulate incoming transaction
    transaction = {
        'id': 'TXN_001',
        'customer_id': 'CUST_123',
        'amount': 1000.0,
        'merchant_id': 'MERCHANT_456',
        'device_id': 'DEVICE_789',
        'location': 'São Paulo',
        'channel': 'PIX',
        'status': 'approved',
        'created_at': datetime.utcnow()
    }

    # Materialize features after transaction
    success = await materializer.materialize_transaction_features(transaction)
    print(f"Features materialized: {success}")

    # Get feature store to retrieve
    feature_store = get_feature_store()

    # Retrieve features
    features = await feature_store.get_features(
        entity_id='CUST_123',
        entity_type='customer',
        windows=['5m', '1h', '24h']
    )

    print(f"Retrieved features: {features}")

    # Stats
    stats = await materializer.get_materialization_stats()
    print(f"Materialization stats: {stats}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
