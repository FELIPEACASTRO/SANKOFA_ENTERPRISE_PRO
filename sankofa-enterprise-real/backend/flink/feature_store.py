"""
Flink Feature Store - Real-time Feature Computation & Serving
Window-based aggregations with <5ms retrieval from Redis
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)


class FlinkFeatureStore:
    """
    Feature Store com features computadas em tempo real via Flink

    Arquitetura:
    1. Flink computa aggregations em windows (5m, 1h, 24h, 7d, 30d)
    2. Features materializadas em Redis (cache layer)
    3. API de retrieval <5ms P95

    Features por window:
    - 5min: velocity, recent_amount_sum
    - 1hour: txn_count, unique_merchants, avg_amount
    - 24hours: daily_volume, device_changes, failed_txns
    - 7days: weekly_pattern, fraud_rate
    - 30days: monthly_avg, seasonal_pattern
    """

    # Feature definitions por window
    FEATURE_WINDOWS = {
        '5m': [
            'velocity_5m',
            'amount_sum_5m',
            'amount_avg_5m',
            'unique_devices_5m',
            'failed_txn_count_5m'
        ],
        '1h': [
            'txn_count_1h',
            'amount_sum_1h',
            'amount_avg_1h',
            'unique_merchants_1h',
            'unique_locations_1h',
            'cross_border_count_1h'
        ],
        '24h': [
            'daily_volume',
            'daily_txn_count',
            'device_changes_24h',
            'failed_txn_count_24h',
            'chargeback_count_24h',
            'max_single_txn_24h'
        ],
        '7d': [
            'weekly_volume',
            'weekly_txn_count',
            'fraud_rate_7d',
            'avg_daily_volume_7d',
            'unique_merchants_7d'
        ],
        '30d': [
            'monthly_volume',
            'monthly_avg_txn',
            'seasonal_pattern',
            'merchant_diversity_30d',
            'chargeback_rate_30d'
        ]
    }

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        ttl_seconds: int = 86400  # 24 hours
    ):
        """
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password
            ttl_seconds: TTL para features (default: 24h)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.ttl_seconds = ttl_seconds

        # Redis client (lazy init)
        self._redis: Optional[redis.Redis] = None

        logger.info(f"Feature Store initialized: redis={redis_host}:{redis_port}")

    async def _get_redis(self) -> redis.Redis:
        """Lazy init Redis connection"""
        if self._redis is None:
            self._redis = await redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True
            )
        return self._redis

    async def get_features(
        self,
        entity_id: str,
        entity_type: str = 'customer',
        windows: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retorna features para uma entidade

        Args:
            entity_id: ID da entidade (customer_id, device_id, etc.)
            entity_type: Tipo de entidade
            windows: Windows para buscar (default: all)

        Returns:
            Dict com features

        Example:
            features = await feature_store.get_features(
                entity_id='CUST_123',
                entity_type='customer',
                windows=['5m', '1h', '24h']
            )
            # Returns:
            # {
            #   'velocity_5m': 3,
            #   'amount_sum_5m': 1500.0,
            #   'txn_count_1h': 5,
            #   'daily_volume': 10000.0,
            #   ...
            # }
        """
        start_time = datetime.now(timezone.utc)

        try:
            r = await self._get_redis()

            # Default: all windows
            if windows is None:
                windows = list(self.FEATURE_WINDOWS.keys())

            features = {}

            # Fetch features para cada window
            for window in windows:
                window_features = await self._get_window_features(
                    r, entity_id, entity_type, window
                )
                features.update(window_features)

            # Calculate retrieval latency
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            logger.debug(
                f"Features retrieved: entity={entity_id}, "
                f"features={len(features)}, latency={latency_ms:.2f}ms"
            )

            # Add metadata
            features['_metadata'] = {
                'entity_id': entity_id,
                'entity_type': entity_type,
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'latency_ms': latency_ms,
                'feature_count': len(features) - 1  # Exclude metadata
            }

            return features

        except Exception as e:
            logger.error(f"Error getting features: {e}")
            return self._get_default_features(windows or [])

    async def _get_window_features(
        self,
        r: redis.Redis,
        entity_id: str,
        entity_type: str,
        window: str
    ) -> Dict[str, Any]:
        """
        Busca features de uma window específica

        Args:
            r: Redis client
            entity_id: Entity ID
            entity_type: Entity type
            window: Window size (5m, 1h, etc.)

        Returns:
            Dict com features da window
        """
        feature_names = self.FEATURE_WINDOWS.get(window, [])
        features = {}

        # Fetch all features em batch (pipelining)
        pipe = r.pipeline()
        for feature_name in feature_names:
            key = self._make_feature_key(entity_id, entity_type, feature_name)
            pipe.get(key)

        results = await pipe.execute()

        # Parse results
        for feature_name, value in zip(feature_names, results):
            if value is not None:
                try:
                    # Try parse as number
                    features[feature_name] = float(value)
                except ValueError:
                    features[feature_name] = value
            else:
                # Default value se feature não existe
                features[feature_name] = 0.0

        return features

    async def set_feature(
        self,
        entity_id: str,
        entity_type: str,
        feature_name: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Define valor de uma feature

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            feature_name: Feature name
            value: Feature value
            ttl: TTL em segundos (opcional)

        Returns:
            True se sucesso
        """
        try:
            r = await self._get_redis()

            key = self._make_feature_key(entity_id, entity_type, feature_name)
            ttl = ttl or self.ttl_seconds

            # Convert value to string
            value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

            # Set with TTL
            await r.setex(key, ttl, value_str)

            return True

        except Exception as e:
            logger.error(f"Error setting feature: {e}")
            return False

    async def set_features_batch(
        self,
        entity_id: str,
        entity_type: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Define múltiplas features em batch

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            features: Dict de features
            ttl: TTL em segundos

        Returns:
            True se sucesso
        """
        try:
            r = await self._get_redis()
            ttl = ttl or self.ttl_seconds

            # Pipeline for batch operations
            pipe = r.pipeline()

            for feature_name, value in features.items():
                if feature_name.startswith('_'):
                    continue  # Skip metadata

                key = self._make_feature_key(entity_id, entity_type, feature_name)
                value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

                pipe.setex(key, ttl, value_str)

            await pipe.execute()

            logger.debug(f"Batch set {len(features)} features for {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error setting features batch: {e}")
            return False

    def _make_feature_key(
        self,
        entity_id: str,
        entity_type: str,
        feature_name: str
    ) -> str:
        """
        Gera Redis key para feature

        Format: features:{entity_type}:{entity_id}:{feature_name}

        Args:
            entity_id: Entity ID
            entity_type: Entity type
            feature_name: Feature name

        Returns:
            Redis key
        """
        return f"features:{entity_type}:{entity_id}:{feature_name}"

    def _get_default_features(self, windows: List[str]) -> Dict[str, Any]:
        """
        Retorna features default (zero) quando erro

        Args:
            windows: Windows

        Returns:
            Dict com features default
        """
        features = {}

        for window in windows:
            feature_names = self.FEATURE_WINDOWS.get(window, [])
            for feature_name in feature_names:
                features[feature_name] = 0.0

        features['_metadata'] = {
            'error': 'Failed to retrieve features, using defaults',
            'retrieved_at': datetime.now(timezone.utc).isoformat()
        }

        return features

    async def compute_features_realtime(
        self,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computa features em tempo real (fallback quando Flink não disponível)

        Args:
            transaction: Transaction data

        Returns:
            Computed features
        """
        customer_id = transaction.get('customer_id', '')

        # Fetch historical transactions (simplificado)
        # Em produção, isso seria feito pelo Flink
        features = {
            # 5 minute window (placeholder)
            'velocity_5m': 1,
            'amount_sum_5m': float(transaction.get('amount', 0)),

            # 1 hour window (placeholder)
            'txn_count_1h': 1,
            'amount_sum_1h': float(transaction.get('amount', 0)),

            # 24 hour window (placeholder)
            'daily_volume': float(transaction.get('amount', 0)),
            'daily_txn_count': 1,

            # Metadata
            '_computed_by': 'realtime_fallback',
            '_timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Materialize features
        await self.set_features_batch(
            entity_id=customer_id,
            entity_type='customer',
            features=features,
            ttl=300  # 5 min TTL para realtime features
        )

        return features

    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            logger.info("Feature Store connection closed")


# Singleton instance
_feature_store_instance: Optional[FlinkFeatureStore] = None


def get_feature_store() -> FlinkFeatureStore:
    """
    Retorna singleton do Feature Store

    Returns:
        FlinkFeatureStore instance
    """
    global _feature_store_instance

    if _feature_store_instance is None:
        # TODO: Get from config
        _feature_store_instance = FlinkFeatureStore(
            redis_host='localhost',
            redis_port=6379
        )

    return _feature_store_instance
