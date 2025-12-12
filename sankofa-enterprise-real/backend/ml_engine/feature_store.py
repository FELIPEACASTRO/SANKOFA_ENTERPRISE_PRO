"""
Feature Store Implementation with Redis
Provides fast feature caching and retrieval for ML predictions
Target: <10ms feature retrieval latency
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
import redis
from utils.structured_logging import get_structured_logger

logger = get_structured_logger("feature_store")


class FeatureStore:
    """
    Redis-based Feature Store for ML predictions

    Features:
    - Fast feature caching (TTL 24h)
    - Customer-level feature aggregation
    - Automatic cache invalidation
    - Hit rate monitoring

    Performance target: <10ms retrieval
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize Feature Store

        Args:
            redis_client: Optional Redis client. If None, creates new connection.
        """
        if redis_client:
            self.redis = redis_client
        else:
            import os
            self.redis = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD'),
                db=1,  # Use DB 1 for feature store
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

        self.default_ttl = 86400  # 24 hours
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Feature Store initialized", ttl_hours=self.default_ttl / 3600)

    def _make_key(self, customer_id: str, feature_type: str = "aggregated") -> str:
        """Generate Redis key for feature storage"""
        return f"feature:{feature_type}:{customer_id}"

    def get_features(self, customer_id: str, feature_type: str = "aggregated") -> Optional[Dict[str, Any]]:
        """
        Retrieve features from cache

        Args:
            customer_id: Customer identifier (CPF or internal ID)
            feature_type: Type of features ('aggregated', 'behavioral', 'temporal')

        Returns:
            Dictionary of features or None if cache miss
        """
        try:
            key = self._make_key(customer_id, feature_type)
            cached = self.redis.get(key)

            if cached:
                self.cache_hits += 1
                features = json.loads(cached)
                logger.debug(
                    "Feature cache hit",
                    customer_id=customer_id[:8] + "***",
                    feature_type=feature_type,
                    hit_rate=self.get_hit_rate()
                )
                return features
            else:
                self.cache_misses += 1
                logger.debug(
                    "Feature cache miss",
                    customer_id=customer_id[:8] + "***",
                    feature_type=feature_type
                )
                return None

        except redis.RedisError as e:
            logger.error(f"Redis error getting features: {e}")
            self.cache_misses += 1
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None

    def set_features(
        self,
        customer_id: str,
        features: Dict[str, Any],
        feature_type: str = "aggregated",
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store features in cache

        Args:
            customer_id: Customer identifier
            features: Dictionary of feature values
            feature_type: Type of features
            ttl: Time-to-live in seconds (default: 24 hours)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_key(customer_id, feature_type)
            ttl = ttl or self.default_ttl

            # Add metadata
            features_with_meta = {
                **features,
                "_cached_at": datetime.now(timezone.utc).isoformat(),
                "_ttl": ttl
            }

            self.redis.setex(
                key,
                ttl,
                json.dumps(features_with_meta, default=str)
            )

            logger.debug(
                "Features cached",
                customer_id=customer_id[:8] + "***",
                feature_type=feature_type,
                feature_count=len(features),
                ttl_hours=ttl / 3600
            )
            return True

        except redis.RedisError as e:
            logger.error(f"Redis error setting features: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error: {e}")
            return False

    def get_or_compute(
        self,
        customer_id: str,
        compute_fn: callable,
        feature_type: str = "aggregated",
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get features from cache or compute and cache them

        This is the main interface for feature retrieval with automatic caching.

        Args:
            customer_id: Customer identifier
            compute_fn: Function that computes features if cache miss
            feature_type: Type of features
            ttl: Cache TTL

        Returns:
            Dictionary of features

        Example:
            >>> features = feature_store.get_or_compute(
            ...     customer_id="12345678901",
            ...     compute_fn=lambda: compute_customer_features(customer_id),
            ...     feature_type="aggregated"
            ... )
        """
        # Try cache first
        cached_features = self.get_features(customer_id, feature_type)

        if cached_features is not None:
            # Remove metadata before returning
            cached_features.pop('_cached_at', None)
            cached_features.pop('_ttl', None)
            return cached_features

        # Cache miss - compute features
        logger.info(
            "Computing features (cache miss)",
            customer_id=customer_id[:8] + "***",
            feature_type=feature_type
        )

        computed_features = compute_fn()

        # Cache for next time
        self.set_features(customer_id, computed_features, feature_type, ttl)

        return computed_features

    def invalidate(self, customer_id: str, feature_type: Optional[str] = None) -> int:
        """
        Invalidate cached features

        Args:
            customer_id: Customer identifier
            feature_type: Optional specific feature type to invalidate.
                         If None, invalidates all feature types.

        Returns:
            Number of keys deleted
        """
        try:
            if feature_type:
                key = self._make_key(customer_id, feature_type)
                deleted = self.redis.delete(key)
            else:
                # Delete all feature types for this customer
                pattern = f"feature:*:{customer_id}"
                keys = self.redis.keys(pattern)
                deleted = self.redis.delete(*keys) if keys else 0

            logger.info(
                "Features invalidated",
                customer_id=customer_id[:8] + "***",
                feature_type=feature_type or "all",
                keys_deleted=deleted
            )
            return deleted

        except redis.RedisError as e:
            logger.error(f"Redis error invalidating features: {e}")
            return 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(self.get_hit_rate(), 2),
            "total_requests": self.cache_hits + self.cache_misses
        }

    def bulk_get(self, customer_ids: List[str], feature_type: str = "aggregated") -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for multiple customers in one operation

        Args:
            customer_ids: List of customer identifiers
            feature_type: Type of features

        Returns:
            Dictionary mapping customer_id to features (None if not cached)
        """
        try:
            keys = [self._make_key(cid, feature_type) for cid in customer_ids]
            values = self.redis.mget(keys)

            results = {}
            for customer_id, value in zip(customer_ids, values):
                if value:
                    results[customer_id] = json.loads(value)
                    self.cache_hits += 1
                else:
                    results[customer_id] = None
                    self.cache_misses += 1

            logger.debug(
                "Bulk feature retrieval",
                customer_count=len(customer_ids),
                hits=sum(1 for v in results.values() if v is not None),
                misses=sum(1 for v in results.values() if v is None)
            )

            return results

        except redis.RedisError as e:
            logger.error(f"Redis error in bulk get: {e}")
            return {cid: None for cid in customer_ids}

    def health_check(self) -> bool:
        """Check if feature store is healthy"""
        try:
            self.redis.ping()
            return True
        except redis.RedisError:
            return False


# Singleton instance
_feature_store_instance = None


def get_feature_store() -> FeatureStore:
    """Get singleton feature store instance"""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore()
    return _feature_store_instance
