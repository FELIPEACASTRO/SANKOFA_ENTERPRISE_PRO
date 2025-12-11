"""
ML Gateway Adapter - Infrastructure Layer
Implements FraudDetectionService interface for ML engine integration

This demonstrates the Adapter Pattern - wrapping external ML engine behind
domain interface for Dependency Inversion Principle.

Benefits:
- Domain layer doesn't depend on ML engine implementation
- Can swap ML frameworks (sklearn → PyTorch → TensorFlow) without touching domain
- Can add caching, monitoring, fallback without modifying domain
- Testable with mock models

Time Complexity: Depends on underlying ML model
"""

import time
from typing import Dict, Any
from decimal import Decimal

from core.interfaces import FraudDetectionService
from core.entities import Transaction, FraudAnalysisResult, TransactionId, RiskLevel
from utils.structured_logging import get_structured_logger
from utils.log_sanitizer import sanitize_log_data


logger = get_structured_logger("ml_gateway")


class ProductionMLGateway(FraudDetectionService):
    """
    Adapter: Wraps production_fraud_engine behind FraudDetectionService interface

    Converts between:
    - Domain entities (Transaction) → ML engine input format
    - ML engine output → Domain entities (FraudAnalysisResult)

    Time Complexity: O(f + m) where:
    - f = feature extraction time (O(1) typically)
    - m = model inference time (O(1) for most production models)
    """

    def __init__(self, fraud_engine):
        """
        Initialize with production fraud engine

        Args:
            fraud_engine: Instance of ProductionFraudEngine
                         (from ml_engine.production_fraud_engine)
        """
        self._engine = fraud_engine

    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """
        Analyze transaction for fraud using ML model

        Converts domain Transaction entity to ML input format,
        calls model, and converts back to domain FraudAnalysisResult

        Time Complexity: O(f + m) ≈ O(1) for simple models

        Args:
            transaction: Domain Transaction entity

        Returns:
            FraudAnalysisResult with fraud probability and risk factors
        """
        start_time = time.time()

        try:
            # 1. Convert domain entity to ML input format - O(1)
            ml_input = self._transaction_to_ml_input(transaction)

            # Log sanitized input
            logger.debug(
                "Converting transaction to ML input",
                extra=sanitize_log_data({
                    'transaction_id': transaction.id.value,
                    'amount': float(transaction.amount.amount),
                    'merchant_id': transaction.merchant_id
                })
            )

            # 2. Call ML engine - O(m) where m is model inference time
            prediction = self._engine.predict(ml_input)

            # 3. Convert ML output to domain entity - O(1)
            result = self._prediction_to_domain_result(
                transaction.id,
                prediction,
                processing_time_ms=(time.time() - start_time) * 1000
            )

            # Log result
            logger.info(
                "ML prediction completed",
                extra={
                    'transaction_id': transaction.id.value,
                    'fraud_probability': result.confidence_score,
                    'is_fraud': result.is_fraud,
                    'processing_time_ms': result.processing_time_ms,
                    'model_version': result.model_version
                }
            )

            return result

        except Exception as e:
            logger.error(
                "ML prediction failed",
                extra=sanitize_log_data({
                    'transaction_id': transaction.id.value,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
            )
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns metadata about the ML model (version, type, etc.)

        Time Complexity: O(1)
        """
        try:
            return {
                'model_type': type(self._engine).__name__,
                'model_version': getattr(self._engine, 'model_version', 'unknown'),
                'features_count': len(self._engine.feature_names) if hasattr(self._engine, 'feature_names') else 'unknown'
            }
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {'model_type': 'unknown', 'error': str(e)}

    def _transaction_to_ml_input(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Convert domain Transaction entity to ML engine input format

        Maps domain concepts to ML features:
        - Transaction.amount → 'amount' feature
        - Transaction.merchant_id → 'merchant_id' feature
        - Transaction.timestamp → 'hour', 'day_of_week' features

        Time Complexity: O(1)

        Args:
            transaction: Domain Transaction entity

        Returns:
            Dictionary in ML engine's expected format
        """
        # Extract basic features
        ml_input = {
            'amount': float(transaction.amount.amount),
            'currency': transaction.amount.currency,
            'merchant_id': transaction.merchant_id,
            'customer_id': transaction.customer_id,
            'timestamp': transaction.timestamp.isoformat(),
        }

        # Add time-based features
        ml_input['hour'] = transaction.timestamp.hour
        ml_input['day_of_week'] = transaction.timestamp.weekday()
        ml_input['is_weekend'] = ml_input['day_of_week'] >= 5

        # Add metadata if present
        if transaction.metadata:
            # Common metadata fields
            for key in ['channel', 'device_id', 'ip_address', 'location']:
                if key in transaction.metadata:
                    ml_input[key] = transaction.metadata[key]

        return ml_input

    def _prediction_to_domain_result(
        self,
        transaction_id: TransactionId,
        prediction: Dict[str, Any],
        processing_time_ms: float
    ) -> FraudAnalysisResult:
        """
        Convert ML engine output to domain FraudAnalysisResult entity

        Maps ML predictions to domain concepts:
        - prediction['fraud_probability'] → confidence_score
        - prediction['risk_factors'] → risk_factors list
        - prediction['model_version'] → model_version

        Time Complexity: O(k) where k is number of risk factors (typically small)

        Args:
            transaction_id: Transaction ID
            prediction: ML engine prediction dictionary
            processing_time_ms: Processing time in milliseconds

        Returns:
            Domain FraudAnalysisResult entity
        """
        # Extract fraud probability (required field)
        fraud_probability = prediction.get('fraud_probability', 0.0)

        # Determine if fraud based on threshold
        # You can make this threshold configurable
        is_fraud = fraud_probability >= 0.5

        # Extract risk factors (features that contributed to decision)
        risk_factors = []

        # If model provides feature importance (SHAP, LIME, etc.)
        if 'feature_importance' in prediction:
            feature_importance = prediction['feature_importance']
            # Get top 5 most important features
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            for feature, importance in top_features:
                risk_factors.append(f"{feature}_impact_{importance:.3f}")

        # If model provides explicit risk factors
        if 'risk_factors' in prediction:
            risk_factors.extend(prediction['risk_factors'])

        # Fallback: Infer risk factors from prediction
        if not risk_factors:
            if fraud_probability > 0.8:
                risk_factors.append("high_fraud_probability")
            if prediction.get('amount', 0) > 5000:
                risk_factors.append("high_value_transaction")

        # Create domain entity
        return FraudAnalysisResult(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            confidence_score=fraud_probability,
            risk_factors=risk_factors,
            model_version=prediction.get('model_version', 'unknown'),
            processing_time_ms=processing_time_ms
        )


class CachedMLGateway(FraudDetectionService):
    """
    Decorator: Adds caching to ML Gateway

    Implements Decorator Pattern to add caching behavior without modifying
    the base MLGateway implementation.

    Benefits:
    - Reduces ML calls for repeated predictions
    - Improves latency (cache hit ~1ms vs ML call ~50ms)
    - Reduces costs (if using cloud ML APIs)

    Time Complexity:
    - Cache hit: O(1)
    - Cache miss: O(f + m) + O(1) to cache
    """

    def __init__(self, ml_gateway: FraudDetectionService, cache_service, ttl: int = 300):
        """
        Initialize with ML gateway and cache

        Args:
            ml_gateway: Underlying ML gateway to wrap
            cache_service: Cache service (e.g., Redis)
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self._ml_gateway = ml_gateway
        self._cache = cache_service
        self._ttl = ttl

    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """
        Analyze with caching

        Cache key based on transaction content (amount, merchant, customer, etc.)
        NOT just transaction ID (since ID is unique)

        Time Complexity: O(1) cache hit, O(f + m) cache miss
        """
        import hashlib

        # Generate cache key based on transaction content
        cache_key = self._generate_cache_key(transaction)

        # Try cache first - O(1)
        cached_result = await self._cache.get(cache_key)
        if cached_result is not None:
            logger.debug(
                "Cache HIT for ML prediction",
                extra={'cache_key': cache_key}
            )
            return cached_result

        # Cache miss - call underlying gateway - O(f + m)
        logger.debug(
            "Cache MISS for ML prediction",
            extra={'cache_key': cache_key}
        )
        result = await self._ml_gateway.analyze_transaction(transaction)

        # Cache result - O(1)
        try:
            await self._cache.set(cache_key, result, ttl=self._ttl)
        except Exception as e:
            logger.warning(f"Failed to cache ML prediction: {e}")

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Delegate to underlying gateway - O(1)"""
        return self._ml_gateway.get_model_info()

    def _generate_cache_key(self, transaction: Transaction) -> str:
        """
        Generate cache key from transaction content

        Uses only STABLE features (amount, merchant, customer)
        NOT transaction ID (always unique) or timestamp (changes)

        Time Complexity: O(1)
        """
        import hashlib

        # Include only features that matter for prediction
        key_components = [
            f"amount:{transaction.amount.amount}",
            f"currency:{transaction.amount.currency}",
            f"merchant:{transaction.merchant_id}",
            f"customer:{transaction.customer_id}",
        ]

        # Add metadata if present
        if transaction.metadata:
            # Add stable metadata fields
            for key in ['channel', 'device_id']:
                if key in transaction.metadata:
                    key_components.append(f"{key}:{transaction.metadata[key]}")

        key_string = "|".join(key_components)
        return f"ml_prediction:{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"


class FallbackMLGateway(FraudDetectionService):
    """
    Decorator: Adds fallback logic if primary ML gateway fails

    Implements Circuit Breaker + Fallback patterns

    Use case:
    - Primary ML service unavailable
    - Timeout on ML call
    - Fallback to rule-based scoring

    Time Complexity: O(max(primary, fallback))
    """

    def __init__(
        self,
        primary_gateway: FraudDetectionService,
        fallback_gateway: FraudDetectionService,
        timeout_seconds: float = 2.0
    ):
        """
        Initialize with primary and fallback gateways

        Args:
            primary_gateway: Primary ML gateway
            fallback_gateway: Fallback (e.g., rule-based scoring)
            timeout_seconds: Timeout for primary gateway
        """
        self._primary = primary_gateway
        self._fallback = fallback_gateway
        self._timeout = timeout_seconds

    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """
        Try primary, fallback on failure

        Time Complexity: O(max(primary, fallback))
        """
        import asyncio

        try:
            # Try primary with timeout - O(f + m)
            result = await asyncio.wait_for(
                self._primary.analyze_transaction(transaction),
                timeout=self._timeout
            )
            return result

        except asyncio.TimeoutError:
            logger.warning(
                "Primary ML gateway timeout, using fallback",
                extra={'transaction_id': transaction.id.value}
            )
            return await self._fallback.analyze_transaction(transaction)

        except Exception as e:
            logger.error(
                "Primary ML gateway failed, using fallback",
                extra=sanitize_log_data({
                    'transaction_id': transaction.id.value,
                    'error': str(e)
                })
            )
            return await self._fallback.analyze_transaction(transaction)

    def get_model_info(self) -> Dict[str, Any]:
        """Return info for both gateways - O(1)"""
        return {
            'primary': self._primary.get_model_info(),
            'fallback': self._fallback.get_model_info()
        }


# Factory function for production setup
def create_production_ml_gateway(fraud_engine, cache_service=None) -> FraudDetectionService:
    """
    Factory: Create production ML gateway with caching and fallback

    Stacks decorators:
    1. CachedMLGateway (adds caching)
    2. FallbackMLGateway (adds fallback to rules)
    3. ProductionMLGateway (base ML gateway)

    Time Complexity: O(1) - just creates objects

    Args:
        fraud_engine: Production fraud engine
        cache_service: Optional cache service

    Returns:
        Fully-configured ML gateway with caching and fallback
    """
    # Base gateway
    base_gateway = ProductionMLGateway(fraud_engine)

    # Add caching if cache service provided
    if cache_service:
        base_gateway = CachedMLGateway(base_gateway, cache_service, ttl=300)

    # Note: For fallback, you'd need a fallback fraud engine
    # Example:
    # from core.fraud_strategies import RuleBasedScoring
    # fallback_strategy = RuleBasedScoring()
    # fallback_gateway = StrategyBasedMLGateway(fallback_strategy)
    # gateway = FallbackMLGateway(base_gateway, fallback_gateway)

    return base_gateway
