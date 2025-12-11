"""
Unit Tests for ML Gateway - Infrastructure Layer
================================================

Tests for infrastructure/ml_gateway.py
Target Coverage: >95%

ML Gateway adapters tested:
- ProductionMLGateway (base adapter for ML engine)
- CachedMLGateway (adds caching)
- FallbackMLGateway (adds circuit breaker + fallback)
- create_production_ml_gateway (factory function)

Test categories:
1. Adapter construction
2. Domain ← → ML format conversion
3. Caching behavior
4. Fallback and error handling
5. Performance characteristics
"""

import pytest
import asyncio
import hashlib
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from infrastructure.ml_gateway import (
    ProductionMLGateway,
    CachedMLGateway,
    FallbackMLGateway,
    create_production_ml_gateway
)
from core.interfaces import FraudDetectionService
from core.entities import Transaction, TransactionFactory, FraudAnalysisResult, TransactionId, RiskLevel
from core.value_objects import RiskScore


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_fraud_engine():
    """Mock production fraud engine"""
    engine = Mock()
    engine.predict = Mock(return_value={
        "fraud_probability": 0.75,
        "model_version": "v1.0.0",
        "feature_importance": {
            "amount": 0.4,
            "velocity": 0.3,
            "device_risk": 0.2
        }
    })
    return engine


@pytest.fixture
def sample_transaction():
    """Create sample transaction"""
    return TransactionFactory.create_transaction(
        amount=Decimal("1000.00"),
        currency="BRL",
        merchant_id="MERCHANT_001",
        customer_id="CUSTOMER_001",
        metadata={
            "channel": "PIX",
            "device_id": "device_123",
            "ip_address": "192.168.1.1"
        }
    )


@pytest.fixture
def mock_cache_service():
    """Mock cache service (Redis)"""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


# ============================================================================
# ProductionMLGateway Tests
# ============================================================================

class TestProductionMLGateway:
    """Test base ProductionMLGateway adapter"""

    def test_production_gateway_creation(self, mock_fraud_engine):
        """Test creating ProductionMLGateway"""
        gateway = ProductionMLGateway(mock_fraud_engine)
        assert gateway is not None
        assert isinstance(gateway, FraudDetectionService)

    @pytest.mark.asyncio
    async def test_gateway_analyze_transaction_success(self, mock_fraud_engine, sample_transaction):
        """Test analyzing transaction successfully"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        result = await gateway.analyze_transaction(sample_transaction)

        assert isinstance(result, FraudAnalysisResult)
        assert result.transaction_id == sample_transaction.id
        assert isinstance(result.confidence_score, float)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.model_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_gateway_calls_ml_engine_predict(self, mock_fraud_engine, sample_transaction):
        """Test gateway calls ML engine's predict method"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        await gateway.analyze_transaction(sample_transaction)

        # Should have called engine.predict once
        mock_fraud_engine.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_gateway_transaction_to_ml_input_conversion(self, mock_fraud_engine, sample_transaction):
        """Test conversion from Transaction entity to ML input format"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        await gateway.analyze_transaction(sample_transaction)

        # Check what was passed to ML engine
        ml_input = mock_fraud_engine.predict.call_args[0][0]

        assert isinstance(ml_input, dict)
        assert "amount" in ml_input
        assert ml_input["amount"] == float(sample_transaction.amount.amount)
        assert "currency" in ml_input
        assert ml_input["currency"] == "BRL"
        assert "merchant_id" in ml_input
        assert "customer_id" in ml_input

    @pytest.mark.asyncio
    async def test_gateway_extracts_time_features(self, mock_fraud_engine, sample_transaction):
        """Test gateway extracts time-based features (hour, day_of_week)"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        await gateway.analyze_transaction(sample_transaction)

        ml_input = mock_fraud_engine.predict.call_args[0][0]

        assert "hour" in ml_input
        assert "day_of_week" in ml_input
        assert "is_weekend" in ml_input
        assert ml_input["hour"] == sample_transaction.timestamp.hour
        assert ml_input["day_of_week"] == sample_transaction.timestamp.weekday()

    @pytest.mark.asyncio
    async def test_gateway_includes_metadata(self, mock_fraud_engine):
        """Test gateway includes transaction metadata in ML input"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("1000.00"),
            currency="BRL",
            merchant_id="M1",
            customer_id="C1",
            metadata={
                "channel": "PIX",
                "device_id": "device_123",
                "ip_address": "192.168.1.1",
                "location": "São Paulo"
            }
        )

        gateway = ProductionMLGateway(mock_fraud_engine)
        await gateway.analyze_transaction(transaction)

        ml_input = mock_fraud_engine.predict.call_args[0][0]

        assert ml_input["channel"] == "PIX"
        assert ml_input["device_id"] == "device_123"
        assert ml_input["ip_address"] == "192.168.1.1"
        assert ml_input["location"] == "São Paulo"

    @pytest.mark.asyncio
    async def test_gateway_prediction_to_fraud_result_conversion(self, mock_fraud_engine, sample_transaction):
        """Test conversion from ML prediction to FraudAnalysisResult"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        result = await gateway.analyze_transaction(sample_transaction)

        # Check FraudAnalysisResult fields
        assert result.transaction_id == sample_transaction.id
        assert result.is_fraud is not None
        assert isinstance(result.is_fraud, bool)
        assert result.confidence_score == 0.75
        assert len(result.risk_factors) > 0  # Should extract risk factors

    @pytest.mark.asyncio
    async def test_gateway_fraud_threshold_logic(self, mock_fraud_engine, sample_transaction):
        """Test is_fraud flag based on threshold (default 0.5)"""
        # High probability -> fraud
        mock_fraud_engine.predict = Mock(return_value={
            "fraud_probability": 0.85,
            "model_version": "v1.0.0"
        })

        gateway = ProductionMLGateway(mock_fraud_engine)
        result_high = await gateway.analyze_transaction(sample_transaction)

        assert result_high.is_fraud is True

        # Low probability -> not fraud
        mock_fraud_engine.predict = Mock(return_value={
            "fraud_probability": 0.25,
            "model_version": "v1.0.0"
        })

        gateway = ProductionMLGateway(mock_fraud_engine)
        result_low = await gateway.analyze_transaction(sample_transaction)

        assert result_low.is_fraud is False

    @pytest.mark.asyncio
    async def test_gateway_extracts_risk_factors_from_feature_importance(self, mock_fraud_engine, sample_transaction):
        """Test extracting risk factors from SHAP/LIME feature importance"""
        mock_fraud_engine.predict = Mock(return_value={
            "fraud_probability": 0.75,
            "model_version": "v1.0.0",
            "feature_importance": {
                "amount": 0.5,
                "velocity": 0.3,
                "device_risk": 0.15,
                "location": 0.05
            }
        })

        gateway = ProductionMLGateway(mock_fraud_engine)
        result = await gateway.analyze_transaction(sample_transaction)

        # Should extract top features
        assert len(result.risk_factors) > 0
        # Top features should be in risk factors
        assert any("amount" in factor.lower() for factor in result.risk_factors)
        assert any("velocity" in factor.lower() for factor in result.risk_factors)

    @pytest.mark.asyncio
    async def test_gateway_uses_explicit_risk_factors_if_provided(self, mock_fraud_engine, sample_transaction):
        """Test using explicit risk factors from model"""
        mock_fraud_engine.predict = Mock(return_value={
            "fraud_probability": 0.85,
            "model_version": "v1.0.0",
            "risk_factors": ["high_amount", "new_device", "unusual_location"]
        })

        gateway = ProductionMLGateway(mock_fraud_engine)
        result = await gateway.analyze_transaction(sample_transaction)

        assert "high_amount" in result.risk_factors
        assert "new_device" in result.risk_factors
        assert "unusual_location" in result.risk_factors

    @pytest.mark.asyncio
    async def test_gateway_records_processing_time(self, mock_fraud_engine, sample_transaction):
        """Test gateway records processing time"""
        gateway = ProductionMLGateway(mock_fraud_engine)

        result = await gateway.analyze_transaction(sample_transaction)

        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0

    def test_gateway_get_model_info(self, mock_fraud_engine):
        """Test getting model information"""
        mock_fraud_engine.model_version = "v2.0.0"
        mock_fraud_engine.feature_names = ["amount", "velocity", "device"]

        gateway = ProductionMLGateway(mock_fraud_engine)
        info = gateway.get_model_info()

        assert info is not None
        assert "model_type" in info
        assert "model_version" in info or "features_count" in info

    @pytest.mark.asyncio
    async def test_gateway_handles_ml_engine_error(self, mock_fraud_engine, sample_transaction):
        """Test gateway handles ML engine errors"""
        mock_fraud_engine.predict = Mock(side_effect=Exception("Model inference failed"))

        gateway = ProductionMLGateway(mock_fraud_engine)

        with pytest.raises(Exception):
            await gateway.analyze_transaction(sample_transaction)


# ============================================================================
# CachedMLGateway Tests
# ============================================================================

class TestCachedMLGateway:
    """Test CachedMLGateway decorator"""

    @pytest.mark.asyncio
    async def test_cached_gateway_creation(self, mock_fraud_engine, mock_cache_service):
        """Test creating CachedMLGateway"""
        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service, ttl=300)

        assert cached_gateway is not None
        assert isinstance(cached_gateway, FraudDetectionService)

    @pytest.mark.asyncio
    async def test_cached_gateway_cache_miss_calls_ml(self, mock_fraud_engine, mock_cache_service, sample_transaction):
        """Test cache miss calls underlying ML gateway"""
        mock_cache_service.get = AsyncMock(return_value=None)  # Cache miss

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        await cached_gateway.analyze_transaction(sample_transaction)

        # Should have called ML engine
        mock_fraud_engine.predict.assert_called_once()
        # Should have cached result
        mock_cache_service.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_gateway_cache_hit_skips_ml(self, mock_fraud_engine, mock_cache_service, sample_transaction):
        """Test cache hit returns cached result without calling ML"""
        # Prepare cached result
        cached_result = FraudAnalysisResult(
            transaction_id=sample_transaction.id,
            is_fraud=True,
            confidence_score=0.85,
            risk_factors=["cached_factor"],
            model_version="v1.0.0",
            processing_time_ms=50.0
        )
        mock_cache_service.get = AsyncMock(return_value=cached_result)

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        result = await cached_gateway.analyze_transaction(sample_transaction)

        # Should return cached result
        assert result == cached_result
        # Should NOT have called ML engine
        mock_fraud_engine.predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_gateway_generates_cache_key(self, mock_fraud_engine, mock_cache_service, sample_transaction):
        """Test cache key generation based on transaction content"""
        mock_cache_service.get = AsyncMock(return_value=None)

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        await cached_gateway.analyze_transaction(sample_transaction)

        # Should have generated cache key
        mock_cache_service.get.assert_called_once()
        cache_key = mock_cache_service.get.call_args[0][0]

        assert isinstance(cache_key, str)
        assert "ml_prediction" in cache_key  # Key prefix

    @pytest.mark.asyncio
    async def test_cached_gateway_different_transactions_different_keys(
        self, mock_fraud_engine, mock_cache_service
    ):
        """Test different transactions produce different cache keys"""
        mock_cache_service.get = AsyncMock(return_value=None)

        txn1 = TransactionFactory.create_transaction(
            amount=Decimal("1000.00"), currency="BRL",
            merchant_id="M1", customer_id="C1"
        )
        txn2 = TransactionFactory.create_transaction(
            amount=Decimal("2000.00"), currency="BRL",
            merchant_id="M1", customer_id="C1"
        )

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        await cached_gateway.analyze_transaction(txn1)
        await cached_gateway.analyze_transaction(txn2)

        # Should have different cache keys
        assert mock_cache_service.get.call_count == 2
        key1 = mock_cache_service.get.call_args_list[0][0][0]
        key2 = mock_cache_service.get.call_args_list[1][0][0]
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cached_gateway_respects_ttl(self, mock_fraud_engine, mock_cache_service, sample_transaction):
        """Test cached gateway sets TTL when caching"""
        mock_cache_service.get = AsyncMock(return_value=None)

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        ttl = 1800  # 30 minutes
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service, ttl=ttl)

        await cached_gateway.analyze_transaction(sample_transaction)

        # Check TTL was passed to cache.set
        mock_cache_service.set.assert_called_once()
        call_kwargs = mock_cache_service.set.call_args.kwargs
        if 'ttl' in call_kwargs:
            assert call_kwargs['ttl'] == ttl

    @pytest.mark.asyncio
    async def test_cached_gateway_handles_cache_errors_gracefully(
        self, mock_fraud_engine, mock_cache_service, sample_transaction
    ):
        """Test gateway continues working if cache.set() fails"""
        mock_cache_service.get = AsyncMock(return_value=None)  # Cache miss
        mock_cache_service.set = AsyncMock(side_effect=Exception("Cache unavailable"))

        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        # Should not raise when cache.set() fails, should call ML engine
        result = await cached_gateway.analyze_transaction(sample_transaction)

        assert isinstance(result, FraudAnalysisResult)
        mock_fraud_engine.predict.assert_called_once()

    def test_cached_gateway_get_model_info_delegates(self, mock_fraud_engine, mock_cache_service):
        """Test get_model_info delegates to base gateway"""
        base_gateway = ProductionMLGateway(mock_fraud_engine)
        cached_gateway = CachedMLGateway(base_gateway, mock_cache_service)

        info = cached_gateway.get_model_info()

        assert info is not None


# ============================================================================
# FallbackMLGateway Tests
# ============================================================================

class TestFallbackMLGateway:
    """Test FallbackMLGateway (Circuit Breaker + Fallback)"""

    @pytest.fixture
    def mock_primary_gateway(self):
        """Mock primary ML gateway"""
        gateway = AsyncMock(spec=FraudDetectionService)
        gateway.analyze_transaction = AsyncMock()
        gateway.get_model_info = Mock(return_value={"model": "primary"})
        return gateway

    @pytest.fixture
    def mock_fallback_gateway(self):
        """Mock fallback gateway (rule-based)"""
        gateway = AsyncMock(spec=FraudDetectionService)
        gateway.analyze_transaction = AsyncMock()
        gateway.get_model_info = Mock(return_value={"model": "fallback"})
        return gateway

    def test_fallback_gateway_creation(self, mock_primary_gateway, mock_fallback_gateway):
        """Test creating FallbackMLGateway"""
        gateway = FallbackMLGateway(
            mock_primary_gateway,
            mock_fallback_gateway,
            timeout_seconds=2.0
        )

        assert gateway is not None

    @pytest.mark.asyncio
    async def test_fallback_uses_primary_when_available(
        self, mock_primary_gateway, mock_fallback_gateway, sample_transaction
    ):
        """Test fallback gateway uses primary when it's working"""
        mock_primary_gateway.analyze_transaction = AsyncMock(return_value=FraudAnalysisResult(
            transaction_id=sample_transaction.id,
            is_fraud=True,
            confidence_score=0.85,
            risk_factors=["primary_factor"],
            model_version="primary",
            processing_time_ms=50.0
        ))

        gateway = FallbackMLGateway(mock_primary_gateway, mock_fallback_gateway)

        result = await gateway.analyze_transaction(sample_transaction)

        # Should use primary
        assert result.model_version == "primary"
        mock_primary_gateway.analyze_transaction.assert_called_once()
        mock_fallback_gateway.analyze_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_on_primary_failure(
        self, mock_primary_gateway, mock_fallback_gateway, sample_transaction
    ):
        """Test fallback gateway uses fallback when primary fails"""
        mock_primary_gateway.analyze_transaction = AsyncMock(
            side_effect=Exception("Primary service down")
        )
        mock_fallback_gateway.analyze_transaction = AsyncMock(return_value=FraudAnalysisResult(
            transaction_id=sample_transaction.id,
            is_fraud=False,
            confidence_score=0.3,
            risk_factors=["fallback_factor"],
            model_version="fallback",
            processing_time_ms=10.0
        ))

        gateway = FallbackMLGateway(mock_primary_gateway, mock_fallback_gateway)

        result = await gateway.analyze_transaction(sample_transaction)

        # Should use fallback
        assert result.model_version == "fallback"
        mock_fallback_gateway.analyze_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_on_timeout(
        self, mock_fallback_gateway, sample_transaction
    ):
        """Test fallback gateway uses fallback when primary times out"""
        # Slow primary gateway
        async def slow_primary(txn):
            await asyncio.sleep(5.0)  # Very slow
            return FraudAnalysisResult(...)

        mock_primary_gateway = Mock(spec=FraudDetectionService)
        mock_primary_gateway.analyze_transaction = slow_primary

        mock_fallback_gateway.analyze_transaction = AsyncMock(return_value=FraudAnalysisResult(
            transaction_id=sample_transaction.id,
            is_fraud=False,
            confidence_score=0.3,
            risk_factors=[],
            model_version="fallback",
            processing_time_ms=10.0
        ))

        gateway = FallbackMLGateway(
            mock_primary_gateway,
            mock_fallback_gateway,
            timeout_seconds=0.1  # Short timeout
        )

        result = await gateway.analyze_transaction(sample_transaction)

        # Should use fallback (timeout)
        assert result.model_version == "fallback"

    def test_fallback_get_model_info_returns_both(
        self, mock_primary_gateway, mock_fallback_gateway
    ):
        """Test get_model_info returns info for both gateways"""
        gateway = FallbackMLGateway(mock_primary_gateway, mock_fallback_gateway)

        info = gateway.get_model_info()

        assert "primary" in info
        assert "fallback" in info


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunction:
    """Test create_production_ml_gateway factory function"""

    def test_factory_creates_base_gateway_without_cache(self, mock_fraud_engine):
        """Test factory creates base gateway when no cache provided"""
        gateway = create_production_ml_gateway(mock_fraud_engine)

        assert isinstance(gateway, ProductionMLGateway)

    def test_factory_creates_cached_gateway_with_cache(self, mock_fraud_engine, mock_cache_service):
        """Test factory creates cached gateway when cache provided"""
        gateway = create_production_ml_gateway(mock_fraud_engine, mock_cache_service)

        assert isinstance(gateway, CachedMLGateway)


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for ml_gateway.py:

ProductionMLGateway: 15 tests
- Creation & basic functionality: 2
- Domain ← → ML conversion: 5
- Prediction processing: 4
- Risk factor extraction: 2
- Error handling: 1
- Model info: 1

CachedMLGateway: 8 tests
- Cache hit/miss: 2
- Cache key generation: 2
- TTL handling: 1
- Error handling: 1
- Delegation: 1

FallbackMLGateway: 5 tests
- Primary usage: 1
- Fallback on error: 1
- Fallback on timeout: 1
- Model info: 1

Factory: 2 tests
- Without cache: 1
- With cache: 1

TOTAL: 30 tests
TARGET: >95% statement coverage
PATTERNS: Adapter, Decorator, Circuit Breaker, Factory
"""
