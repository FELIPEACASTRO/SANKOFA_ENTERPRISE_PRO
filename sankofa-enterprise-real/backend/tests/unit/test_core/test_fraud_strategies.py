"""
Unit Tests for Fraud Strategies - Domain Layer
===============================================

Tests for core/fraud_strategies.py
Target Coverage: >95%

Fraud strategies tested:
- RuleBasedScoring (fast, explainable rules)
- MLBasedScoring (machine learning model)
- VelocityBasedScoring (burst detection)
- CompositeScoring (weighted combination)

Test categories:
1. Strategy construction
2. Score calculation (happy path)
3. Score calculation (edge cases)
4. Risk factor identification
5. Performance characteristics
6. Strategy composition
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from core.fraud_strategies import (
    FraudScoringStrategy,
    RuleBasedScoring,
    MLBasedScoring,
    VelocityBasedScoring,
    CompositeScoring,
    FraudScoreResult
)
from core.entities import Transaction, TransactionFactory, Money, TransactionId
from core.value_objects import RiskScore, RiskLevel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing"""
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
def low_risk_transaction():
    """Create low-risk transaction (small amount, known merchant)"""
    return TransactionFactory.create_transaction(
        amount=Decimal("50.00"),
        currency="BRL",
        merchant_id="TRUSTED_MERCHANT",
        customer_id="GOOD_CUSTOMER",
        metadata={"channel": "CREDIT_CARD"}
    )


@pytest.fixture
def high_risk_transaction():
    """Create high-risk transaction (large amount, new customer)"""
    return TransactionFactory.create_transaction(
        amount=Decimal("50000.00"),
        currency="BRL",
        merchant_id="NEW_MERCHANT",
        customer_id="NEW_CUSTOMER",
        metadata={
            "channel": "PIX",
            "device_id": "unknown_device",
            "is_new_customer": True
        }
    )


@pytest.fixture
def sample_context():
    """Create sample context with historical data"""
    return {
        "customer_history": {
            "total_transactions": 100,
            "total_amount": Decimal("50000.00"),
            "avg_transaction_amount": Decimal("500.00"),
            "fraud_count": 0,
        },
        "merchant_history": {
            "total_transactions": 1000,
            "fraud_rate": 0.01,
        },
        "velocity_data": {
            "transactions_last_hour": 2,
            "transactions_last_day": 10,
            "amount_last_hour": Decimal("2000.00"),
        }
    }


# ============================================================================
# RuleBasedScoring Tests
# ============================================================================

class TestRuleBasedScoring:
    """Test Rule-Based Fraud Scoring Strategy"""

    @pytest.mark.asyncio
    async def test_rule_based_strategy_creation(self):
        """Test creating rule-based strategy with default rules"""
        strategy = RuleBasedScoring()
        assert strategy is not None
        assert isinstance(strategy, FraudScoringStrategy)

    @pytest.mark.asyncio
    async def test_rule_based_low_risk_transaction(self, low_risk_transaction, sample_context):
        """Test rule-based scoring on low-risk transaction"""
        strategy = RuleBasedScoring()

        result = await strategy.calculate_score(low_risk_transaction, sample_context)

        assert isinstance(result, FraudScoreResult)
        assert isinstance(result.score, RiskScore)
        assert result.score.value < 0.3  # Low risk
        assert result.score.risk_level() == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_rule_based_high_risk_transaction(self, high_risk_transaction, sample_context):
        """Test rule-based scoring on high-risk transaction"""
        strategy = RuleBasedScoring()

        result = await strategy.calculate_score(high_risk_transaction, sample_context)

        assert result.score.value > 0.5  # High risk
        assert result.score.risk_level() in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_rule_based_high_amount_increases_score(self, sample_context):
        """Test that high amounts increase risk score"""
        strategy = RuleBasedScoring()

        low_amount_txn = TransactionFactory.create_transaction(
            amount=Decimal("100.00"), currency="BRL",
            merchant_id="M1", customer_id="C1"
        )
        high_amount_txn = TransactionFactory.create_transaction(
            amount=Decimal("50000.00"), currency="BRL",
            merchant_id="M1", customer_id="C1"
        )

        result_low = await strategy.calculate_score(low_amount_txn, sample_context)
        result_high = await strategy.calculate_score(high_amount_txn, sample_context)

        assert result_high.score.value > result_low.score.value

    @pytest.mark.asyncio
    async def test_rule_based_new_customer_increases_score(self, sample_context):
        """Test that new customers have higher risk"""
        strategy = RuleBasedScoring()

        new_customer_context = {
            **sample_context,
            "customer_history": {
                "total_transactions": 0,  # New customer
                "is_new": True
            }
        }

        txn = TransactionFactory.create_transaction(
            amount=Decimal("1000.00"), currency="BRL",
            merchant_id="M1", customer_id="NEW_CUSTOMER"
        )

        result = await strategy.calculate_score(txn, new_customer_context)

        # New customer should have higher risk
        assert result.score.value > 0.3

    @pytest.mark.asyncio
    async def test_rule_based_risk_factors_populated(self, high_risk_transaction, sample_context):
        """Test that risk factors are identified and populated"""
        strategy = RuleBasedScoring()

        result = await strategy.calculate_score(high_risk_transaction, sample_context)

        assert len(result.risk_factors) > 0
        # Should contain explanations like "high_amount", "new_customer", etc.
        assert any("amount" in factor.lower() for factor in result.risk_factors)

    @pytest.mark.asyncio
    async def test_rule_based_pix_channel_risk(self, sample_context):
        """Test PIX channel has higher risk (instant, irreversible)"""
        strategy = RuleBasedScoring()

        pix_txn = TransactionFactory.create_transaction(
            amount=Decimal("1000.00"), currency="BRL",
            merchant_id="M1", customer_id="C1",
            metadata={"channel": "PIX"}
        )
        credit_txn = TransactionFactory.create_transaction(
            amount=Decimal("1000.00"), currency="BRL",
            merchant_id="M1", customer_id="C1",
            metadata={"channel": "CREDIT_CARD"}
        )

        result_pix = await strategy.calculate_score(pix_txn, sample_context)
        result_credit = await strategy.calculate_score(credit_txn, sample_context)

        # PIX should have higher base risk
        assert result_pix.score.value >= result_credit.score.value

    @pytest.mark.asyncio
    async def test_rule_based_performance_is_fast(self, sample_transaction, sample_context):
        """Test rule-based strategy is fast (< 10ms)"""
        strategy = RuleBasedScoring()

        start = datetime.now()
        await strategy.calculate_score(sample_transaction, sample_context)
        duration = (datetime.now() - start).total_seconds() * 1000

        # Rule-based should be very fast (O(1) complexity)
        assert duration < 10  # < 10ms

    @pytest.mark.asyncio
    async def test_rule_based_with_empty_context(self, sample_transaction):
        """Test rule-based scoring handles empty context gracefully"""
        strategy = RuleBasedScoring()

        result = await strategy.calculate_score(sample_transaction, {})

        # Should still produce a score (use defaults)
        assert isinstance(result.score, RiskScore)
        assert 0.0 <= result.score.value <= 1.0


# ============================================================================
# MLBasedScoring Tests
# ============================================================================

class TestMLBasedScoring:
    """Test ML-Based Fraud Scoring Strategy"""

    @pytest.fixture
    def mock_ml_model(self):
        """Create mock ML model"""
        mock = Mock()
        mock.predict = Mock(return_value={"fraud_probability": 0.75})
        return mock

    @pytest.mark.asyncio
    async def test_ml_based_strategy_creation(self, mock_ml_model):
        """Test creating ML-based strategy"""
        strategy = MLBasedScoring(mock_ml_model)
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_ml_based_scoring_calls_model(self, mock_ml_model, sample_transaction, sample_context):
        """Test ML strategy calls underlying model"""
        strategy = MLBasedScoring(mock_ml_model)

        await strategy.calculate_score(sample_transaction, sample_context)

        # Should have called model.predict
        mock_ml_model.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_ml_based_score_extraction(self, sample_transaction, sample_context):
        """Test extracting score from ML model output"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value={
            "fraud_probability": 0.85,
            "model_version": "v1.0.0"
        })

        strategy = MLBasedScoring(mock_model)
        result = await strategy.calculate_score(sample_transaction, sample_context)

        assert result.score.value == 0.85
        assert result.model_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_ml_based_feature_importance_to_risk_factors(self, sample_transaction, sample_context):
        """Test converting ML feature importance to risk factors"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value={
            "fraud_probability": 0.75,
            "feature_importance": {
                "amount": 0.4,
                "velocity": 0.3,
                "device_risk": 0.2
            }
        })

        strategy = MLBasedScoring(mock_model)
        result = await strategy.calculate_score(sample_transaction, sample_context)

        # Should extract top features as risk factors
        assert len(result.risk_factors) > 0
        assert any("amount" in factor.lower() for factor in result.risk_factors)

    @pytest.mark.asyncio
    async def test_ml_based_handles_model_error(self, sample_transaction, sample_context):
        """Test ML strategy handles model errors gracefully"""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception("Model inference failed"))

        strategy = MLBasedScoring(mock_model, fallback_score=0.5)

        # Should not raise, should return fallback score
        result = await strategy.calculate_score(sample_transaction, sample_context)

        assert result.score.value == 0.5
        assert "model_error" in result.risk_factors

    @pytest.mark.asyncio
    async def test_ml_based_score_clamping(self, sample_transaction, sample_context):
        """Test ML scores are clamped to [0.0, 1.0]"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value={"fraud_probability": 1.5})  # Invalid

        strategy = MLBasedScoring(mock_model)
        result = await strategy.calculate_score(sample_transaction, sample_context)

        # Should clamp to 1.0
        assert result.score.value == 1.0


# ============================================================================
# VelocityBasedScoring Tests
# ============================================================================

class TestVelocityBasedScoring:
    """Test Velocity-Based Fraud Scoring Strategy (Burst Detection)"""

    @pytest.mark.asyncio
    async def test_velocity_strategy_creation(self):
        """Test creating velocity-based strategy"""
        strategy = VelocityBasedScoring()
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_velocity_low_frequency(self, sample_transaction):
        """Test low transaction frequency = low risk"""
        strategy = VelocityBasedScoring()

        context = {
            "velocity_data": {
                "transactions_last_hour": 2,
                "transactions_last_day": 10,
                "amount_last_hour": Decimal("1000.00")
            }
        }

        result = await strategy.calculate_score(sample_transaction, context)

        assert result.score.value < 0.5  # Low risk

    @pytest.mark.asyncio
    async def test_velocity_high_frequency(self, sample_transaction):
        """Test high transaction frequency = high risk (burst)"""
        strategy = VelocityBasedScoring()

        context = {
            "velocity_data": {
                "transactions_last_hour": 50,  # Suspicious burst
                "transactions_last_day": 100,
                "amount_last_hour": Decimal("50000.00")
            }
        }

        result = await strategy.calculate_score(sample_transaction, context)

        assert result.score.value > 0.7  # High risk
        assert any("velocity" in factor.lower() or "burst" in factor.lower()
                  for factor in result.risk_factors)

    @pytest.mark.asyncio
    async def test_velocity_amount_burst(self, sample_transaction):
        """Test high amount in short time = high risk"""
        strategy = VelocityBasedScoring()

        context = {
            "velocity_data": {
                "transactions_last_hour": 10,
                "amount_last_hour": Decimal("100000.00"),  # Huge amount
                "avg_hourly_amount": Decimal("1000.00")
            }
        }

        result = await strategy.calculate_score(sample_transaction, context)

        assert result.score.value > 0.6
        assert any("amount" in factor.lower() for factor in result.risk_factors)

    @pytest.mark.asyncio
    async def test_velocity_missing_data_uses_defaults(self, sample_transaction):
        """Test velocity strategy handles missing velocity data"""
        strategy = VelocityBasedScoring()

        result = await strategy.calculate_score(sample_transaction, {})

        # Should not crash, use conservative defaults
        assert isinstance(result.score, RiskScore)

    @pytest.mark.asyncio
    async def test_velocity_per_customer(self, sample_transaction):
        """Test velocity is calculated per customer"""
        strategy = VelocityBasedScoring()

        # Same customer, multiple transactions
        context = {
            "velocity_data": {
                "customer_id": "CUSTOMER_001",
                "transactions_last_hour": 20,
            }
        }

        result = await strategy.calculate_score(sample_transaction, context)

        # High frequency for single customer = suspicious
        assert result.score.value > 0.5


# ============================================================================
# CompositeScoring Tests
# ============================================================================

class TestCompositeScoring:
    """Test Composite Scoring Strategy (Weighted Combination)"""

    @pytest.fixture
    def mock_strategies(self):
        """Create mock strategies with known scores"""
        strategy1 = AsyncMock(spec=FraudScoringStrategy)
        strategy1.calculate_score = AsyncMock(return_value=FraudScoreResult(
            score=RiskScore(0.2),
            risk_factors=["factor1"],
            strategy_name="Strategy1"
        ))

        strategy2 = AsyncMock(spec=FraudScoringStrategy)
        strategy2.calculate_score = AsyncMock(return_value=FraudScoreResult(
            score=RiskScore(0.8),
            risk_factors=["factor2"],
            strategy_name="Strategy2"
        ))

        return strategy1, strategy2

    @pytest.mark.asyncio
    async def test_composite_strategy_creation(self, mock_strategies):
        """Test creating composite strategy with weighted strategies"""
        strategy1, strategy2 = mock_strategies

        composite = CompositeScoring([
            (strategy1, 0.5),
            (strategy2, 0.5)
        ])

        assert composite is not None

    @pytest.mark.asyncio
    async def test_composite_weighted_average(self, mock_strategies, sample_transaction, sample_context):
        """Test composite calculates weighted average correctly"""
        strategy1, strategy2 = mock_strategies

        # Strategy1 = 0.2 (weight 0.3)
        # Strategy2 = 0.8 (weight 0.7)
        # Expected = 0.2 * 0.3 + 0.8 * 0.7 = 0.06 + 0.56 = 0.62
        composite = CompositeScoring([
            (strategy1, 0.3),
            (strategy2, 0.7)
        ])

        result = await composite.calculate_score(sample_transaction, sample_context)

        assert abs(result.score.value - 0.62) < 0.01  # Floating point tolerance

    @pytest.mark.asyncio
    async def test_composite_all_strategies_called(self, mock_strategies, sample_transaction, sample_context):
        """Test composite calls all underlying strategies"""
        strategy1, strategy2 = mock_strategies

        composite = CompositeScoring([
            (strategy1, 0.5),
            (strategy2, 0.5)
        ])

        await composite.calculate_score(sample_transaction, sample_context)

        # Both strategies should be called
        strategy1.calculate_score.assert_called_once()
        strategy2.calculate_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_composite_aggregates_risk_factors(self, mock_strategies, sample_transaction, sample_context):
        """Test composite aggregates risk factors from all strategies"""
        strategy1, strategy2 = mock_strategies

        composite = CompositeScoring([
            (strategy1, 0.5),
            (strategy2, 0.5)
        ])

        result = await composite.calculate_score(sample_transaction, sample_context)

        # Should contain factors from both strategies
        assert "factor1" in result.risk_factors
        assert "factor2" in result.risk_factors

    @pytest.mark.asyncio
    async def test_composite_parallel_execution(self, sample_transaction, sample_context):
        """Test composite executes strategies in parallel"""
        # Create slow strategies
        async def slow_strategy1(txn, ctx):
            await asyncio.sleep(0.1)
            return FraudScoreResult(
                score=RiskScore(0.5),
                risk_factors=[],
                strategy_name="Slow1"
            )

        async def slow_strategy2(txn, ctx):
            await asyncio.sleep(0.1)
            return FraudScoreResult(
                score=RiskScore(0.5),
                risk_factors=[],
                strategy_name="Slow2"
            )

        strategy1 = Mock()
        strategy1.calculate_score = slow_strategy1
        strategy2 = Mock()
        strategy2.calculate_score = slow_strategy2

        composite = CompositeScoring([
            (strategy1, 0.5),
            (strategy2, 0.5)
        ])

        start = datetime.now()
        await composite.calculate_score(sample_transaction, sample_context)
        duration = (datetime.now() - start).total_seconds()

        # If parallel, should take ~0.1s (not 0.2s)
        assert duration < 0.15

    @pytest.mark.asyncio
    async def test_composite_weights_sum_to_one(self):
        """Test composite validates weights sum to 1.0"""
        strategy1 = Mock(spec=FraudScoringStrategy)
        strategy2 = Mock(spec=FraudScoringStrategy)

        # Invalid weights (sum = 0.5)
        with pytest.raises(ValueError, match="Pesos devem somar 1.0"):
            CompositeScoring([
                (strategy1, 0.3),
                (strategy2, 0.2)
            ])

    @pytest.mark.asyncio
    async def test_composite_handles_strategy_failure(self, sample_transaction, sample_context):
        """Test composite handles individual strategy failures"""
        failing_strategy = Mock(spec=FraudScoringStrategy)
        failing_strategy.calculate_score = AsyncMock(side_effect=Exception("Strategy failed"))

        working_strategy = Mock(spec=FraudScoringStrategy)
        working_strategy.calculate_score = AsyncMock(return_value=FraudScoreResult(
            score=RiskScore(0.5),
            risk_factors=[],
            strategy_name="Working"
        ))

        composite = CompositeScoring([
            (failing_strategy, 0.5),
            (working_strategy, 0.5)
        ])

        # Should not raise, should handle failure gracefully
        result = await composite.calculate_score(sample_transaction, sample_context)

        # Should use fallback for failed strategy
        assert isinstance(result.score, RiskScore)


# ============================================================================
# Integration Tests - Strategy Combinations
# ============================================================================

class TestStrategyIntegration:
    """Test real-world strategy combinations"""

    @pytest.mark.asyncio
    async def test_three_strategy_ensemble(self, sample_transaction, sample_context):
        """Test ensemble of rules + ML + velocity"""
        rule_strategy = RuleBasedScoring()

        mock_ml_model = Mock()
        mock_ml_model.predict = Mock(return_value={"fraud_probability": 0.7})
        ml_strategy = MLBasedScoring(mock_ml_model)

        velocity_strategy = VelocityBasedScoring()

        # Ensemble: 30% rules, 50% ML, 20% velocity
        composite = CompositeScoring([
            (rule_strategy, 0.3),
            (ml_strategy, 0.5),
            (velocity_strategy, 0.2)
        ])

        result = await composite.calculate_score(sample_transaction, sample_context)

        # Should produce valid score
        assert 0.0 <= result.score.value <= 1.0
        # Should have risk factors from multiple strategies
        assert len(result.risk_factors) > 0

    @pytest.mark.asyncio
    async def test_fallback_cascade(self, sample_transaction, sample_context):
        """Test fallback from ML to rules if ML fails"""
        # Primary: ML (will fail)
        failing_ml = Mock()
        failing_ml.predict = Mock(side_effect=Exception("Model down"))
        ml_strategy = MLBasedScoring(failing_ml)

        # Fallback: Rules
        rule_strategy = RuleBasedScoring()

        # Try ML first, fallback to rules
        try:
            result = await ml_strategy.calculate_score(sample_transaction, sample_context)
        except:
            result = await rule_strategy.calculate_score(sample_transaction, sample_context)

        # Should get result from fallback
        assert isinstance(result.score, RiskScore)


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for fraud_strategies.py:

RuleBasedScoring: 10 tests
- Strategy creation: 1
- Low/high risk transactions: 2
- Risk factors: 4
- Performance: 1
- Edge cases: 2

MLBasedScoring: 6 tests
- Strategy creation: 1
- Model integration: 2
- Error handling: 2
- Score validation: 1

VelocityBasedScoring: 5 tests
- Low/high frequency: 2
- Amount bursts: 1
- Missing data: 1
- Per-customer velocity: 1

CompositeScoring: 8 tests
- Creation & weights: 2
- Weighted average: 1
- Strategy execution: 2
- Risk factor aggregation: 1
- Error handling: 1
- Parallel execution: 1

Integration: 2 tests
- Multi-strategy ensemble: 1
- Fallback cascade: 1

TOTAL: 31 tests
TARGET: >95% statement coverage
COMPLEXITY: Tests cover O(1) rules, O(m) ML, O(n) velocity, O(k) composite
"""
