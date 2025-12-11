"""
Fraud Detection Strategies - Strategy Pattern
Different fraud scoring algorithms implemented as interchangeable strategies

This demonstrates the Strategy Pattern - one of the most important GoF patterns
for systems with multiple algorithms.

Benefits:
- Open/Closed Principle: Add new strategies without modifying existing code
- Easy A/B testing: Switch strategies at runtime
- Testable: Each strategy can be tested independently
- Composable: Combine multiple strategies with different weights

Time Complexity: Documented per strategy
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta

from .entities import Transaction, RiskLevel
from .value_objects import CPF, Amount, RiskScore, TimeWindow


@dataclass
class FraudScoreResult:
    """
    Result of fraud scoring

    Contains score, confidence, and explanation
    """

    score: RiskScore
    confidence: float  # 0.0 to 1.0
    risk_factors: List[str]
    strategy_name: str
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FraudScoringStrategy(ABC):
    """
    Abstract base class for fraud scoring strategies

    Implements Strategy Pattern - defines interface for all fraud detection algorithms

    Each concrete strategy must:
    1. Implement calculate_score()
    2. Provide strategy_name
    3. Document time complexity
    """

    @abstractmethod
    async def calculate_score(self, transaction: Transaction, context: Dict[str, Any]) -> FraudScoreResult:
        """
        Calculate fraud score for transaction

        Args:
            transaction: Transaction entity to analyze
            context: Additional context (customer history, device info, etc.)

        Returns:
            FraudScoreResult with score and explanation

        Time Complexity: Varies by strategy
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name for logging/metrics"""
        pass


class RuleBasedScoring(FraudScoringStrategy):
    """
    Rule-based fraud scoring using hard-coded business rules

    Pros:
    - Fast (O(1))
    - Explainable
    - No training required

    Cons:
    - Manual rule maintenance
    - Can't adapt to new patterns
    - High false positive rate

    Time Complexity: O(1) - all rules are simple comparisons
    """

    def __init__(self, rules_config: Optional[Dict[str, Any]] = None):
        self._config = rules_config or self._default_config()

    @property
    def strategy_name(self) -> str:
        return "rule_based_scoring"

    async def calculate_score(self, transaction: Transaction, context: Dict[str, Any]) -> FraudScoreResult:
        """
        Calculate score using rule-based logic

        Time Complexity: O(1) - fixed number of rule checks
        """
        import time
        start_time = time.time()

        score = 0.0
        risk_factors = []

        # Rule 1: High value transactions - O(1)
        if transaction.is_high_value(Amount(Decimal("5000"))):
            score += 0.3
            risk_factors.append("high_value_transaction")

        # Rule 2: Unusual hour (late night) - O(1)
        hour = transaction.timestamp.hour
        if hour < 6 or hour > 23:
            score += 0.2
            risk_factors.append("unusual_hour")

        # Rule 3: New customer with high value - O(1)
        customer_txn_count = context.get('customer_transaction_count', 0)
        if customer_txn_count < 5 and transaction.is_high_value(Amount(Decimal("1000"))):
            score += 0.25
            risk_factors.append("new_customer_high_value")

        # Rule 4: Rapid transactions (velocity check) - O(1)
        recent_txn_count = context.get('recent_transaction_count_5min', 0)
        if recent_txn_count > 3:
            score += 0.3
            risk_factors.append("high_velocity")

        # Rule 5: Different device than usual - O(1)
        is_known_device = context.get('is_known_device', True)
        if not is_known_device:
            score += 0.15
            risk_factors.append("unknown_device")

        # Normalize score to [0, 1]
        score = min(score, 1.0)

        processing_time = (time.time() - start_time) * 1000

        return FraudScoreResult(
            score=RiskScore(score),
            confidence=0.7,  # Rule-based has medium confidence
            risk_factors=risk_factors,
            strategy_name=self.strategy_name,
            processing_time_ms=processing_time,
            metadata={'rules_triggered': len(risk_factors)}
        )

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            'high_value_threshold': 5000,
            'velocity_window_minutes': 5,
            'velocity_threshold': 3
        }


class MLBasedScoring(FraudScoringStrategy):
    """
    ML-based fraud scoring using trained model

    Pros:
    - Adapts to patterns
    - Higher accuracy
    - Lower false positives

    Cons:
    - Requires training data
    - Less explainable
    - Slower than rules

    Time Complexity: O(f) where f is feature extraction + model inference time
    Typically O(1) for pre-extracted features with simple models
    """

    def __init__(self, model_gateway):
        """
        Initialize with ML model gateway

        Args:
            model_gateway: Gateway to ML model (Dependency Inversion)
        """
        self._model = model_gateway

    @property
    def strategy_name(self) -> str:
        return "ml_based_scoring"

    async def calculate_score(self, transaction: Transaction, context: Dict[str, Any]) -> FraudScoreResult:
        """
        Calculate score using ML model

        Time Complexity: O(f + m) where:
        - f = feature extraction time (typically O(1) for precomputed)
        - m = model inference time (O(1) for linear/tree models, O(n) for neural nets)
        """
        import time
        start_time = time.time()

        # Extract features - O(1) typically
        features = self._extract_features(transaction, context)

        # Model prediction - O(1) for most models in production
        prediction = await self._model.predict(features)

        # Extract risk factors from model (if available)
        risk_factors = self._extract_risk_factors(prediction, features)

        processing_time = (time.time() - start_time) * 1000

        return FraudScoreResult(
            score=RiskScore(prediction['fraud_probability']),
            confidence=prediction.get('confidence', 0.85),
            risk_factors=risk_factors,
            strategy_name=self.strategy_name,
            processing_time_ms=processing_time,
            metadata={
                'model_version': prediction.get('model_version', 'unknown'),
                'feature_count': len(features)
            }
        )

    def _extract_features(self, transaction: Transaction, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features for ML model - O(1)

        In production, many features should be precomputed
        """
        return {
            'amount': float(transaction.amount.amount),
            'hour': float(transaction.timestamp.hour),
            'customer_txn_count': float(context.get('customer_transaction_count', 0)),
            'recent_txn_count': float(context.get('recent_transaction_count_5min', 0)),
            'is_known_device': 1.0 if context.get('is_known_device', True) else 0.0,
            'customer_age_days': float(context.get('customer_age_days', 0)),
        }

    def _extract_risk_factors(self, prediction: Dict[str, Any], features: Dict[str, float]) -> List[str]:
        """
        Extract risk factors from model prediction - O(n) where n is features

        Uses SHAP values or feature importance if available
        """
        risk_factors = []

        # If model provides feature importance
        feature_importance = prediction.get('feature_importance', {})

        for feature, importance in sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]:  # Top 5 features
            risk_factors.append(f"{feature}_impact_{importance:.2f}")

        return risk_factors


class VelocityBasedScoring(FraudScoringStrategy):
    """
    Velocity-based fraud detection

    Detects rapid sequences of transactions (velocity attacks)

    Pros:
    - Catches burst patterns
    - Simple to implement
    - Real-time detection

    Cons:
    - Needs transaction history
    - Can have false positives for legitimate bulk operations

    Time Complexity: O(log n + k) where:
    - log n = query for transactions in time window
    - k = number of transactions in window
    """

    def __init__(self, velocity_rules: Optional[Dict[str, Any]] = None):
        self._rules = velocity_rules or self._default_rules()

    @property
    def strategy_name(self) -> str:
        return "velocity_based_scoring"

    async def calculate_score(self, transaction: Transaction, context: Dict[str, Any]) -> FraudScoreResult:
        """
        Calculate score based on transaction velocity

        Time Complexity: O(1) if counts are precomputed, O(log n + k) otherwise
        """
        import time
        start_time = time.time()

        score = 0.0
        risk_factors = []

        # Check 5-minute velocity - O(1) if precomputed
        count_5min = context.get('recent_transaction_count_5min', 0)
        if count_5min > self._rules['threshold_5min']:
            score += 0.4
            risk_factors.append(f"high_velocity_5min_{count_5min}_txns")

        # Check 1-hour velocity - O(1) if precomputed
        count_1hr = context.get('recent_transaction_count_1hr', 0)
        if count_1hr > self._rules['threshold_1hr']:
            score += 0.3
            risk_factors.append(f"high_velocity_1hr_{count_1hr}_txns")

        # Check total amount in window - O(1) if precomputed
        amount_1hr = context.get('recent_amount_1hr', 0.0)
        if amount_1hr > self._rules['amount_threshold_1hr']:
            score += 0.3
            risk_factors.append(f"high_amount_velocity_R${amount_1hr:.2f}")

        # Normalize
        score = min(score, 1.0)

        processing_time = (time.time() - start_time) * 1000

        return FraudScoreResult(
            score=RiskScore(score),
            confidence=0.8,
            risk_factors=risk_factors,
            strategy_name=self.strategy_name,
            processing_time_ms=processing_time,
            metadata={
                'velocity_5min': count_5min,
                'velocity_1hr': count_1hr,
                'amount_1hr': amount_1hr
            }
        )

    @staticmethod
    def _default_rules() -> Dict[str, Any]:
        return {
            'threshold_5min': 3,
            'threshold_1hr': 10,
            'amount_threshold_1hr': 50000.0
        }


class CompositeScoring(FraudScoringStrategy):
    """
    Composite strategy combining multiple strategies with weights

    Implements Composite Pattern + Strategy Pattern

    This is the most powerful approach:
    - Combines strengths of multiple strategies
    - Weighted voting reduces false positives/negatives
    - Can be tuned for precision vs recall

    Time Complexity: O(n * s) where:
    - n = number of strategies
    - s = time complexity of each strategy
    """

    def __init__(self, strategies: List[tuple[FraudScoringStrategy, float]]):
        """
        Initialize with weighted strategies

        Args:
            strategies: List of (strategy, weight) tuples
                       Weights should sum to 1.0

        Example:
            CompositeScoring([
                (RuleBasedScoring(), 0.3),
                (MLBasedScoring(model), 0.5),
                (VelocityBasedScoring(), 0.2)
            ])
        """
        self._strategies = strategies

        # Validate weights
        total_weight = sum(weight for _, weight in strategies)
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    @property
    def strategy_name(self) -> str:
        return "composite_scoring"

    async def calculate_score(self, transaction: Transaction, context: Dict[str, Any]) -> FraudScoreResult:
        """
        Calculate weighted average of all strategies

        Time Complexity: O(n * s) where n is strategies, s is avg strategy time
        In practice, with asyncio, this can be O(max(s)) if run in parallel
        """
        import time
        import asyncio
        start_time = time.time()

        # Run all strategies in parallel - O(max(s)) instead of O(sum(s))
        tasks = [
            strategy.calculate_score(transaction, context)
            for strategy, _ in self._strategies
        ]
        results = await asyncio.gather(*tasks)

        # Weighted average - O(n)
        weighted_score = 0.0
        all_risk_factors = []
        strategy_details = []

        for (strategy, weight), result in zip(self._strategies, results):
            weighted_score += float(result.score.value) * weight
            all_risk_factors.extend(result.risk_factors)
            strategy_details.append({
                'strategy': result.strategy_name,
                'score': result.score.value,
                'weight': weight,
                'contribution': result.score.value * weight
            })

        # Deduplicate risk factors
        unique_risk_factors = list(dict.fromkeys(all_risk_factors))

        processing_time = (time.time() - start_time) * 1000

        return FraudScoreResult(
            score=RiskScore(weighted_score),
            confidence=0.9,  # Composite has high confidence
            risk_factors=unique_risk_factors,
            strategy_name=self.strategy_name,
            processing_time_ms=processing_time,
            metadata={
                'strategy_count': len(self._strategies),
                'strategy_details': strategy_details
            }
        )


# Convenience factory functions
def create_default_scoring_strategy(model_gateway) -> FraudScoringStrategy:
    """
    Factory: Create default composite strategy for production

    Combines:
    - 30% Rule-based (fast, explainable)
    - 50% ML-based (accurate)
    - 20% Velocity-based (catches bursts)

    Time Complexity: O(1) - just creates objects
    """
    return CompositeScoring([
        (RuleBasedScoring(), 0.3),
        (MLBasedScoring(model_gateway), 0.5),
        (VelocityBasedScoring(), 0.2)
    ])


def create_conservative_strategy(model_gateway) -> FraudScoringStrategy:
    """
    Factory: Create conservative strategy (higher recall, more false positives)

    Emphasizes catching fraud over convenience

    Time Complexity: O(1)
    """
    return CompositeScoring([
        (RuleBasedScoring(), 0.4),
        (MLBasedScoring(model_gateway), 0.4),
        (VelocityBasedScoring(), 0.2)
    ])


def create_aggressive_strategy(model_gateway) -> FraudScoringStrategy:
    """
    Factory: Create aggressive strategy (higher precision, fewer false positives)

    Emphasizes user experience over fraud detection

    Time Complexity: O(1)
    """
    return CompositeScoring([
        (RuleBasedScoring(), 0.2),
        (MLBasedScoring(model_gateway), 0.7),
        (VelocityBasedScoring(), 0.1)
    ])
