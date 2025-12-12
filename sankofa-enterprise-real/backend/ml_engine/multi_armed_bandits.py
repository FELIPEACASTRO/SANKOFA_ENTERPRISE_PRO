"""
Multi-Armed Bandits - Optimize step-up MFA decisions
Thompson Sampling for adaptive challenge selection
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    """Types of authentication challenges"""
    NONE = "none"                          # No challenge
    SMS_OTP = "sms_otp"                    # SMS one-time password
    PUSH_NOTIFICATION = "push_notification"  # Push approve/deny
    BIOMETRIC = "biometric"                # Fingerprint/Face ID
    SECURITY_QUESTIONS = "security_questions"  # Knowledge-based
    EMAIL_OTP = "email_otp"                # Email verification
    TOTP = "totp"                          # Authenticator app


@dataclass
class BanditArm:
    """
    Represents one "arm" of the bandit (one challenge type)

    Uses Beta distribution for Thompson Sampling:
    - Alpha: successes (legitimate users who completed challenge)
    - Beta: failures (legitimate users who abandoned + fraudsters)
    """
    challenge_type: ChallengeType
    alpha: float = 1.0  # Prior: 1 success
    beta: float = 1.0   # Prior: 1 failure

    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)"""
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """
        Update distribution based on reward

        Args:
            reward: 1.0 for success, 0.0 for failure
        """
        if reward > 0:
            self.alpha += reward
        else:
            self.beta += (1.0 - reward)

    def get_expected_value(self) -> float:
        """Get expected value (mean of Beta distribution)"""
        return self.alpha / (self.alpha + self.beta)

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for conversion rate"""
        # Simplified: use quantiles of Beta distribution
        lower = np.percentile(np.random.beta(self.alpha, self.beta, 10000), (1 - confidence) / 2 * 100)
        upper = np.percentile(np.random.beta(self.alpha, self.beta, 10000), (1 + confidence) / 2 * 100)
        return (lower, upper)


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit for MFA challenge selection

    Context features:
    - Risk score (0-1)
    - Amount bucket (low/medium/high)
    - Time of day (business hours / off-hours)
    - Channel (PIX, credit card, etc.)
    - Device trust (trusted / new / suspicious)

    Goal: Maximize legitimate transaction approval while minimizing fraud
    """

    def __init__(
        self,
        challenge_types: List[ChallengeType],
        context_keys: Optional[List[str]] = None,
        exploration_rate: float = 0.1
    ):
        """
        Args:
            challenge_types: Available challenge types
            context_keys: Context features for segmentation
            exploration_rate: Epsilon for exploration (0-1)
        """
        self.challenge_types = challenge_types
        self.context_keys = context_keys or ['risk_level', 'amount_bucket', 'time_period']
        self.exploration_rate = exploration_rate

        # Arms per context (nested dict)
        # Structure: {context_hash: {challenge_type: BanditArm}}
        self.arms: Dict[str, Dict[ChallengeType, BanditArm]] = defaultdict(
            lambda: {ct: BanditArm(ct) for ct in challenge_types}
        )

        # Metrics
        self.total_selections = 0
        self.total_rewards = 0.0

        logger.info(
            f"Contextual Bandit initialized: "
            f"challenges={[ct.value for ct in challenge_types]}, "
            f"exploration_rate={exploration_rate}"
        )

    def _extract_context(self, transaction: Dict[str, Any]) -> str:
        """
        Extract context from transaction and hash it

        Args:
            transaction: Transaction data

        Returns:
            Context hash
        """
        context = {}

        # Risk level bucket
        risk_score = transaction.get('risk_score', 0.5)
        if risk_score < 0.3:
            context['risk_level'] = 'low'
        elif risk_score < 0.7:
            context['risk_level'] = 'medium'
        else:
            context['risk_level'] = 'high'

        # Amount bucket
        amount = transaction.get('amount', 0)
        if amount < 100:
            context['amount_bucket'] = 'low'
        elif amount < 1000:
            context['amount_bucket'] = 'medium'
        else:
            context['amount_bucket'] = 'high'

        # Time period
        hour = datetime.now(timezone.utc).hour
        if 9 <= hour < 18:
            context['time_period'] = 'business_hours'
        elif 18 <= hour < 23:
            context['time_period'] = 'evening'
        else:
            context['time_period'] = 'night'

        # Hash context
        context_hash = json.dumps(context, sort_keys=True)

        return context_hash

    async def select_challenge(
        self,
        transaction: Dict[str, Any],
        eligible_challenges: Optional[List[ChallengeType]] = None
    ) -> ChallengeType:
        """
        Select optimal challenge using Thompson Sampling

        Args:
            transaction: Transaction data
            eligible_challenges: Subset of eligible challenges (optional)

        Returns:
            Selected challenge type
        """
        # Get context
        context = self._extract_context(transaction)

        # Get arms for this context
        context_arms = self.arms[context]

        # Filter to eligible challenges
        if eligible_challenges:
            available_arms = {
                ct: arm for ct, arm in context_arms.items()
                if ct in eligible_challenges
            }
        else:
            available_arms = context_arms

        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            # Explore: random challenge
            selected = np.random.choice(list(available_arms.keys()))
            logger.debug(f"Exploration: selected {selected.value}")
        else:
            # Exploit: Thompson Sampling
            samples = {ct: arm.sample() for ct, arm in available_arms.items()}
            selected = max(samples, key=samples.get)
            logger.debug(f"Exploitation: selected {selected.value} (sample={samples[selected]:.3f})")

        self.total_selections += 1

        return selected

    async def update_reward(
        self,
        transaction: Dict[str, Any],
        challenge_type: ChallengeType,
        outcome: str
    ) -> None:
        """
        Update bandit based on challenge outcome

        Args:
            transaction: Transaction data
            challenge_type: Challenge that was used
            outcome: 'approved', 'abandoned', 'fraud_blocked'
        """
        # Get context
        context = self._extract_context(transaction)

        # Calculate reward
        if outcome == 'approved':
            # Legitimate user completed challenge successfully
            reward = 1.0
        elif outcome == 'abandoned':
            # User abandoned (friction too high)
            reward = 0.0
        elif outcome == 'fraud_blocked':
            # Fraud blocked by challenge (good!)
            reward = 0.8
        else:
            reward = 0.0

        # Update arm
        arm = self.arms[context][challenge_type]
        arm.update(reward)

        self.total_rewards += reward

        logger.debug(
            f"Reward update: challenge={challenge_type.value}, "
            f"outcome={outcome}, reward={reward}, "
            f"arm_stats=(alpha={arm.alpha:.1f}, beta={arm.beta:.1f})"
        )

    def get_arm_stats(self, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for all arms

        Args:
            context: Specific context to get stats for (optional)

        Returns:
            Arm statistics
        """
        if context:
            # Stats for specific context
            arms = self.arms.get(context, {})
            return {
                ct.value: {
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'expected_value': arm.get_expected_value(),
                    'confidence_interval': arm.get_confidence_interval()
                }
                for ct, arm in arms.items()
            }
        else:
            # Aggregate stats across all contexts
            stats = {}
            for context_hash, arms in self.arms.items():
                for ct, arm in arms.items():
                    if ct.value not in stats:
                        stats[ct.value] = {
                            'total_alpha': 0.0,
                            'total_beta': 0.0,
                            'contexts': 0
                        }
                    stats[ct.value]['total_alpha'] += arm.alpha
                    stats[ct.value]['total_beta'] += arm.beta
                    stats[ct.value]['contexts'] += 1

            # Calculate aggregates
            for ct_value, data in stats.items():
                avg_alpha = data['total_alpha'] / data['contexts']
                avg_beta = data['total_beta'] / data['contexts']
                stats[ct_value]['avg_expected_value'] = avg_alpha / (avg_alpha + avg_beta)

            return stats

    def get_metrics(self) -> Dict[str, Any]:
        """Get overall bandit metrics"""
        avg_reward = (
            self.total_rewards / self.total_selections
            if self.total_selections > 0
            else 0.0
        )

        return {
            'total_selections': self.total_selections,
            'total_rewards': self.total_rewards,
            'avg_reward': avg_reward,
            'exploration_rate': self.exploration_rate,
            'contexts_learned': len(self.arms)
        }


class StepUpMFAOptimizer:
    """
    Step-up MFA optimizer using Multi-Armed Bandits

    Decides:
    1. Whether to challenge (based on risk score)
    2. Which challenge type to use (bandit optimization)
    3. Dynamic threshold adjustment

    Goals:
    - Maximize legitimate approval rate
    - Minimize fraud loss
    - Minimize user friction
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        high_risk_threshold: float = 0.8,
        friction_weight: float = 0.3
    ):
        """
        Args:
            base_threshold: Risk score threshold for MFA
            high_risk_threshold: Risk score for mandatory strong MFA
            friction_weight: Weight for friction in reward calculation
        """
        self.base_threshold = base_threshold
        self.high_risk_threshold = high_risk_threshold
        self.friction_weight = friction_weight

        # Initialize bandit
        self.bandit = ContextualBandit(
            challenge_types=[
                ChallengeType.NONE,
                ChallengeType.SMS_OTP,
                ChallengeType.PUSH_NOTIFICATION,
                ChallengeType.BIOMETRIC,
                ChallengeType.SECURITY_QUESTIONS
            ],
            exploration_rate=0.1
        )

        # Dynamic threshold tracking
        self.threshold_adjustments: List[Tuple[datetime, float]] = []

        logger.info(
            f"Step-up MFA Optimizer initialized: "
            f"base_threshold={base_threshold}, high_risk_threshold={high_risk_threshold}"
        )

    async def decide_challenge(
        self,
        transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decide if and how to challenge transaction

        Args:
            transaction: Transaction data

        Returns:
            Challenge decision
        """
        risk_score = transaction.get('risk_score', 0.5)

        # No challenge for low risk
        if risk_score < self.base_threshold:
            return {
                'require_challenge': False,
                'challenge_type': ChallengeType.NONE,
                'risk_score': risk_score,
                'reason': 'low_risk'
            }

        # Mandatory strong challenge for very high risk
        if risk_score >= self.high_risk_threshold:
            # Select strong challenge
            eligible = [
                ChallengeType.BIOMETRIC,
                ChallengeType.PUSH_NOTIFICATION,
                ChallengeType.TOTP
            ]
            challenge = await self.bandit.select_challenge(transaction, eligible)

            return {
                'require_challenge': True,
                'challenge_type': challenge,
                'risk_score': risk_score,
                'reason': 'high_risk'
            }

        # Medium risk: bandit decides
        challenge = await self.bandit.select_challenge(transaction)

        return {
            'require_challenge': challenge != ChallengeType.NONE,
            'challenge_type': challenge,
            'risk_score': risk_score,
            'reason': 'bandit_optimized'
        }

    async def record_outcome(
        self,
        transaction: Dict[str, Any],
        challenge_type: ChallengeType,
        outcome: str,
        is_fraud: bool = False
    ) -> None:
        """
        Record challenge outcome for learning

        Args:
            transaction: Transaction data
            challenge_type: Challenge used
            outcome: 'approved', 'abandoned', 'fraud_blocked'
            is_fraud: Whether transaction was actually fraud
        """
        # Update bandit
        await self.bandit.update_reward(transaction, challenge_type, outcome)

        # Adjust threshold if needed
        await self._adjust_threshold(outcome, is_fraud)

    async def _adjust_threshold(self, outcome: str, is_fraud: bool) -> None:
        """
        Dynamically adjust base threshold based on outcomes

        Args:
            outcome: Challenge outcome
            is_fraud: Fraud indicator
        """
        # Simplified threshold adjustment
        # In production: use more sophisticated adaptive algorithms

        if outcome == 'abandoned' and not is_fraud:
            # Too much friction for legitimate users - lower threshold
            adjustment = -0.01
        elif is_fraud and outcome == 'approved':
            # Fraud passed through - raise threshold
            adjustment = +0.02
        else:
            adjustment = 0.0

        if adjustment != 0.0:
            self.base_threshold = max(0.3, min(0.9, self.base_threshold + adjustment))
            self.threshold_adjustments.append((datetime.now(timezone.utc), self.base_threshold))

            logger.info(f"Threshold adjusted: new_threshold={self.base_threshold:.3f}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get MFA optimization performance metrics"""
        bandit_metrics = self.bandit.get_metrics()
        arm_stats = self.bandit.get_arm_stats()

        return {
            'current_threshold': self.base_threshold,
            'high_risk_threshold': self.high_risk_threshold,
            'bandit_metrics': bandit_metrics,
            'arm_stats': arm_stats,
            'threshold_adjustments': len(self.threshold_adjustments)
        }


# Singleton instance
_mfa_optimizer_instance: Optional[StepUpMFAOptimizer] = None


def get_mfa_optimizer() -> StepUpMFAOptimizer:
    """
    Get singleton MFA optimizer

    Returns:
        StepUpMFAOptimizer instance
    """
    global _mfa_optimizer_instance

    if _mfa_optimizer_instance is None:
        _mfa_optimizer_instance = StepUpMFAOptimizer(
            base_threshold=0.5,
            high_risk_threshold=0.8,
            friction_weight=0.3
        )

    return _mfa_optimizer_instance


# Example usage
async def example_mfa_optimization():
    """Example: Step-up MFA optimization"""

    optimizer = get_mfa_optimizer()

    # Simulate transactions
    transactions = [
        {'id': 'TXN_001', 'amount': 500, 'risk_score': 0.3},    # Low risk
        {'id': 'TXN_002', 'amount': 2000, 'risk_score': 0.6},   # Medium risk
        {'id': 'TXN_003', 'amount': 10000, 'risk_score': 0.9},  # High risk
    ]

    for txn in transactions:
        # Decide challenge
        decision = await optimizer.decide_challenge(txn)

        print(f"\nTransaction {txn['id']}:")
        print(f"  Risk Score: {txn['risk_score']}")
        print(f"  Challenge: {decision['challenge_type'].value}")
        print(f"  Reason: {decision['reason']}")

        # Simulate outcome
        if decision['require_challenge']:
            # Simulate: high risk more likely fraud
            is_fraud = txn['risk_score'] > 0.7
            if is_fraud:
                outcome = 'fraud_blocked'
            else:
                outcome = 'approved' if np.random.random() > 0.1 else 'abandoned'

            await optimizer.record_outcome(txn, decision['challenge_type'], outcome, is_fraud)

    # Get metrics
    metrics = optimizer.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_mfa_optimization())
