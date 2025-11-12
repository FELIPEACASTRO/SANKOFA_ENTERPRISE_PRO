"""
Core Entities - Clean Architecture
Domain entities representing business objects with business rules
Time Complexity: O(1) for all entity operations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class TransactionStatus(Enum):
    """Transaction status enumeration"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class RiskLevel(Enum):
    """Risk level enumeration"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)  # Immutable entity
class TransactionId:
    """Value Object for Transaction ID - DDD Pattern"""

    value: str

    def __post_init__(self):
        if not self.value or len(self.value) < 10:
            raise ValueError("Transaction ID must be at least 10 characters")


@dataclass(frozen=True)
class Money:
    """Value Object for Money - DDD Pattern
    Time Complexity: O(1) for all operations
    """

    amount: Decimal
    currency: str = "BRL"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be 3 characters")

    def add(self, other: "Money") -> "Money":
        """Add two Money objects - O(1)"""
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def is_greater_than(self, other: "Money") -> bool:
        """Compare Money objects - O(1)"""
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount > other.amount


@dataclass
class Transaction:
    """
    Core Transaction Entity - Clean Architecture
    Contains business rules and invariants
    Time Complexity: O(1) for all operations
    """

    id: TransactionId
    amount: Money
    merchant_id: str
    customer_id: str
    timestamp: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate business invariants - O(1)"""
        if self.risk_score < 0 or self.risk_score > 1:
            raise ValueError("Risk score must be between 0 and 1")
        if not self.merchant_id:
            raise ValueError("Merchant ID is required")
        if not self.customer_id:
            raise ValueError("Customer ID is required")

    def mark_as_fraud(self, reason: str) -> None:
        """Business rule: Mark transaction as fraudulent - O(1)"""
        self.status = TransactionStatus.REJECTED
        self.risk_level = RiskLevel.CRITICAL
        self.metadata["fraud_reason"] = reason
        self.metadata["flagged_at"] = datetime.utcnow().isoformat()

    def approve(self) -> None:
        """Business rule: Approve transaction - O(1)"""
        if self.risk_level == RiskLevel.CRITICAL:
            raise ValueError("Cannot approve critical risk transactions")
        self.status = TransactionStatus.APPROVED
        self.metadata["approved_at"] = datetime.utcnow().isoformat()

    def is_high_value(self, threshold: Money) -> bool:
        """Business rule: Check if transaction is high value - O(1)"""
        return self.amount.is_greater_than(threshold)

    def requires_manual_review(self) -> bool:
        """Business rule: Check if manual review is required - O(1)"""
        return (
            self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            or self.risk_score > 0.7
            or self.is_high_value(Money(Decimal("10000")))
        )


@dataclass
class FraudAnalysisResult:
    """
    Entity representing fraud analysis result
    Time Complexity: O(1) for all operations
    """

    transaction_id: TransactionId
    is_fraud: bool
    confidence_score: float
    risk_factors: List[str] = field(default_factory=list)
    model_version: str = "1.0"
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0

    def __post_init__(self):
        """Validate analysis result - O(1)"""
        if self.confidence_score < 0 or self.confidence_score > 1:
            raise ValueError("Confidence score must be between 0 and 1")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")

    def add_risk_factor(self, factor: str) -> None:
        """Add risk factor - O(1) amortized"""
        if factor not in self.risk_factors:
            self.risk_factors.append(factor)

    def get_risk_level(self) -> RiskLevel:
        """Determine risk level based on confidence - O(1)"""
        if self.confidence_score >= 0.9:
            return RiskLevel.CRITICAL
        elif self.confidence_score >= 0.7:
            return RiskLevel.HIGH
        elif self.confidence_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class Customer:
    """
    Customer Entity with behavioral patterns
    Time Complexity: O(1) for most operations
    """

    id: str
    email: str
    created_at: datetime
    risk_profile: RiskLevel = RiskLevel.LOW
    transaction_history: List[TransactionId] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate customer data - O(1)"""
        if not self.id:
            raise ValueError("Customer ID is required")
        if "@" not in self.email:
            raise ValueError("Invalid email format")

    def add_transaction(self, transaction_id: TransactionId) -> None:
        """Add transaction to history - O(1) amortized"""
        self.transaction_history.append(transaction_id)

    def get_transaction_count(self) -> int:
        """Get total transaction count - O(1)"""
        return len(self.transaction_history)

    def update_risk_profile(self, new_risk: RiskLevel) -> None:
        """Update customer risk profile - O(1)"""
        self.risk_profile = new_risk
        self.metadata["risk_updated_at"] = datetime.utcnow().isoformat()


# Domain Events - Event Sourcing Pattern
class DomainEvent(ABC):
    """Base class for domain events"""

    def __init__(self):
        self.event_id = uuid4()
        self.occurred_at = datetime.utcnow()

    @abstractmethod
    def event_type(self) -> str:
        pass


@dataclass
class TransactionCreated(DomainEvent):
    """Domain event: Transaction was created"""

    transaction_id: TransactionId
    amount: Money
    customer_id: str

    def event_type(self) -> str:
        return "transaction_created"


@dataclass
class FraudDetected(DomainEvent):
    """Domain event: Fraud was detected"""

    transaction_id: TransactionId
    confidence_score: float
    risk_factors: List[str]

    def event_type(self) -> str:
        return "fraud_detected"


@dataclass
class TransactionApproved(DomainEvent):
    """Domain event: Transaction was approved"""

    transaction_id: TransactionId
    approved_by: str

    def event_type(self) -> str:
        return "transaction_approved"


# Aggregate Root - DDD Pattern
class TransactionAggregate:
    """
    Transaction Aggregate Root - DDD Pattern
    Ensures consistency and encapsulates business rules
    Time Complexity: O(1) for most operations
    """

    def __init__(self, transaction: Transaction):
        self._transaction = transaction
        self._events: List[DomainEvent] = []
        self._version = 0

    @property
    def transaction(self) -> Transaction:
        return self._transaction

    @property
    def events(self) -> List[DomainEvent]:
        return self._events.copy()

    @property
    def version(self) -> int:
        return self._version

    def process_fraud_analysis(self, result: FraudAnalysisResult) -> None:
        """Process fraud analysis result - O(1)"""
        self._transaction.risk_score = result.confidence_score
        self._transaction.risk_level = result.get_risk_level()

        if result.is_fraud:
            self._transaction.mark_as_fraud("ML Model Detection")
            self._add_event(
                FraudDetected(self._transaction.id, result.confidence_score, result.risk_factors)
            )

        self._version += 1

    def approve_transaction(self, approved_by: str) -> None:
        """Approve transaction with business rules - O(1)"""
        if self._transaction.requires_manual_review():
            raise ValueError("Transaction requires manual review before approval")

        self._transaction.approve()
        self._add_event(TransactionApproved(self._transaction.id, approved_by))
        self._version += 1

    def _add_event(self, event: DomainEvent) -> None:
        """Add domain event - O(1)"""
        self._events.append(event)

    def clear_events(self) -> None:
        """Clear events after publishing - O(1)"""
        self._events.clear()


# Factory Pattern for creating entities
class TransactionFactory:
    """
    Factory for creating Transaction entities
    Implements Factory Pattern
    Time Complexity: O(1)
    """

    @staticmethod
    def create_transaction(
        amount: Decimal,
        currency: str,
        merchant_id: str,
        customer_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Transaction:
        """Create a new transaction - O(1)"""
        transaction_id = TransactionId(f"TXN_{uuid4().hex[:12].upper()}")
        money = Money(amount, currency)

        return Transaction(
            id=transaction_id,
            amount=money,
            merchant_id=merchant_id,
            customer_id=customer_id,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

    @staticmethod
    def create_aggregate(transaction: Transaction) -> TransactionAggregate:
        """Create transaction aggregate - O(1)"""
        aggregate = TransactionAggregate(transaction)
        aggregate._add_event(
            TransactionCreated(transaction.id, transaction.amount, transaction.customer_id)
        )
        return aggregate


# Specification Pattern for business rules
class Specification(ABC):
    """Base specification for business rules"""

    @abstractmethod
    def is_satisfied_by(self, transaction: Transaction) -> bool:
        pass

    def and_(self, other: "Specification") -> "AndSpecification":
        return AndSpecification(self, other)

    def or_(self, other: "Specification") -> "OrSpecification":
        return OrSpecification(self, other)


class HighValueTransactionSpec(Specification):
    """Specification for high value transactions - O(1)"""

    def __init__(self, threshold: Money):
        self.threshold = threshold

    def is_satisfied_by(self, transaction: Transaction) -> bool:
        return transaction.is_high_value(self.threshold)


class HighRiskTransactionSpec(Specification):
    """Specification for high risk transactions - O(1)"""

    def is_satisfied_by(self, transaction: Transaction) -> bool:
        return transaction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class AndSpecification(Specification):
    """AND specification combinator - O(1)"""

    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right

    def is_satisfied_by(self, transaction: Transaction) -> bool:
        return self.left.is_satisfied_by(transaction) and self.right.is_satisfied_by(transaction)


class OrSpecification(Specification):
    """OR specification combinator - O(1)"""

    def __init__(self, left: Specification, right: Specification):
        self.left = left
        self.right = right

    def is_satisfied_by(self, transaction: Transaction) -> bool:
        return self.left.is_satisfied_by(transaction) or self.right.is_satisfied_by(transaction)
