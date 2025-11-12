"""
Core Interfaces - Clean Architecture
Abstract interfaces defining contracts between layers
Implements Dependency Inversion Principle (SOLID)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
from .entities import (
    Transaction,
    TransactionId,
    FraudAnalysisResult,
    Customer,
    TransactionAggregate,
    DomainEvent,
)


# Repository Interfaces - Repository Pattern
class TransactionRepository(ABC):
    """
    Abstract repository for Transaction entities
    Time Complexity: Depends on implementation
    """

    @abstractmethod
    async def save(self, transaction: Transaction) -> None:
        """Save transaction - O(log n) typical"""
        pass

    @abstractmethod
    async def find_by_id(self, transaction_id: TransactionId) -> Optional[Transaction]:
        """Find transaction by ID - O(log n) typical"""
        pass

    @abstractmethod
    async def find_by_customer(self, customer_id: str, limit: int = 100) -> List[Transaction]:
        """Find transactions by customer - O(log n + k) where k is result size"""
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[Transaction]:
        """Find transactions by date range - O(log n + k)"""
        pass

    @abstractmethod
    async def count_by_customer(self, customer_id: str) -> int:
        """Count transactions by customer - O(log n)"""
        pass


class CustomerRepository(ABC):
    """Abstract repository for Customer entities"""

    @abstractmethod
    async def save(self, customer: Customer) -> None:
        """Save customer - O(log n)"""
        pass

    @abstractmethod
    async def find_by_id(self, customer_id: str) -> Optional[Customer]:
        """Find customer by ID - O(log n)"""
        pass

    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[Customer]:
        """Find customer by email - O(log n)"""
        pass


class EventStore(ABC):
    """Abstract event store for domain events"""

    @abstractmethod
    async def save_events(self, aggregate_id: str, events: List[DomainEvent]) -> None:
        """Save domain events - O(k) where k is number of events"""
        pass

    @abstractmethod
    async def get_events(self, aggregate_id: str) -> List[DomainEvent]:
        """Get events for aggregate - O(k)"""
        pass


# Use Case Interfaces - Clean Architecture
class FraudDetectionService(ABC):
    """
    Abstract fraud detection service
    Implements Strategy Pattern for different ML models
    """

    @abstractmethod
    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """
        Analyze transaction for fraud
        Time Complexity: O(f) where f is feature extraction + model inference
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - O(1)"""
        pass


class NotificationService(ABC):
    """Abstract notification service"""

    @abstractmethod
    async def send_fraud_alert(
        self, transaction: Transaction, analysis: FraudAnalysisResult
    ) -> None:
        """Send fraud alert - O(1) async"""
        pass

    @abstractmethod
    async def send_approval_notification(self, transaction: Transaction) -> None:
        """Send approval notification - O(1) async"""
        pass


class AuditService(ABC):
    """Abstract audit service for compliance"""

    @abstractmethod
    async def log_transaction_event(
        self, transaction_id: TransactionId, event_type: str, details: Dict[str, Any]
    ) -> None:
        """Log audit event - O(1) async"""
        pass

    @abstractmethod
    async def log_fraud_detection(
        self, transaction: Transaction, analysis: FraudAnalysisResult
    ) -> None:
        """Log fraud detection for compliance - O(1) async"""
        pass


class CacheService(ABC):
    """Abstract cache service"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache - O(1) average"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set in cache - O(1) average"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete from cache - O(1) average"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists - O(1) average"""
        pass


# Event Handling Interfaces
class EventHandler(ABC):
    """Abstract event handler"""

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event - O(1) typically"""
        pass

    @abstractmethod
    def can_handle(self, event: DomainEvent) -> bool:
        """Check if can handle event - O(1)"""
        pass


class EventPublisher(ABC):
    """Abstract event publisher"""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish event - O(1) async"""
        pass

    @abstractmethod
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events - O(k) where k is number of events"""
        pass


# Command and Query Interfaces - CQRS Pattern
class Command(ABC):
    """Base command interface"""

    pass


class Query(ABC):
    """Base query interface"""

    pass


class CommandHandler(Protocol):
    """Command handler protocol"""

    async def handle(self, command: Command) -> Any:
        """Handle command"""
        pass


class QueryHandler(Protocol):
    """Query handler protocol"""

    async def handle(self, query: Query) -> Any:
        """Handle query"""
        pass


# Specific Commands
class ProcessTransactionCommand(Command):
    """Command to process a new transaction"""

    def __init__(
        self,
        amount: float,
        currency: str,
        merchant_id: str,
        customer_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.amount = amount
        self.currency = currency
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.metadata = metadata or {}


class ApproveTransactionCommand(Command):
    """Command to approve a transaction"""

    def __init__(self, transaction_id: str, approved_by: str):
        self.transaction_id = transaction_id
        self.approved_by = approved_by


class RejectTransactionCommand(Command):
    """Command to reject a transaction"""

    def __init__(self, transaction_id: str, reason: str, rejected_by: str):
        self.transaction_id = transaction_id
        self.reason = reason
        self.rejected_by = rejected_by


# Specific Queries
class GetTransactionQuery(Query):
    """Query to get a transaction by ID"""

    def __init__(self, transaction_id: str):
        self.transaction_id = transaction_id


class GetCustomerTransactionsQuery(Query):
    """Query to get customer transactions"""

    def __init__(self, customer_id: str, limit: int = 100, offset: int = 0):
        self.customer_id = customer_id
        self.limit = limit
        self.offset = offset


class GetFraudStatisticsQuery(Query):
    """Query to get fraud statistics"""

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date


# External Service Interfaces
class PaymentGateway(ABC):
    """Abstract payment gateway interface"""

    @abstractmethod
    async def process_payment(self, transaction: Transaction) -> Dict[str, Any]:
        """Process payment - O(1) network call"""
        pass

    @abstractmethod
    async def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Refund payment - O(1) network call"""
        pass


class ComplianceService(ABC):
    """Abstract compliance service"""

    @abstractmethod
    async def report_suspicious_activity(
        self, transaction: Transaction, analysis: FraudAnalysisResult
    ) -> None:
        """Report to regulatory authorities - O(1) async"""
        pass

    @abstractmethod
    async def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction against compliance rules - O(1)"""
        pass


# Metrics and Monitoring Interfaces
class MetricsCollector(ABC):
    """Abstract metrics collector"""

    @abstractmethod
    def increment_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter - O(1)"""
        pass

    @abstractmethod
    def record_histogram(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram value - O(1)"""
        pass

    @abstractmethod
    def record_gauge(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record gauge value - O(1)"""
        pass


class HealthChecker(ABC):
    """Abstract health checker"""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check service health - O(1)"""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get service name - O(1)"""
        pass


# Configuration Interface
class ConfigurationService(ABC):
    """Abstract configuration service"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value - O(1)"""
        pass

    @abstractmethod
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration - O(1)"""
        pass

    @abstractmethod
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration - O(1)"""
        pass

    @abstractmethod
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration - O(1)"""
        pass


# Feature Flag Interface
class FeatureFlags(ABC):
    """Abstract feature flags service"""

    @abstractmethod
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if feature is enabled - O(1)"""
        pass

    @abstractmethod
    def get_variant(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get feature variant - O(1)"""
        pass
