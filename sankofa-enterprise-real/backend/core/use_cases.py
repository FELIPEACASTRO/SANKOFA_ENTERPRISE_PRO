"""
Use Cases - Clean Architecture Application Layer
Implements business use cases orchestrating domain entities
Follows Single Responsibility Principle (SOLID)
Time Complexity analysis provided for each use case
"""

import asyncio
import time
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .entities import (
    Transaction, TransactionId, FraudAnalysisResult, Customer,
    TransactionFactory, TransactionAggregate, RiskLevel, Money
)
from .interfaces import (
    TransactionRepository, CustomerRepository, EventStore,
    FraudDetectionService, NotificationService, AuditService,
    CacheService, EventPublisher, MetricsCollector,
    ProcessTransactionCommand, ApproveTransactionCommand, RejectTransactionCommand,
    GetTransactionQuery, GetCustomerTransactionsQuery, GetFraudStatisticsQuery,
    CommandHandler, QueryHandler
)


class ProcessTransactionUseCase:
    """
    Use Case: Process a new transaction
    Orchestrates fraud detection, risk assessment, and decision making
    Time Complexity: O(f + log n) where f is fraud detection time, n is repository size
    """
    
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        customer_repo: CustomerRepository,
        fraud_service: FraudDetectionService,
        notification_service: NotificationService,
        audit_service: AuditService,
        cache_service: CacheService,
        event_publisher: EventPublisher,
        metrics_collector: MetricsCollector
    ):
        # Dependency Injection - Dependency Inversion Principle
        self._transaction_repo = transaction_repo
        self._customer_repo = customer_repo
        self._fraud_service = fraud_service
        self._notification_service = notification_service
        self._audit_service = audit_service
        self._cache_service = cache_service
        self._event_publisher = event_publisher
        self._metrics_collector = metrics_collector
    
    async def execute(self, command: ProcessTransactionCommand) -> Dict[str, Any]:
        """
        Execute transaction processing use case
        Time Complexity: O(f + log n) where f is ML inference time
        """
        start_time = time.time()
        
        try:
            # 1. Create transaction entity - O(1)
            transaction = TransactionFactory.create_transaction(
                amount=Decimal(str(command.amount)),
                currency=command.currency,
                merchant_id=command.merchant_id,
                customer_id=command.customer_id,
                metadata=command.metadata
            )
            
            # 2. Create aggregate - O(1)
            aggregate = TransactionFactory.create_aggregate(transaction)
            
            # 3. Get customer info (with caching) - O(1) cache hit, O(log n) cache miss
            customer = await self._get_customer_with_cache(command.customer_id)
            
            # 4. Fraud detection - O(f) where f is feature extraction + ML inference
            fraud_analysis = await self._fraud_service.analyze_transaction(transaction)
            
            # 5. Process analysis result - O(1)
            aggregate.process_fraud_analysis(fraud_analysis)
            
            # 6. Apply business rules - O(1)
            decision = await self._apply_business_rules(aggregate, customer, fraud_analysis)
            
            # 7. Save transaction - O(log n)
            await self._transaction_repo.save(aggregate.transaction)
            
            # 8. Update customer - O(log n)
            customer.add_transaction(transaction.id)
            await self._customer_repo.save(customer)
            
            # 9. Publish events - O(k) where k is number of events
            for event in aggregate.events:
                await self._event_publisher.publish(event)
            aggregate.clear_events()
            
            # 10. Send notifications if needed - O(1) async
            await self._handle_notifications(aggregate.transaction, fraud_analysis, decision)
            
            # 11. Audit logging - O(1) async
            await self._audit_service.log_transaction_event(
                transaction.id,
                "transaction_processed",
                {
                    "decision": decision,
                    "risk_score": fraud_analysis.confidence_score,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
            
            # 12. Metrics - O(1)
            self._record_metrics(decision, fraud_analysis, time.time() - start_time)
            
            return {
                "transaction_id": transaction.id.value,
                "status": transaction.status.value,
                "risk_level": transaction.risk_level.value,
                "risk_score": fraud_analysis.confidence_score,
                "decision": decision,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            # Error handling and metrics
            self._metrics_collector.increment_counter("transaction_processing_errors")
            await self._audit_service.log_transaction_event(
                TransactionId("ERROR"),
                "processing_error",
                {"error": str(e), "command": command.__dict__}
            )
            raise
    
    async def _get_customer_with_cache(self, customer_id: str) -> Customer:
        """Get customer with caching - O(1) cache hit, O(log n) cache miss"""
        cache_key = f"customer:{customer_id}"
        
        # Try cache first - O(1)
        cached_customer = await self._cache_service.get(cache_key)
        if cached_customer:
            return cached_customer
        
        # Cache miss, get from repository - O(log n)
        customer = await self._customer_repo.find_by_id(customer_id)
        if not customer:
            # Create new customer if not exists
            customer = Customer(
                id=customer_id,
                email=f"customer_{customer_id}@example.com",
                created_at=datetime.utcnow()
            )
            await self._customer_repo.save(customer)
        
        # Cache for future use - O(1)
        await self._cache_service.set(cache_key, customer, ttl=3600)
        return customer
    
    async def _apply_business_rules(
        self, 
        aggregate: TransactionAggregate, 
        customer: Customer,
        fraud_analysis: FraudAnalysisResult
    ) -> str:
        """Apply business rules and make decision - O(1)"""
        transaction = aggregate.transaction
        
        # High risk transactions require manual review
        if fraud_analysis.confidence_score > 0.8:
            return "manual_review_required"
        
        # High value transactions from new customers
        if (transaction.is_high_value(Money(Decimal("5000"))) and 
            customer.get_transaction_count() < 5):
            return "manual_review_required"
        
        # Auto-approve low risk
        if fraud_analysis.confidence_score < 0.3:
            try:
                aggregate.approve_transaction("system")
                return "auto_approved"
            except ValueError:
                return "manual_review_required"
        
        return "pending_review"
    
    async def _handle_notifications(
        self, 
        transaction: Transaction, 
        fraud_analysis: FraudAnalysisResult,
        decision: str
    ) -> None:
        """Handle notifications based on decision - O(1) async"""
        if fraud_analysis.confidence_score > 0.7:
            await self._notification_service.send_fraud_alert(transaction, fraud_analysis)
        
        if decision == "auto_approved":
            await self._notification_service.send_approval_notification(transaction)
    
    def _record_metrics(self, decision: str, fraud_analysis: FraudAnalysisResult, duration: float) -> None:
        """Record metrics - O(1)"""
        self._metrics_collector.increment_counter("transactions_processed")
        self._metrics_collector.increment_counter(f"transactions_{decision}")
        self._metrics_collector.record_histogram("transaction_processing_duration", duration * 1000)
        self._metrics_collector.record_histogram("fraud_score", fraud_analysis.confidence_score)


class ApproveTransactionUseCase:
    """
    Use Case: Approve a transaction manually
    Time Complexity: O(log n) for repository operations
    """
    
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        audit_service: AuditService,
        notification_service: NotificationService,
        event_publisher: EventPublisher,
        metrics_collector: MetricsCollector
    ):
        self._transaction_repo = transaction_repo
        self._audit_service = audit_service
        self._notification_service = notification_service
        self._event_publisher = event_publisher
        self._metrics_collector = metrics_collector
    
    async def execute(self, command: ApproveTransactionCommand) -> Dict[str, Any]:
        """Execute approval use case - O(log n)"""
        start_time = time.time()
        
        try:
            # 1. Find transaction - O(log n)
            transaction_id = TransactionId(command.transaction_id)
            transaction = await self._transaction_repo.find_by_id(transaction_id)
            
            if not transaction:
                raise ValueError(f"Transaction {command.transaction_id} not found")
            
            # 2. Create aggregate and approve - O(1)
            aggregate = TransactionFactory.create_aggregate(transaction)
            aggregate.approve_transaction(command.approved_by)
            
            # 3. Save updated transaction - O(log n)
            await self._transaction_repo.save(aggregate.transaction)
            
            # 4. Publish events - O(k)
            for event in aggregate.events:
                await self._event_publisher.publish(event)
            
            # 5. Send notification - O(1) async
            await self._notification_service.send_approval_notification(transaction)
            
            # 6. Audit log - O(1) async
            await self._audit_service.log_transaction_event(
                transaction_id,
                "transaction_approved",
                {
                    "approved_by": command.approved_by,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
            
            # 7. Metrics - O(1)
            self._metrics_collector.increment_counter("transactions_manually_approved")
            self._metrics_collector.record_histogram(
                "approval_processing_duration", 
                (time.time() - start_time) * 1000
            )
            
            return {
                "transaction_id": command.transaction_id,
                "status": "approved",
                "approved_by": command.approved_by,
                "approved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._metrics_collector.increment_counter("approval_errors")
            raise


class GetTransactionUseCase:
    """
    Query Use Case: Get transaction by ID
    Time Complexity: O(1) cache hit, O(log n) cache miss
    """
    
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        cache_service: CacheService
    ):
        self._transaction_repo = transaction_repo
        self._cache_service = cache_service
    
    async def execute(self, query: GetTransactionQuery) -> Optional[Dict[str, Any]]:
        """Execute get transaction query - O(1) or O(log n)"""
        # Try cache first - O(1)
        cache_key = f"transaction:{query.transaction_id}"
        cached_transaction = await self._cache_service.get(cache_key)
        
        if cached_transaction:
            return cached_transaction
        
        # Cache miss, get from repository - O(log n)
        transaction_id = TransactionId(query.transaction_id)
        transaction = await self._transaction_repo.find_by_id(transaction_id)
        
        if not transaction:
            return None
        
        # Convert to dict and cache - O(1)
        result = {
            "id": transaction.id.value,
            "amount": float(transaction.amount.amount),
            "currency": transaction.amount.currency,
            "merchant_id": transaction.merchant_id,
            "customer_id": transaction.customer_id,
            "status": transaction.status.value,
            "risk_level": transaction.risk_level.value,
            "risk_score": transaction.risk_score,
            "timestamp": transaction.timestamp.isoformat(),
            "metadata": transaction.metadata
        }
        
        await self._cache_service.set(cache_key, result, ttl=1800)
        return result


class GetFraudStatisticsUseCase:
    """
    Query Use Case: Get fraud statistics for a date range
    Time Complexity: O(log n + k) where k is result size
    """
    
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        cache_service: CacheService
    ):
        self._transaction_repo = transaction_repo
        self._cache_service = cache_service
    
    async def execute(self, query: GetFraudStatisticsQuery) -> Dict[str, Any]:
        """Execute fraud statistics query - O(log n + k)"""
        # Create cache key based on date range
        cache_key = f"fraud_stats:{query.start_date.date()}:{query.end_date.date()}"
        
        # Try cache first - O(1)
        cached_stats = await self._cache_service.get(cache_key)
        if cached_stats:
            return cached_stats
        
        # Get transactions in date range - O(log n + k)
        transactions = await self._transaction_repo.find_by_date_range(
            query.start_date, 
            query.end_date,
            limit=10000  # Reasonable limit
        )
        
        # Calculate statistics - O(k)
        total_transactions = len(transactions)
        fraud_transactions = sum(1 for t in transactions if t.risk_level == RiskLevel.CRITICAL)
        high_risk_transactions = sum(1 for t in transactions if t.risk_level == RiskLevel.HIGH)
        
        total_amount = sum(float(t.amount.amount) for t in transactions)
        fraud_amount = sum(
            float(t.amount.amount) for t in transactions 
            if t.risk_level == RiskLevel.CRITICAL
        )
        
        # Calculate rates - O(1)
        fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
        fraud_amount_rate = (fraud_amount / total_amount * 100) if total_amount > 0 else 0
        
        # Risk distribution - O(k)
        risk_distribution = {
            "low": sum(1 for t in transactions if t.risk_level == RiskLevel.LOW),
            "medium": sum(1 for t in transactions if t.risk_level == RiskLevel.MEDIUM),
            "high": high_risk_transactions,
            "critical": fraud_transactions
        }
        
        result = {
            "period": {
                "start_date": query.start_date.isoformat(),
                "end_date": query.end_date.isoformat()
            },
            "summary": {
                "total_transactions": total_transactions,
                "fraud_transactions": fraud_transactions,
                "fraud_rate_percent": round(fraud_rate, 2),
                "total_amount": round(total_amount, 2),
                "fraud_amount": round(fraud_amount, 2),
                "fraud_amount_rate_percent": round(fraud_amount_rate, 2)
            },
            "risk_distribution": risk_distribution,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache for 30 minutes - O(1)
        await self._cache_service.set(cache_key, result, ttl=1800)
        return result


# Command and Query Handlers implementing CQRS pattern
class TransactionCommandHandler:
    """
    Command handler for transaction commands
    Implements Command Handler pattern
    """
    
    def __init__(
        self,
        process_use_case: ProcessTransactionUseCase,
        approve_use_case: ApproveTransactionUseCase
    ):
        self._process_use_case = process_use_case
        self._approve_use_case = approve_use_case
    
    async def handle(self, command) -> Any:
        """Handle command based on type - O(1) dispatch + use case complexity"""
        if isinstance(command, ProcessTransactionCommand):
            return await self._process_use_case.execute(command)
        elif isinstance(command, ApproveTransactionCommand):
            return await self._approve_use_case.execute(command)
        else:
            raise ValueError(f"Unknown command type: {type(command)}")


class TransactionQueryHandler:
    """
    Query handler for transaction queries
    Implements Query Handler pattern
    """
    
    def __init__(
        self,
        get_transaction_use_case: GetTransactionUseCase,
        get_fraud_stats_use_case: GetFraudStatisticsUseCase
    ):
        self._get_transaction_use_case = get_transaction_use_case
        self._get_fraud_stats_use_case = get_fraud_stats_use_case
    
    async def handle(self, query) -> Any:
        """Handle query based on type - O(1) dispatch + use case complexity"""
        if isinstance(query, GetTransactionQuery):
            return await self._get_transaction_use_case.execute(query)
        elif isinstance(query, GetFraudStatisticsQuery):
            return await self._get_fraud_stats_use_case.execute(query)
        else:
            raise ValueError(f"Unknown query type: {type(query)}")


# Saga Pattern for complex workflows
class TransactionProcessingSaga:
    """
    Saga for handling complex transaction processing workflows
    Implements Saga Pattern for distributed transactions
    Time Complexity: O(n) where n is number of steps
    """
    
    def __init__(
        self,
        process_use_case: ProcessTransactionUseCase,
        audit_service: AuditService
    ):
        self._process_use_case = process_use_case
        self._audit_service = audit_service
        self._steps = []
        self._compensations = []
    
    async def execute_transaction_processing(
        self, 
        command: ProcessTransactionCommand
    ) -> Dict[str, Any]:
        """Execute saga with compensation on failure - O(n)"""
        saga_id = f"saga_{int(time.time() * 1000)}"
        
        try:
            # Step 1: Process transaction
            result = await self._process_use_case.execute(command)
            self._steps.append("transaction_processed")
            
            # Step 2: Additional validations (if needed)
            # This is where you'd add more steps
            
            await self._audit_service.log_transaction_event(
                TransactionId(result["transaction_id"]),
                "saga_completed",
                {"saga_id": saga_id, "steps": self._steps}
            )
            
            return result
            
        except Exception as e:
            # Execute compensations in reverse order - O(n)
            await self._compensate(saga_id, str(e))
            raise
    
    async def _compensate(self, saga_id: str, error: str) -> None:
        """Execute compensation actions - O(n)"""
        for compensation in reversed(self._compensations):
            try:
                await compensation()
            except Exception as comp_error:
                # Log compensation failure but continue
                await self._audit_service.log_transaction_event(
                    TransactionId("SAGA_ERROR"),
                    "compensation_failed",
                    {
                        "saga_id": saga_id,
                        "original_error": error,
                        "compensation_error": str(comp_error)
                    }
                )