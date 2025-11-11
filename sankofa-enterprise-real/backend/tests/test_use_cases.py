"""
Integration Tests for Use Cases
Tests the orchestration of business logic
Follows Test-Driven Development (TDD) principles
"""

import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime

from ..core.entities import (
    Transaction, TransactionId, Money, Customer,
    TransactionStatus, RiskLevel, FraudAnalysisResult
)
from ..core.interfaces import (
    ProcessTransactionCommand, ApproveTransactionCommand,
    GetTransactionQuery, GetFraudStatisticsQuery
)
from ..core.use_cases import (
    ProcessTransactionUseCase, ApproveTransactionUseCase,
    GetTransactionUseCase, GetFraudStatisticsUseCase
)


class TestProcessTransactionUseCase:
    """Test ProcessTransactionUseCase integration"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for use case"""
        return {
            'transaction_repo': AsyncMock(),
            'customer_repo': AsyncMock(),
            'fraud_service': AsyncMock(),
            'notification_service': AsyncMock(),
            'audit_service': AsyncMock(),
            'cache_service': AsyncMock(),
            'event_publisher': AsyncMock(),
            'metrics_collector': Mock()
        }
    
    @pytest.fixture
    def use_case(self, mock_dependencies):
        """Create ProcessTransactionUseCase with mocked dependencies"""
        return ProcessTransactionUseCase(**mock_dependencies)
    
    @pytest.fixture
    def sample_command(self):
        """Create sample ProcessTransactionCommand"""
        return ProcessTransactionCommand(
            amount=100.0,
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            metadata={"channel": "online"}
        )
    
    @pytest.fixture
    def sample_customer(self):
        """Create sample customer"""
        return Customer(
            id="CUSTOMER_456",
            email="test@example.com",
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def low_risk_fraud_result(self):
        """Create low risk fraud analysis result"""
        return FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=False,
            confidence_score=0.2,
            processing_time_ms=15.0
        )
    
    @pytest.fixture
    def high_risk_fraud_result(self):
        """Create high risk fraud analysis result"""
        return FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=True,
            confidence_score=0.9,
            risk_factors=["HIGH_VELOCITY", "UNUSUAL_LOCATION"],
            processing_time_ms=25.0
        )
    
    @pytest.mark.asyncio
    async def test_process_low_risk_transaction(
        self, 
        use_case, 
        mock_dependencies, 
        sample_command, 
        sample_customer, 
        low_risk_fraud_result
    ):
        """Test processing low risk transaction - should auto-approve"""
        # Setup mocks
        mock_dependencies['customer_repo'].find_by_id.return_value = sample_customer
        mock_dependencies['fraud_service'].analyze_transaction.return_value = low_risk_fraud_result
        mock_dependencies['cache_service'].get.return_value = None  # Cache miss
        
        # Execute use case
        result = await use_case.execute(sample_command)
        
        # Verify result
        assert result['decision'] == 'auto_approved'
        assert result['risk_score'] == 0.2
        assert 'transaction_id' in result
        assert 'processing_time_ms' in result
        
        # Verify interactions
        mock_dependencies['transaction_repo'].save.assert_called_once()
        mock_dependencies['customer_repo'].save.assert_called_once()
        mock_dependencies['fraud_service'].analyze_transaction.assert_called_once()
        mock_dependencies['notification_service'].send_approval_notification.assert_called_once()
        mock_dependencies['audit_service'].log_transaction_event.assert_called_once()
        mock_dependencies['metrics_collector'].increment_counter.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_high_risk_transaction(
        self, 
        use_case, 
        mock_dependencies, 
        sample_command, 
        sample_customer, 
        high_risk_fraud_result
    ):
        """Test processing high risk transaction - should require manual review"""
        # Setup mocks
        mock_dependencies['customer_repo'].find_by_id.return_value = sample_customer
        mock_dependencies['fraud_service'].analyze_transaction.return_value = high_risk_fraud_result
        mock_dependencies['cache_service'].get.return_value = None
        
        # Execute use case
        result = await use_case.execute(sample_command)
        
        # Verify result
        assert result['decision'] == 'manual_review_required'
        assert result['risk_score'] == 0.9
        
        # Verify fraud alert was sent
        mock_dependencies['notification_service'].send_fraud_alert.assert_called_once()
        
        # Verify no approval notification
        mock_dependencies['notification_service'].send_approval_notification.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_high_value_new_customer(
        self, 
        use_case, 
        mock_dependencies, 
        sample_customer, 
        low_risk_fraud_result
    ):
        """Test high value transaction from new customer - should require review"""
        # High value command
        command = ProcessTransactionCommand(
            amount=6000.0,  # Above 5000 threshold
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        
        # New customer (0 transactions)
        sample_customer.transaction_history = []
        
        # Setup mocks
        mock_dependencies['customer_repo'].find_by_id.return_value = sample_customer
        mock_dependencies['fraud_service'].analyze_transaction.return_value = low_risk_fraud_result
        mock_dependencies['cache_service'].get.return_value = None
        
        # Execute use case
        result = await use_case.execute(command)
        
        # Verify result
        assert result['decision'] == 'manual_review_required'
    
    @pytest.mark.asyncio
    async def test_customer_caching(
        self, 
        use_case, 
        mock_dependencies, 
        sample_command, 
        sample_customer, 
        low_risk_fraud_result
    ):
        """Test customer caching mechanism"""
        # Setup cache hit
        mock_dependencies['cache_service'].get.return_value = sample_customer
        mock_dependencies['fraud_service'].analyze_transaction.return_value = low_risk_fraud_result
        
        # Execute use case
        await use_case.execute(sample_command)
        
        # Verify cache was checked
        mock_dependencies['cache_service'].get.assert_called_with("customer:CUSTOMER_456")
        
        # Verify repository was not called (cache hit)
        mock_dependencies['customer_repo'].find_by_id.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_new_customer_creation(
        self, 
        use_case, 
        mock_dependencies, 
        sample_command, 
        low_risk_fraud_result
    ):
        """Test automatic customer creation for new customers"""
        # Setup cache miss and repository miss
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['customer_repo'].find_by_id.return_value = None
        mock_dependencies['fraud_service'].analyze_transaction.return_value = low_risk_fraud_result
        
        # Execute use case
        result = await use_case.execute(sample_command)
        
        # Verify customer was created and saved
        assert mock_dependencies['customer_repo'].save.call_count == 2  # Create + update
        
        # Verify customer was cached
        mock_dependencies['cache_service'].set.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self, 
        use_case, 
        mock_dependencies, 
        sample_command
    ):
        """Test error handling in use case"""
        # Setup fraud service to raise exception
        mock_dependencies['fraud_service'].analyze_transaction.side_effect = Exception("ML service error")
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['customer_repo'].find_by_id.return_value = None
        
        # Execute use case and expect exception
        with pytest.raises(Exception, match="ML service error"):
            await use_case.execute(sample_command)
        
        # Verify error metrics were recorded
        mock_dependencies['metrics_collector'].increment_counter.assert_called_with("transaction_processing_errors")
        mock_dependencies['audit_service'].log_transaction_event.assert_called()


class TestApproveTransactionUseCase:
    """Test ApproveTransactionUseCase"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        return {
            'transaction_repo': AsyncMock(),
            'audit_service': AsyncMock(),
            'notification_service': AsyncMock(),
            'event_publisher': AsyncMock(),
            'metrics_collector': Mock()
        }
    
    @pytest.fixture
    def use_case(self, mock_dependencies):
        """Create ApproveTransactionUseCase"""
        return ApproveTransactionUseCase(**mock_dependencies)
    
    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction"""
        return Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("100.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            status=TransactionStatus.PENDING,
            risk_level=RiskLevel.MEDIUM
        )
    
    @pytest.mark.asyncio
    async def test_approve_valid_transaction(
        self, 
        use_case, 
        mock_dependencies, 
        sample_transaction
    ):
        """Test approving valid transaction"""
        command = ApproveTransactionCommand(
            transaction_id="TXN_123456789",
            approved_by="admin_user"
        )
        
        # Setup mocks
        mock_dependencies['transaction_repo'].find_by_id.return_value = sample_transaction
        
        # Execute use case
        result = await use_case.execute(command)
        
        # Verify result
        assert result['transaction_id'] == "TXN_123456789"
        assert result['status'] == "approved"
        assert result['approved_by'] == "admin_user"
        assert 'approved_at' in result
        
        # Verify interactions
        mock_dependencies['transaction_repo'].save.assert_called_once()
        mock_dependencies['notification_service'].send_approval_notification.assert_called_once()
        mock_dependencies['audit_service'].log_transaction_event.assert_called_once()
        mock_dependencies['event_publisher'].publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_approve_nonexistent_transaction(
        self, 
        use_case, 
        mock_dependencies
    ):
        """Test approving non-existent transaction"""
        command = ApproveTransactionCommand(
            transaction_id="TXN_NONEXISTENT",
            approved_by="admin_user"
        )
        
        # Setup mock to return None
        mock_dependencies['transaction_repo'].find_by_id.return_value = None
        
        # Execute and expect error
        with pytest.raises(ValueError, match="Transaction TXN_NONEXISTENT not found"):
            await use_case.execute(command)
    
    @pytest.mark.asyncio
    async def test_approve_critical_risk_transaction(
        self, 
        use_case, 
        mock_dependencies, 
        sample_transaction
    ):
        """Test approving critical risk transaction - should fail"""
        # Set transaction to critical risk
        sample_transaction.risk_level = RiskLevel.CRITICAL
        
        command = ApproveTransactionCommand(
            transaction_id="TXN_123456789",
            approved_by="admin_user"
        )
        
        # Setup mocks
        mock_dependencies['transaction_repo'].find_by_id.return_value = sample_transaction
        
        # Execute and expect error
        with pytest.raises(ValueError, match="Transaction requires manual review"):
            await use_case.execute(command)


class TestGetTransactionUseCase:
    """Test GetTransactionUseCase"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        return {
            'transaction_repo': AsyncMock(),
            'cache_service': AsyncMock()
        }
    
    @pytest.fixture
    def use_case(self, mock_dependencies):
        """Create GetTransactionUseCase"""
        return GetTransactionUseCase(**mock_dependencies)
    
    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction"""
        return Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("100.00"), "USD"),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            status=TransactionStatus.APPROVED,
            risk_level=RiskLevel.LOW,
            risk_score=0.2,
            metadata={"channel": "online"}
        )
    
    @pytest.mark.asyncio
    async def test_get_transaction_cache_hit(
        self, 
        use_case, 
        mock_dependencies
    ):
        """Test getting transaction with cache hit"""
        query = GetTransactionQuery("TXN_123456789")
        
        cached_result = {
            "id": "TXN_123456789",
            "amount": 100.0,
            "currency": "USD",
            "status": "approved"
        }
        
        # Setup cache hit
        mock_dependencies['cache_service'].get.return_value = cached_result
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify result
        assert result == cached_result
        
        # Verify repository was not called
        mock_dependencies['transaction_repo'].find_by_id.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_transaction_cache_miss(
        self, 
        use_case, 
        mock_dependencies, 
        sample_transaction
    ):
        """Test getting transaction with cache miss"""
        query = GetTransactionQuery("TXN_123456789")
        
        # Setup cache miss
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['transaction_repo'].find_by_id.return_value = sample_transaction
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify result structure
        assert result['id'] == "TXN_123456789"
        assert result['amount'] == 100.0
        assert result['currency'] == "USD"
        assert result['merchant_id'] == "MERCHANT_123"
        assert result['customer_id'] == "CUSTOMER_456"
        assert result['status'] == "approved"
        assert result['risk_level'] == "low"
        assert result['risk_score'] == 0.2
        assert result['metadata'] == {"channel": "online"}
        
        # Verify cache was populated
        mock_dependencies['cache_service'].set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_transaction(
        self, 
        use_case, 
        mock_dependencies
    ):
        """Test getting non-existent transaction"""
        query = GetTransactionQuery("TXN_NONEXISTENT")
        
        # Setup cache miss and repository miss
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['transaction_repo'].find_by_id.return_value = None
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify result is None
        assert result is None


class TestGetFraudStatisticsUseCase:
    """Test GetFraudStatisticsUseCase"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        return {
            'transaction_repo': AsyncMock(),
            'cache_service': AsyncMock()
        }
    
    @pytest.fixture
    def use_case(self, mock_dependencies):
        """Create GetFraudStatisticsUseCase"""
        return GetFraudStatisticsUseCase(**mock_dependencies)
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transactions for statistics"""
        transactions = []
        
        # Normal transactions
        for i in range(80):
            tx = Transaction(
                id=TransactionId(f"TXN_{i:06d}"),
                amount=Money(Decimal("100.00")),
                merchant_id="MERCHANT_123",
                customer_id=f"CUSTOMER_{i}",
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.LOW
            )
            transactions.append(tx)
        
        # Fraud transactions
        for i in range(80, 100):
            tx = Transaction(
                id=TransactionId(f"TXN_{i:06d}"),
                amount=Money(Decimal("500.00")),
                merchant_id="MERCHANT_123",
                customer_id=f"CUSTOMER_{i}",
                timestamp=datetime.utcnow(),
                risk_level=RiskLevel.CRITICAL
            )
            transactions.append(tx)
        
        return transactions
    
    @pytest.mark.asyncio
    async def test_get_fraud_statistics_cache_miss(
        self, 
        use_case, 
        mock_dependencies, 
        sample_transactions
    ):
        """Test getting fraud statistics with cache miss"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        query = GetFraudStatisticsQuery(start_date, end_date)
        
        # Setup cache miss
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['transaction_repo'].find_by_date_range.return_value = sample_transactions
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify result structure
        assert 'period' in result
        assert 'summary' in result
        assert 'risk_distribution' in result
        assert 'generated_at' in result
        
        # Verify statistics calculations
        summary = result['summary']
        assert summary['total_transactions'] == 100
        assert summary['fraud_transactions'] == 20
        assert summary['fraud_rate_percent'] == 20.0
        assert summary['total_amount'] == 18000.0  # 80*100 + 20*500
        assert summary['fraud_amount'] == 10000.0  # 20*500
        assert abs(summary['fraud_amount_rate_percent'] - 55.56) < 0.1
        
        # Verify risk distribution
        risk_dist = result['risk_distribution']
        assert risk_dist['low'] == 80
        assert risk_dist['critical'] == 20
        assert risk_dist['medium'] == 0
        assert risk_dist['high'] == 0
        
        # Verify cache was populated
        mock_dependencies['cache_service'].set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_fraud_statistics_cache_hit(
        self, 
        use_case, 
        mock_dependencies
    ):
        """Test getting fraud statistics with cache hit"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        query = GetFraudStatisticsQuery(start_date, end_date)
        
        cached_result = {
            "summary": {"fraud_rate_percent": 15.0},
            "cached": True
        }
        
        # Setup cache hit
        mock_dependencies['cache_service'].get.return_value = cached_result
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify cached result was returned
        assert result == cached_result
        
        # Verify repository was not called
        mock_dependencies['transaction_repo'].find_by_date_range.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_fraud_statistics_empty_results(
        self, 
        use_case, 
        mock_dependencies
    ):
        """Test getting fraud statistics with no transactions"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        query = GetFraudStatisticsQuery(start_date, end_date)
        
        # Setup cache miss and empty results
        mock_dependencies['cache_service'].get.return_value = None
        mock_dependencies['transaction_repo'].find_by_date_range.return_value = []
        
        # Execute query
        result = await use_case.execute(query)
        
        # Verify zero statistics
        summary = result['summary']
        assert summary['total_transactions'] == 0
        assert summary['fraud_transactions'] == 0
        assert summary['fraud_rate_percent'] == 0
        assert summary['total_amount'] == 0
        assert summary['fraud_amount'] == 0
        assert summary['fraud_amount_rate_percent'] == 0