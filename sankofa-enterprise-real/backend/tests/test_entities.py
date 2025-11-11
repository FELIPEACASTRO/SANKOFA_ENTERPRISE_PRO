"""
Unit Tests for Core Entities
Tests business logic and domain rules
Follows Test-Driven Development (TDD) principles
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock

from ..core.entities import (
    Transaction, TransactionId, Money, Customer,
    TransactionStatus, RiskLevel, FraudAnalysisResult,
    TransactionFactory, TransactionAggregate,
    HighValueTransactionSpec, HighRiskTransactionSpec
)


class TestTransactionId:
    """Test TransactionId value object"""
    
    def test_valid_transaction_id(self):
        """Test creating valid transaction ID - O(1)"""
        tx_id = TransactionId("TXN_123456789")
        assert tx_id.value == "TXN_123456789"
    
    def test_invalid_transaction_id_too_short(self):
        """Test validation of short transaction ID - O(1)"""
        with pytest.raises(ValueError, match="Transaction ID must be at least 10 characters"):
            TransactionId("SHORT")
    
    def test_invalid_transaction_id_empty(self):
        """Test validation of empty transaction ID - O(1)"""
        with pytest.raises(ValueError, match="Transaction ID must be at least 10 characters"):
            TransactionId("")


class TestMoney:
    """Test Money value object"""
    
    def test_valid_money_creation(self):
        """Test creating valid Money object - O(1)"""
        money = Money(Decimal("100.50"), "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"
    
    def test_default_currency(self):
        """Test default currency is BRL - O(1)"""
        money = Money(Decimal("100.00"))
        assert money.currency == "BRL"
    
    def test_negative_amount_validation(self):
        """Test negative amount validation - O(1)"""
        with pytest.raises(ValueError, match="Amount cannot be negative"):
            Money(Decimal("-10.00"))
    
    def test_invalid_currency_validation(self):
        """Test currency validation - O(1)"""
        with pytest.raises(ValueError, match="Currency must be 3 characters"):
            Money(Decimal("100.00"), "US")
    
    def test_money_addition_same_currency(self):
        """Test adding money with same currency - O(1)"""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")
        result = money1.add(money2)
        
        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"
    
    def test_money_addition_different_currency(self):
        """Test adding money with different currencies - O(1)"""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")
        
        with pytest.raises(ValueError, match="Cannot add different currencies"):
            money1.add(money2)
    
    def test_money_comparison(self):
        """Test money comparison - O(1)"""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")
        
        assert money1.is_greater_than(money2)
        assert not money2.is_greater_than(money1)
    
    def test_money_comparison_different_currency(self):
        """Test comparison with different currencies - O(1)"""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")
        
        with pytest.raises(ValueError, match="Cannot compare different currencies"):
            money1.is_greater_than(money2)


class TestTransaction:
    """Test Transaction entity"""
    
    def test_valid_transaction_creation(self):
        """Test creating valid transaction - O(1)"""
        tx_id = TransactionId("TXN_123456789")
        amount = Money(Decimal("100.00"))
        timestamp = datetime.utcnow()
        
        transaction = Transaction(
            id=tx_id,
            amount=amount,
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=timestamp
        )
        
        assert transaction.id == tx_id
        assert transaction.amount == amount
        assert transaction.status == TransactionStatus.PENDING
        assert transaction.risk_level == RiskLevel.LOW
        assert transaction.risk_score == 0.0
    
    def test_transaction_validation_empty_merchant(self):
        """Test merchant ID validation - O(1)"""
        with pytest.raises(ValueError, match="Merchant ID is required"):
            Transaction(
                id=TransactionId("TXN_123456789"),
                amount=Money(Decimal("100.00")),
                merchant_id="",
                customer_id="CUSTOMER_456",
                timestamp=datetime.utcnow()
            )
    
    def test_transaction_validation_empty_customer(self):
        """Test customer ID validation - O(1)"""
        with pytest.raises(ValueError, match="Customer ID is required"):
            Transaction(
                id=TransactionId("TXN_123456789"),
                amount=Money(Decimal("100.00")),
                merchant_id="MERCHANT_123",
                customer_id="",
                timestamp=datetime.utcnow()
            )
    
    def test_transaction_validation_invalid_risk_score(self):
        """Test risk score validation - O(1)"""
        with pytest.raises(ValueError, match="Risk score must be between 0 and 1"):
            Transaction(
                id=TransactionId("TXN_123456789"),
                amount=Money(Decimal("100.00")),
                merchant_id="MERCHANT_123",
                customer_id="CUSTOMER_456",
                timestamp=datetime.utcnow(),
                risk_score=1.5
            )
    
    def test_mark_as_fraud(self):
        """Test marking transaction as fraud - O(1)"""
        transaction = self._create_valid_transaction()
        
        transaction.mark_as_fraud("Suspicious pattern detected")
        
        assert transaction.status == TransactionStatus.REJECTED
        assert transaction.risk_level == RiskLevel.CRITICAL
        assert transaction.metadata["fraud_reason"] == "Suspicious pattern detected"
        assert "flagged_at" in transaction.metadata
    
    def test_approve_transaction(self):
        """Test approving transaction - O(1)"""
        transaction = self._create_valid_transaction()
        
        transaction.approve()
        
        assert transaction.status == TransactionStatus.APPROVED
        assert "approved_at" in transaction.metadata
    
    def test_approve_critical_risk_transaction(self):
        """Test approving critical risk transaction fails - O(1)"""
        transaction = self._create_valid_transaction()
        transaction.risk_level = RiskLevel.CRITICAL
        
        with pytest.raises(ValueError, match="Cannot approve critical risk transactions"):
            transaction.approve()
    
    def test_is_high_value(self):
        """Test high value transaction detection - O(1)"""
        transaction = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("15000.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow()
        )
        
        threshold = Money(Decimal("10000.00"))
        assert transaction.is_high_value(threshold)
        
        low_threshold = Money(Decimal("20000.00"))
        assert not transaction.is_high_value(low_threshold)
    
    def test_requires_manual_review_high_risk(self):
        """Test manual review requirement for high risk - O(1)"""
        transaction = self._create_valid_transaction()
        transaction.risk_level = RiskLevel.HIGH
        
        assert transaction.requires_manual_review()
    
    def test_requires_manual_review_high_score(self):
        """Test manual review requirement for high score - O(1)"""
        transaction = self._create_valid_transaction()
        transaction.risk_score = 0.8
        
        assert transaction.requires_manual_review()
    
    def test_requires_manual_review_high_value(self):
        """Test manual review requirement for high value - O(1)"""
        transaction = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("15000.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow()
        )
        
        assert transaction.requires_manual_review()
    
    def _create_valid_transaction(self) -> Transaction:
        """Helper to create valid transaction - O(1)"""
        return Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("100.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow()
        )


class TestFraudAnalysisResult:
    """Test FraudAnalysisResult entity"""
    
    def test_valid_fraud_result_creation(self):
        """Test creating valid fraud analysis result - O(1)"""
        tx_id = TransactionId("TXN_123456789")
        result = FraudAnalysisResult(
            transaction_id=tx_id,
            is_fraud=True,
            confidence_score=0.85,
            processing_time_ms=15.5
        )
        
        assert result.transaction_id == tx_id
        assert result.is_fraud is True
        assert result.confidence_score == 0.85
        assert result.processing_time_ms == 15.5
        assert result.risk_factors == []
    
    def test_invalid_confidence_score(self):
        """Test confidence score validation - O(1)"""
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            FraudAnalysisResult(
                transaction_id=TransactionId("TXN_123456789"),
                is_fraud=True,
                confidence_score=1.5
            )
    
    def test_invalid_processing_time(self):
        """Test processing time validation - O(1)"""
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            FraudAnalysisResult(
                transaction_id=TransactionId("TXN_123456789"),
                is_fraud=True,
                confidence_score=0.5,
                processing_time_ms=-10.0
            )
    
    def test_add_risk_factor(self):
        """Test adding risk factors - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=True,
            confidence_score=0.85
        )
        
        result.add_risk_factor("HIGH_VELOCITY")
        result.add_risk_factor("UNUSUAL_LOCATION")
        
        assert "HIGH_VELOCITY" in result.risk_factors
        assert "UNUSUAL_LOCATION" in result.risk_factors
        assert len(result.risk_factors) == 2
    
    def test_add_duplicate_risk_factor(self):
        """Test adding duplicate risk factor - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=True,
            confidence_score=0.85
        )
        
        result.add_risk_factor("HIGH_VELOCITY")
        result.add_risk_factor("HIGH_VELOCITY")  # Duplicate
        
        assert len(result.risk_factors) == 1
    
    def test_get_risk_level_critical(self):
        """Test risk level calculation - critical - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=True,
            confidence_score=0.95
        )
        
        assert result.get_risk_level() == RiskLevel.CRITICAL
    
    def test_get_risk_level_high(self):
        """Test risk level calculation - high - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=True,
            confidence_score=0.75
        )
        
        assert result.get_risk_level() == RiskLevel.HIGH
    
    def test_get_risk_level_medium(self):
        """Test risk level calculation - medium - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=False,
            confidence_score=0.45
        )
        
        assert result.get_risk_level() == RiskLevel.MEDIUM
    
    def test_get_risk_level_low(self):
        """Test risk level calculation - low - O(1)"""
        result = FraudAnalysisResult(
            transaction_id=TransactionId("TXN_123456789"),
            is_fraud=False,
            confidence_score=0.15
        )
        
        assert result.get_risk_level() == RiskLevel.LOW


class TestCustomer:
    """Test Customer entity"""
    
    def test_valid_customer_creation(self):
        """Test creating valid customer - O(1)"""
        customer = Customer(
            id="CUSTOMER_123",
            email="test@example.com",
            created_at=datetime.utcnow()
        )
        
        assert customer.id == "CUSTOMER_123"
        assert customer.email == "test@example.com"
        assert customer.risk_profile == RiskLevel.LOW
        assert customer.get_transaction_count() == 0
    
    def test_invalid_customer_id(self):
        """Test customer ID validation - O(1)"""
        with pytest.raises(ValueError, match="Customer ID is required"):
            Customer(
                id="",
                email="test@example.com",
                created_at=datetime.utcnow()
            )
    
    def test_invalid_email(self):
        """Test email validation - O(1)"""
        with pytest.raises(ValueError, match="Invalid email format"):
            Customer(
                id="CUSTOMER_123",
                email="invalid-email",
                created_at=datetime.utcnow()
            )
    
    def test_add_transaction(self):
        """Test adding transaction to customer - O(1)"""
        customer = Customer(
            id="CUSTOMER_123",
            email="test@example.com",
            created_at=datetime.utcnow()
        )
        
        tx_id = TransactionId("TXN_123456789")
        customer.add_transaction(tx_id)
        
        assert customer.get_transaction_count() == 1
        assert tx_id in customer.transaction_history
    
    def test_update_risk_profile(self):
        """Test updating customer risk profile - O(1)"""
        customer = Customer(
            id="CUSTOMER_123",
            email="test@example.com",
            created_at=datetime.utcnow()
        )
        
        customer.update_risk_profile(RiskLevel.HIGH)
        
        assert customer.risk_profile == RiskLevel.HIGH
        assert "risk_updated_at" in customer.metadata


class TestTransactionFactory:
    """Test TransactionFactory"""
    
    def test_create_transaction(self):
        """Test creating transaction via factory - O(1)"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("100.00"),
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        
        assert isinstance(transaction.id, TransactionId)
        assert transaction.amount.amount == Decimal("100.00")
        assert transaction.amount.currency == "USD"
        assert transaction.merchant_id == "MERCHANT_123"
        assert transaction.customer_id == "CUSTOMER_456"
        assert transaction.status == TransactionStatus.PENDING
    
    def test_create_aggregate(self):
        """Test creating transaction aggregate - O(1)"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("100.00"),
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        
        aggregate = TransactionFactory.create_aggregate(transaction)
        
        assert aggregate.transaction == transaction
        assert len(aggregate.events) == 1  # TransactionCreated event
        assert aggregate.version == 0


class TestTransactionAggregate:
    """Test TransactionAggregate"""
    
    def test_process_fraud_analysis(self):
        """Test processing fraud analysis - O(1)"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("100.00"),
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        
        aggregate = TransactionFactory.create_aggregate(transaction)
        initial_version = aggregate.version
        
        fraud_result = FraudAnalysisResult(
            transaction_id=transaction.id,
            is_fraud=True,
            confidence_score=0.85
        )
        
        aggregate.process_fraud_analysis(fraud_result)
        
        assert aggregate.transaction.risk_score == 0.85
        assert aggregate.transaction.risk_level == RiskLevel.HIGH
        assert aggregate.version == initial_version + 1
        assert len(aggregate.events) == 2  # Original + FraudDetected
    
    def test_approve_transaction(self):
        """Test approving transaction via aggregate - O(1)"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("100.00"),
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        
        aggregate = TransactionFactory.create_aggregate(transaction)
        
        aggregate.approve_transaction("admin_user")
        
        assert aggregate.transaction.status == TransactionStatus.APPROVED
        assert aggregate.version == 1
    
    def test_approve_transaction_requiring_review(self):
        """Test approving transaction that requires review - O(1)"""
        transaction = TransactionFactory.create_transaction(
            amount=Decimal("100.00"),
            currency="USD",
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456"
        )
        transaction.risk_level = RiskLevel.CRITICAL
        
        aggregate = TransactionFactory.create_aggregate(transaction)
        
        with pytest.raises(ValueError, match="Transaction requires manual review"):
            aggregate.approve_transaction("admin_user")


class TestSpecifications:
    """Test Specification pattern implementations"""
    
    def test_high_value_specification(self):
        """Test high value transaction specification - O(1)"""
        threshold = Money(Decimal("1000.00"))
        spec = HighValueTransactionSpec(threshold)
        
        high_value_tx = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("1500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow()
        )
        
        low_value_tx = Transaction(
            id=TransactionId("TXN_987654321"),
            amount=Money(Decimal("500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow()
        )
        
        assert spec.is_satisfied_by(high_value_tx)
        assert not spec.is_satisfied_by(low_value_tx)
    
    def test_high_risk_specification(self):
        """Test high risk transaction specification - O(1)"""
        spec = HighRiskTransactionSpec()
        
        high_risk_tx = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("100.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.HIGH
        )
        
        low_risk_tx = Transaction(
            id=TransactionId("TXN_987654321"),
            amount=Money(Decimal("100.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.LOW
        )
        
        assert spec.is_satisfied_by(high_risk_tx)
        assert not spec.is_satisfied_by(low_risk_tx)
    
    def test_and_specification(self):
        """Test AND specification combinator - O(1)"""
        threshold = Money(Decimal("1000.00"))
        high_value_spec = HighValueTransactionSpec(threshold)
        high_risk_spec = HighRiskTransactionSpec()
        
        combined_spec = high_value_spec.and_(high_risk_spec)
        
        # High value AND high risk
        tx1 = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("1500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.HIGH
        )
        
        # High value but low risk
        tx2 = Transaction(
            id=TransactionId("TXN_987654321"),
            amount=Money(Decimal("1500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.LOW
        )
        
        assert combined_spec.is_satisfied_by(tx1)
        assert not combined_spec.is_satisfied_by(tx2)
    
    def test_or_specification(self):
        """Test OR specification combinator - O(1)"""
        threshold = Money(Decimal("1000.00"))
        high_value_spec = HighValueTransactionSpec(threshold)
        high_risk_spec = HighRiskTransactionSpec()
        
        combined_spec = high_value_spec.or_(high_risk_spec)
        
        # Low value but high risk
        tx1 = Transaction(
            id=TransactionId("TXN_123456789"),
            amount=Money(Decimal("500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.HIGH
        )
        
        # Low value and low risk
        tx2 = Transaction(
            id=TransactionId("TXN_987654321"),
            amount=Money(Decimal("500.00")),
            merchant_id="MERCHANT_123",
            customer_id="CUSTOMER_456",
            timestamp=datetime.utcnow(),
            risk_level=RiskLevel.LOW
        )
        
        assert combined_spec.is_satisfied_by(tx1)
        assert not combined_spec.is_satisfied_by(tx2)