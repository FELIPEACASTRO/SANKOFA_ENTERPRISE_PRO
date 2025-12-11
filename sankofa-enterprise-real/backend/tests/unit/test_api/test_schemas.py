"""
Unit Tests for Pydantic Schemas
Tests all input validation schemas for API endpoints
"""

import pytest
from pydantic import ValidationError
from datetime import datetime, timedelta

from api.schemas import (
    TransactionRequest,
    FraudPredictionBatchRequest,
    HardRuleCreate,
    HardRuleUpdate,
    VipListCreate,
    HotListCreate,
    UserLogin,
    ManualReviewCreate,
    FeedbackCreate,
    SettingsUpdate,
    InvestigationCreate,
    DSRAccessRequest,
    DSRDeletionRequest,
    validate_sql_fields,
)


# =============================================================================
# TRANSACTIONREQUEST TESTS
# =============================================================================

class TestTransactionRequest:
    """Tests for TransactionRequest schema"""

    def test_valid_transaction(self):
        """Test valid transaction request"""
        data = {
            'amount': 1000.0,
            'cpf': '12345678909',
            'channel': 'PIX',
        }

        txn = TransactionRequest(**data)

        assert txn.amount == 1000.0
        assert txn.cpf == '12345678909'
        assert txn.channel == 'PIX'

    def test_cpf_validation_11_digits(self):
        """Test CPF must be exactly 11 digits"""
        # Valid
        txn = TransactionRequest(amount=100, cpf='12345678909', channel='PIX')
        assert txn.cpf == '12345678909'

        # Invalid - too short
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(amount=100, cpf='123', channel='PIX')

        errors = exc_info.value.errors()
        assert any('cpf' in str(e['loc']) for e in errors)

    def test_cpf_check_digits(self):
        """Test CPF check digit validation"""
        # Valid CPF with correct check digits
        txn = TransactionRequest(amount=100, cpf='12345678909', channel='PIX')
        assert txn.cpf == '12345678909'

        # Invalid - all same digits
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(amount=100, cpf='11111111111', channel='PIX')

        errors = exc_info.value.errors()
        assert any('cpf' in str(e['loc']) for e in errors)

    def test_amount_validation(self):
        """Test amount must be positive and within range"""
        # Valid amount
        txn = TransactionRequest(amount=500.0, cpf='12345678909', channel='PIX')
        assert txn.amount == 500.0

        # Invalid - negative amount
        with pytest.raises(ValidationError):
            TransactionRequest(amount=-100.0, cpf='12345678909', channel='PIX')

        # Invalid - zero amount
        with pytest.raises(ValidationError):
            TransactionRequest(amount=0.0, cpf='12345678909', channel='PIX')

        # Invalid - exceeds max
        with pytest.raises(ValidationError):
            TransactionRequest(amount=2000000.0, cpf='12345678909', channel='PIX')

    def test_channel_validation(self):
        """Test channel must be valid enum"""
        valid_channels = ['PIX', 'TED', 'DOC', 'BOLETO', 'CARTAO_CREDITO', 'CARTAO_DEBITO', 'APP', 'WEB', 'ATM']

        for channel in valid_channels:
            txn = TransactionRequest(amount=100, cpf='12345678909', channel=channel)
            assert txn.channel == channel

        # Invalid channel
        with pytest.raises(ValidationError):
            TransactionRequest(amount=100, cpf='12345678909', channel='INVALID_CHANNEL')

    def test_optional_fields(self):
        """Test optional fields work correctly"""
        data = {
            'amount': 1000.0,
            'cpf': '12345678909',
            'channel': 'PIX',
            'location': 'São Paulo',
            'device_id': 'device_123',
            'ip_address': '192.168.1.1',
        }

        txn = TransactionRequest(**data)

        assert txn.location == 'São Paulo'
        assert txn.device_id == 'device_123'
        assert txn.ip_address == '192.168.1.1'

    def test_ip_address_validation(self):
        """Test IP address format validation"""
        # Valid IP
        txn = TransactionRequest(
            amount=100,
            cpf='12345678909',
            channel='PIX',
            ip_address='192.168.1.1'
        )
        assert txn.ip_address == '192.168.1.1'

        # Invalid IP
        with pytest.raises(ValidationError):
            TransactionRequest(
                amount=100,
                cpf='12345678909',
                channel='PIX',
                ip_address='invalid_ip'
            )


# =============================================================================
# FRAUDPREDICTIONBATCHREQUEST TESTS
# =============================================================================

class TestFraudPredictionBatchRequest:
    """Tests for FraudPredictionBatchRequest schema"""

    def test_valid_batch_request(self):
        """Test valid batch prediction request"""
        data = {
            'transactions': [
                {'amount': 100, 'cpf': '12345678909', 'channel': 'PIX'},
                {'amount': 200, 'cpf': '98765432100', 'channel': 'TED'},
            ],
            'include_explanation': True,
            'fast_mode': True,
        }

        batch = FraudPredictionBatchRequest(**data)

        assert len(batch.transactions) == 2
        assert batch.include_explanation is True
        assert batch.fast_mode is True

    def test_empty_transactions_fails(self):
        """Test empty transactions array fails validation"""
        with pytest.raises(ValidationError):
            FraudPredictionBatchRequest(transactions=[])

    def test_batch_size_limits(self):
        """Test batch size within limits"""
        # Valid - within limit
        transactions = [
            {'amount': 100, 'cpf': '12345678909', 'channel': 'PIX'}
            for _ in range(100)
        ]
        batch = FraudPredictionBatchRequest(transactions=transactions)
        assert len(batch.transactions) == 100

        # Invalid - exceeds max
        transactions = [
            {'amount': 100, 'cpf': '12345678909', 'channel': 'PIX'}
            for _ in range(1001)
        ]
        with pytest.raises(ValidationError):
            FraudPredictionBatchRequest(transactions=transactions)


# =============================================================================
# HARDRULECREATE TESTS
# =============================================================================

class TestHardRuleCreate:
    """Tests for HardRuleCreate schema"""

    def test_valid_hard_rule(self):
        """Test valid hard rule creation"""
        data = {
            'name': 'Test Rule',
            'description': 'Test rule for validation',
            'condition': 'amount > 5000',
            'action': 'block',
            'priority': 1,
            'enabled': True,
        }

        rule = HardRuleCreate(**data)

        assert rule.name == 'Test Rule'
        assert rule.action == 'block'
        assert rule.priority == 1
        assert rule.enabled is True

    def test_action_validation(self):
        """Test action must be valid enum"""
        valid_actions = ['block', 'allow', 'review', 'score']

        for action in valid_actions:
            rule = HardRuleCreate(
                name='Test',
                description='Test',
                condition='amount > 100',
                action=action,
                priority=1,
                enabled=True
            )
            assert rule.action == action

        # Invalid action
        with pytest.raises(ValidationError):
            HardRuleCreate(
                name='Test',
                description='Test',
                condition='amount > 100',
                action='invalid_action',
                priority=1,
                enabled=True
            )

    def test_priority_validation(self):
        """Test priority must be positive"""
        # Valid priority
        rule = HardRuleCreate(
            name='Test',
            description='Test',
            condition='amount > 100',
            action='block',
            priority=5,
            enabled=True
        )
        assert rule.priority == 5

        # Invalid - negative priority
        with pytest.raises(ValidationError):
            HardRuleCreate(
                name='Test',
                description='Test',
                condition='amount > 100',
                action='block',
                priority=-1,
                enabled=True
            )

    def test_conditions_json_validation(self):
        """Test conditions_json structure"""
        data = {
            'name': 'Test',
            'description': 'Test',
            'condition': 'amount > 5000',
            'conditions_json': [
                {'field': 'amount', 'operator': '>', 'value': 5000}
            ],
            'action': 'block',
            'priority': 1,
            'enabled': True
        }

        rule = HardRuleCreate(**data)
        assert len(rule.conditions_json) == 1
        assert rule.conditions_json[0]['field'] == 'amount'


# =============================================================================
# USERLOGIN TESTS
# =============================================================================

class TestUserLogin:
    """Tests for UserLogin schema"""

    def test_valid_login(self):
        """Test valid login request"""
        data = {
            'username': 'test_user',
            'password': 'SecurePassword123!'
        }

        login = UserLogin(**data)

        assert login.username == 'test_user'
        assert login.password == 'SecurePassword123!'

    def test_username_required(self):
        """Test username is required"""
        with pytest.raises(ValidationError):
            UserLogin(password='password')

    def test_password_required(self):
        """Test password is required"""
        with pytest.raises(ValidationError):
            UserLogin(username='test_user')

    def test_username_length(self):
        """Test username length validation"""
        # Too short
        with pytest.raises(ValidationError):
            UserLogin(username='ab', password='password')

        # Valid
        login = UserLogin(username='abc', password='password')
        assert login.username == 'abc'


# =============================================================================
# VIP/HOT LIST TESTS
# =============================================================================

class TestVipListCreate:
    """Tests for VipListCreate schema"""

    def test_valid_vip_entry(self):
        """Test valid VIP list entry"""
        data = {
            'cpf': '12345678909',
            'reason': 'Premium customer',
        }

        vip = VipListCreate(**data)

        assert vip.cpf == '12345678909'
        assert vip.reason == 'Premium customer'

    def test_cpf_required(self):
        """Test CPF is required"""
        with pytest.raises(ValidationError):
            VipListCreate(reason='Premium customer')

    def test_reason_required(self):
        """Test reason is required"""
        with pytest.raises(ValidationError):
            VipListCreate(cpf='12345678909')


class TestHotListCreate:
    """Tests for HotListCreate schema"""

    def test_valid_hot_entry(self):
        """Test valid hot list entry"""
        data = {
            'cpf': '12345678909',
            'reason': 'Confirmed fraudster',
            'severity': 'HIGH',
        }

        hot = HotListCreate(**data)

        assert hot.cpf == '12345678909'
        assert hot.severity == 'HIGH'

    def test_severity_validation(self):
        """Test severity enum validation"""
        valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

        for severity in valid_severities:
            hot = HotListCreate(
                cpf='12345678909',
                reason='Test',
                severity=severity
            )
            assert hot.severity == severity

        # Invalid severity
        with pytest.raises(ValidationError):
            HotListCreate(
                cpf='12345678909',
                reason='Test',
                severity='INVALID'
            )


# =============================================================================
# DSR (LGPD) TESTS
# =============================================================================

class TestDSRAccessRequest:
    """Tests for DSR access request (LGPD Art. 18, I)"""

    def test_valid_dsr_access(self):
        """Test valid DSR access request"""
        data = {
            'cpf': '12345678909',
            'request_reason': 'I want to access my personal data stored in the system',
            'requester_email': 'user@example.com'
        }

        dsr = DSRAccessRequest(**data)

        assert dsr.cpf == '12345678909'
        assert 'personal data' in dsr.request_reason
        assert dsr.requester_email == 'user@example.com'

    def test_reason_min_length(self):
        """Test request reason must be at least 20 characters"""
        # Too short
        with pytest.raises(ValidationError):
            DSRAccessRequest(
                cpf='12345678909',
                request_reason='Too short',
                requester_email='user@example.com'
            )

        # Valid
        dsr = DSRAccessRequest(
            cpf='12345678909',
            request_reason='This is a valid reason with sufficient length',
            requester_email='user@example.com'
        )
        assert dsr.request_reason

    def test_email_validation(self):
        """Test email format validation"""
        # Invalid email
        with pytest.raises(ValidationError):
            DSRAccessRequest(
                cpf='12345678909',
                request_reason='Valid reason with sufficient length here',
                requester_email='invalid_email'
            )


# =============================================================================
# SQL INJECTION PREVENTION TESTS
# =============================================================================

class TestValidateSQLFields:
    """Tests for SQL field whitelist validation"""

    def test_valid_fields_accepted(self):
        """Test valid fields pass validation"""
        allowed = {'amount', 'channel', 'cpf', 'status'}
        fields = ['amount', 'channel']

        validated = validate_sql_fields(fields, allowed)

        assert validated == ['amount', 'channel']

    def test_invalid_fields_rejected(self):
        """Test invalid fields raise ValueError"""
        allowed = {'amount', 'channel'}
        fields = ['amount', 'malicious_field']

        with pytest.raises(ValueError) as exc_info:
            validate_sql_fields(fields, allowed)

        assert 'Invalid fields' in str(exc_info.value)
        assert 'malicious_field' in str(exc_info.value)

    def test_sql_injection_attempt_blocked(self):
        """Test SQL injection attempts are blocked"""
        allowed = {'amount', 'channel'}
        malicious_fields = [
            'amount',
            "'; DROP TABLE users; --"
        ]

        with pytest.raises(ValueError):
            validate_sql_fields(malicious_fields, allowed)


# =============================================================================
# SETTINGS UPDATE TESTS
# =============================================================================

class TestSettingsUpdate:
    """Tests for SettingsUpdate schema"""

    def test_valid_settings_update(self):
        """Test valid settings update"""
        data = {
            'fraud_threshold': 0.7,
            'auto_block_enabled': True,
            'email_notifications': False,
        }

        settings = SettingsUpdate(**data)

        assert settings.fraud_threshold == 0.7
        assert settings.auto_block_enabled is True
        assert settings.email_notifications is False

    def test_fraud_threshold_range(self):
        """Test fraud_threshold must be between 0 and 1"""
        # Valid
        settings = SettingsUpdate(fraud_threshold=0.5)
        assert settings.fraud_threshold == 0.5

        # Invalid - below range
        with pytest.raises(ValidationError):
            SettingsUpdate(fraud_threshold=-0.1)

        # Invalid - above range
        with pytest.raises(ValidationError):
            SettingsUpdate(fraud_threshold=1.5)

    def test_optional_fields(self):
        """Test all fields are optional"""
        # Empty update should be valid
        settings = SettingsUpdate()
        assert settings.fraud_threshold is None
        assert settings.auto_block_enabled is None


# =============================================================================
# INVESTIGATION CREATE TESTS
# =============================================================================

class TestInvestigationCreate:
    """Tests for InvestigationCreate schema"""

    def test_valid_investigation(self):
        """Test valid investigation creation"""
        data = {
            'transaction_id': 'TXN_12345',
            'investigation_type': 'FRAUD',
            'description': 'Suspicious activity detected on this transaction',
            'priority': 'HIGH',
        }

        inv = InvestigationCreate(**data)

        assert inv.transaction_id == 'TXN_12345'
        assert inv.investigation_type == 'FRAUD'
        assert inv.priority == 'HIGH'

    def test_investigation_type_validation(self):
        """Test investigation_type enum"""
        valid_types = ['FRAUD', 'COMPLIANCE', 'SECURITY', 'OTHER']

        for inv_type in valid_types:
            inv = InvestigationCreate(
                transaction_id='TXN_123',
                investigation_type=inv_type,
                description='Valid description with sufficient length',
                priority='MEDIUM'
            )
            assert inv.investigation_type == inv_type

    def test_priority_validation(self):
        """Test priority enum"""
        valid_priorities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

        for priority in valid_priorities:
            inv = InvestigationCreate(
                transaction_id='TXN_123',
                investigation_type='FRAUD',
                description='Valid description with sufficient length',
                priority=priority
            )
            assert inv.priority == priority


# =============================================================================
# FEEDBACK CREATE TESTS
# =============================================================================

class TestFeedbackCreate:
    """Tests for FeedbackCreate schema"""

    def test_valid_feedback(self):
        """Test valid feedback submission"""
        data = {
            'transaction_id': 'TXN_123',
            'feedback_type': 'correction',
            'correct_label': 'fraud',
            'comments': 'This was actually a fraud case'
        }

        feedback = FeedbackCreate(**data)

        assert feedback.transaction_id == 'TXN_123'
        assert feedback.feedback_type == 'correction'
        assert feedback.correct_label == 'fraud'

    def test_feedback_type_validation(self):
        """Test feedback_type enum"""
        valid_types = ['correction', 'comment', 'quality']

        for fb_type in valid_types:
            feedback = FeedbackCreate(
                transaction_id='TXN_123',
                feedback_type=fb_type,
                correct_label='fraud'
            )
            assert feedback.feedback_type == fb_type

    def test_correct_label_validation(self):
        """Test correct_label enum"""
        valid_labels = ['fraud', 'legitimate', 'uncertain']

        for label in valid_labels:
            feedback = FeedbackCreate(
                transaction_id='TXN_123',
                feedback_type='correction',
                correct_label=label
            )
            assert feedback.correct_label == label
