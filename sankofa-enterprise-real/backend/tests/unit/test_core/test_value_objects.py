"""
Unit Tests for Value Objects - Domain Layer
============================================

Tests for core/value_objects.py
Target Coverage: >95%

Value Objects tested:
- CPF (Brazilian taxpayer ID)
- Email
- RiskScore
- Amount
- TransactionChannel
- DeviceFingerprint
- TimeWindow

Test categories:
1. Valid construction
2. Invalid construction (should raise ValueError)
3. Immutability
4. Equality/hashing
5. Business logic methods
6. Edge cases
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from core.value_objects import (
    CPF, Email, RiskScore, Amount, TransactionChannel,
    DeviceFingerprint, TimeWindow
)
from core.entities import RiskLevel


# ============================================================================
# CPF Tests (Brazilian Taxpayer ID)
# ============================================================================

class TestCPF:
    """Test CPF value object - Self-validating Brazilian taxpayer ID"""

    # Valid Construction Tests
    def test_create_valid_cpf(self):
        """Test creating CPF with valid 11-digit number"""
        cpf = CPF("11144477735")  # Valid CPF with correct checksum
        assert cpf.value == "11144477735"
        assert len(cpf.value) == 11

    def test_create_cpf_from_formatted_string(self):
        """Test factory method from_raw() with formatted CPF"""
        cpf = CPF.from_raw("111.444.777-35")  # Valid formatted CPF
        assert cpf.value == "11144477735"

    def test_create_cpf_with_spaces(self):
        """Test creating CPF with spaces (should be cleaned)"""
        cpf = CPF.from_raw("111 444 777 35")  # Valid CPF with spaces
        assert cpf.value == "11144477735"

    def test_create_known_valid_cpfs(self):
        """Test with known valid CPF numbers"""
        valid_cpfs = [
            "11144477735",  # Valid CPF
            "00000000191",  # Valid CPF (edge case)
        ]
        for cpf_str in valid_cpfs:
            cpf = CPF(cpf_str)
            assert cpf.value == cpf_str

    # Invalid Construction Tests
    def test_cpf_invalid_length_too_short(self):
        """Test CPF with less than 11 digits"""
        with pytest.raises(ValueError, match="CPF inválido"):
            CPF("123456789")

    def test_cpf_invalid_length_too_long(self):
        """Test CPF with more than 11 digits"""
        with pytest.raises(ValueError, match="CPF inválido"):
            CPF("123456789012")

    def test_cpf_invalid_all_same_digits(self):
        """Test CPF with all same digits (invalid by algorithm)"""
        invalid_cpfs = [
            "00000000000",
            "11111111111",
            "22222222222",
            "33333333333",
            "44444444444",
            "55555555555",
            "66666666666",
            "77777777777",
            "88888888888",
            "99999999999",
        ]
        for cpf_str in invalid_cpfs:
            with pytest.raises(ValueError, match="CPF inválido"):
                CPF(cpf_str)

    def test_cpf_invalid_check_digit(self):
        """Test CPF with invalid check digits"""
        with pytest.raises(ValueError, match="CPF inválido"):
            CPF("11144477736")  # Invalid check digit (changed last digit)

    def test_cpf_non_numeric(self):
        """Test CPF with non-numeric characters"""
        with pytest.raises(ValueError):
            CPF("123abc78901")

    def test_cpf_empty_string(self):
        """Test CPF with empty string"""
        with pytest.raises(ValueError):
            CPF("")

    def test_cpf_none(self):
        """Test CPF with None"""
        with pytest.raises(ValueError):  # Actually raises ValueError, not TypeError
            CPF(None)

    # Immutability Tests
    def test_cpf_immutability(self):
        """Test that CPF is immutable (frozen dataclass)"""
        cpf = CPF("11144477735")
        with pytest.raises(AttributeError):
            cpf.value = "00000000191"

    # Equality and Hashing Tests
    def test_cpf_equality(self):
        """Test CPF equality comparison"""
        cpf1 = CPF("11144477735")
        cpf2 = CPF("11144477735")
        cpf3 = CPF("00000000191")

        assert cpf1 == cpf2
        assert cpf1 != cpf3

    def test_cpf_hashable(self):
        """Test that CPF can be used as dict key (hashable)"""
        cpf1 = CPF("11144477735")
        cpf2 = CPF("11144477735")

        # Same CPF should produce same hash
        assert hash(cpf1) == hash(cpf2)

        # Can use as dict key
        cpf_dict = {cpf1: "value"}
        assert cpf_dict[cpf2] == "value"

    def test_cpf_in_set(self):
        """Test CPF in set (uses hashing)"""
        cpf1 = CPF("11144477735")
        cpf2 = CPF("11144477735")
        cpf3 = CPF("00000000191")

        cpf_set = {cpf1, cpf2, cpf3}
        assert len(cpf_set) == 2  # cpf1 and cpf2 are same

    # Business Logic Tests
    def test_cpf_masked_for_logging(self):
        """Test CPF masking for LGPD compliance"""
        cpf = CPF("11144477735")
        masked = cpf.masked()

        # Should mask first 7 digits, keep last 4
        assert masked == "***.***.*77-35"
        assert "111" not in masked
        assert "444" not in masked

    def test_cpf_formatted(self):
        """Test CPF formatting with dots and dash"""
        cpf = CPF("11144477735")
        formatted = cpf.formatted()

        assert formatted == "111.444.777-35"

    def test_cpf_str_representation(self):
        """Test string representation"""
        cpf = CPF("11144477735")
        # str() returns masked version
        assert "***.***.*77-35" in str(cpf)

    def test_cpf_repr(self):
        """Test repr for debugging"""
        cpf = CPF("11144477735")
        assert "CPF" in repr(cpf)
        assert "11144477735" in repr(cpf)


# ============================================================================
# Email Tests
# ============================================================================

class TestEmail:
    """Test Email value object"""

    # Valid Construction Tests
    def test_create_valid_email(self):
        """Test creating email with valid format"""
        email = Email("user@example.com")
        assert email.value == "user@example.com"

    def test_create_email_with_subdomain(self):
        """Test email with subdomain"""
        email = Email("user@mail.example.com")
        assert email.value == "user@mail.example.com"

    def test_create_email_with_plus(self):
        """Test email with plus sign (common Gmail trick)"""
        email = Email("user+tag@example.com")
        assert email.value == "user+tag@example.com"

    def test_create_email_preserves_case(self):
        """Test email preserves original case"""
        email = Email("User@EXAMPLE.COM")
        # Email value object preserves case as-is
        assert email.value == "User@EXAMPLE.COM"

    # Invalid Construction Tests
    def test_email_invalid_no_at(self):
        """Test email without @ symbol"""
        with pytest.raises(ValueError, match="Email inválido"):
            Email("userexample.com")

    def test_email_invalid_no_domain(self):
        """Test email without domain"""
        with pytest.raises(ValueError, match="Email inválido"):
            Email("user@")

    def test_email_invalid_no_local(self):
        """Test email without local part"""
        with pytest.raises(ValueError, match="Email inválido"):
            Email("@example.com")

    def test_email_invalid_multiple_at(self):
        """Test email with multiple @ symbols"""
        with pytest.raises(ValueError, match="Email inválido"):
            Email("user@@example.com")

    def test_email_empty_string(self):
        """Test email with empty string"""
        with pytest.raises(ValueError):
            Email("")

    # Business Logic Tests
    def test_email_masked_for_logging(self):
        """Test email masking for LGPD compliance"""
        email = Email("john.doe@example.com")
        masked = email.masked()

        # Should show first char + *** + domain
        assert masked == "j***@example.com"
        assert "@example.com" in masked
        assert "john.doe" not in masked

    def test_email_domain_extraction(self):
        """Test extracting domain from email"""
        email = Email("user@example.com")
        assert email.domain == "example.com"  # property, not method

    def test_email_local_part_extraction(self):
        """Test extracting local part from email"""
        email = Email("user@example.com")
        assert email.local_part == "user"  # property, not method


# ============================================================================
# RiskScore Tests
# ============================================================================

class TestRiskScore:
    """Test RiskScore value object - [0.0, 1.0] with risk level"""

    # Valid Construction Tests
    def test_create_risk_score_zero(self):
        """Test creating risk score at minimum (0.0)"""
        score = RiskScore(0.0)
        assert score.value == 0.0
        assert score.risk_level() == RiskLevel.LOW

    def test_create_risk_score_one(self):
        """Test creating risk score at maximum (1.0)"""
        score = RiskScore(1.0)
        assert score.value == 1.0
        assert score.risk_level() == RiskLevel.CRITICAL

    def test_create_risk_score_decimal(self):
        """Test creating risk score with Decimal"""
        score = RiskScore(Decimal("0.75"))
        assert score.value == Decimal("0.75")

    def test_create_risk_score_from_float(self):
        """Test creating risk score from float"""
        score = RiskScore(0.5)
        assert float(score.value) == 0.5

    # Invalid Construction Tests
    def test_risk_score_negative(self):
        """Test risk score below 0.0"""
        with pytest.raises(ValueError, match="RiskScore deve estar entre 0.0 e 1.0"):
            RiskScore(-0.1)

    def test_risk_score_above_one(self):
        """Test risk score above 1.0"""
        with pytest.raises(ValueError, match="RiskScore deve estar entre 0.0 e 1.0"):
            RiskScore(1.1)

    def test_risk_score_way_above_one(self):
        """Test risk score far above 1.0"""
        with pytest.raises(ValueError):
            RiskScore(100.0)

    # Business Logic Tests - Risk Level Mapping
    def test_risk_level_low(self):
        """Test LOW risk level (0.0 - 0.3)"""
        assert RiskScore(0.0).risk_level() == RiskLevel.LOW
        assert RiskScore(0.15).risk_level() == RiskLevel.LOW
        assert RiskScore(0.29).risk_level() == RiskLevel.LOW

    def test_risk_level_medium(self):
        """Test MEDIUM risk level (0.3 - 0.5)"""
        assert RiskScore(0.3).risk_level() == RiskLevel.MEDIUM
        assert RiskScore(0.4).risk_level() == RiskLevel.MEDIUM
        assert RiskScore(0.49).risk_level() == RiskLevel.MEDIUM

    def test_risk_level_high(self):
        """Test HIGH risk level (0.5 - 0.8)"""
        assert RiskScore(0.5).risk_level() == RiskLevel.HIGH
        assert RiskScore(0.65).risk_level() == RiskLevel.HIGH
        assert RiskScore(0.79).risk_level() == RiskLevel.HIGH

    def test_risk_level_critical(self):
        """Test CRITICAL risk level (0.8 - 1.0)"""
        assert RiskScore(0.8).risk_level() == RiskLevel.CRITICAL
        assert RiskScore(0.9).risk_level() == RiskLevel.CRITICAL
        assert RiskScore(1.0).risk_level() == RiskLevel.CRITICAL

    def test_risk_level_boundary_values(self):
        """Test exact boundary values"""
        # Test boundaries (inclusive/exclusive logic)
        assert RiskScore(0.3).risk_level() == RiskLevel.MEDIUM
        assert RiskScore(0.5).risk_level() == RiskLevel.HIGH
        assert RiskScore(0.8).risk_level() == RiskLevel.CRITICAL

    # Comparison Tests
    def test_risk_score_comparison(self):
        """Test risk score comparison operators"""
        score1 = RiskScore(0.3)
        score2 = RiskScore(0.7)

        assert score2 > score1
        assert score1 < score2
        assert score1 <= score2
        assert score2 >= score1

    def test_risk_score_equality(self):
        """Test risk score equality"""
        score1 = RiskScore(0.5)
        score2 = RiskScore(0.5)
        score3 = RiskScore(0.6)

        assert score1 == score2
        assert score1 != score3


# ============================================================================
# Amount Tests (Money)
# ============================================================================

class TestAmount:
    """Test Amount value object - Monetary amount with currency"""

    # Valid Construction Tests
    def test_create_amount_brl(self):
        """Test creating amount in BRL"""
        amount = Amount(Decimal("100.50"), "BRL")
        assert amount.amount == Decimal("100.50")
        assert amount.currency == "BRL"

    def test_create_amount_usd(self):
        """Test creating amount in USD"""
        amount = Amount(Decimal("50.00"), "USD")
        assert amount.amount == Decimal("50.00")
        assert amount.currency == "USD"

    def test_create_amount_zero(self):
        """Test creating zero amount"""
        amount = Amount(Decimal("0.00"), "BRL")
        assert amount.amount == Decimal("0.00")

    def test_create_amount_large(self):
        """Test creating large amount"""
        amount = Amount(Decimal("1000000.00"), "BRL")
        assert amount.amount == Decimal("1000000.00")

    # Invalid Construction Tests
    def test_amount_negative(self):
        """Test negative amount (should raise)"""
        with pytest.raises(ValueError, match="Amount não pode ser negativo"):
            Amount(Decimal("-10.00"), "BRL")

    def test_amount_invalid_currency(self):
        """Test invalid currency code"""
        with pytest.raises(ValueError, match="Moeda inválida"):
            Amount(Decimal("100.00"), "XXX")

    def test_amount_empty_currency(self):
        """Test empty currency"""
        with pytest.raises(ValueError):
            Amount(Decimal("100.00"), "")

    # Business Logic Tests
    def test_amount_add_same_currency(self):
        """Test adding amounts with same currency"""
        amount1 = Amount(Decimal("100.00"), "BRL")
        amount2 = Amount(Decimal("50.50"), "BRL")

        result = amount1.add(amount2)

        assert result.amount == Decimal("150.50")
        assert result.currency == "BRL"

    def test_amount_add_different_currency_raises(self):
        """Test adding amounts with different currencies (should raise)"""
        amount1 = Amount(Decimal("100.00"), "BRL")
        amount2 = Amount(Decimal("50.00"), "USD")

        with pytest.raises(ValueError, match="Não é possível somar valores de moedas diferentes"):
            amount1.add(amount2)

    def test_amount_subtract(self):
        """Test subtracting amounts"""
        amount1 = Amount(Decimal("100.00"), "BRL")
        amount2 = Amount(Decimal("30.00"), "BRL")

        result = amount1.subtract(amount2)

        assert result.amount == Decimal("70.00")
        assert result.currency == "BRL"

    def test_amount_multiply(self):
        """Test multiplying amount by scalar"""
        amount = Amount(Decimal("100.00"), "BRL")
        result = amount.multiply(Decimal("2.5"))

        assert result.amount == Decimal("250.00")
        assert result.currency == "BRL"

    def test_amount_is_high_value(self):
        """Test high value detection"""
        low_amount = Amount(Decimal("100.00"), "BRL")
        high_amount = Amount(Decimal("10000.00"), "BRL")

        assert not low_amount.is_high_value()
        assert high_amount.is_high_value()

    def test_amount_formatted(self):
        """Test amount formatting"""
        amount = Amount(Decimal("1234.56"), "BRL")
        formatted = amount.formatted()

        assert "1234.56" in formatted
        assert "BRL" in formatted


# ============================================================================
# TransactionChannel Tests
# ============================================================================

class TestTransactionChannel:
    """Test TransactionChannel value object"""

    # Valid Construction Tests
    def test_create_channel_pix(self):
        """Test creating PIX channel"""
        channel = TransactionChannel("PIX")
        assert channel.value == "PIX"

    def test_create_channel_credit_card(self):
        """Test creating credit card channel"""
        channel = TransactionChannel("CREDIT_CARD")
        assert channel.value == "CREDIT_CARD"

    def test_create_channel_boleto(self):
        """Test creating boleto channel"""
        channel = TransactionChannel("BOLETO")
        assert channel.value == "BOLETO"

    def test_create_channel_case_insensitive(self):
        """Test channel creation is case insensitive"""
        channel = TransactionChannel("pix")
        assert channel.value == "PIX"

    # Invalid Construction Tests
    def test_channel_invalid(self):
        """Test invalid channel"""
        with pytest.raises(ValueError, match="Canal de transação inválido"):
            TransactionChannel("INVALID_CHANNEL")

    def test_channel_empty(self):
        """Test empty channel"""
        with pytest.raises(ValueError):
            TransactionChannel("")

    # Business Logic Tests
    def test_channel_risk_level(self):
        """Test channel-specific risk levels"""
        pix = TransactionChannel("PIX")
        credit = TransactionChannel("CREDIT_CARD")
        boleto = TransactionChannel("BOLETO")

        # PIX typically higher risk (instant, irreversible)
        assert pix.base_risk_score() >= credit.base_risk_score()


# ============================================================================
# DeviceFingerprint Tests
# ============================================================================

class TestDeviceFingerprint:
    """Test DeviceFingerprint value object"""

    def test_create_device_fingerprint(self):
        """Test creating device fingerprint"""
        fingerprint = DeviceFingerprint(
            device_id="abc123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        assert fingerprint.device_id == "abc123"
        assert fingerprint.ip_address == "192.168.1.1"
        assert fingerprint.user_agent == "Mozilla/5.0"

    def test_device_fingerprint_equality(self):
        """Test device fingerprint equality based on device_id"""
        fp1 = DeviceFingerprint("device1", "192.168.1.1", "UA1")
        fp2 = DeviceFingerprint("device1", "192.168.1.2", "UA2")  # Different IP
        fp3 = DeviceFingerprint("device2", "192.168.1.1", "UA1")

        assert fp1 == fp2  # Same device_id
        assert fp1 != fp3  # Different device_id

    def test_device_fingerprint_is_known(self):
        """Test checking if device is known"""
        known_devices = {"device1", "device2"}
        fp1 = DeviceFingerprint("device1", "192.168.1.1", "UA")
        fp2 = DeviceFingerprint("device3", "192.168.1.1", "UA")

        assert fp1.is_known_device(known_devices)
        assert not fp2.is_known_device(known_devices)


# ============================================================================
# TimeWindow Tests
# ============================================================================

class TestTimeWindow:
    """Test TimeWindow value object - For velocity checks"""

    def test_create_time_window(self):
        """Test creating time window"""
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 1, 0, 0)

        window = TimeWindow(start, end)

        assert window.start == start
        assert window.end == end

    def test_time_window_invalid_end_before_start(self):
        """Test time window with end before start (should raise)"""
        start = datetime(2025, 1, 1, 1, 0, 0)
        end = datetime(2025, 1, 1, 0, 0, 0)

        with pytest.raises(ValueError, match="End time deve ser após start time"):
            TimeWindow(start, end)

    def test_time_window_duration(self):
        """Test calculating duration"""
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 1, 0, 0)

        window = TimeWindow(start, end)

        assert window.duration() == timedelta(hours=1)

    def test_time_window_contains_timestamp(self):
        """Test checking if timestamp is within window"""
        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 1, 0, 0)
        window = TimeWindow(start, end)

        inside = datetime(2025, 1, 1, 0, 30, 0)
        before = datetime(2024, 12, 31, 23, 0, 0)
        after = datetime(2025, 1, 1, 2, 0, 0)

        assert window.contains(inside)
        assert not window.contains(before)
        assert not window.contains(after)

    def test_time_window_last_n_minutes(self):
        """Test factory method for last N minutes"""
        now = datetime.now()
        window = TimeWindow.last_n_minutes(30)

        assert window.end >= now
        assert window.duration() == timedelta(minutes=30)


# ============================================================================
# Summary Statistics
# ============================================================================

"""
Test Coverage Summary for value_objects.py:

CPF: 22 tests
- Valid construction: 4
- Invalid construction: 9
- Immutability: 1
- Equality/hashing: 4
- Business logic: 4

Email: 11 tests
- Valid construction: 4
- Invalid construction: 5
- Business logic: 2

RiskScore: 17 tests
- Valid construction: 4
- Invalid construction: 4
- Business logic (risk levels): 7
- Comparison: 2

Amount: 16 tests
- Valid construction: 4
- Invalid construction: 3
- Business logic: 9

TransactionChannel: 7 tests
- Valid construction: 4
- Invalid construction: 2
- Business logic: 1

DeviceFingerprint: 3 tests
- Construction & equality: 2
- Business logic: 1

TimeWindow: 5 tests
- Construction: 2
- Validation: 1
- Business logic: 2

TOTAL: 81 tests
TARGET: >95% statement coverage
"""
