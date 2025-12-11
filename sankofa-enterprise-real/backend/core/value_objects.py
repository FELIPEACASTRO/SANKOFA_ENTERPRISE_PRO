"""
Core Value Objects - Domain-Driven Design
Immutable, self-validating value objects encapsulating domain validation logic
Time Complexity: O(1) for all validations
"""

from dataclasses import dataclass
from typing import ClassVar, Optional
import re
from decimal import Decimal
from datetime import datetime


@dataclass(frozen=True)
class CPF:
    """
    Value Object: CPF Brasileiro

    Encapsulates CPF validation logic in one place (DRY principle)
    Currently duplicated in 15+ places across production_api.py

    Benefits:
    - Immutable (frozen=True)
    - Self-validating (__post_init__)
    - Reusable across all layers
    - Type-safe (CPF vs str)

    Time Complexity: O(1) for all operations
    """

    value: str

    _REGEX: ClassVar = re.compile(r'^\d{11}$')
    _BLACKLIST: ClassVar = {
        '00000000000', '11111111111', '22222222222', '33333333333',
        '44444444444', '55555555555', '66666666666', '77777777777',
        '88888888888', '99999999999'
    }

    def __post_init__(self):
        """Validate CPF on creation - O(1)"""
        if not self._is_valid(self.value):
            raise ValueError(f"CPF inválido: {self.value}")

    @classmethod
    def from_raw(cls, raw: str) -> 'CPF':
        """
        Factory method: Create CPF from formatted string

        Examples:
            CPF.from_raw("123.456.789-09")  # Returns CPF("12345678909")
            CPF.from_raw("12345678909")     # Returns CPF("12345678909")

        Time Complexity: O(n) where n is string length (max 14)
        """
        # Remove formatting characters
        cleaned = re.sub(r'\D', '', raw)
        return cls(cleaned)

    def masked(self) -> str:
        """
        Return masked CPF for logging/display (LGPD compliance)

        Example: ***.***.*67-89

        Time Complexity: O(1)
        """
        return f"***.***.*{self.value[7:9]}-{self.value[9:]}"

    def formatted(self) -> str:
        """
        Return formatted CPF: 123.456.789-09

        Time Complexity: O(1)
        """
        return f"{self.value[:3]}.{self.value[3:6]}.{self.value[6:9]}-{self.value[9:]}"

    @staticmethod
    def _is_valid(cpf: str) -> bool:
        """
        Validate CPF using official algorithm

        Time Complexity: O(1) - fixed 11 digits
        """
        if not cpf or len(cpf) != 11:
            return False

        if not cpf.isdigit():
            return False

        if cpf in CPF._BLACKLIST:
            return False

        # Calculate first check digit
        sum1 = sum(int(cpf[i]) * (10 - i) for i in range(9))
        digit1 = (sum1 * 10 % 11) % 10

        if int(cpf[9]) != digit1:
            return False

        # Calculate second check digit
        sum2 = sum(int(cpf[i]) * (11 - i) for i in range(10))
        digit2 = (sum2 * 10 % 11) % 10

        return int(cpf[10]) == digit2

    def __str__(self) -> str:
        return self.masked()  # Always mask by default for security


@dataclass(frozen=True)
class Email:
    """
    Value Object: Email address

    Simple email validation with basic regex
    For production, consider using email-validator library

    Time Complexity: O(n) where n is email length
    """

    value: str

    _REGEX: ClassVar = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    def __post_init__(self):
        """Validate email on creation - O(n)"""
        if not self._REGEX.match(self.value):
            raise ValueError(f"Email inválido: {self.value}")

        if len(self.value) > 254:  # RFC 5321
            raise ValueError("Email muito longo (max 254 caracteres)")

    @property
    def domain(self) -> str:
        """Extract domain from email - O(n)"""
        return self.value.split('@')[1]

    @property
    def local_part(self) -> str:
        """Extract local part from email - O(n)"""
        return self.value.split('@')[0]

    def masked(self) -> str:
        """
        Return masked email for logging (LGPD compliance)

        Example: j***@example.com

        Time Complexity: O(n)
        """
        local, domain = self.value.split('@')
        masked_local = f"{local[0]}***" if len(local) > 0 else "***"
        return f"{masked_local}@{domain}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RiskScore:
    """
    Value Object: Risk score between 0.0 and 1.0

    Encapsulates risk score validation and risk level calculation

    Time Complexity: O(1) for all operations
    """

    value: float

    def __post_init__(self):
        """Validate risk score - O(1)"""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Risk score must be numeric, got {type(self.value)}")

        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Risk score must be between 0.0 and 1.0, got {self.value}")

    def get_risk_level(self) -> str:
        """
        Calculate risk level from score

        Thresholds:
        - CRITICAL: >= 0.9
        - HIGH: >= 0.7
        - MEDIUM: >= 0.3
        - LOW: < 0.3

        Time Complexity: O(1)
        """
        if self.value >= 0.9:
            return "CRITICAL"
        elif self.value >= 0.7:
            return "HIGH"
        elif self.value >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def is_critical(self) -> bool:
        """Check if score is critical - O(1)"""
        return self.value >= 0.9

    def is_high_risk(self) -> bool:
        """Check if score is high or critical - O(1)"""
        return self.value >= 0.7

    def requires_manual_review(self) -> bool:
        """Business rule: Check if manual review required - O(1)"""
        return self.value >= 0.6

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.3f} ({self.get_risk_level()})"


@dataclass(frozen=True)
class Amount:
    """
    Value Object: Monetary amount with validation

    Extends Money from entities.py with business-specific validations

    Time Complexity: O(1) for all operations
    """

    value: Decimal
    currency: str = "BRL"

    # Transaction limits (in BRL)
    _MIN_AMOUNT: ClassVar[Decimal] = Decimal("0.01")
    _MAX_AMOUNT: ClassVar[Decimal] = Decimal("100000.00")
    _HIGH_VALUE_THRESHOLD: ClassVar[Decimal] = Decimal("5000.00")

    def __post_init__(self):
        """Validate amount - O(1)"""
        if not isinstance(self.value, Decimal):
            # Convert to Decimal if numeric
            if isinstance(self.value, (int, float)):
                object.__setattr__(self, 'value', Decimal(str(self.value)))
            else:
                raise ValueError(f"Amount must be Decimal or numeric, got {type(self.value)}")

        if self.value < self._MIN_AMOUNT:
            raise ValueError(f"Amount too small: {self.value} (min: {self._MIN_AMOUNT})")

        if self.value > self._MAX_AMOUNT:
            raise ValueError(f"Amount too large: {self.value} (max: {self._MAX_AMOUNT})")

        if len(self.currency) != 3:
            raise ValueError(f"Currency must be 3 characters, got: {self.currency}")

    def is_high_value(self) -> bool:
        """Business rule: Check if transaction is high value - O(1)"""
        return self.value >= self._HIGH_VALUE_THRESHOLD

    def formatted(self) -> str:
        """
        Format amount with currency

        Example: R$ 1.234,56

        Time Complexity: O(1)
        """
        if self.currency == "BRL":
            return f"R$ {self.value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            return f"{self.currency} {self.value:,.2f}"

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class TransactionChannel:
    """
    Value Object: Transaction channel with validation

    Encapsulates channel validation and risk mapping

    Time Complexity: O(1) for all operations
    """

    value: str

    _VALID_CHANNELS: ClassVar = {
        'PIX', 'TED', 'DOC', 'BOLETO', 'CARTAO_CREDITO',
        'CARTAO_DEBITO', 'TRANSFERENCIA', 'DEPOSITO'
    }

    _HIGH_RISK_CHANNELS: ClassVar = {'PIX', 'TED'}

    def __post_init__(self):
        """Validate channel - O(1)"""
        normalized = self.value.upper().replace(' ', '_')

        if normalized not in self._VALID_CHANNELS:
            raise ValueError(
                f"Canal inválido: {self.value}. "
                f"Válidos: {', '.join(self._VALID_CHANNELS)}"
            )

        # Normalize to uppercase
        object.__setattr__(self, 'value', normalized)

    def is_high_risk_channel(self) -> bool:
        """Business rule: Check if channel is high risk - O(1)"""
        return self.value in self._HIGH_RISK_CHANNELS

    def is_instant_payment(self) -> bool:
        """Check if channel is instant payment (PIX) - O(1)"""
        return self.value == 'PIX'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DeviceFingerprint:
    """
    Value Object: Device fingerprint for fraud detection

    Encapsulates device identification and risk scoring

    Time Complexity: O(1) for all operations
    """

    device_id: str
    ip_address: str
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None

    def __post_init__(self):
        """Validate device fingerprint - O(1)"""
        if not self.device_id or len(self.device_id) < 10:
            raise ValueError("Device ID must be at least 10 characters")

        # Validate IP address (simple check)
        if not self._is_valid_ip(self.ip_address):
            raise ValueError(f"Invalid IP address: {self.ip_address}")

    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Validate IP address - O(1)"""
        parts = ip.split('.')
        if len(parts) != 4:
            return False

        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False

    def get_hash(self) -> str:
        """
        Generate hash for device fingerprint matching

        Time Complexity: O(n) where n is total string length
        """
        import hashlib
        components = [
            self.device_id,
            self.ip_address,
            self.user_agent or '',
            self.screen_resolution or '',
            self.timezone or ''
        ]
        fingerprint_str = '|'.join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def masked_ip(self) -> str:
        """
        Mask IP address for logging (LGPD compliance)

        Example: 192.168.*.*

        Time Complexity: O(1)
        """
        parts = self.ip_address.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.*.*"
        return "***.***.***"

    def __str__(self) -> str:
        return f"Device({self.get_hash()}, IP: {self.masked_ip()})"


@dataclass(frozen=True)
class TimeWindow:
    """
    Value Object: Time window for behavioral analysis

    Used for velocity checks and pattern detection

    Time Complexity: O(1) for all operations
    """

    start: datetime
    end: datetime

    def __post_init__(self):
        """Validate time window - O(1)"""
        if self.end <= self.start:
            raise ValueError("End time must be after start time")

        duration = self.duration_minutes()
        if duration > 43200:  # 30 days in minutes
            raise ValueError(f"Time window too large: {duration} minutes (max: 43200)")

    def duration_seconds(self) -> float:
        """Calculate duration in seconds - O(1)"""
        return (self.end - self.start).total_seconds()

    def duration_minutes(self) -> float:
        """Calculate duration in minutes - O(1)"""
        return self.duration_seconds() / 60

    def duration_hours(self) -> float:
        """Calculate duration in hours - O(1)"""
        return self.duration_minutes() / 60

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within window - O(1)"""
        return self.start <= timestamp <= self.end

    def overlaps(self, other: 'TimeWindow') -> bool:
        """Check if windows overlap - O(1)"""
        return (
            self.contains(other.start) or
            self.contains(other.end) or
            other.contains(self.start) or
            other.contains(self.end)
        )

    def __str__(self) -> str:
        return f"TimeWindow({self.start.isoformat()} - {self.end.isoformat()})"


# Convenience factory functions
def create_cpf(raw_input: str) -> CPF:
    """Factory: Create CPF from any input format - O(1)"""
    return CPF.from_raw(raw_input)


def create_amount(value: float | Decimal | int, currency: str = "BRL") -> Amount:
    """Factory: Create Amount from numeric value - O(1)"""
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return Amount(value, currency)


def create_risk_score(value: float) -> RiskScore:
    """Factory: Create RiskScore with validation - O(1)"""
    return RiskScore(value)
