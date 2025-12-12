"""
Sankofa Enterprise Pro - LGPD-Compliant Logging Utility
Sanitizes PII (CPF, email, phone) from logs according to LGPD Art. 46
"""

import hashlib
import logging
import re
from typing import Any, Dict, Union
from decimal import Decimal

logger = logging.getLogger(__name__)


class LGPDLogger:
    """
    Utility class for LGPD-compliant logging
    - Hashes CPF/email/phone numbers
    - Converts amounts to buckets (not exact values)
    - Masks sensitive data
    """

    @staticmethod
    def sanitize_cpf(cpf: str) -> str:
        """
        Hash CPF to irreversible format

        Args:
            cpf: CPF number (11 digits)

        Returns:
            Hashed CPF (first 16 chars of SHA-256)

        Example:
            >>> LGPDLogger.sanitize_cpf("12345678901")
            "a3f2b1c4d5e6f7a8"
        """
        if not cpf:
            return "unknown"

        # Remove non-digits
        cpf_clean = re.sub(r'\D', '', str(cpf))

        # Hash
        return hashlib.sha256(cpf_clean.encode()).hexdigest()[:16]

    @staticmethod
    def sanitize_email(email: str) -> str:
        """
        Hash email preserving domain

        Args:
            email: Email address

        Returns:
            Hashed email like "a3f2b1@domain.com"

        Example:
            >>> LGPDLogger.sanitize_email("user@example.com")
            "a3f2b1c4@example.com"
        """
        if not email or '@' not in email:
            return "unknown"

        local, domain = email.split('@', 1)
        local_hash = hashlib.sha256(local.encode()).hexdigest()[:8]

        return f"{local_hash}@{domain}"

    @staticmethod
    def sanitize_phone(phone: str) -> str:
        """
        Mask phone number

        Args:
            phone: Phone number

        Returns:
            Masked phone like "+55 11 *****-1234"

        Example:
            >>> LGPDLogger.sanitize_phone("+5511987654321")
            "+55 11 *****-4321"
        """
        if not phone:
            return "unknown"

        # Keep only last 4 digits
        phone_clean = re.sub(r'\D', '', str(phone))
        if len(phone_clean) >= 4:
            return f"***-{phone_clean[-4:]}"
        return "***"

    @staticmethod
    def sanitize_amount(amount: Union[float, Decimal, int]) -> str:
        """
        Convert exact amount to bucket range

        Args:
            amount: Transaction amount

        Returns:
            Amount bucket string

        Example:
            >>> LGPDLogger.sanitize_amount(1500.50)
            "1k-10k"
        """
        if amount is None:
            return "unknown"

        amount = float(amount)

        if amount < 0:
            return "negative"
        elif amount < 100:
            return "0-100"
        elif amount < 1000:
            return "100-1k"
        elif amount < 10000:
            return "1k-10k"
        elif amount < 50000:
            return "10k-50k"
        elif amount < 100000:
            return "50k-100k"
        else:
            return "100k+"

    @staticmethod
    def sanitize_ip(ip: str) -> str:
        """
        Mask last octet of IP address

        Args:
            ip: IP address

        Returns:
            Masked IP like "192.168.1.***"

        Example:
            >>> LGPDLogger.sanitize_ip("192.168.1.100")
            "192.168.1.***"
        """
        if not ip:
            return "unknown"

        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.***"
        return "***"

    @staticmethod
    def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize entire log dictionary

        Automatically detects and sanitizes:
        - Fields containing 'cpf'
        - Fields containing 'email'
        - Fields containing 'phone' or 'telefone'
        - Fields containing 'amount' or 'valor'
        - Fields containing 'ip'

        Args:
            data: Dictionary with log data

        Returns:
            Sanitized dictionary safe for logging

        Example:
            >>> data = {
            ...     "customer_cpf": "12345678901",
            ...     "amount": 1500.50,
            ...     "email": "user@example.com"
            ... }
            >>> LGPDLogger.sanitize_log_data(data)
            {
                "customer_cpf_hash": "a3f2b1c4d5e6f7a8",
                "amount_bucket": "1k-10k",
                "email_masked": "a3f2b1c4@example.com"
            }
        """
        sanitized = {}

        for key, value in data.items():
            key_lower = key.lower()

            # CPF fields
            if 'cpf' in key_lower:
                sanitized[f"{key}_hash"] = LGPDLogger.sanitize_cpf(str(value) if value else "")

            # Email fields
            elif 'email' in key_lower:
                sanitized[f"{key}_masked"] = LGPDLogger.sanitize_email(str(value) if value else "")

            # Phone fields
            elif 'phone' in key_lower or 'telefone' in key_lower or 'celular' in key_lower:
                sanitized[f"{key}_masked"] = LGPDLogger.sanitize_phone(str(value) if value else "")

            # Amount fields
            elif 'amount' in key_lower or 'valor' in key_lower:
                sanitized[f"{key}_bucket"] = LGPDLogger.sanitize_amount(value)

            # IP fields
            elif 'ip' in key_lower and 'zip' not in key_lower:
                sanitized[f"{key}_masked"] = LGPDLogger.sanitize_ip(str(value) if value else "")

            # Safe fields (IDs, timestamps, booleans, etc.)
            elif any(safe in key_lower for safe in ['id', 'timestamp', 'created', 'updated', 'is_', 'has_', 'status', 'type', 'channel']):
                sanitized[key] = value

            # Unknown fields - hash if string, keep if number/bool
            else:
                if isinstance(value, str) and len(str(value)) > 20:
                    # Potentially sensitive long string - hash it
                    sanitized[f"{key}_hash"] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
                elif isinstance(value, (int, float, bool)):
                    sanitized[key] = value
                else:
                    sanitized[key] = type(value).__name__

        return sanitized


# Convenience function for quick logging
def lgpd_log(level: str, message: str, **kwargs):
    """
    Log message with automatic PII sanitization

    Args:
        level: Log level ('info', 'warning', 'error', 'debug')
        message: Log message
        **kwargs: Additional data to log (will be sanitized)

    Example:
        >>> lgpd_log('info', 'Transaction processed',
        ...          customer_cpf='12345678901',
        ...          amount=1500.50,
        ...          transaction_id='TXN123')
        # Logs: Transaction processed | customer_cpf_hash=a3f2b1... amount_bucket=1k-10k transaction_id=TXN123
    """
    sanitized = LGPDLogger.sanitize_log_data(kwargs)

    # Convert to log-friendly format
    extra_str = " | ".join(f"{k}={v}" for k, v in sanitized.items())
    full_message = f"{message} | {extra_str}" if extra_str else message

    log_func = getattr(logger, level.lower(), logger.info)
    log_func(full_message)


# Example usage for migration:
#
# BEFORE (INSECURE):
# print(f"Transaction {txn_id} de CPF {cpf} valor {valor}")
#
# AFTER (LGPD-COMPLIANT):
# from utils.lgpd_logger import lgpd_log
# lgpd_log('info', 'Transaction processed',
#          transaction_id=txn_id,
#          customer_cpf=cpf,
#          amount=valor)
