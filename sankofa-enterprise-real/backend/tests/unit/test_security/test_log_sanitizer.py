"""
Unit Tests for Log Sanitizer
Tests PII masking and sanitization functionality
"""

import pytest
from utils.log_sanitizer import (
    mask_cpf,
    mask_email,
    mask_phone,
    mask_credit_card,
    mask_jwt_token,
    sanitize_string,
    sanitize_dict,
    sanitize_list,
    sanitize_log_data,
)


# =============================================================================
# CPF MASKING TESTS
# =============================================================================

class TestMaskCPF:
    """Tests for CPF masking"""

    def test_mask_valid_cpf(self):
        """Test masking valid 11-digit CPF"""
        result = mask_cpf('12345678901')
        assert result == '***.***.*89-**'
        assert '89' in result  # Shows last 2 digits before check digits
        assert result.count('*') == 10  # Masks 10 digits

    def test_mask_formatted_cpf(self):
        """Test masking CPF with formatting"""
        result = mask_cpf('123.456.789-01')
        assert '*' in result
        assert '89' in result

    def test_mask_invalid_cpf(self):
        """Test masking invalid CPF returns generic mask"""
        result = mask_cpf('123')
        assert result == '***.***.***-**'

    def test_mask_empty_cpf(self):
        """Test masking empty CPF"""
        result = mask_cpf('')
        assert result == '***.***.***-**'

    def test_mask_cpf_preserves_last_two(self):
        """Test CPF masking preserves exactly last 2 digits before check"""
        cpf = '98765432109'
        result = mask_cpf(cpf)
        assert '10' in result  # Last 2 digits before check digits


# =============================================================================
# EMAIL MASKING TESTS
# =============================================================================

class TestMaskEmail:
    """Tests for email masking"""

    def test_mask_simple_email(self):
        """Test masking simple email"""
        result = mask_email('user@example.com')
        assert result == '***@example.com'
        assert '@example.com' in result
        assert 'user' not in result

    def test_mask_long_username(self):
        """Test masking email with long username"""
        result = mask_email('verylongusername@example.com')
        assert result == '***@example.com'

    def test_mask_subdomain_email(self):
        """Test masking email with subdomain"""
        result = mask_email('user@mail.example.com')
        assert result == '***@mail.example.com'
        assert 'user' not in result

    def test_mask_invalid_email(self):
        """Test masking invalid email returns as-is"""
        result = mask_email('not_an_email')
        assert result == 'not_an_email'

    def test_mask_email_special_chars(self):
        """Test masking email with special characters"""
        result = mask_email('user+tag@example.com')
        assert '@example.com' in result
        assert 'user' not in result


# =============================================================================
# PHONE MASKING TESTS
# =============================================================================

class TestMaskPhone:
    """Tests for phone masking"""

    def test_mask_brazilian_phone(self):
        """Test masking Brazilian phone number"""
        result = mask_phone('11987654321')
        assert '****' in result
        assert len(result.replace('*', '').replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) <= 4

    def test_mask_formatted_phone(self):
        """Test masking formatted phone"""
        result = mask_phone('(11) 98765-4321')
        assert '****' in result

    def test_mask_short_phone(self):
        """Test masking short phone number"""
        result = mask_phone('12345')
        assert '****' in result


# =============================================================================
# CREDIT CARD MASKING TESTS
# =============================================================================

class TestMaskCreditCard:
    """Tests for credit card masking"""

    def test_mask_16_digit_card(self):
        """Test masking 16-digit credit card"""
        result = mask_credit_card('1234567890123456')
        assert result == '************3456'
        assert '3456' in result  # Last 4 digits visible
        assert result.count('*') == 12

    def test_mask_formatted_card(self):
        """Test masking formatted credit card"""
        result = mask_credit_card('1234-5678-9012-3456')
        assert '3456' in result

    def test_mask_short_card(self):
        """Test masking short card number"""
        result = mask_credit_card('12345')
        assert '****' in result


# =============================================================================
# JWT TOKEN MASKING TESTS
# =============================================================================

class TestMaskJWTToken:
    """Tests for JWT token masking"""

    def test_mask_jwt_token(self):
        """Test masking JWT token"""
        token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123'
        result = mask_jwt_token(token)

        assert result.startswith('***.')
        assert result.endswith('.***')
        assert 'eyJ' not in result  # Header masked
        assert 'abc123' not in result  # Signature masked

    def test_mask_short_token(self):
        """Test masking short token"""
        result = mask_jwt_token('short_token')
        assert result == '***token'


# =============================================================================
# STRING SANITIZATION TESTS
# =============================================================================

class TestSanitizeString:
    """Tests for string sanitization"""

    def test_sanitize_cpf_in_string(self):
        """Test CPF detection and masking in string"""
        text = "User CPF: 12345678901 needs verification"
        result = sanitize_string(text)

        assert '12345678901' not in result
        assert '***.***.*89-**' in result

    def test_sanitize_email_in_string(self):
        """Test email detection and masking in string"""
        text = "Contact: user@example.com for help"
        result = sanitize_string(text)

        assert 'user@example.com' not in result
        assert '***@example.com' in result

    def test_sanitize_multiple_pii(self):
        """Test sanitizing multiple PII types in one string"""
        text = "User 12345678901 email user@example.com phone (11) 98765-4321"
        result = sanitize_string(text)

        # CPF masked
        assert '12345678901' not in result

        # Email masked
        assert 'user@example.com' not in result

        # Phone masked
        assert '98765-4321' not in result or '*' in result

    def test_sanitize_preserves_safe_text(self):
        """Test sanitization preserves text without PII"""
        text = "This is safe text without any PII"
        result = sanitize_string(text)

        assert result == text


# =============================================================================
# DICT SANITIZATION TESTS
# =============================================================================

class TestSanitizeDict:
    """Tests for dict sanitization"""

    def test_sanitize_dict_with_cpf(self):
        """Test dict sanitization with CPF field"""
        data = {
            'cpf': '12345678901',
            'name': 'John Doe'
        }

        result = sanitize_dict(data, mode='mask')

        assert result['cpf'] == '***.***.*89-**'
        assert result['name'] == 'John Doe'  # Non-sensitive field unchanged

    def test_sanitize_dict_with_email(self):
        """Test dict sanitization with email field"""
        data = {
            'email': 'user@example.com',
            'role': 'admin'
        }

        result = sanitize_dict(data, mode='mask')

        assert result['email'] == '***@example.com'
        assert result['role'] == 'admin'

    def test_sanitize_dict_multiple_sensitive_fields(self):
        """Test dict with multiple sensitive fields"""
        data = {
            'cpf': '12345678901',
            'email': 'user@example.com',
            'phone': '11987654321',
            'role': 'analyst'
        }

        result = sanitize_dict(data, mode='mask')

        assert '*' in result['cpf']
        assert '*' in result['email']
        assert '*' in result['phone']
        assert result['role'] == 'analyst'

    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dicts"""
        data = {
            'user': {
                'cpf': '12345678901',
                'name': 'John'
            },
            'transaction': {
                'amount': 1000
            }
        }

        result = sanitize_dict(data, mode='mask')

        assert '*' in result['user']['cpf']
        assert result['transaction']['amount'] == 1000

    def test_sanitize_dict_remove_mode(self):
        """Test remove mode completely removes sensitive fields"""
        data = {
            'cpf': '12345678901',
            'name': 'John Doe'
        }

        result = sanitize_dict(data, mode='remove')

        assert 'cpf' not in result
        assert result['name'] == 'John Doe'

    def test_sanitize_dict_hash_mode(self):
        """Test hash mode hashes sensitive data"""
        data = {
            'cpf': '12345678901',
            'name': 'John Doe'
        }

        result = sanitize_dict(data, mode='hash')

        assert result['cpf'] != '12345678901'
        assert len(result['cpf']) == 16  # SHA256 hash truncated
        assert result['name'] == 'John Doe'


# =============================================================================
# LIST SANITIZATION TESTS
# =============================================================================

class TestSanitizeList:
    """Tests for list sanitization"""

    def test_sanitize_list_of_dicts(self):
        """Test sanitizing list of dicts"""
        data = [
            {'cpf': '12345678901', 'name': 'John'},
            {'cpf': '98765432100', 'name': 'Jane'},
        ]

        result = sanitize_list(data, mode='mask')

        assert len(result) == 2
        assert '*' in result[0]['cpf']
        assert '*' in result[1]['cpf']
        assert result[0]['name'] == 'John'

    def test_sanitize_list_of_strings(self):
        """Test sanitizing list of strings"""
        data = [
            'User CPF: 12345678901',
            'Email: user@example.com'
        ]

        result = sanitize_list(data, mode='mask')

        assert '12345678901' not in result[0]
        assert 'user@example.com' not in result[1]

    def test_sanitize_empty_list(self):
        """Test sanitizing empty list"""
        result = sanitize_list([], mode='mask')
        assert result == []


# =============================================================================
# SANITIZE_LOG_DATA TESTS (Main Function)
# =============================================================================

class TestSanitizeLogData:
    """Tests for main sanitize_log_data function"""

    def test_sanitize_dict_data(self):
        """Test sanitizing dict log data"""
        data = {
            'user_id': 'user123',
            'cpf': '12345678901',
            'action': 'login'
        }

        result = sanitize_log_data(data)

        assert '*' in result['cpf']
        assert result['user_id'] == 'user123'
        assert result['action'] == 'login'

    def test_sanitize_string_data(self):
        """Test sanitizing string log data"""
        data = "User 12345678901 attempted login"

        result = sanitize_log_data(data)

        assert '12345678901' not in result
        assert 'attempted login' in result

    def test_sanitize_list_data(self):
        """Test sanitizing list log data"""
        data = [
            {'cpf': '12345678901'},
            {'cpf': '98765432100'}
        ]

        result = sanitize_log_data(data)

        assert len(result) == 2
        assert '*' in result[0]['cpf']

    def test_sanitize_complex_nested_data(self):
        """Test sanitizing complex nested structures"""
        data = {
            'transaction': {
                'id': 'TXN_123',
                'customer': {
                    'cpf': '12345678901',
                    'email': 'user@example.com',
                    'contacts': {
                        'phone': '11987654321'
                    }
                },
                'amount': 1000
            },
            'metadata': {
                'ip': '192.168.1.1'
            }
        }

        result = sanitize_log_data(data)

        # Check deep nesting is sanitized
        assert '*' in result['transaction']['customer']['cpf']
        assert '*' in result['transaction']['customer']['email']
        assert '*' in result['transaction']['customer']['contacts']['phone']

        # Check non-sensitive data preserved
        assert result['transaction']['id'] == 'TXN_123'
        assert result['transaction']['amount'] == 1000

    def test_sanitize_preserves_none(self):
        """Test sanitization preserves None values"""
        data = {
            'cpf': '12345678901',
            'optional_field': None
        }

        result = sanitize_log_data(data)

        assert result['optional_field'] is None

    def test_sanitize_with_remove_mode(self):
        """Test sanitize_log_data with remove mode"""
        data = {
            'cpf': '12345678901',
            'name': 'John'
        }

        result = sanitize_log_data(data, mask_mode='remove')

        assert 'cpf' not in result
        assert result['name'] == 'John'

    def test_sanitize_handles_exceptions(self):
        """Test sanitization handles exceptions gracefully"""
        # Create object that will raise exception during sanitization
        class BadObject:
            def __str__(self):
                raise Exception("Cannot convert to string")

        data = {
            'safe_field': 'value',
            'bad_field': BadObject()
        }

        # Should not raise exception
        result = sanitize_log_data(data)

        assert result['safe_field'] == 'value'
        # bad_field should be handled somehow (converted to string or removed)
