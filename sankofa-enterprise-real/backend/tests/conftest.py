"""
Pytest Configuration and Shared Fixtures
Provides common fixtures for all tests
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


# =============================================================================
# APPLICATION FIXTURES
# =============================================================================

@pytest.fixture(scope='session')
def app():
    """
    Flask application instance for testing
    """
    # Import here to avoid circular imports
    from api.production_api import app as flask_app

    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    flask_app.config['DEBUG'] = False

    return flask_app


@pytest.fixture
def client(app):
    """
    Flask test client
    """
    return app.test_client()


@pytest.fixture
def authenticated_client(client):
    """
    Flask test client with valid JWT token

    Use this for endpoints that require authentication
    """
    from api.production_api import pyjwt, config

    # Generate valid JWT token
    token_payload = {
        "sub": "test_user",
        "user_id": "test_123",
        "name": "Test User",
        "role": "analyst",
        "roles": ["analyst"],
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    token = pyjwt.encode(
        token_payload,
        config.security.jwt_secret,
        algorithm=config.security.jwt_algorithm
    )

    # Set authorization header
    client.environ_base['HTTP_AUTHORIZATION'] = f'Bearer {token}'

    return client


@pytest.fixture
def admin_client(client):
    """
    Flask test client with admin JWT token
    """
    from api.production_api import pyjwt, config

    token_payload = {
        "sub": "admin_user",
        "user_id": "admin_123",
        "name": "Admin User",
        "role": "admin",
        "roles": ["admin", "analyst"],
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }

    token = pyjwt.encode(
        token_payload,
        config.security.jwt_secret,
        algorithm=config.security.jwt_algorithm
    )

    client.environ_base['HTTP_AUTHORIZATION'] = f'Bearer {token}'

    return client


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_transaction():
    """
    Sample transaction for testing fraud detection
    """
    return {
        'amount': 1000.0,
        'cpf': '12345678909',  # Valid CPF with check digits
        'channel': 'PIX',
        'tipo_transacao': 'PIX',
        'location': 'São Paulo',
        'device_id': 'device_123',
        'ip_address': '192.168.1.1',
    }


@pytest.fixture
def sample_transactions_batch():
    """
    Batch of sample transactions for batch processing
    """
    return [
        {
            'amount': 500.0,
            'cpf': '12345678909',
            'channel': 'PIX',
        },
        {
            'amount': 1500.0,
            'cpf': '98765432100',
            'channel': 'TED',
        },
        {
            'amount': 250.0,
            'cpf': '11122233344',
            'channel': 'DOC',
        },
    ]


@pytest.fixture
def sample_hard_rule():
    """
    Sample hard rule for testing
    """
    return {
        'name': 'Test Rule',
        'description': 'Test rule for automated testing',
        'condition': 'amount > 5000 AND channel == PIX',
        'conditions_json': [
            {
                'field': 'amount',
                'operator': '>',
                'value': 5000
            },
            {
                'field': 'channel',
                'operator': '==',
                'value': 'PIX'
            }
        ],
        'logic_operator': 'AND',
        'action': 'block',
        'action_config': {},
        'rule_type': 'blocking',
        'priority': 1,
        'enabled': True
    }


@pytest.fixture
def sample_user():
    """
    Sample user for authentication testing
    """
    return {
        'username': 'test_analyst',
        'password': 'SecurePassword123!',
        'name': 'Test Analyst',
        'email': 'test@sankofa.com',
        'role': 'analyst',
        'roles': ['analyst']
    }


@pytest.fixture
def sample_vip_entry():
    """
    Sample VIP list entry
    """
    return {
        'cpf': '12345678909',
        'reason': 'Premium customer - VIP status',
        'expires_at': (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    }


@pytest.fixture
def sample_hot_entry():
    """
    Sample Hot list entry (blacklist)
    """
    return {
        'cpf': '98765432100',
        'reason': 'Confirmed fraudster - multiple fraud cases',
        'severity': 'CRITICAL',
        'expires_at': (datetime.now(timezone.utc) + timedelta(days=180)).isoformat()
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_fraud_engine():
    """
    Mock fraud detection engine
    """
    mock_engine = Mock()
    mock_engine.VERSION = '1.0.0'
    mock_engine.threshold = 0.5

    # Mock predict method
    def mock_predict(data):
        return {
            'is_fraud': False,
            'fraud_probability': 0.3,
            'risk_score': 0.35,
            'model_version': '1.0.0',
            'risk_factors': ['high_amount'],
            'transaction_id': 'TXN_TEST_123'
        }

    mock_engine.predict = Mock(side_effect=mock_predict)
    mock_engine.predict_detailed = Mock(return_value=[])

    return mock_engine


@pytest.fixture
def mock_postgres_store():
    """
    Mock PostgreSQL store
    """
    mock_store = Mock()

    # Mock common methods
    mock_store.get_hard_rules = Mock(return_value=[])
    mock_store.get_vip_list = Mock(return_value=[])
    mock_store.get_hot_list = Mock(return_value=[])
    mock_store.get_settings = Mock(return_value={})
    mock_store.add_audit_log = Mock(return_value=True)
    mock_store.save_transaction = Mock(return_value=True)

    return mock_store


@pytest.fixture
def mock_cache():
    """
    Mock cache manager
    """
    mock_cache_mgr = Mock()

    # Simple in-memory cache
    cache_dict = {}

    def mock_get(key, default=None):
        return cache_dict.get(key, default)

    def mock_set(key, value, ttl=None):
        cache_dict[key] = value
        return True

    def mock_delete(key):
        if key in cache_dict:
            del cache_dict[key]
        return True

    mock_cache_mgr.cache = Mock()
    mock_cache_mgr.cache.get = Mock(side_effect=mock_get)
    mock_cache_mgr.cache.set = Mock(side_effect=mock_set)
    mock_cache_mgr.cache.delete = Mock(side_effect=mock_delete)

    return mock_cache_mgr


# =============================================================================
# PYDANTIC SCHEMA FIXTURES
# =============================================================================

@pytest.fixture
def valid_transaction_request():
    """
    Valid TransactionRequest for Pydantic testing
    """
    return {
        'amount': 1000.0,
        'cpf': '12345678909',
        'channel': 'PIX',
        'tipo_transacao': 'PIX',
        'location': 'São Paulo',
        'device_id': 'device_123',
        'ip_address': '192.168.1.1'
    }


@pytest.fixture
def invalid_transaction_request():
    """
    Invalid TransactionRequest for Pydantic testing
    """
    return {
        'amount': -100.0,  # Invalid: negative amount
        'cpf': 'invalid',  # Invalid: not 11 digits
        'channel': 'INVALID_CHANNEL',  # Invalid: not in enum
    }


# =============================================================================
# TEMPORARY FILE FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """
    Temporary directory for test files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file():
    """
    Temporary file for testing
    """
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield Path(path)
    if Path(path).exists():
        Path(path).unlink()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def mock_db_connection():
    """
    Mock database connection
    """
    mock_conn = Mock()

    # Mock cursor
    mock_cursor = Mock()
    mock_cursor.fetchone = Mock(return_value=None)
    mock_cursor.fetchall = Mock(return_value=[])
    mock_cursor.rowcount = 0

    mock_conn.cursor = Mock(return_value=mock_cursor)
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()

    return mock_conn


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_caches():
    """
    Reset all caches between tests
    """
    yield
    # Cleanup code here if needed


@pytest.fixture(autouse=True)
def reset_mocks():
    """
    Reset all mocks between tests
    """
    yield
    # Any cleanup needed


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook
    """
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically
    """
    for item in items:
        # Auto-mark based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
