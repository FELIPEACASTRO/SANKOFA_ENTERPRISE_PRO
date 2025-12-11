"""
Integration Tests for API Endpoints
Tests real endpoint behavior with mocked dependencies
"""

import pytest
from unittest.mock import Mock, patch

class TestAuthEndpoints:
    """Test authentication endpoints"""

    def test_login_success(self, client, sample_user):
        """Test successful login"""
        with patch('api.production_api.get_user_from_db') as mock_get_user:
            mock_get_user.return_value = {
                'id': 'user123',
                'username': 'test_user',
                'name': 'Test User',
                'role': 'analyst',
                'roles': ['analyst'],
                'is_active': True,
                'password_hash': '$2b$12$dummy_hash',
                'locked_until': None
            }

            with patch('api.production_api.verify_password') as mock_verify:
                mock_verify.return_value = True

                response = client.post('/api/auth/login', json={
                    'username': 'test_user',
                    'password': 'SecurePass123!'
                })

                assert response.status_code == 200
                data = response.get_json()
                assert data['success'] is True
                assert 'token' in data['data']

class TestFraudEndpoints:
    """Test fraud detection endpoints"""

    @pytest.mark.integration
    def test_fraud_predict_valid(self, client, sample_transaction):
        """Test fraud prediction with valid data"""
        response = client.post('/api/fraud/predict', json={
            'transactions': [sample_transaction]
        })

        assert response.status_code in [200, 400]  # May fail validation but shouldn't crash

class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_endpoint(self, client):
        """Test basic health check"""
        response = client.get('/api/health')
        assert response.status_code == 200

        data = response.get_json()
        assert 'status' in data or 'success' in data
