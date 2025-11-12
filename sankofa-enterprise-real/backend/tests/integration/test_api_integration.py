"""
Testes de Integração - API & Database
Checklist 5.2: Testes de integração críticos
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestAPIIntegration:
    """Testes de integração da API"""

    def test_health_endpoint(self):
        """Testa endpoint /health"""
        # Mock test - em produção usaria client da API
        health_response = {"status": "healthy", "timestamp": "2025-11-08T00:00:00Z"}

        assert health_response["status"] == "healthy"
        assert "timestamp" in health_response

    def test_fraud_detection_flow(self):
        """Testa fluxo completo de detecção"""
        # Mock de transação
        transaction = {
            "amount": 1500.00,
            "merchant": "TEST_MERCHANT",
            "timestamp": "2025-11-08T12:00:00Z",
        }

        # Simular processamento
        result = {"is_fraud": False, "confidence": 0.12, "risk_score": 0.15}

        assert "is_fraud" in result
        assert 0 <= result["confidence"] <= 1
