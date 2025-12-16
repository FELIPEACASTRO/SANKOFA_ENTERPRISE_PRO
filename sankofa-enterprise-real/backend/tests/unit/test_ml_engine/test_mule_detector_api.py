"""
A1_UNIT_CORE - MuleDetector API Tests
BackendEngineer_QA + BankingProduction_QA
Seed: 42
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestMuleDetectorDetect:

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.mule_detection.mule_detector import MuleDetector
        return MuleDetector()

    @pytest.fixture
    def sample_account_data(self):
        return {
            'account_id': 'ACC_12345', 'account_age_days': 30,
            'total_transactions': 50, 'avg_transaction_amount': 500.0,
            'country': 'BR'
        }

    @pytest.fixture
    def sample_history(self):
        return [
            {'amount': 100.0, 'timestamp': '2025-01-10T10:00:00', 'type': 'PIX_IN'},
            {'amount': 95.0, 'timestamp': '2025-01-10T10:05:00', 'type': 'PIX_OUT'},
        ]

    def test_MULE_001_has_detect_method(self, detector):
        """MULE-001: DEVE ter metodo detect"""
        assert hasattr(detector, 'detect')
        assert callable(getattr(detector, 'detect'))

    def test_MULE_002_detect_signature(self, detector, sample_account_data, sample_history):
        """MULE-002: detect(account_id, account_data, transaction_history)"""
        result = detector.detect(
            account_id='ACC_12345',
            account_data=sample_account_data,
            transaction_history=sample_history
        )
        assert result is not None

    def test_MULE_003_detect_has_score(self, detector, sample_account_data, sample_history):
        """MULE-003: resultado DEVE ter score"""
        result = detector.detect('ACC_12345', sample_account_data, sample_history)
        has_score = hasattr(result, 'score') or hasattr(result, 'mule_probability') or (isinstance(result, dict) and 'score' in result)
        assert has_score, f"CRITICAL: resultado sem 'score': {type(result)}"

    def test_MULE_004_score_in_range(self, detector, sample_account_data, sample_history):
        """MULE-004: score em [0, 1]"""
        result = detector.detect('ACC_12345', sample_account_data, sample_history)
        if hasattr(result, 'score'):
            score = result.score
        elif hasattr(result, 'mule_probability'):
            score = result.mule_probability
        elif isinstance(result, dict):
            score = result.get('score', result.get('mule_probability', 0.5))
        else:
            score = float(result)
        assert 0.0 <= score <= 1.0, f"Score fora do range: {score}"

    def test_MULE_005_has_add_suspicious_account(self, detector):
        """MULE-005: DEVE ter add_suspicious_account"""
        assert hasattr(detector, 'add_suspicious_account')

    def test_MULE_006_add_suspicious_accepts_score(self, detector):
        """MULE-006: add_suspicious_account(account_id, score)"""
        try:
            detector.add_suspicious_account('SUSP_001', score=0.85)
        except TypeError as e:
            pytest.fail(f"add_suspicious_account requer score: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
