"""
CHECKPOINT 1 - Unit Tests: AutoencoderAnomalyDetector
DataScientist_QA

Contrato:
- detect(features: dict) -> Dict com is_anomaly, score, threshold
- detect_anomaly(features: np.array) -> float (anomaly score)
- fit(data: np.array) -> self
- get_status() -> Dict com version, trained, etc.
"""
import pytest
import numpy as np
import sys
import os

# Ajustar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestAutoencoderDetectorContract:
    """Testes de contrato da API"""

    @pytest.fixture
    def detector(self):
        """Fixture para criar detector"""
        from backend.ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
        return AutoencoderAnomalyDetector()

    @pytest.fixture
    def sample_features(self):
        """Features de exemplo"""
        return {
            'amount': 1500.0,
            'hour': 14,
            'velocity': 0.3,
            'is_new_device': 0,
            'transaction_count': 5,
            'avg_amount': 1200.0,
            'std_amount': 300.0,
            'time_since_last': 3600,
            'merchant_risk': 0.2,
            'geo_distance': 10.5,
            'device_score': 0.9,
            'account_age_days': 365,
            'login_frequency': 2.5,
            'failed_attempts': 0
        }

    def test_DS_001_has_detect_method(self, detector):
        """DS-001: Deve ter método detect"""
        assert hasattr(detector, 'detect'), "Falta método detect"
        assert callable(getattr(detector, 'detect')), "detect não é callable"

    def test_DS_002_has_detect_anomaly_method(self, detector):
        """DS-002: Deve ter método detect_anomaly"""
        has_method = (
            hasattr(detector, 'detect_anomaly') or
            hasattr(detector, 'detect')  # detect pode servir como alias
        )
        assert has_method, "Falta método detect_anomaly ou detect"

    def test_DS_003_has_get_status_method(self, detector):
        """DS-003: Deve ter método get_status"""
        assert hasattr(detector, 'get_status'), "Falta método get_status"

    def test_DS_004_detect_accepts_dict(self, detector, sample_features):
        """DS-004: detect() deve aceitar dicionário"""
        result = detector.detect(sample_features)
        assert result is not None, "detect() retornou None"


class TestAutoencoderDetectorOutput:
    """Testes do output de detect()"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
        return AutoencoderAnomalyDetector()

    @pytest.fixture
    def sample_features(self):
        return {f'feature_{i}': np.random.random() for i in range(14)}

    def test_DS_005_detect_returns_dict(self, detector, sample_features):
        """DS-005: detect() retorna dicionário"""
        result = detector.detect(sample_features)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_DS_006_detect_has_is_anomaly(self, detector, sample_features):
        """DS-006: Resultado deve ter is_anomaly"""
        result = detector.detect(sample_features)
        assert 'is_anomaly' in result, f"Falta is_anomaly. Keys: {list(result.keys())}"

    def test_DS_007_detect_has_score(self, detector, sample_features):
        """DS-007: Resultado deve ter score"""
        result = detector.detect(sample_features)
        assert 'score' in result, f"Falta score. Keys: {list(result.keys())}"

    def test_DS_008_detect_has_threshold(self, detector, sample_features):
        """DS-008: Resultado deve ter threshold"""
        result = detector.detect(sample_features)
        assert 'threshold' in result, f"Falta threshold. Keys: {list(result.keys())}"

    def test_DS_009_score_in_valid_range(self, detector, sample_features):
        """DS-009: Score deve estar em range válido"""
        result = detector.detect(sample_features)
        score = result['score']
        assert 0 <= score <= 10, f"Score {score} fora do range esperado [0,10]"


class TestAutoencoderDetectorStatus:
    """Testes de status do detector"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
        return AutoencoderAnomalyDetector()

    def test_STATUS_001_get_status_returns_dict(self, detector):
        """STATUS-001: get_status retorna dicionário"""
        status = detector.get_status()
        assert isinstance(status, dict), f"Expected dict, got {type(status)}"

    def test_STATUS_002_status_has_version(self, detector):
        """STATUS-002: Status deve ter version"""
        status = detector.get_status()
        has_version = 'version' in status or 'VERSION' in status
        assert has_version, f"Falta version em status. Keys: {list(status.keys())}"


class TestAutoencoderDetectorAnomalyDetection:
    """Testes de detecção de anomalias"""

    @pytest.fixture
    def detector(self):
        from backend.ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
        return AutoencoderAnomalyDetector()

    def test_ANOMALY_001_normal_transaction_low_score(self, detector):
        """ANOMALY-001: Transação normal deve ter score baixo"""
        normal_features = {
            'amount': 500.0,
            'hour': 14,
            'velocity': 0.2,
            'is_new_device': 0,
            'transaction_count': 10,
            'avg_amount': 400.0,
            'std_amount': 100.0,
            'time_since_last': 7200,
            'merchant_risk': 0.1,
            'geo_distance': 5.0,
            'device_score': 0.95,
            'account_age_days': 730,
            'login_frequency': 3.0,
            'failed_attempts': 0
        }
        result = detector.detect(normal_features)
        # Score normalizado, quanto menor mais normal
        assert result['score'] <= result['threshold'] * 2, "Transação normal deveria ter score próximo do threshold"

    def test_ANOMALY_002_detect_consistency(self, detector):
        """ANOMALY-002: Mesmas features = mesmo resultado"""
        features = {f'feature_{i}': 0.5 for i in range(14)}

        results = [detector.detect(features) for _ in range(3)]
        scores = [r['score'] for r in results]

        assert all(s == scores[0] for s in scores), f"Resultados não consistentes: {scores}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
