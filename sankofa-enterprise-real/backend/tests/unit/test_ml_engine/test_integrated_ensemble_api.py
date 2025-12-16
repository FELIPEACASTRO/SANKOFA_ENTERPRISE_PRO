"""
A1_UNIT_CORE - IntegratedEnsemble API Tests
MLEngineer_QA + DataScientist_QA
Seed: 42
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

np.random.seed(42)


class TestIntegratedEnsemblePredictCombined:

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    @pytest.fixture
    def sample_transaction(self):
        return {
            'amount': 1500.0, 'hour': 3, 'day_of_week': 6,
            'is_international': 1, 'merchant_risk': 0.8,
            'velocity_1h': 8, 'velocity_24h': 25
        }

    def test_ENS_001_has_predict_combined_method(self, ensemble):
        """ENS-001: DEVE ter metodo predict_combined"""
        assert hasattr(ensemble, 'predict_combined')
        assert callable(getattr(ensemble, 'predict_combined'))

    def test_ENS_002_predict_combined_signature(self, ensemble, sample_transaction):
        """ENS-002: predict_combined(transaction, base_probability)"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.3)
        assert result is not None

    def test_ENS_003_predict_combined_returns_structured(self, ensemble, sample_transaction):
        """ENS-003: predict_combined retorna dict ou dataclass"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.3)
        is_dict = isinstance(result, dict)
        has_score = hasattr(result, 'fraud_probability') or hasattr(result, 'final_score') or hasattr(result, 'combined_score')
        assert is_dict or has_score, f"Expected dict/dataclass, got {type(result)}"

    def test_ENS_004_predict_combined_score_range(self, ensemble, sample_transaction):
        """ENS-004: score final em [0, 1]"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.5)
        if isinstance(result, dict):
            score = result.get('fraud_probability') or result.get('final_score') or result.get('combined_score') or result.get('score', 0.5)
        else:
            score = getattr(result, 'fraud_probability', getattr(result, 'final_score', getattr(result, 'combined_score', 0.5)))
        assert 0.0 <= score <= 1.0, f"Score fora do range: {score}"


class TestIntegratedEnsemblePredictAlias:

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    def test_ENS_005_has_predict_alias(self, ensemble):
        """ENS-005: DEVE ter predict() alias - HIGH se ausente"""
        has_predict = hasattr(ensemble, 'predict')
        assert has_predict, "HIGH: IntegratedEnsemble.predict() alias NAO EXISTE"

    def test_ENS_006_predict_alias_callable(self, ensemble):
        """ENS-006: predict() alias DEVE ser callable"""
        if not hasattr(ensemble, 'predict'):
            pytest.fail("HIGH: predict() alias NAO EXISTE")
        assert callable(getattr(ensemble, 'predict'))

    def test_ENS_006b_predict_alias_works(self, ensemble):
        """ENS-006b: predict() alias DEVE funcionar"""
        transaction = {'amount': 100.0, 'hour': 10}
        result = ensemble.predict(transaction)
        assert result is not None


class TestIntegratedEnsembleCalibrateWeights:

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    def test_ENS_007_has_calibrate_weights(self, ensemble):
        """ENS-007: DEVE ter calibrate_weights"""
        assert hasattr(ensemble, 'calibrate_weights')

    def test_ENS_008_calibrate_weights_returns_dict(self, ensemble):
        """ENS-008: calibrate_weights retorna dict"""
        X_val = pd.DataFrame({'amount': [100.0, 200.0], 'hour': [10, 14]})
        y_val = np.array([0, 1])
        base_predictions = np.array([0.2, 0.8])
        result = ensemble.calibrate_weights(X_val, y_val, base_predictions)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
