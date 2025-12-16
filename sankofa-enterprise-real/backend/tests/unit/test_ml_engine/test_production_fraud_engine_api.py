"""
A1_UNIT_CORE - ProductionFraudEngine API Tests
MLEngineer_QA + BackendEngineer_QA
Seed: 42
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

np.random.seed(42)


class TestProductionFraudEnginePredict:

    @pytest.fixture
    def engine(self):
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame([{
            'amount': 150.00,
            'hour': 14,
            'day_of_week': 2,
            'is_international': 0,
            'merchant_risk': 0.1,
            'velocity_1h': 1,
            'velocity_24h': 3
        }])

    @pytest.fixture
    def valid_dict(self):
        return {
            'amount': 150.00,
            'hour': 14,
            'day_of_week': 2,
            'is_international': 0,
            'merchant_risk': 0.1,
            'velocity_1h': 1,
            'velocity_24h': 3
        }

    def test_API_001_predict_accepts_dataframe(self, engine, valid_df):
        """API-001: predict() DEVE aceitar DataFrame"""
        result = engine.predict(valid_df)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_API_002_predict_returns_ndarray(self, engine, valid_df):
        """API-002: predict() DEVE retornar np.ndarray"""
        result = engine.predict(valid_df)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"

    def test_API_003_predict_scores_in_valid_range(self, engine, valid_df):
        """API-003: scores DEVEM estar em [0, 1]"""
        result = engine.predict(valid_df)
        assert np.all(result >= 0), f"Score < 0: {result}"
        assert np.all(result <= 1), f"Score > 1: {result}"

    def test_API_004_predict_with_dict_behavior(self, engine, valid_dict):
        """API-004: predict() com dict - FAIL se aceitar sem validacao"""
        try:
            result = engine.predict(valid_dict)
            pytest.fail("CRITICAL: predict() aceitou dict sem validacao - GAP de contrato")
        except (TypeError, ValueError, AttributeError):
            pass
        except Exception as e:
            pytest.fail(f"CRITICAL: Erro inesperado: {type(e).__name__}: {e}")

    def test_API_005_has_predict_proba_method(self, engine):
        """API-005: DEVE ter metodo predict_proba - CRITICAL se ausente"""
        has_method = hasattr(engine, 'predict_proba')
        assert has_method, "CRITICAL: predict_proba NAO IMPLEMENTADO"

    def test_API_006_predict_proba_returns_2d_array(self, engine, valid_df):
        """API-006: predict_proba DEVE retornar array 2D [P(0), P(1)]"""
        if not hasattr(engine, 'predict_proba'):
            pytest.fail("CRITICAL: predict_proba NAO IMPLEMENTADO")
        result = engine.predict_proba(valid_df)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2, f"Expected 2D, got shape {result.shape}"
        assert result.shape[1] == 2, f"Expected 2 cols, got {result.shape[1]}"

    def test_API_007_predict_proba_sums_to_one(self, engine, valid_df):
        """API-007: predict_proba linhas DEVEM somar 1"""
        if not hasattr(engine, 'predict_proba'):
            pytest.fail("CRITICAL: predict_proba NAO IMPLEMENTADO")
        result = engine.predict_proba(valid_df)
        row_sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)), decimal=5)


class TestProductionFraudEngineDeterminism:

    @pytest.fixture
    def engine(self):
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    def test_DET_001_same_input_same_output(self, engine):
        """DET-001: mesma entrada DEVE produzir mesma saida"""
        df = pd.DataFrame([{
            'amount': 500.0, 'hour': 12, 'day_of_week': 3,
            'is_international': 0, 'merchant_risk': 0.3,
            'velocity_1h': 2, 'velocity_24h': 8
        }])
        r1 = engine.predict(df)
        r2 = engine.predict(df)
        np.testing.assert_array_almost_equal(r1, r2, decimal=6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
