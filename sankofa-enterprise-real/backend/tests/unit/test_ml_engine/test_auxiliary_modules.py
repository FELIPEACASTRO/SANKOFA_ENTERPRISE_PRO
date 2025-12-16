"""
Unit tests for auxiliary ML modules
Tests: MuleDetector, HardRulesEngine, CatBoost, Explainability
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestMuleDetector:
    """BE-003: Test MuleDetector"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.mule_detection import MuleDetector
        self.detector = MuleDetector()

    def test_has_detect_method(self):
        """MuleDetector has detect method"""
        assert hasattr(self.detector, 'detect')
        assert callable(getattr(self.detector, 'detect'))

    def test_has_add_known_mule(self):
        """MuleDetector has add_known_mule method"""
        assert hasattr(self.detector, 'add_known_mule')

    def test_has_get_stats(self):
        """MuleDetector has get_stats method"""
        assert hasattr(self.detector, 'get_stats')


class TestHardRulesEngine:
    """BE-004: Test HardRulesEngine"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.hard_rules_engine import HardRulesEngine
        self.engine = HardRulesEngine()

    def test_has_evaluate_method(self):
        """HardRulesEngine has evaluate method"""
        assert hasattr(self.engine, 'evaluate')
        assert callable(getattr(self.engine, 'evaluate'))

    def test_evaluate_returns_result(self):
        """evaluate() returns result"""
        tx = {
            "amount": 50000.0,
            "hour": 3,
            "is_new_device": True
        }

        result = self.engine.evaluate(tx)

        assert result is not None

    def test_evaluate_high_amount(self):
        """evaluate handles high amount transaction"""
        tx = {"amount": 100000.0, "hour": 14}

        result = self.engine.evaluate(tx)

        assert result is not None


class TestCatBoostFraudModel:
    """ML-004: Test CatBoostFraudModel"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.catboost_model import CatBoostFraudModel
        self.model = CatBoostFraudModel()

    def test_has_predict_method(self):
        """CatBoostFraudModel has predict method"""
        assert hasattr(self.model, 'predict')

    def test_has_get_feature_importance(self):
        """CatBoostFraudModel has feature importance method"""
        assert hasattr(self.model, 'get_feature_importance')

    def test_has_get_shap_values(self):
        """CatBoostFraudModel has SHAP values method"""
        assert hasattr(self.model, 'get_shap_values')


class TestExplainabilityEngine:
    """BANK-003: Test ExplainabilityEngine"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.explainability_engine import ExplainabilityEngine
        self.engine = ExplainabilityEngine()

    def test_has_explain_prediction(self):
        """ExplainabilityEngine has explain_prediction method"""
        assert hasattr(self.engine, 'explain_prediction')
        assert callable(getattr(self.engine, 'explain_prediction'))

    def test_has_explain_batch(self):
        """ExplainabilityEngine has explain_batch method"""
        assert hasattr(self.engine, 'explain_batch')

    def test_has_get_top_features(self):
        """ExplainabilityEngine has get_top_features method"""
        assert hasattr(self.engine, 'get_top_features')

    def test_has_compliance_report(self):
        """ExplainabilityEngine has compliance report method"""
        assert hasattr(self.engine, 'to_compliance_report')


class TestDeviceFingerprintAnalyzer:
    """BANK-005: Test DeviceFingerprintAnalyzer"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.device_fingerprint import DeviceFingerprintAnalyzer
        self.analyzer = DeviceFingerprintAnalyzer()

    def test_has_analyze_method(self):
        """DeviceFingerprintAnalyzer has analyze method"""
        assert hasattr(self.analyzer, 'analyze')
        assert callable(getattr(self.analyzer, 'analyze'))

    def test_has_get_device_stats(self):
        """DeviceFingerprintAnalyzer has get_device_stats method"""
        assert hasattr(self.analyzer, 'get_device_stats')

    def test_has_register_trusted_device(self):
        """DeviceFingerprintAnalyzer has register_trusted_device method"""
        assert hasattr(self.analyzer, 'register_trusted_device')


class TestONNXModelConverter:
    """OPS-003: Test ONNXModelConverter"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.onnx_serving import ONNXModelConverter
        self.converter = ONNXModelConverter()

    def test_has_convert_sklearn_method(self):
        """ONNXModelConverter has convert_sklearn_model method"""
        assert hasattr(self.converter, 'convert_sklearn_model')
        assert callable(getattr(self.converter, 'convert_sklearn_model'))

    def test_has_save_onnx_method(self):
        """ONNXModelConverter has save_onnx_model method"""
        assert hasattr(self.converter, 'save_onnx_model')


class TestIntegratedEnsemble:
    """ML-005: Test IntegratedEnsemble"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.ensemble_integration import IntegratedEnsemble
        self.ensemble = IntegratedEnsemble()

    def test_has_predict_combined(self):
        """IntegratedEnsemble has predict_combined method"""
        assert hasattr(self.ensemble, 'predict_combined')

    def test_has_calibrate_weights(self):
        """IntegratedEnsemble has calibrate_weights method"""
        assert hasattr(self.ensemble, 'calibrate_weights')

    def test_has_get_ensemble_info(self):
        """IntegratedEnsemble has get_ensemble_info method"""
        assert hasattr(self.ensemble, 'get_ensemble_info')
