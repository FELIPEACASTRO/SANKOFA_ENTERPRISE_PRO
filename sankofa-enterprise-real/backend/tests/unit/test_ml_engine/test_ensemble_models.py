"""
Unit tests for base ensemble models
Tests: model loading, predictions, consistency
"""
import pytest
import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

BACKEND_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BACKEND_DIR / "models" / "production"


class TestModelPersistence:
    """ML-001: Test model persistence"""

    def test_random_forest_exists(self):
        """Random forest model file exists"""
        path = MODELS_DIR / "random_forest.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_gradient_boosting_exists(self):
        """Gradient boosting model file exists"""
        path = MODELS_DIR / "gradient_boosting.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_extra_trees_exists(self):
        """Extra trees model file exists"""
        path = MODELS_DIR / "extra_trees_gnn.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_mlp_exists(self):
        """MLP model file exists"""
        path = MODELS_DIR / "mlp.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_isolation_forest_exists(self):
        """Isolation forest model file exists"""
        path = MODELS_DIR / "isolation_forest.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_scaler_exists(self):
        """Scaler file exists"""
        path = MODELS_DIR / "scaler.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_autoencoder_exists(self):
        """Autoencoder model file exists"""
        path = MODELS_DIR / "autoencoder_model.pkl"
        assert path.exists(), f"Missing: {path}"

    def test_bilstm_exists(self):
        """BiLSTM model file exists"""
        path = MODELS_DIR / "bilstm_model.pkl"
        assert path.exists(), f"Missing: {path}"


class TestModelLoading:
    """Test model loading"""

    def test_random_forest_loads(self):
        """Random forest model loads correctly"""
        path = MODELS_DIR / "random_forest.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_gradient_boosting_loads(self):
        """Gradient boosting model loads correctly"""
        path = MODELS_DIR / "gradient_boosting.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_mlp_loads(self):
        """MLP model loads correctly"""
        path = MODELS_DIR / "mlp.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_scaler_loads(self):
        """Scaler loads and has n_features_in_"""
        path = MODELS_DIR / "scaler.pkl"
        with open(path, 'rb') as f:
            scaler = pickle.load(f)

        assert scaler is not None
        assert hasattr(scaler, 'n_features_in_')


class TestModelPredictions:
    """ML-003: Test model predictions"""

    @pytest.fixture
    def test_data(self):
        """Create test data with correct feature count"""
        scaler_path = MODELS_DIR / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        n_features = scaler.n_features_in_
        return np.zeros((1, n_features))

    def test_random_forest_predicts(self, test_data):
        """Random forest produces prediction"""
        path = MODELS_DIR / "random_forest.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        pred = model.predict(test_data)

        assert pred is not None
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_gradient_boosting_predicts(self, test_data):
        """Gradient boosting produces prediction"""
        path = MODELS_DIR / "gradient_boosting.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        pred = model.predict(test_data)

        assert pred is not None
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_mlp_predicts(self, test_data):
        """MLP produces prediction"""
        path = MODELS_DIR / "mlp.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        pred = model.predict(test_data)

        assert pred is not None
        assert len(pred) == 1

    def test_isolation_forest_predicts(self, test_data):
        """Isolation forest produces prediction"""
        path = MODELS_DIR / "isolation_forest.pkl"
        with open(path, 'rb') as f:
            model = pickle.load(f)

        pred = model.predict(test_data)

        assert pred is not None
        assert len(pred) == 1
        # Isolation forest returns -1 (anomaly) or 1 (normal)
        assert pred[0] in [-1, 1]


class TestEnsembleConfig:
    """ML-002: Test ensemble configuration"""

    def test_config_exists(self):
        """Ensemble config file exists"""
        path = MODELS_DIR / "ensemble_config.json"
        assert path.exists()

    def test_config_has_weights(self):
        """Config has model weights"""
        import json
        path = MODELS_DIR / "ensemble_config.json"
        with open(path, 'r') as f:
            config = json.load(f)

        assert 'weights' in config

    def test_config_has_auc(self):
        """Config has ensemble AUC"""
        import json
        path = MODELS_DIR / "ensemble_config.json"
        with open(path, 'r') as f:
            config = json.load(f)

        assert 'ensemble_auc' in config or 'auc' in str(config).lower()

    def test_config_has_timestamp(self):
        """Config has training timestamp"""
        import json
        path = MODELS_DIR / "ensemble_config.json"
        with open(path, 'r') as f:
            config = json.load(f)

        assert 'trained_at' in config
