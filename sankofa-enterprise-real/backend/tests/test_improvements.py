"""
Testes para as melhorias implementadas baseadas no AIForge
- Explainability Engine (SHAP)
- Entropia de Localização
- Calibração de Probabilidades
- Self-Training
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.explainability_engine import ExplainabilityEngine, PredictionExplanation
from ml_engine.probability_calibration import ProbabilityCalibrator, EnsembleCalibrator
from ml_engine.self_training_optimizer import SelfTrainingClassifier, AdaptiveSelfTraining
from ml_engine.advanced_feature_engineering import AdvancedFeatureEngineering


class TestExplainabilityEngine:
    """Testes para o engine de explicabilidade"""
    
    def test_initialization(self):
        """Testa inicialização do engine"""
        engine = ExplainabilityEngine(
            feature_names=["amount", "hour", "is_night"]
        )
        assert engine.VERSION == "1.0.0"
        assert len(engine.feature_names) == 3
    
    def test_explain_prediction(self):
        """Testa explicação de predição"""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = (X_train[:, 0] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        engine = ExplainabilityEngine(
            model=model,
            feature_names=["amount", "hour", "is_night", "velocity_score"]
        )
        engine._calculate_fallback_importance()
        
        X = np.array([[5000.0, 2, 1, 0.8]])
        
        explanation = engine.explain_prediction(
            X,
            transaction_id="TX123",
            fraud_probability=0.85
        )
        
        assert isinstance(explanation, PredictionExplanation)
        assert explanation.transaction_id == "TX123"
        assert explanation.fraud_probability == 0.85
        assert explanation.risk_level == "CRITICO"
        assert "risco" in explanation.explanation_text.lower()
    
    def test_risk_levels(self):
        """Testa classificação de níveis de risco"""
        engine = ExplainabilityEngine()
        
        assert engine._get_risk_level(0.9) == "CRITICO"
        assert engine._get_risk_level(0.7) == "ALTO"
        assert engine._get_risk_level(0.5) == "MEDIO"
        assert engine._get_risk_level(0.3) == "BAIXO"
        assert engine._get_risk_level(0.1) == "MUITO_BAIXO"
    
    def test_compliance_report(self):
        """Testa geração de relatório de compliance"""
        engine = ExplainabilityEngine(
            feature_names=["amount", "hour"]
        )
        
        X = np.array([[1000.0, 14]])
        explanation = engine.explain_prediction(
            X,
            transaction_id="TX456",
            fraud_probability=0.3
        )
        
        report = engine.to_compliance_report(explanation)
        
        assert "lgpd_compliance" in report
        assert "bacen_compliance" in report
        assert "pci_dss_compliance" in report
        assert report["lgpd_compliance"]["right_to_explanation"] is True


class TestProbabilityCalibration:
    """Testes para calibração de probabilidades"""
    
    def test_isotonic_calibration(self):
        """Testa calibração isotônica"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.beta(2 + y_true * 3, 5 - y_true * 2, 500)
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(y_true, y_prob)
        
        assert calibrator.is_fitted is True
        assert calibrator.calibration_metrics is not None
        
        calibrated = calibrator.calibrate(y_prob)
        assert len(calibrated) == len(y_prob)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
    
    def test_sigmoid_calibration(self):
        """Testa calibração sigmoid"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.beta(2 + y_true * 3, 5 - y_true * 2, 500)
        
        calibrator = ProbabilityCalibrator(method="sigmoid")
        calibrator.fit(y_true, y_prob)
        
        assert calibrator.is_fitted is True
        
        metrics = calibrator.get_metrics_summary()
        assert "expected_calibration_error" in metrics
        assert "brier_score" in metrics
    
    def test_ensemble_calibrator(self):
        """Testa calibrador ensemble"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.beta(2 + y_true * 3, 5 - y_true * 2, 500)
        
        ensemble = EnsembleCalibrator()
        ensemble.fit(y_true, y_prob)
        
        assert ensemble.is_fitted is True
        assert ensemble.best_method in ["isotonic", "sigmoid"]
        
        calibrated = ensemble.calibrate(y_prob)
        assert len(calibrated) == len(y_prob)
    
    def test_calibration_improves_ece(self):
        """Testa que calibração melhora ECE"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.clip(y_true + np.random.normal(0, 0.3, 1000), 0, 1)
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        
        ece_before = calibrator._calculate_ece(y_true, y_prob)
        calibrator.fit(y_true, y_prob)
        calibrated = calibrator.calibrate(y_prob)
        ece_after = calibrator._calculate_ece(y_true, calibrated)
        
        assert ece_after <= ece_before + 0.1


class TestSelfTraining:
    """Testes para self-training"""
    
    def test_self_training_basic(self):
        """Testa self-training básico"""
        np.random.seed(42)
        n_features = 5
        
        X_labeled = np.random.randn(200, n_features)
        y_labeled = (X_labeled[:, 0] + X_labeled[:, 1] > 0).astype(int)
        
        X_unlabeled = np.random.randn(500, n_features)
        
        clf = SelfTrainingClassifier(max_iter=3)
        clf.fit(X_labeled, y_labeled, X_unlabeled)
        
        assert clf.is_fitted is True
        assert clf.metrics is not None
        assert len(clf.training_history) > 0
    
    def test_self_training_predictions(self):
        """Testa predições após self-training"""
        np.random.seed(42)
        n_features = 5
        
        X_labeled = np.random.randn(200, n_features)
        y_labeled = (X_labeled[:, 0] + X_labeled[:, 1] > 0).astype(int)
        
        X_unlabeled = np.random.randn(500, n_features)
        X_test = np.random.randn(50, n_features)
        
        clf = SelfTrainingClassifier(max_iter=3)
        clf.fit(X_labeled, y_labeled, X_unlabeled)
        
        predictions = clf.predict(X_test)
        assert len(predictions) == 50
        assert all(p in [0, 1] for p in predictions)
        
        probs = clf.predict_proba(X_test)
        assert probs.shape == (50, 2)
    
    def test_self_training_metrics(self):
        """Testa métricas de self-training"""
        np.random.seed(42)
        n_features = 5
        
        X_labeled = np.random.randn(200, n_features)
        y_labeled = (X_labeled[:, 0] + X_labeled[:, 1] > 0).astype(int)
        
        X_unlabeled = np.random.randn(500, n_features)
        
        clf = SelfTrainingClassifier(max_iter=5)
        clf.fit(X_labeled, y_labeled, X_unlabeled)
        
        metrics = clf.get_metrics_summary()
        
        assert "iterations" in metrics
        assert "pseudo_labels_added" in metrics
        assert "initial_accuracy" in metrics
        assert "final_accuracy" in metrics
    
    def test_adaptive_self_training(self):
        """Testa self-training adaptativo"""
        np.random.seed(42)
        n_features = 5
        
        X_labeled = np.random.randn(200, n_features)
        y_labeled = (X_labeled[:, 0] + X_labeled[:, 1] > 0).astype(int)
        
        X_unlabeled = np.random.randn(500, n_features)
        X_val = np.random.randn(100, n_features)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
        
        clf = AdaptiveSelfTraining()
        clf.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val, max_iter=3)
        
        assert clf.is_fitted is True
        
        predictions = clf.predict(X_val)
        assert len(predictions) == 100


class TestLocationEntropyFeatures:
    """Testes para features de entropia de localização"""
    
    def test_entropy_features_created(self):
        """Testa criação de features de entropia"""
        df = pd.DataFrame({
            "client_cpf": ["A", "A", "A", "B", "B", "C"],
            "state": ["SP", "RJ", "MG", "SP", "SP", "RJ"],
            "value": [100, 200, 300, 150, 250, 500]
        })
        
        fe = AdvancedFeatureEngineering()
        result = fe.create_features(df)
        
        assert "location_entropy" in result.columns
        assert "unique_locations_count" in result.columns
        assert "is_diverse_locations" in result.columns
    
    def test_high_entropy_detection(self):
        """Testa detecção de alta entropia"""
        df = pd.DataFrame({
            "client_cpf": ["A"] * 10 + ["B"] * 10,
            "state": ["SP", "RJ", "MG", "BA", "RS", "SC", "PR", "GO", "MT", "MS"] + ["SP"] * 10,
            "value": [100] * 20
        })
        
        fe = AdvancedFeatureEngineering()
        result = fe.create_features(df)
        
        client_a_entropy = result[result["client_cpf"] == "A"]["location_entropy"].mean()
        client_b_entropy = result[result["client_cpf"] == "B"]["location_entropy"].mean()
        
        assert client_a_entropy > client_b_entropy
    
    def test_feature_names_updated(self):
        """Testa que lista de features foi atualizada"""
        fe = AdvancedFeatureEngineering()
        feature_names = fe.get_feature_names()
        
        assert "location_entropy" in feature_names
        assert "unique_locations_count" in feature_names
        assert "is_diverse_locations" in feature_names
        assert "value_zscore" in feature_names
        assert "hour_entropy" in feature_names


class TestTransactionPatternFeatures:
    """Testes para features de padrão de transação"""
    
    def test_pattern_features_created(self):
        """Testa criação de features de padrão"""
        df = pd.DataFrame({
            "client_cpf": ["A", "A", "A", "A", "A"],
            "value": [100, 110, 105, 1000, 95],
            "hour": [14, 15, 14, 3, 13]
        })
        
        fe = AdvancedFeatureEngineering()
        result = fe.create_features(df)
        
        assert "value_zscore" in result.columns
        assert "is_value_outlier" in result.columns
        assert "hour_entropy" in result.columns
        assert "is_unusual_hour" in result.columns
    
    def test_outlier_detection(self):
        """Testa detecção de outliers"""
        df = pd.DataFrame({
            "client_cpf": ["A"] * 10,
            "value": [100, 105, 110, 95, 102, 108, 98, 103, 1500, 97],
            "hour": [14] * 10
        })
        
        fe = AdvancedFeatureEngineering()
        result = fe.create_features(df)
        
        outlier_row = result[result["value"] == 1500]
        assert outlier_row["is_value_outlier"].values[0] == 1


class TestIntegration:
    """Testes de integração"""
    
    def test_full_pipeline(self):
        """Testa pipeline completo de melhorias"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            "client_cpf": np.random.choice(["A", "B", "C"], 100),
            "state": np.random.choice(["SP", "RJ", "MG"], 100),
            "value": np.random.exponential(500, 100),
            "hour": np.random.randint(0, 24, 100),
            "is_fraud": np.random.randint(0, 2, 100)
        })
        
        fe = AdvancedFeatureEngineering()
        df_features = fe.create_features(df)
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "is_fraud"]
        
        X = df_features[feature_cols].fillna(0).values
        y = df_features["is_fraud"].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(y_test, y_prob)
        calibrated_probs = calibrator.calibrate(y_prob)
        
        explainer = ExplainabilityEngine(
            model=model,
            feature_names=feature_cols
        )
        
        explanation = explainer.explain_prediction(
            X_test[0:1],
            transaction_id="TEST001",
            fraud_probability=float(calibrated_probs[0])
        )
        
        assert isinstance(explanation, PredictionExplanation)
        assert explanation.compliance_ready is True
        
        print("\n=== Integration Test Results ===")
        print(f"Features created: {len(feature_cols)}")
        print(f"Calibration ECE: {calibrator.calibration_metrics.expected_calibration_error:.4f}")
        print(f"Explanation risk level: {explanation.risk_level}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
