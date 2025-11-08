"""
Testes Unitários - Production Fraud Engine
Checklist 5.2: Pirâmide de testes equilibrada
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_engine.production_fraud_engine import ProductionFraudEngine, ModelMetrics


class TestProductionFraudEngine:
    """Testes do motor de fraude em produção"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture com dados sintéticos para teste"""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'amount': np.random.lognormal(5, 2, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'merchant_risk': np.random.uniform(0, 1, n_samples),
            'user_history': np.random.uniform(0, 100, n_samples),
            'velocity_1h': np.random.randint(0, 10, n_samples)
        })
        
        # 10% fraude (correlacionado com amount alto)
        y = ((X['amount'] > X['amount'].quantile(0.9)) & 
             (X['merchant_risk'] > 0.7)).astype(int)
        
        return X, y.values
    
    def test_engine_initialization(self):
        """Testa inicialização do engine"""
        engine = ProductionFraudEngine()
        
        assert engine.calibrated_model is None
        assert engine.metrics is None
        assert engine.threshold == 0.5
    
    def test_fit_creates_model(self, sample_data):
        """Testa que fit() cria modelo treinado"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        
        engine.fit(X, y)
        
        assert engine.calibrated_model is not None
        assert engine.metrics is not None
        assert isinstance(engine.metrics, ModelMetrics)
    
    def test_fit_calculates_metrics(self, sample_data):
        """Testa que fit() calcula métricas válidas"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        
        engine.fit(X, y)
        
        # Métricas devem estar entre 0 e 1
        assert 0 <= engine.metrics.accuracy <= 1
        assert 0 <= engine.metrics.precision <= 1
        assert 0 <= engine.metrics.recall <= 1
        assert 0 <= engine.metrics.f1_score <= 1
        assert 0 <= engine.metrics.roc_auc <= 1
    
    def test_predict_without_fit_raises_error(self, sample_data):
        """Testa que predict() sem fit() lança erro"""
        X, _ = sample_data
        engine = ProductionFraudEngine()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.predict(X)
    
    def test_predict_returns_binary(self, sample_data):
        """Testa que predict() retorna 0 ou 1"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        predictions = engine.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_returns_probabilities(self, sample_data):
        """Testa que predict_proba() retorna probabilidades [0, 1]"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        probas = engine.predict_proba(X)
        
        assert isinstance(probas, np.ndarray)
        assert len(probas) == len(X)
        assert np.all((probas >= 0) & (probas <= 1))
    
    def test_different_thresholds_change_predictions(self, sample_data):
        """Testa que threshold diferente muda predições"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        # Threshold alto (conservador)
        preds_high = engine.predict(X, threshold=0.8)
        
        # Threshold baixo (agressivo)
        preds_low = engine.predict(X, threshold=0.2)
        
        # Threshold baixo deve detectar mais fraudes
        assert preds_low.sum() > preds_high.sum()
    
    def test_f1_score_reasonable(self, sample_data):
        """Testa que F1-Score é razoável (>= 0.5) com dados sintéticos"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        # Com dados correlacionados, F1 deve ser >= 0.5
        assert engine.metrics.f1_score >= 0.5, \
            f"F1-Score muito baixo: {engine.metrics.f1_score:.3f}"
    
    def test_model_handles_missing_features(self, sample_data):
        """Testa tratamento de features faltantes"""
        X, y = sample_data
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        # Adicionar NaN
        X_with_nan = X.copy()
        X_with_nan.loc[0, 'amount'] = np.nan
        
        # Deve prever sem crashes (imputer lida com NaN)
        predictions = engine.predict(X_with_nan)
        assert len(predictions) == len(X_with_nan)


class TestModelMetrics:
    """Testes da classe ModelMetrics"""
    
    def test_metrics_initialization(self):
        """Testa criação de métricas"""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            roc_auc=0.93,
            threshold=0.5
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.roc_auc == 0.93
        assert metrics.threshold == 0.5
