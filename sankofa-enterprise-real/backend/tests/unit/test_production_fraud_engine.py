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

    def test_engine_initialization(self, fraud_engine):
        """Testa inicialização do engine"""
        assert fraud_engine.calibrated_model is None
        assert fraud_engine.metrics is None
        assert fraud_engine.threshold == 0.5

    def test_fit_creates_model(self, fraud_engine, small_fraud_dataset):
        """Testa que fit() cria modelo treinado"""
        X, y = small_fraud_dataset
        fraud_engine.fit(X, y)

        assert fraud_engine.calibrated_model is not None
        assert fraud_engine.metrics is not None
        assert isinstance(fraud_engine.metrics, ModelMetrics)
        assert fraud_engine.is_trained is True

    def test_fit_calculates_metrics(self, fraud_engine, small_fraud_dataset):
        """Testa que fit() calcula métricas válidas"""
        X, y = small_fraud_dataset
        fraud_engine.fit(X, y)

        # Verificar que metrics não é None
        assert fraud_engine.metrics is not None

        # Métricas devem estar entre 0 e 1
        assert 0 <= fraud_engine.metrics.accuracy <= 1
        assert 0 <= fraud_engine.metrics.precision <= 1
        assert 0 <= fraud_engine.metrics.recall <= 1
        assert 0 <= fraud_engine.metrics.f1_score <= 1
        assert 0 <= fraud_engine.metrics.roc_auc <= 1

    def test_predict_without_fit_raises_error(self, fraud_engine, small_fraud_dataset):
        """Testa que predict() sem fit() lança erro"""
        X, _ = small_fraud_dataset

        with pytest.raises(ValueError, match="Model not trained"):
            fraud_engine.predict(X)

    def test_predict_returns_binary(self, trained_engine):
        """Testa que predict() retorna 0 ou 1"""
        engine, X, y = trained_engine
        predictions = engine.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_different_thresholds_change_predictions(self, trained_engine):
        """Testa que threshold diferente muda predições"""
        engine, X, y = trained_engine

        # Threshold baixo (mais sensível)
        engine.threshold = 0.3
        preds_low = engine.predict(X)

        # Threshold alto (menos sensível)
        engine.threshold = 0.8
        preds_high = engine.predict(X)

        # Threshold baixo deve detectar mais fraudes
        assert preds_low.sum() >= preds_high.sum()

    def test_f1_score_reasonable(self, trained_engine):
        """Testa que F1-Score é razoável (>= 0.3) com dados sintéticos pequenos"""
        engine, X, y = trained_engine

        # Verificar que metrics não é None
        assert engine.metrics is not None

        # Com 100 samples, F1 >= 0.3 é razoável
        assert (
            engine.metrics.f1_score >= 0.3
        ), f"F1-Score muito baixo: {engine.metrics.f1_score:.3f}"

    def test_model_handles_missing_features(self, trained_engine):
        """Testa tratamento de features faltantes"""
        engine, X, y = trained_engine

        # Adicionar NaN
        X_with_nan = X.copy()
        X_with_nan.loc[0, "amount"] = np.nan

        # Deve prever sem crashes (preprocessing lida com NaN)
        predictions = engine.predict(X_with_nan)
        assert len(predictions) == len(X_with_nan)


class TestModelMetrics:
    """Testes da classe ModelMetrics"""

    def test_metrics_initialization(self):
        """Testa criação de métricas"""
        from datetime import datetime

        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            roc_auc=0.93,
            threshold=0.5,
            timestamp=datetime.now().isoformat(),
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.roc_auc == 0.93
        assert metrics.threshold == 0.5
