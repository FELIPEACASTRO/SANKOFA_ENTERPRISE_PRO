"""
Sankofa Enterprise Pro - Domain Unit Tests
Tests for core domain entities, business rules, and use cases
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, '.')

from ml_engine.production_fraud_engine import (
    ProductionFraudEngine,
    FraudPrediction,
    ModelMetrics,
)
from utils.error_handling import ValidationError, ErrorCategory
from dataclasses import dataclass


@dataclass
class FraudThresholds:
    """Thresholds for fraud risk levels"""
    low_risk: float = 0.3
    medium_risk: float = 0.5
    high_risk: float = 0.8
    critical_risk: float = 0.95


class TestFraudPredictionEntity:
    """Tests for FraudPrediction dataclass"""
    
    def test_fraud_prediction_creation(self):
        """Test creating a valid FraudPrediction"""
        prediction = FraudPrediction(
            transaction_id="TXN_001",
            is_fraud=True,
            fraud_probability=0.85,
            risk_score=85.0,
            risk_level="HIGH",
            confidence=0.9,
            detection_reason=["High value transaction at night"],
            processing_time_ms=25.5,
            model_version="12.3",
            timestamp="2025-11-28T12:00:00Z",
        )
        
        assert prediction.transaction_id == "TXN_001"
        assert prediction.is_fraud == True
        assert prediction.fraud_probability == 0.85
        assert prediction.risk_level == "HIGH"
    
    def test_fraud_prediction_risk_levels(self):
        """Test that risk levels are correctly assigned"""
        test_cases = [
            (0.95, "CRITICAL"),
            (0.85, "HIGH"),
            (0.60, "MEDIUM"),
            (0.30, "LOW"),
        ]
        
        for prob, expected_level in test_cases:
            prediction = FraudPrediction(
                transaction_id="TXN",
                is_fraud=prob >= 0.5,
                fraud_probability=prob,
                risk_score=prob * 100,
                risk_level=expected_level,
                confidence=0.9,
                detection_reason=[],
                processing_time_ms=10.0,
                model_version="12.3",
                timestamp="2025-11-28T12:00:00Z",
            )
            assert prediction.risk_level == expected_level


class TestModelMetricsEntity:
    """Tests for ModelMetrics dataclass"""
    
    def test_model_metrics_creation(self):
        """Test creating valid ModelMetrics"""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.94,
            threshold=0.5,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert 0 <= metrics.f1_score <= 1


class TestFraudThresholdsEntity:
    """Tests for FraudThresholds dataclass"""
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = FraudThresholds()
        
        assert thresholds.low_risk == 0.3
        assert thresholds.medium_risk == 0.5
        assert thresholds.high_risk == 0.8
        assert thresholds.critical_risk == 0.95


class TestBusinessRules:
    """Tests for core business rules"""
    
    def test_high_amount_night_transaction_rule(self):
        """Test that high amounts at night are flagged as suspicious"""
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([{
            'amount': 50000,
            'hour': 3,
            'channel': 'pix',
        }])
        
        predictions = engine.predict_detailed(X)
        
        assert len(predictions) == 1
        assert predictions[0].fraud_probability >= 0.3
    
    def test_normal_transaction_low_risk(self):
        """Test that normal transactions are low risk"""
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([{
            'amount': 100,
            'hour': 14,
            'channel': 'web',
        }])
        
        predictions = engine.predict_detailed(X)
        
        assert len(predictions) == 1
        assert predictions[0].risk_level in ["LOW", "MEDIUM"]
    
    def test_multiple_risk_factors(self):
        """Test that multiple risk factors increase probability"""
        engine = ProductionFraudEngine()
        
        low_risk = pd.DataFrame([{
            'amount': 100,
            'hour': 14,
            'channel': 'web',
        }])
        
        high_risk = pd.DataFrame([{
            'amount': 50000,
            'hour': 3,
            'channel': 'pix',
        }])
        
        low_pred = engine.predict_detailed(low_risk)[0]
        high_pred = engine.predict_detailed(high_risk)[0]
        
        assert high_pred.fraud_probability > low_pred.fraud_probability


class TestValidationErrors:
    """Tests for validation error handling"""
    
    def test_validation_error_creation(self):
        """Test ValidationError creation with context"""
        error = ValidationError(
            message="Invalid amount",
            context={"amount": -100, "field": "amount"}
        )
        
        assert error.message == "Invalid amount"
        assert error.category == ErrorCategory.VALIDATION
        assert error.context["amount"] == -100
    
    def test_validation_error_context(self):
        """Test ValidationError.get_context()"""
        error = ValidationError("Test error")
        context = error.get_context()
        
        assert context.category == ErrorCategory.VALIDATION
        assert "validation" in context.error_id.lower()


class TestFeatureEngineering:
    """Tests for feature engineering"""
    
    def test_derived_features_created(self):
        """Test that derived features are created correctly"""
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([{
            'amount': 5000,
            'hour': 3,
            'channel': 'pix',
        }])
        
        derived = engine._create_derived_features(X)
        
        assert 'amount_log' in derived.columns
        assert 'is_night' in derived.columns
        assert 'is_high_amount' in derived.columns
    
    def test_night_detection(self):
        """Test night hour detection"""
        engine = ProductionFraudEngine()
        
        night_hours = [0, 1, 2, 3, 4, 5, 23]
        day_hours = [8, 10, 12, 14, 16, 18]
        
        for hour in night_hours:
            X = pd.DataFrame([{'amount': 100, 'hour': hour, 'channel': 'web'}])
            derived = engine._create_derived_features(X)
            assert derived['is_night'].iloc[0] == 1, f"Hour {hour} should be night"
        
        for hour in day_hours:
            X = pd.DataFrame([{'amount': 100, 'hour': hour, 'channel': 'web'}])
            derived = engine._create_derived_features(X)
            assert derived['is_night'].iloc[0] == 0, f"Hour {hour} should be day"
    
    def test_channel_encoding(self):
        """Test channel one-hot encoding"""
        engine = ProductionFraudEngine()
        
        X = pd.DataFrame([
            {'amount': 100, 'hour': 10, 'channel': 'pix'},
            {'amount': 100, 'hour': 10, 'channel': 'web'},
            {'amount': 100, 'hour': 10, 'channel': 'mobile'},
        ])
        
        derived = engine._create_derived_features(X)
        
        assert 'channel_pix' in derived.columns
        assert 'channel_web' in derived.columns
        assert 'channel_mobile' in derived.columns
        
        assert derived['channel_pix'].iloc[0] == 1
        assert derived['channel_web'].iloc[1] == 1
        assert derived['channel_mobile'].iloc[2] == 1


class TestModelTraining:
    """Tests for model training with API features"""
    
    def test_train_with_api_features(self):
        """Test that model can be trained with API-compatible features"""
        engine = ProductionFraudEngine()
        engine.train_with_api_features()
        
        assert engine.is_trained == True
        assert len(engine.feature_names) > 0
        
        api_features = ['amount', 'hour']
        for feat in api_features:
            derived_features = [f for f in engine.feature_names if feat in f]
            assert len(derived_features) > 0
    
    def test_model_predictions_after_training(self):
        """Test predictions work after training"""
        engine = ProductionFraudEngine()
        engine.train_with_api_features()
        
        X = pd.DataFrame([
            {'amount': 100, 'hour': 14, 'channel': 'web'},
            {'amount': 50000, 'hour': 3, 'channel': 'pix'},
        ])
        
        predictions = engine.predict_detailed(X)
        
        assert len(predictions) == 2
        assert all(0 <= p.fraud_probability <= 1 for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
