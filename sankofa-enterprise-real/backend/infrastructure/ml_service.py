"""
Machine Learning Service Implementation
Implements Strategy Pattern for different ML models
Follows Single Responsibility and Open/Closed Principles (SOLID)
"""

import asyncio
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging

from ..core.entities import Transaction, FraudAnalysisResult, TransactionId
from ..core.interfaces import FraudDetectionService, MetricsCollector


logger = logging.getLogger(__name__)


# Strategy Pattern for different ML models
class MLModelStrategy(ABC):
    """
    Abstract strategy for ML models
    Time Complexity varies by implementation
    """

    @abstractmethod
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction
        Time Complexity: Depends on model
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - O(1)"""
        pass

    @abstractmethod
    async def extract_features(self, transaction: Transaction) -> np.ndarray:
        """
        Extract features from transaction
        Time Complexity: O(f) where f is number of features
        """
        pass


class RandomForestStrategy(MLModelStrategy):
    """
    Random Forest implementation
    Time Complexity: O(n_trees * log(n_samples) * n_features) for prediction
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            "amount",
            "hour",
            "day_of_week",
            "merchant_risk",
            "customer_transaction_count",
            "amount_zscore",
            "velocity_score",
            "location_risk",
        ]

        if model_path:
            self._load_model(model_path)
        else:
            self._create_default_model()

    def _create_default_model(self) -> None:
        """Create default Random Forest model - O(1)"""
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1  # Use all CPU cores
        )

        # Train with synthetic data for demo
        X_train, y_train = self._generate_training_data()
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def _load_model(self, model_path: str) -> None:
        """Load pre-trained model - O(1)"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_default_model()

    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction using Random Forest
        Time Complexity: O(n_trees * log(n_samples) * n_features)
        Average case: O(100 * log(1000) * 8) ≈ O(8000) operations
        """
        start_time = time.time()

        # Scale features - O(n_features)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get prediction probability - O(n_trees * log(n_samples) * n_features)
        fraud_probability = self.model.predict_proba(features_scaled)[0][1]

        # Get prediction - O(n_trees * log(n_samples) * n_features)
        is_fraud = fraud_probability > 0.5

        # Feature importance for explainability - O(n_features)
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        processing_time = (time.time() - start_time) * 1000

        return {
            "is_fraud": is_fraud,
            "confidence_score": fraud_probability,
            "feature_importance": feature_importance,
            "processing_time_ms": processing_time,
            "model_type": "random_forest",
        }

    async def extract_features(self, transaction: Transaction) -> np.ndarray:
        """
        Extract features from transaction
        Time Complexity: O(f) where f is number of features (8 in this case)
        """
        # Basic features - O(1) each
        amount = float(transaction.amount.amount)
        hour = transaction.timestamp.hour
        day_of_week = transaction.timestamp.weekday()

        # Derived features - O(1) each
        merchant_risk = hash(transaction.merchant_id) % 100 / 100.0  # Simplified
        customer_tx_count = len(transaction.metadata.get("customer_history", []))

        # Statistical features - O(1) each
        amount_zscore = (amount - 1000) / 500  # Simplified z-score
        velocity_score = transaction.metadata.get("velocity_score", 0.0)
        location_risk = transaction.metadata.get("location_risk", 0.0)

        return np.array(
            [
                amount,
                hour,
                day_of_week,
                merchant_risk,
                customer_tx_count,
                amount_zscore,
                velocity_score,
                location_risk,
            ]
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - O(1)"""
        return {
            "model_type": "RandomForest",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }

    def _generate_training_data(self) -> tuple:
        """Generate synthetic training data - O(n)"""
        np.random.seed(42)
        n_samples = 10000

        # Generate features
        X = np.random.randn(n_samples, len(self.feature_names))

        # Generate labels with some pattern
        y = (
            (X[:, 0] > 2)  # High amount
            | (X[:, 1] < -1)  # Late night
            | (X[:, 6] > 1.5)  # High velocity
        ).astype(int)

        return X, y


class IsolationForestStrategy(MLModelStrategy):
    """
    Isolation Forest for anomaly detection
    Time Complexity: O(ψ * log(n)) where ψ is number of trees
    """

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42  # 10% anomalies expected
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            "amount",
            "hour",
            "day_of_week",
            "merchant_risk",
            "customer_transaction_count",
            "velocity_score",
        ]

    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Anomaly detection using Isolation Forest
        Time Complexity: O(ψ * log(n)) ≈ O(100 * log(1000)) ≈ O(1000)
        """
        start_time = time.time()

        if not self.is_fitted:
            # Quick training with current features
            await self._quick_fit(features)

        # Scale features - O(n_features)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get anomaly score - O(ψ * log(n))
        anomaly_score = self.model.decision_function(features_scaled)[0]

        # Convert to probability (higher score = less anomalous)
        # Normalize to 0-1 range where 1 = high fraud probability
        fraud_probability = max(0, min(1, (0.5 - anomaly_score) / 0.5))

        is_fraud = anomaly_score < 0  # Negative scores are anomalies

        processing_time = (time.time() - start_time) * 1000

        return {
            "is_fraud": is_fraud,
            "confidence_score": fraud_probability,
            "anomaly_score": anomaly_score,
            "processing_time_ms": processing_time,
            "model_type": "isolation_forest",
        }

    async def extract_features(self, transaction: Transaction) -> np.ndarray:
        """
        Extract features for anomaly detection
        Time Complexity: O(f) where f is number of features
        """
        amount = float(transaction.amount.amount)
        hour = transaction.timestamp.hour
        day_of_week = transaction.timestamp.weekday()
        merchant_risk = hash(transaction.merchant_id) % 100 / 100.0
        customer_tx_count = len(transaction.metadata.get("customer_history", []))
        velocity_score = transaction.metadata.get("velocity_score", 0.0)

        return np.array(
            [amount, hour, day_of_week, merchant_risk, customer_tx_count, velocity_score]
        )

    async def _quick_fit(self, features: np.ndarray) -> None:
        """Quick fit with synthetic data - O(n * ψ * log(n))"""
        # Generate synthetic data for fitting
        np.random.seed(42)
        X_train = np.random.randn(1000, len(self.feature_names))

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled)
        self.is_fitted = True

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - O(1)"""
        return {
            "model_type": "IsolationForest",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }


class EnsembleMLStrategy(MLModelStrategy):
    """
    Ensemble of multiple models
    Time Complexity: O(sum of all model complexities)
    """

    def __init__(self):
        self.strategies = [RandomForestStrategy(), IsolationForestStrategy()]
        self.weights = [0.7, 0.3]  # Random Forest gets more weight

    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Ensemble prediction
        Time Complexity: O(sum of model complexities)
        """
        start_time = time.time()

        predictions = []
        model_results = {}

        # Get predictions from all models - O(sum of complexities)
        for i, strategy in enumerate(self.strategies):
            result = await strategy.predict(features)
            predictions.append(result["confidence_score"])
            model_results[f"model_{i}"] = result

        # Weighted average - O(n_models)
        ensemble_score = sum(pred * weight for pred, weight in zip(predictions, self.weights))

        is_fraud = ensemble_score > 0.5

        processing_time = (time.time() - start_time) * 1000

        return {
            "is_fraud": is_fraud,
            "confidence_score": ensemble_score,
            "individual_predictions": predictions,
            "model_results": model_results,
            "processing_time_ms": processing_time,
            "model_type": "ensemble",
        }

    async def extract_features(self, transaction: Transaction) -> np.ndarray:
        """Use the first strategy's feature extraction - O(f)"""
        return await self.strategies[0].extract_features(transaction)

    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble information - O(n_models)"""
        return {
            "model_type": "Ensemble",
            "strategies": [s.get_model_info() for s in self.strategies],
            "weights": self.weights,
            "n_models": len(self.strategies),
        }


# Context class for Strategy Pattern
class MLFraudDetectionService(FraudDetectionService):
    """
    Main fraud detection service using Strategy Pattern
    Time Complexity: Depends on selected strategy
    """

    def __init__(
        self, strategy: MLModelStrategy, metrics_collector: Optional[MetricsCollector] = None
    ):
        self._strategy = strategy
        self._metrics_collector = metrics_collector

    async def analyze_transaction(self, transaction: Transaction) -> FraudAnalysisResult:
        """
        Analyze transaction using selected strategy
        Time Complexity: O(f + model_complexity) where f is feature extraction
        """
        start_time = time.time()

        try:
            # Extract features - O(f)
            features = await self._strategy.extract_features(transaction)

            # Make prediction - O(model_complexity)
            prediction = await self._strategy.predict(features)

            # Create result
            result = FraudAnalysisResult(
                transaction_id=transaction.id,
                is_fraud=prediction["is_fraud"],
                confidence_score=prediction["confidence_score"],
                model_version=prediction.get("model_type", "unknown"),
                processing_time_ms=prediction.get("processing_time_ms", 0),
            )

            # Add risk factors based on prediction
            if prediction["is_fraud"]:
                result.add_risk_factor("ML_MODEL_DETECTION")

                # Add specific risk factors based on feature importance
                if "feature_importance" in prediction:
                    top_features = sorted(
                        prediction["feature_importance"].items(), key=lambda x: x[1], reverse=True
                    )[:3]

                    for feature, importance in top_features:
                        if importance > 0.1:  # Significant importance
                            result.add_risk_factor(f"HIGH_{feature.upper()}")

            # Record metrics
            if self._metrics_collector:
                self._record_metrics(result, prediction)

            return result

        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")

            # Return safe default
            return FraudAnalysisResult(
                transaction_id=transaction.id,
                is_fraud=False,
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information - O(1)"""
        return self._strategy.get_model_info()

    def set_strategy(self, strategy: MLModelStrategy) -> None:
        """Change ML strategy at runtime - O(1)"""
        self._strategy = strategy

    def _record_metrics(self, result: FraudAnalysisResult, prediction: Dict[str, Any]) -> None:
        """Record metrics - O(1)"""
        if not self._metrics_collector:
            return

        self._metrics_collector.increment_counter("fraud_analysis_requests")
        self._metrics_collector.record_histogram(
            "fraud_analysis_duration", result.processing_time_ms
        )
        self._metrics_collector.record_histogram("fraud_confidence_score", result.confidence_score)

        if result.is_fraud:
            self._metrics_collector.increment_counter("fraud_detected")


# Factory for creating ML services
class MLServiceFactory:
    """
    Factory for creating ML services
    Implements Factory Pattern
    """

    @staticmethod
    def create_fraud_service(
        model_type: str = "ensemble", metrics_collector: Optional[MetricsCollector] = None
    ) -> MLFraudDetectionService:
        """
        Create fraud detection service
        Time Complexity: O(1) + model initialization time
        """
        if model_type == "random_forest":
            strategy = RandomForestStrategy()
        elif model_type == "isolation_forest":
            strategy = IsolationForestStrategy()
        elif model_type == "ensemble":
            strategy = EnsembleMLStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return MLFraudDetectionService(strategy, metrics_collector)

    @staticmethod
    def create_strategy(model_type: str) -> MLModelStrategy:
        """Create ML strategy - O(1) + initialization"""
        if model_type == "random_forest":
            return RandomForestStrategy()
        elif model_type == "isolation_forest":
            return IsolationForestStrategy()
        elif model_type == "ensemble":
            return EnsembleMLStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Singleton Pattern for model registry
class ModelRegistry:
    """
    Singleton registry for ML models
    Implements Singleton Pattern
    """

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_model(self, name: str, strategy: MLModelStrategy) -> None:
        """Register a model strategy - O(1)"""
        self._models[name] = strategy

    def get_model(self, name: str) -> Optional[MLModelStrategy]:
        """Get registered model - O(1)"""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """List all registered models - O(n)"""
        return list(self._models.keys())

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model information - O(1)"""
        model = self._models.get(name)
        return model.get_model_info() if model else None
