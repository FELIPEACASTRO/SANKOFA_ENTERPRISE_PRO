import logging

logger = logging.getLogger(__name__)
"""
Sankofa Enterprise Pro - Production Fraud Engine
Motor consolidado de detecção de fraude enterprise-grade
Substitui os 15 engines anteriores com implementação única e otimizada
"""

import sys
from pathlib import Path

# Add parent to path FIRST (before any local imports)
sys.path.append(str(Path(__file__).parent.parent))

import time
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV

# Internal imports (after path setup)
from utils.structured_logging import get_structured_logger, log_execution_time
from config.settings import get_config

warnings.filterwarnings("ignore")

logger = get_structured_logger("fraud_engine", "INFO")


@dataclass
class FraudPrediction:
    """Resultado de predição de fraude"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    model_version: str
    detection_reason: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return asdict(self)


@dataclass
class ModelMetrics:
    """Métricas do modelo"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    threshold: float
    timestamp: str


class ProductionFraudEngine:
    """
    Motor de Detecção de Fraude Production-Grade

    Features:
    - Ensemble stacking (RF, GB, LR)
    - Calibração de probabilidades
    - Threshold dinâmico
    - Logging estruturado
    - Error handling robusto
    - Métricas de performance
    - Versionamento de modelos
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o engine

        Args:
            config: Configuração customizada (opcional)
        """
        if config is None:
            app_config = get_config()
            self.confidence_threshold = app_config.ml.confidence_threshold
            self.model_path_str = app_config.ml.model_path
        else:
            self.confidence_threshold = config.get("confidence_threshold", 0.5)
            self.model_path_str = config.get("model_path", "./models")

        # Modelos base do ensemble
        self.base_models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.8, random_state=42
            ),
        }

        # Meta-modelo (stacking)
        self.meta_model = LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        )

        # Ensemble stacking
        self.ensemble = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=self.meta_model,
            cv=5,
            stack_method="predict_proba",
        )

        # Calibração de probabilidades
        self.calibrated_model = None

        # Preprocessamento
        self.scaler = StandardScaler()
        self.feature_names = []

        # Estado e métricas
        self.is_trained = False
        self.threshold = self.confidence_threshold
        self.metrics: Optional[ModelMetrics] = None
        self.model_path = Path(self.model_path_str)

        # Rules para precision boosting
        self.precision_rules = self._initialize_precision_rules()

        logger.info(
            "Production Fraud Engine initialized",
            version=self.VERSION,
            threshold=self.threshold,
            model_path=str(self.model_path),
        )

    def _initialize_precision_rules(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa regras de alta precisão"""
        return {
            "extreme_amount_suspicious_hour": {
                "amount_threshold": 50000,
                "suspicious_hours": [0, 1, 2, 3, 4, 23],
                "probability_boost": 0.3,
            },
            "velocity_burst": {
                "frequency_threshold": 50,
                "time_window_hours": 0.5,
                "probability_boost": 0.4,
            },
            "high_risk_combination": {
                "location_risk_threshold": 0.9,
                "device_risk_threshold": 0.9,
                "probability_boost": 0.5,
            },
        }

    @log_execution_time(logger)
    def _preprocess_data(self, X: pd.DataFrame, fit_transform: bool = False) -> np.ndarray:
        """
        Pré-processa dados para o modelo

        Args:
            X: DataFrame com features
            fit_transform: Se True, fit + transform. Se False, apenas transform

        Returns:
            Array numpy processado
        """
        try:
            # Selecionar features numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                raise ValueError("No numeric features found in input data")

            X_selected = X[numeric_cols].copy()

            # Tratar valores ausentes
            X_selected = X_selected.fillna(X_selected.median())

            # Scaling
            if fit_transform:
                X_scaled = self.scaler.fit_transform(X_selected)
                self.feature_names = numeric_cols
                logger.info(
                    "Features preprocessed", num_features=len(numeric_cols), num_samples=len(X)
                )
            else:
                X_scaled = self.scaler.transform(X_selected)

            return np.asarray(X_scaled)

        except Exception as e:
            logger.error("Preprocessing failed", exception=e)
            raise

    def _apply_precision_rules(self, X: pd.DataFrame, base_probabilities: np.ndarray) -> np.ndarray:
        """
        Aplica regras de alta precisão para boost de probabilidade

        Args:
            X: DataFrame original
            base_probabilities: Probabilidades base do modelo

        Returns:
            Probabilidades ajustadas
        """
        adjusted_probs = base_probabilities.copy()
        boost = np.zeros(len(X))

        # Regra 1: Valor extremo em horário suspeito
        if "amount" in X.columns and "hour" in X.columns:
            rule1 = self.precision_rules["extreme_amount_suspicious_hour"]
            extreme_amount = X["amount"] >= rule1["amount_threshold"]
            suspicious_hour = X["hour"].isin(rule1["suspicious_hours"])
            match = extreme_amount & suspicious_hour
            boost += match.astype(float) * rule1["probability_boost"]

        # Regra 2: Burst de velocidade
        if (
            "transaction_frequency_7d" in X.columns
            and "time_since_last_transaction_hours" in X.columns
        ):
            rule2 = self.precision_rules["velocity_burst"]
            high_freq = X["transaction_frequency_7d"] >= rule2["frequency_threshold"]
            short_time = X["time_since_last_transaction_hours"] <= rule2["time_window_hours"]
            match = high_freq & short_time
            boost += match.astype(float) * rule2["probability_boost"]

        # Regra 3: Combinação de riscos altos
        if "location_risk_score" in X.columns and "device_risk_score" in X.columns:
            rule3 = self.precision_rules["high_risk_combination"]
            high_location = X["location_risk_score"] >= rule3["location_risk_threshold"]
            high_device = X["device_risk_score"] >= rule3["device_risk_threshold"]
            match = high_location & high_device
            boost += match.astype(float) * rule3["probability_boost"]

        # Aplicar boost (max 1.0)
        adjusted_probs = np.clip(adjusted_probs + boost, 0.0, 1.0)

        return adjusted_probs

    def _calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Calibra threshold para otimizar F1-Score

        Args:
            X_val: Features de validação
            y_val: Labels de validação

        Returns:
            Threshold otimizado
        """
        try:
            if self.calibrated_model is None:
                raise ValueError("Calibrated model not initialized")

            probas = self.calibrated_model.predict_proba(X_val)[:, 1]

            best_threshold = 0.5
            best_f1 = 0.0

            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred = (probas >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division="warn")

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            logger.info(
                "Threshold calibrated",
                threshold=round(best_threshold, 3),
                f1_score=round(best_f1, 3),
            )

            return float(best_threshold)

        except Exception as e:
            logger.error("Threshold calibration failed", exception=e)
            return 0.5

    @log_execution_time(logger)
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ProductionFraudEngine":
        """
        Treina o modelo

        Args:
            X: DataFrame com features
            y: Labels (0 = legítimo, 1 = fraude)

        Returns:
            Self (permite chaining)
        """
        try:
            logger.info(
                "Starting model training", num_samples=len(X), fraud_rate=round(y.mean() * 100, 2)
            )

            # Preprocessamento
            X_processed = self._preprocess_data(X, fit_transform=True)

            # Split para validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )

            # Converter para arrays numpy se necessário
            X_val_array = np.asarray(X_val)
            y_val_array = np.asarray(y_val)

            # Treinar ensemble
            logger.info("Training ensemble...")
            self.ensemble.fit(X_train, y_train)

            # Calibrar probabilidades
            logger.info("Calibrating probabilities...")
            self.calibrated_model = CalibratedClassifierCV(self.ensemble, cv=3, method="isotonic")
            self.calibrated_model.fit(X_train, y_train)

            # Calibrar threshold
            self.threshold = self._calibrate_threshold(X_val_array, y_val_array)

            # Calcular métricas finais
            if self.calibrated_model is None:
                raise ValueError("Calibrated model not initialized")

            y_pred_proba = self.calibrated_model.predict_proba(X_val_array)[:, 1]
            y_pred = (y_pred_proba >= self.threshold).astype(int)

            self.metrics = ModelMetrics(
                accuracy=float(accuracy_score(y_val_array, y_pred)),
                precision=float(precision_score(y_val_array, y_pred, zero_division="warn")),
                recall=float(recall_score(y_val_array, y_pred, zero_division="warn")),
                f1_score=float(f1_score(y_val_array, y_pred, zero_division="warn")),
                roc_auc=float(roc_auc_score(y_val_array, y_pred_proba)),
                threshold=self.threshold,
                timestamp=datetime.utcnow().isoformat() + "Z",
            )

            self.is_trained = True

            logger.info(
                "Model training completed",
                accuracy=round(self.metrics.accuracy, 3),
                precision=round(self.metrics.precision, 3),
                recall=round(self.metrics.recall, 3),
                f1_score=round(self.metrics.f1_score, 3),
                roc_auc=round(self.metrics.roc_auc, 3),
            )

            return self

        except Exception as e:
            logger.error("Model training failed", exception=e)
            raise

    @log_execution_time(logger)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições de fraude (sklearn-compatible)

        Args:
            X: DataFrame com features

        Returns:
            Array numpy com labels binários (0 = legítimo, 1 = fraude)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        try:
            # Preprocessar
            X_processed = self._preprocess_data(X, fit_transform=False)

            # Validar modelo
            if self.calibrated_model is None:
                raise ValueError("Calibrated model not initialized. Call fit() first.")

            # Predições
            y_proba = self.calibrated_model.predict_proba(X_processed)[:, 1]
            y_pred = (y_proba >= self.threshold).astype(int)

            logger.info(
                "Predictions completed",
                num_predictions=len(y_pred),
                num_frauds=int(y_pred.sum()),
                avg_time_ms=round((time.time() * 1000) / len(X) if len(X) > 0 else 0, 2),
            )

            return y_pred

        except Exception as e:
            logger.error("Prediction failed", exception=e)
            raise

    @log_execution_time(logger)
    def predict_detailed(self, X: pd.DataFrame, apply_rules: bool = True) -> List[FraudPrediction]:
        """
        Faz predições detalhadas de fraude com metadados completos

        Args:
            X: DataFrame com features
            apply_rules: Se True, aplica regras de precision boosting

        Returns:
            Lista de predições detalhadas com risk levels, razões, etc.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        try:
            start_time = time.time()

            # Preprocessar
            X_processed = self._preprocess_data(X, fit_transform=False)

            # Validar modelo
            if self.calibrated_model is None:
                raise ValueError("Calibrated model not initialized. Call fit() first.")

            # Predições
            y_proba = self.calibrated_model.predict_proba(X_processed)[:, 1]

            # Aplicar regras de precisão
            if apply_rules:
                y_proba = self._apply_precision_rules(X, y_proba)

            # Tempo total
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(X)

            # Criar predições
            predictions = []
            for idx, proba in enumerate(y_proba):
                is_fraud = proba >= self.threshold

                # Determinar risk level
                if proba >= 0.95:
                    risk_level = "CRITICAL"
                elif proba >= 0.80:
                    risk_level = "HIGH"
                elif proba >= 0.50:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"

                # Razões de detecção
                reasons = []
                if is_fraud:
                    reasons.append(f"High fraud probability: {proba:.2%}")
                    if apply_rules:
                        # Verificar quais regras matcharam
                        row = X.iloc[idx]
                        if "amount" in X.columns and "hour" in X.columns:
                            if row["amount"] >= 50000 and row["hour"] in [0, 1, 2, 3, 4, 23]:
                                reasons.append("Extreme amount in suspicious hour")

                prediction = FraudPrediction(
                    transaction_id=str(idx),
                    is_fraud=bool(is_fraud),
                    fraud_probability=float(proba),
                    risk_score=float(proba),
                    risk_level=risk_level,
                    confidence=float(self.metrics.f1_score) if self.metrics else 0.0,
                    processing_time_ms=round(avg_time, 2),
                    model_version=self.VERSION,
                    detection_reason=reasons,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )

                predictions.append(prediction)

            logger.info(
                "Detailed predictions completed",
                num_predictions=len(predictions),
                num_frauds=sum(1 for p in predictions if p.is_fraud),
                avg_time_ms=round(avg_time, 2),
            )

            return predictions

        except Exception as e:
            logger.error("Detailed prediction failed", exception=e)
            raise

    def save(self, filepath: Optional[str] = None):
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        try:
            if filepath is None:
                filepath = str(self.model_path / f"fraud_engine_v{self.VERSION}.joblib")

            model_data = {
                "version": self.VERSION,
                "ensemble": self.ensemble,
                "calibrated_model": self.calibrated_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "threshold": self.threshold,
                "metrics": asdict(self.metrics) if self.metrics else None,
                "precision_rules": self.precision_rules,
            }

            joblib.dump(model_data, filepath)

            logger.info("Model saved", filepath=str(filepath), version=self.VERSION)

        except Exception as e:
            logger.error("Model save failed", exception=e)
            raise

    def load(self, filepath: str):
        """Carrega modelo salvo"""
        try:
            model_data = joblib.load(filepath)

            self.ensemble = model_data["ensemble"]
            self.calibrated_model = model_data["calibrated_model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.threshold = model_data["threshold"]
            self.precision_rules = model_data["precision_rules"]

            if model_data["metrics"]:
                self.metrics = ModelMetrics(**model_data["metrics"])

            self.is_trained = True

            logger.info("Model loaded", filepath=filepath, version=model_data["version"])

        except Exception as e:
            logger.error("Model load failed", exception=e)
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance"""
        if not self.metrics:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "version": self.VERSION,
            "metrics": asdict(self.metrics),
            "threshold": self.threshold,
            "feature_count": len(self.feature_names),
        }


# Instância global (singleton)
_engine_instance: Optional[ProductionFraudEngine] = None


def get_fraud_engine() -> ProductionFraudEngine:
    """Factory para obter instância do engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ProductionFraudEngine()
    return _engine_instance
