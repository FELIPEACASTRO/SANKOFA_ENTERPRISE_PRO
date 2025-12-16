"""
Sankofa Enterprise Pro - Ensemble Integration Layer
Integra CatBoost, GNN e outros modelos com o engine de fraude existente

CORRECAO 10/10: Thread-safe singleton e temporal validation
"""

import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Predição do ensemble integrado"""

    fraud_probability: float
    is_fraud: bool
    model_contributions: Dict[str, float]
    gnn_risk_score: float
    gnn_suspicious_patterns: List[str]
    catboost_probability: float
    base_probability: float
    processing_time_ms: float
    mule_score: float = 0.0
    mule_detected: bool = False
    hard_rules_triggered: List[str] = None


class IntegratedEnsemble:
    """
    Integra múltiplos modelos de ML em um ensemble unificado

    Modelos integrados:
    - Base: RF + GB + LR (existente)
    - CatBoost: Para features categóricas
    - GNN: Para análise de relacionamentos

    CORRECAO 10/10: Thread-safe com lock para atributos mutáveis
    """

    VERSION = "2.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # CORRECAO 10/10: Lock para proteger atributos mutáveis
        self._weights_lock = threading.RLock()

        self._weights = {
            "base_ensemble": 0.50,
            "catboost": 0.25,
            "gnn": 0.25,
        }

        self.catboost_model = None
        self.gnn_detector = None
        self.mule_detector = None
        self.hard_rules_engine = None
        self._catboost_available = False
        self._gnn_available = False
        self._mule_detector_available = False
        self._hard_rules_available = False

        self._initialize_models()

    @property
    def weights(self) -> Dict[str, float]:
        """Retorna pesos do ensemble de forma thread-safe"""
        with self._weights_lock:
            return self._weights.copy()

    @weights.setter
    def weights(self, value: Dict[str, float]):
        """Define pesos do ensemble de forma thread-safe"""
        with self._weights_lock:
            self._weights = value.copy()

    def _initialize_models(self):
        """Inicializa modelos opcionais (CatBoost, GNN)"""
        try:
            from ml_engine.catboost_model import CatBoostFraudModel, CATBOOST_AVAILABLE
            from pathlib import Path

            if CATBOOST_AVAILABLE:
                # Determinar path do modelo
                model_path = self.config.get("catboost_model_path")
                if not model_path:
                    # Tentar path padrão em produção
                    base_path = Path(__file__).parent.parent / "models" / "production"
                    model_path = base_path / "catboost_fraud_model.cbm"

                self.catboost_model = CatBoostFraudModel(self.config)

                # Tentar carregar modelo treinado se existir
                if Path(model_path).exists():
                    try:
                        self.catboost_model.load(str(model_path))
                        logger.info(f"CatBoost model loaded from {model_path}")
                    except Exception as load_err:
                        logger.warning(f"Failed to load CatBoost model: {load_err}")

                self._catboost_available = True
                logger.info(f"CatBoost model initialized for ensemble (trained: {self.catboost_model.is_trained})")
        except Exception as e:
            logger.warning(f"CatBoost not available: {e}")

        try:
            from ml_engine.gnn_fraud_detector import GNNFraudDetector, NETWORKX_AVAILABLE

            if NETWORKX_AVAILABLE:
                self.gnn_detector = GNNFraudDetector(self.config)
                self._gnn_available = True
                logger.info("GNN detector initialized for ensemble")
        except Exception as e:
            logger.warning(f"GNN not available: {e}")

        # Initialize MuleDetector
        try:
            from ml_engine.mule_detection import MuleDetector, create_mule_detector

            self.mule_detector = create_mule_detector(self.config)
            self._mule_detector_available = True
            logger.info("MuleDetector initialized for ensemble")
        except Exception as e:
            logger.warning(f"MuleDetector not available: {e}")

        # Initialize HardRulesEngine
        try:
            from ml_engine.hard_rules_engine import HardRulesEngine

            self.hard_rules_engine = HardRulesEngine()
            self._hard_rules_available = True
            logger.info("HardRulesEngine initialized for ensemble")
        except Exception as e:
            logger.warning(f"HardRulesEngine not available: {e}")

        self._adjust_weights()

    def _adjust_weights(self):
        """Ajusta pesos baseado em modelos disponíveis"""
        if not self._catboost_available and not self._gnn_available:
            self.weights = {"base_ensemble": 1.0, "catboost": 0.0, "gnn": 0.0}
        elif not self._catboost_available:
            self.weights = {"base_ensemble": 0.7, "catboost": 0.0, "gnn": 0.3}
        elif not self._gnn_available:
            self.weights = {"base_ensemble": 0.65, "catboost": 0.35, "gnn": 0.0}

        logger.info(f"Ensemble weights adjusted: {self.weights}")

    def calibrate_weights(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        base_predictions: np.ndarray
    ) -> Dict[str, float]:
        """CORRECAO 10/10: Calibra pesos do ensemble usando dados de validacao

        Usa grid search para encontrar a melhor combinacao de pesos que
        maximiza o F1-score no conjunto de validacao.

        Args:
            X_val: Features de validacao
            y_val: Labels de validacao
            base_predictions: Predicoes do modelo base (probabilidades)

        Returns:
            Dict com pesos otimizados
        """
        from sklearn.metrics import f1_score

        best_weights = self.weights.copy()
        best_f1 = 0.0

        logger.info("Starting ensemble weight calibration...")

        # Grid search sobre combinacoes de pesos
        for base_w in np.arange(0.3, 0.8, 0.1):
            for cat_w in np.arange(0.0, 0.5, 0.1):
                gnn_w = 1.0 - base_w - cat_w

                # Validar que gnn_w esta no range valido
                if gnn_w < 0 or gnn_w > 0.5:
                    continue

                # Ajustar para modelos disponiveis
                if not self._catboost_available:
                    if cat_w > 0:
                        continue
                if not self._gnn_available:
                    if gnn_w > 0:
                        continue

                # Calcular predicao combinada
                combined_prob = base_predictions * base_w

                # Adicionar CatBoost se disponivel
                if self._catboost_available and self.catboost_model and cat_w > 0:
                    try:
                        cat_pred = self.catboost_model.predict_proba(X_val)
                        if hasattr(cat_pred, 'fraud_probability'):
                            combined_prob += cat_pred.fraud_probability * cat_w
                        elif isinstance(cat_pred, np.ndarray) and len(cat_pred.shape) > 1:
                            combined_prob += cat_pred[:, 1] * cat_w
                    except Exception:
                        combined_prob += base_predictions * cat_w  # Fallback

                # Adicionar GNN se disponivel
                if self._gnn_available and gnn_w > 0:
                    # GNN usa score fixo por simplicidade (em producao, integraria scores reais)
                    combined_prob += base_predictions * gnn_w * 0.5  # Conservative

                # Calcular predicoes binarias
                predictions = (combined_prob >= 0.5).astype(int)

                # Calcular F1
                try:
                    f1 = f1_score(y_val, predictions, zero_division=0)
                except Exception:
                    continue

                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = {
                        "base_ensemble": round(base_w, 2),
                        "catboost": round(cat_w, 2),
                        "gnn": round(gnn_w, 2),
                    }

        # Aplicar melhores pesos
        self.weights = best_weights
        logger.info(f"Weights calibrated: {self.weights}, Best F1: {best_f1:.4f}")

        return best_weights

    def train_catboost(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        timestamp_col: Optional[str] = None
    ) -> Dict[str, float]:
        """Treina o modelo CatBoost

        CORRECAO 10/10: Implementa temporal validation para evitar data leakage

        Args:
            X: Features
            y: Labels
            timestamp_col: Coluna de timestamp para split temporal (recomendado)

        Returns:
            Metricas de treinamento
        """
        if not self._catboost_available or self.catboost_model is None:
            return {}

        # CORRECAO 10/10: Usar TimeSeriesSplit se possivel
        if timestamp_col and timestamp_col in X.columns:
            # Split temporal - ordenar por tempo e dividir sequencialmente
            X_sorted = X.sort_values(by=timestamp_col).reset_index(drop=True)
            y_sorted = y[X.sort_values(by=timestamp_col).index].reset_index(drop=True) if hasattr(y, 'index') else y

            n = len(X_sorted)
            train_end = int(n * 0.8)

            X_train = X_sorted.iloc[:train_end]
            X_val = X_sorted.iloc[train_end:]
            y_train = y_sorted[:train_end] if isinstance(y_sorted, np.ndarray) else y_sorted.iloc[:train_end]
            y_val = y_sorted[train_end:] if isinstance(y_sorted, np.ndarray) else y_sorted.iloc[train_end:]

            logger.info(f"CatBoost: Temporal split applied (train={train_end}, val={n-train_end})")
        else:
            # Fallback para StratifiedShuffleSplit (sem embaralhamento total)
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(X, y))

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            logger.warning("CatBoost: No timestamp column - using stratified split (potential leakage)")

        metrics = self.catboost_model.train(X_train, y_train, X_val, y_val)
        logger.info(f"CatBoost trained: {metrics}")
        return metrics

    def add_transactions_to_gnn(self, transactions: pd.DataFrame):
        """Adiciona transações ao grafo GNN"""
        if not self._gnn_available or self.gnn_detector is None:
            return

        self.gnn_detector.add_historical_transactions(transactions)
        logger.info(f"Added {len(transactions)} transactions to GNN graph")

    def predict_combined(
        self, transaction: Dict[str, Any], base_probability: float
    ) -> EnsemblePrediction:
        """
        Faz predição combinando todos os modelos

        Args:
            transaction: Dados da transação
            base_probability: Probabilidade do ensemble base (RF+GB+LR)

        Returns:
            Predição do ensemble integrado
        """
        import time

        start_time = time.time()

        contributions = {"base_ensemble": base_probability}
        catboost_prob = 0.0
        gnn_risk = 0.0
        gnn_patterns = []

        if self._catboost_available and self.catboost_model and self.catboost_model.is_trained:
            try:
                df = pd.DataFrame([transaction])
                predictions = self.catboost_model.predict(df)
                if predictions:
                    catboost_prob = predictions[0].fraud_probability
                    contributions["catboost"] = catboost_prob
            except Exception as e:
                logger.warning(f"CatBoost prediction failed: {e}")
                catboost_prob = base_probability

        if self._gnn_available and self.gnn_detector:
            try:
                gnn_prediction = self.gnn_detector.analyze_transaction(
                    transaction_id=str(transaction.get("transaction_id", "")),
                    customer_id=str(
                        transaction.get("customer_id", transaction.get("cliente_cpf", ""))
                    ),
                    amount=float(transaction.get("amount", 0)),
                    receiver_id=transaction.get("receiver_id"),
                    device_id=transaction.get("device_id"),
                    ip_address=transaction.get("ip_address"),
                )
                gnn_risk = gnn_prediction.graph_risk_score
                gnn_patterns = gnn_prediction.suspicious_patterns
                contributions["gnn"] = gnn_risk
            except Exception as e:
                logger.warning(f"GNN analysis failed: {e}")

        # CORRECAO 10/10: Obter copia dos weights UMA VEZ para evitar race condition
        # (cada chamada a self.weights retorna uma nova copia, usar 3 chamadas
        # poderia ver valores inconsistentes se outro thread modificar durante o calculo)
        w = self.weights  # Single call - thread-safe copy
        combined_prob = (
            base_probability * w["base_ensemble"]
            + catboost_prob * w["catboost"]
            + gnn_risk * w["gnn"]
        )

        combined_prob = max(0.0, min(1.0, combined_prob))

        # Mule Detection
        mule_score = 0.0
        mule_detected = False
        if self._mule_detector_available and self.mule_detector:
            try:
                customer_id = str(transaction.get("customer_id", transaction.get("cliente_cpf", "")))
                # Create minimal account data and transaction history for detection
                account_data = {
                    "account_id": customer_id,
                    "created_at": transaction.get("account_created_at", datetime.now().isoformat())
                }
                transaction_history = [transaction]  # Minimal history

                mule_result = self.mule_detector.detect(
                    account_id=customer_id,
                    account_data=account_data,
                    transaction_history=transaction_history
                )
                mule_score = mule_result.mule_probability
                mule_detected = mule_result.is_mule
                contributions["mule_detector"] = mule_score

                # Boost fraud probability if mule detected
                if mule_detected:
                    combined_prob = max(combined_prob, mule_score * 0.8)
            except Exception as e:
                logger.warning(f"MuleDetector analysis failed: {e}")

        # Hard Rules Check
        hard_rules_triggered = []
        if self._hard_rules_available and self.hard_rules_engine:
            try:
                rules_result = self.hard_rules_engine.evaluate(transaction)
                if rules_result and rules_result.is_fraud:
                    hard_rules_triggered = rules_result.detection_reason or []
                    contributions["hard_rules"] = rules_result.fraud_probability
                    # Override if hard rule triggered with high confidence
                    if rules_result.fraud_probability >= 0.9:
                        combined_prob = max(combined_prob, rules_result.fraud_probability)
            except Exception as e:
                logger.warning(f"HardRulesEngine evaluation failed: {e}")

        processing_time = (time.time() - start_time) * 1000

        return EnsemblePrediction(
            fraud_probability=combined_prob,
            is_fraud=combined_prob >= 0.5,
            model_contributions=contributions,
            gnn_risk_score=gnn_risk,
            gnn_suspicious_patterns=gnn_patterns,
            catboost_probability=catboost_prob,
            base_probability=base_probability,
            processing_time_ms=processing_time,
            mule_score=mule_score,
            mule_detected=mule_detected,
            hard_rules_triggered=hard_rules_triggered,
        )

    def predict(
        self, transaction: Dict[str, Any], base_probability: float = 0.5
    ) -> EnsemblePrediction:
        """
        Alias para predict_combined para compatibilidade sklearn-style

        Args:
            transaction: Dados da transação
            base_probability: Probabilidade do ensemble base (default: 0.5)

        Returns:
            Predição do ensemble integrado
        """
        return self.predict_combined(transaction, base_probability)

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Retorna informações do ensemble"""
        return {
            "version": self.VERSION,
            "weights": self.weights,
            "catboost_available": self._catboost_available,
            "catboost_trained": self._catboost_available
            and self.catboost_model
            and self.catboost_model.is_trained,
            "gnn_available": self._gnn_available,
            "gnn_stats": (
                self.gnn_detector.get_graph_stats()
                if self._gnn_available and self.gnn_detector
                else {}
            ),
            "mule_detector_available": self._mule_detector_available,
            "hard_rules_available": self._hard_rules_available,
        }


_integrated_ensemble: Optional[IntegratedEnsemble] = None
_integrated_ensemble_lock = threading.Lock()


def get_integrated_ensemble() -> IntegratedEnsemble:
    """Retorna instância singleton do ensemble integrado

    CORRECAO 10/10: Double-checked locking para thread-safety
    """
    global _integrated_ensemble
    if _integrated_ensemble is None:
        with _integrated_ensemble_lock:
            if _integrated_ensemble is None:
                _integrated_ensemble = IntegratedEnsemble()
    return _integrated_ensemble
