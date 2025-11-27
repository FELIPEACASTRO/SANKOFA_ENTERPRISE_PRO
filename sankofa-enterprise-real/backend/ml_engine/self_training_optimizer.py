"""
Sankofa Enterprise Pro - Self-Training Optimizer
Aprendizado semi-supervisionado para detecção de fraude
Baseado em: AIForge - Semi-Supervised Learning for Fraud Detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class SelfTrainingMetrics:
    """Métricas de self-training"""
    iterations: int
    pseudo_labels_added: int
    initial_accuracy: float
    final_accuracy: float
    improvement: float
    high_confidence_ratio: float


class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador com Self-Training para Detecção de Fraude
    
    Usa dados não rotulados para melhorar o modelo através de pseudo-labeling.
    
    Funcionalidades:
    - Pseudo-labeling com threshold de confiança
    - Iterações controladas
    - Validação de qualidade dos pseudo-labels
    - Suporte a class imbalance
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        base_classifier=None,
        threshold: float = 0.95,
        max_iter: int = 10,
        min_samples_per_iter: int = 10,
        fraud_threshold: float = 0.9,
        non_fraud_threshold: float = 0.1
    ):
        """
        Inicializa o classificador
        
        Args:
            base_classifier: Classificador base (default: RandomForest)
            threshold: Threshold de confiança para pseudo-labeling
            max_iter: Máximo de iterações
            min_samples_per_iter: Mínimo de amostras por iteração
            fraud_threshold: Threshold para considerar fraude
            non_fraud_threshold: Threshold para considerar não-fraude
        """
        self.base_classifier = base_classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.threshold = threshold
        self.max_iter = max_iter
        self.min_samples_per_iter = min_samples_per_iter
        self.fraud_threshold = fraud_threshold
        self.non_fraud_threshold = non_fraud_threshold
        
        self.is_fitted = False
        self.training_history: List[Dict[str, Any]] = []
        self.metrics: Optional[SelfTrainingMetrics] = None
        
        logger.info(f"SelfTrainingClassifier initialized v{self.VERSION}")
    
    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray = None
    ) -> "SelfTrainingClassifier":
        """
        Treina o modelo com self-training
        
        Args:
            X_labeled: Features rotuladas
            y_labeled: Labels
            X_unlabeled: Features não rotuladas
            
        Returns:
            Self
        """
        logger.info(
            f"Starting self-training: labeled_samples={len(X_labeled)}, "
            f"unlabeled_samples={len(X_unlabeled) if X_unlabeled is not None else 0}"
        )
        
        self.base_classifier.fit(X_labeled, y_labeled)
        
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.is_fitted = True
            logger.info("No unlabeled data, training completed without self-training")
            return self
        
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        
        total_pseudo_labels = 0
        initial_score = self._evaluate_model(X_labeled, y_labeled)
        
        for iteration in range(self.max_iter):
            if len(X_pool) == 0:
                logger.info(f"No more unlabeled samples at iteration {iteration}")
                break
            
            probs = self.base_classifier.predict_proba(X_pool)[:, 1]
            
            high_fraud_mask = probs >= self.fraud_threshold
            high_non_fraud_mask = probs <= self.non_fraud_threshold
            high_confidence_mask = high_fraud_mask | high_non_fraud_mask
            
            if high_confidence_mask.sum() < self.min_samples_per_iter:
                logger.info(f"Not enough high-confidence samples at iteration {iteration}")
                break
            
            X_pseudo = X_pool[high_confidence_mask]
            y_pseudo = (probs[high_confidence_mask] >= 0.5).astype(int)
            
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])
            
            X_pool = X_pool[~high_confidence_mask]
            
            self.base_classifier.fit(X_train, y_train)
            
            iter_score = self._evaluate_model(X_labeled, y_labeled)
            
            self.training_history.append({
                "iteration": iteration + 1,
                "pseudo_labels_added": len(X_pseudo),
                "total_training_samples": len(X_train),
                "remaining_unlabeled": len(X_pool),
                "validation_score": iter_score,
                "fraud_pseudo_labels": y_pseudo.sum(),
                "non_fraud_pseudo_labels": len(y_pseudo) - y_pseudo.sum()
            })
            
            total_pseudo_labels += len(X_pseudo)
            
            logger.info(
                f"Iteration {iteration + 1}: added {len(X_pseudo)} pseudo-labels, "
                f"score: {iter_score:.4f}"
            )
        
        final_score = self._evaluate_model(X_labeled, y_labeled)
        
        self.metrics = SelfTrainingMetrics(
            iterations=len(self.training_history),
            pseudo_labels_added=total_pseudo_labels,
            initial_accuracy=initial_score,
            final_accuracy=final_score,
            improvement=final_score - initial_score,
            high_confidence_ratio=total_pseudo_labels / (len(X_unlabeled) + 1e-6)
        )
        
        self.is_fitted = True
        
        logger.info(
            f"Self-training completed: iterations={self.metrics.iterations}, "
            f"pseudo_labels={self.metrics.pseudo_labels_added}, "
            f"improvement={self.metrics.improvement:.4f}"
        )
        
        return self
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Avalia o modelo atual"""
        try:
            y_pred_proba = self.base_classifier.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred_proba)
        except Exception:
            return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.base_classifier.predict_proba(X)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de treinamento"""
        return self.training_history
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas"""
        if self.metrics is None:
            return {"status": "not_fitted"}
        
        return {
            "iterations": self.metrics.iterations,
            "pseudo_labels_added": self.metrics.pseudo_labels_added,
            "initial_accuracy": round(self.metrics.initial_accuracy, 4),
            "final_accuracy": round(self.metrics.final_accuracy, 4),
            "improvement": round(self.metrics.improvement, 4),
            "high_confidence_ratio": round(self.metrics.high_confidence_ratio, 4),
            "version": self.VERSION
        }


class AdaptiveSelfTraining:
    """
    Self-Training Adaptativo com Validação de Qualidade
    
    Monitora a qualidade dos pseudo-labels e ajusta thresholds automaticamente.
    """
    
    def __init__(
        self,
        base_classifier=None,
        initial_threshold: float = 0.95,
        adaptation_rate: float = 0.02,
        quality_check_interval: int = 3
    ):
        """
        Inicializa o self-training adaptativo
        
        Args:
            base_classifier: Classificador base
            initial_threshold: Threshold inicial de confiança
            adaptation_rate: Taxa de adaptação do threshold
            quality_check_interval: Intervalo para verificar qualidade
        """
        self.base_classifier = base_classifier or RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.quality_check_interval = quality_check_interval
        
        self.is_fitted = False
        self.threshold_history: List[float] = []
        self.quality_history: List[float] = []
    
    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        max_iter: int = 10
    ) -> "AdaptiveSelfTraining":
        """
        Treina com adaptação automática de threshold
        """
        self.base_classifier.fit(X_labeled, y_labeled)
        
        if X_unlabeled is None or len(X_unlabeled) == 0:
            self.is_fitted = True
            return self
        
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        
        self.current_threshold = self.initial_threshold
        
        for iteration in range(max_iter):
            if len(X_pool) == 0:
                break
            
            probs = self.base_classifier.predict_proba(X_pool)[:, 1]
            
            high_confidence_mask = (probs >= self.current_threshold) | (probs <= (1 - self.current_threshold))
            
            if high_confidence_mask.sum() == 0:
                self.current_threshold -= self.adaptation_rate
                self.current_threshold = max(0.7, self.current_threshold)
                continue
            
            X_pseudo = X_pool[high_confidence_mask]
            y_pseudo = (probs[high_confidence_mask] >= 0.5).astype(int)
            
            X_train = np.vstack([X_train, X_pseudo])
            y_train = np.concatenate([y_train, y_pseudo])
            X_pool = X_pool[~high_confidence_mask]
            
            self.base_classifier.fit(X_train, y_train)
            
            if (iteration + 1) % self.quality_check_interval == 0 and X_val is not None:
                quality = self._check_quality(X_val, y_val)
                self.quality_history.append(quality)
                
                if len(self.quality_history) > 1:
                    if quality < self.quality_history[-2]:
                        self.current_threshold += self.adaptation_rate
                        self.current_threshold = min(0.99, self.current_threshold)
                    else:
                        self.current_threshold -= self.adaptation_rate * 0.5
                        self.current_threshold = max(0.7, self.current_threshold)
            
            self.threshold_history.append(self.current_threshold)
        
        self.is_fitted = True
        return self
    
    def _check_quality(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Verifica qualidade do modelo atual"""
        try:
            y_pred_proba = self.base_classifier.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
        except Exception:
            return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        return self.base_classifier.predict_proba(X)


def create_unlabeled_from_production(
    df: pd.DataFrame,
    labeled_mask: np.ndarray = None,
    sample_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria conjuntos rotulados e não-rotulados a partir de dados de produção
    
    Args:
        df: DataFrame com dados
        labeled_mask: Máscara indicando quais são rotulados
        sample_ratio: Proporção de dados rotulados
        
    Returns:
        Tuple (X_labeled, y_labeled, X_unlabeled, y_unlabeled_true)
    """
    if labeled_mask is None:
        n_samples = len(df)
        labeled_mask = np.random.random(n_samples) < sample_ratio
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_col = None
    for col in ["is_fraud", "fraud", "label", "target"]:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("No target column found")
    
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    X_unlabeled = X[~labeled_mask]
    y_unlabeled_true = y[~labeled_mask]
    
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled_true


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Self-Training Classifier ===\n")
    
    np.random.seed(42)
    n_labeled = 500
    n_unlabeled = 2000
    n_features = 10
    
    X_labeled = np.random.randn(n_labeled, n_features)
    y_labeled = (X_labeled[:, 0] + X_labeled[:, 1] > 0).astype(int)
    y_labeled[np.random.random(n_labeled) < 0.1] = 1 - y_labeled[np.random.random(n_labeled) < 0.1]
    
    X_unlabeled = np.random.randn(n_unlabeled, n_features)
    y_unlabeled_true = (X_unlabeled[:, 0] + X_unlabeled[:, 1] > 0).astype(int)
    
    clf = SelfTrainingClassifier(
        threshold=0.95,
        max_iter=5,
        fraud_threshold=0.9,
        non_fraud_threshold=0.1
    )
    
    clf.fit(X_labeled, y_labeled, X_unlabeled)
    
    print("\nTraining History:")
    for record in clf.get_training_history():
        print(f"  Iteration {record['iteration']}: "
              f"added {record['pseudo_labels_added']} labels, "
              f"score {record['validation_score']:.4f}")
    
    print("\nMetrics Summary:")
    metrics = clf.get_metrics_summary()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    X_test = np.random.randn(100, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
