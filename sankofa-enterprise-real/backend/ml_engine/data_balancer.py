import logging

logger = logging.getLogger(__name__)
"""
Data Balancer - Balanceia dataset de fraude usando técnicas de resampling
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")


class DataBalancer:
    """
    Balanceia dataset de fraude usando técnicas de resampling ou class weights.
    """

    def __init__(self, method: str = "class_weights"):
        """
        Args:
            method: 'class_weights', 'undersample', 'oversample', 'hybrid'
        """
        self.method = method
        self.class_weights = None

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balanceia o dataset.

        Args:
            X: Features
            y: Labels (0=legítimo, 1=fraude)

        Returns:
            X_balanced, y_balanced
        """
        logger.info(f"Dataset original: {len(y)} samples")
        logger.info(f"  - Legítimas: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        logger.info(f"  - Fraudes: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

        if self.method == "class_weights":
            # Apenas calcular pesos, não modificar dataset
            classes = np.unique(y)
            class_weights = compute_class_weight("balanced", classes=classes, y=y)
            self.class_weights = {classes[i]: class_weights[i] for i in range(len(classes))}
            logger.info(f"\nClass weights calculados: {self.class_weights}")
            return X, y

        elif self.method == "undersample":
            return self._undersample(X, y)

        elif self.method == "oversample":
            return self._oversample(X, y)

        elif self.method == "hybrid":
            return self._hybrid_sampling(X, y)

        else:
            raise ValueError(f"Método inválido: {self.method}")

    def _undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random under-sampling da classe majoritária."""
        # Índices de cada classe
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        # Reduzir classe majoritária para o tamanho da minoritária
        if len(legit_idx) > len(fraud_idx):
            legit_idx_sampled = np.random.choice(legit_idx, size=len(fraud_idx), replace=False)
            balanced_idx = np.concatenate([fraud_idx, legit_idx_sampled])
        else:
            fraud_idx_sampled = np.random.choice(fraud_idx, size=len(legit_idx), replace=False)
            balanced_idx = np.concatenate([fraud_idx_sampled, legit_idx])

        # Embaralhar
        np.random.shuffle(balanced_idx)

        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]

        logger.info(f"\nDataset balanceado (undersample): {len(y_balanced)} samples")
        logger.info(
            f"  - Legítimas: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)"
        )
        logger.info(
            f"  - Fraudes: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)"
        )

        return X_balanced, y_balanced

    def _oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random over-sampling da classe minoritária."""
        # Índices de cada classe
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        # Aumentar classe minoritária para o tamanho da majoritária
        if len(fraud_idx) < len(legit_idx):
            fraud_idx_sampled = np.random.choice(fraud_idx, size=len(legit_idx), replace=True)
            balanced_idx = np.concatenate([fraud_idx_sampled, legit_idx])
        else:
            legit_idx_sampled = np.random.choice(legit_idx, size=len(fraud_idx), replace=True)
            balanced_idx = np.concatenate([fraud_idx, legit_idx_sampled])

        # Embaralhar
        np.random.shuffle(balanced_idx)

        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]

        logger.info(f"\nDataset balanceado (oversample): {len(y_balanced)} samples")
        logger.info(
            f"  - Legítimas: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)"
        )
        logger.info(
            f"  - Fraudes: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)"
        )

        return X_balanced, y_balanced

    def _hybrid_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Híbrido: undersample + oversample para balancear no meio."""
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        # Calcular tamanho alvo (média entre as duas classes)
        target_size = (len(fraud_idx) + len(legit_idx)) // 2

        # Ajustar ambas as classes para o tamanho alvo
        if len(fraud_idx) < target_size:
            fraud_idx_sampled = np.random.choice(fraud_idx, size=target_size, replace=True)
        else:
            fraud_idx_sampled = np.random.choice(fraud_idx, size=target_size, replace=False)

        if len(legit_idx) < target_size:
            legit_idx_sampled = np.random.choice(legit_idx, size=target_size, replace=True)
        else:
            legit_idx_sampled = np.random.choice(legit_idx, size=target_size, replace=False)

        balanced_idx = np.concatenate([fraud_idx_sampled, legit_idx_sampled])
        np.random.shuffle(balanced_idx)

        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]

        logger.info(f"\nDataset balanceado (hybrid): {len(y_balanced)} samples")
        logger.info(
            f"  - Legítimas: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)"
        )
        logger.info(
            f"  - Fraudes: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)"
        )

        return X_balanced, y_balanced

    def get_class_weights(self) -> dict:
        """Retorna os class weights calculados."""
        return self.class_weights
