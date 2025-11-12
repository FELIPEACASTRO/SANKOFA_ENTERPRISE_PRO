import logging

logger = logging.getLogger(__name__)
"""
Threshold Optimizer - Otimiza o threshold de decisão para maximizar F1-Score
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class ThresholdOptimizer:
    """
    Otimiza o threshold de decisão para maximizar F1-Score.
    """

    def __init__(self, target_precision: float = 0.80, target_recall: float = 0.75):
        """
        Args:
            target_precision: Precision mínima desejada (0.80 = 80%)
            target_recall: Recall mínimo desejado (0.75 = 75%)
        """
        self.target_precision = target_precision
        self.target_recall = target_recall

    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """
        Encontra o threshold ótimo que maximiza F1-Score.

        Args:
            y_true: Labels verdadeiros (0 ou 1)
            y_proba: Probabilidades preditas (0.0 a 1.0)

        Returns:
            Dict com threshold ótimo e métricas
        """
        # Calcular precision e recall para diferentes thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Calcular F1-Score para cada threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Encontrar threshold que maximiza F1-Score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        # Encontrar threshold que atende aos requisitos mínimos
        valid_idx = np.where(
            (precisions[:-1] >= self.target_precision) & (recalls[:-1] >= self.target_recall)
        )[0]

        if len(valid_idx) > 0:
            # Usar o threshold que maximiza F1 entre os válidos
            valid_f1 = f1_scores[valid_idx]
            best_valid_idx = valid_idx[np.argmax(valid_f1)]
            recommended_threshold = thresholds[best_valid_idx]
            recommended_f1 = f1_scores[best_valid_idx]
            recommended_precision = precisions[best_valid_idx]
            recommended_recall = recalls[best_valid_idx]
        else:
            # Nenhum threshold atende aos requisitos, usar o melhor F1
            recommended_threshold = best_threshold
            recommended_f1 = best_f1
            recommended_precision = best_precision
            recommended_recall = best_recall

        return {
            "optimal_threshold": float(recommended_threshold),
            "f1_score": float(recommended_f1),
            "precision": float(recommended_precision),
            "recall": float(recommended_recall),
            "meets_requirements": (
                recommended_precision >= self.target_precision
                and recommended_recall >= self.target_recall
            ),
        }

    def plot_threshold_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None
    ):
        """
        Plota análise de threshold (Precision-Recall e ROC).
        """
        # Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Precision, Recall, F1 vs Threshold
        axes[0].plot(thresholds, precisions[:-1], label="Precision", linewidth=2)
        axes[0].plot(thresholds, recalls[:-1], label="Recall", linewidth=2)
        axes[0].plot(thresholds, f1_scores[:-1], label="F1-Score", linewidth=2, linestyle="--")
        axes[0].axhline(
            y=self.target_precision,
            color="r",
            linestyle=":",
            label=f"Target Precision ({self.target_precision})",
        )
        axes[0].axhline(
            y=self.target_recall,
            color="g",
            linestyle=":",
            label=f"Target Recall ({self.target_recall})",
        )
        axes[0].set_xlabel("Threshold", fontsize=12)
        axes[0].set_ylabel("Score", fontsize=12)
        axes[0].set_title("Métricas vs Threshold", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Precision-Recall curve
        axes[1].plot(recalls, precisions, linewidth=2)
        axes[1].set_xlabel("Recall", fontsize=12)
        axes[1].set_ylabel("Precision", fontsize=12)
        axes[1].set_title("Curva Precision-Recall", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Gráfico salvo em: {save_path}")

        plt.close()
