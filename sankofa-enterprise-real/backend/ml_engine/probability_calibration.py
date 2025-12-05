"""
Sankofa Enterprise Pro - Probability Calibration Module
Calibração avançada de probabilidades para detecção de fraude
Baseado em: AIForge - MLOps Best Practices
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Métricas de calibração"""

    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    reliability_diagram: Dict[str, List[float]]
    is_well_calibrated: bool


class ProbabilityCalibrator:
    """
    Calibrador de Probabilidades para Modelos de Fraude

    Funcionalidades:
    - Calibração isotônica
    - Calibração Platt (sigmoid)
    - Avaliação de calibração
    - Métricas ECE e MCE
    """

    VERSION = "1.0.0"

    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        """
        Inicializa o calibrador

        Args:
            method: Método de calibração ('isotonic', 'sigmoid')
            n_bins: Número de bins para métricas de calibração
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrator = None
        self.is_fitted = False
        self.calibration_metrics: Optional[CalibrationMetrics] = None

        logger.info(f"ProbabilityCalibrator initialized v{self.VERSION} method={method}")

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """
        Ajusta o calibrador

        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades previstas

        Returns:
            Self
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_true)
        elif self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

        calibrated_probs = self.calibrate(y_prob)
        self.calibration_metrics = self.evaluate_calibration(y_true, calibrated_probs)

        logger.info(
            f"Calibrator fitted: method={self.method}, "
            f"ece={round(self.calibration_metrics.expected_calibration_error, 4)}, "
            f"well_calibrated={self.calibration_metrics.is_well_calibrated}"
        )

        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibra probabilidades

        Args:
            y_prob: Probabilidades não calibradas

        Returns:
            Probabilidades calibradas
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original probabilities")
            return y_prob

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(y_prob)
        elif self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        else:
            calibrated = y_prob

        return np.clip(calibrated, 0, 1)

    def evaluate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationMetrics:
        """
        Avalia a qualidade da calibração

        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades

        Returns:
            CalibrationMetrics
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins)

        ece = self._calculate_ece(y_true, y_prob)
        mce = self._calculate_mce(prob_true, prob_pred)
        brier = self._calculate_brier_score(y_true, y_prob)

        is_well_calibrated = ece < 0.1 and brier < 0.25

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            reliability_diagram={
                "mean_predicted_value": prob_pred.tolist(),
                "fraction_of_positives": prob_true.tolist(),
            },
            is_well_calibrated=is_well_calibrated,
        )

    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calcula Expected Calibration Error (ECE)

        ECE = sum(|accuracy_bin - confidence_bin| * n_samples_bin) / n_total
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)

            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def _calculate_mce(self, prob_true: np.ndarray, prob_pred: np.ndarray) -> float:
        """
        Calcula Maximum Calibration Error (MCE)
        """
        if len(prob_true) == 0 or len(prob_pred) == 0:
            return 0.0
        return float(np.max(np.abs(prob_true - prob_pred)))

    def _calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calcula Brier Score
        """
        return float(np.mean((y_prob - y_true) ** 2))

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retorna resumo das métricas de calibração"""
        if self.calibration_metrics is None:
            return {"status": "not_fitted"}

        return {
            "method": self.method,
            "expected_calibration_error": self.calibration_metrics.expected_calibration_error,
            "maximum_calibration_error": self.calibration_metrics.maximum_calibration_error,
            "brier_score": self.calibration_metrics.brier_score,
            "is_well_calibrated": self.calibration_metrics.is_well_calibrated,
            "version": self.VERSION,
        }


class EnsembleCalibrator:
    """
    Calibrador para Ensemble de Modelos

    Combina múltiplas técnicas de calibração para melhor resultado.
    """

    def __init__(self):
        self.isotonic_calibrator = ProbabilityCalibrator(method="isotonic")
        self.sigmoid_calibrator = ProbabilityCalibrator(method="sigmoid")
        self.best_method: Optional[str] = None
        self.is_fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "EnsembleCalibrator":
        """
        Ajusta ambos calibradores e seleciona o melhor
        """
        self.isotonic_calibrator.fit(y_true, y_prob)
        self.sigmoid_calibrator.fit(y_true, y_prob)

        iso_ece = self.isotonic_calibrator.calibration_metrics.expected_calibration_error
        sig_ece = self.sigmoid_calibrator.calibration_metrics.expected_calibration_error

        if iso_ece <= sig_ece:
            self.best_method = "isotonic"
        else:
            self.best_method = "sigmoid"

        self.is_fitted = True

        logger.info(
            f"EnsembleCalibrator fitted: best_method={self.best_method}, "
            f"isotonic_ece={round(iso_ece, 4)}, sigmoid_ece={round(sig_ece, 4)}"
        )

        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibra usando o melhor método
        """
        if not self.is_fitted:
            return y_prob

        if self.best_method == "isotonic":
            return self.isotonic_calibrator.calibrate(y_prob)
        else:
            return self.sigmoid_calibrator.calibrate(y_prob)

    def get_best_calibrator(self) -> ProbabilityCalibrator:
        """Retorna o melhor calibrador"""
        if self.best_method == "isotonic":
            return self.isotonic_calibrator
        return self.sigmoid_calibrator


def calibrate_model_predictions(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    method: str = "isotonic",
) -> Tuple[Any, ProbabilityCalibrator]:
    """
    Função utilitária para calibrar predições de um modelo

    Args:
        model: Modelo sklearn-compatible
        X_train: Features de treino
        y_train: Labels de treino
        X_val: Features de validação (opcional)
        y_val: Labels de validação (opcional)
        method: Método de calibração

    Returns:
        Tuple (CalibratedClassifierCV, ProbabilityCalibrator)
    """
    calibrated_model = CalibratedClassifierCV(model, cv=3, method=method)
    calibrated_model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_prob = calibrated_model.predict_proba(X_val)[:, 1]

        post_calibrator = ProbabilityCalibrator(method=method)
        post_calibrator.fit(y_val, y_prob)
    else:
        post_calibrator = None

    return calibrated_model, post_calibrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.beta(2 + y_true * 3, 5 - y_true * 2, 1000)

    print("=== Testing Probability Calibration ===\n")

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(y_true, y_prob)

    calibrated_probs = calibrator.calibrate(y_prob)

    metrics = calibrator.get_metrics_summary()
    print("Calibration Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n=== Testing Ensemble Calibrator ===\n")

    ensemble_cal = EnsembleCalibrator()
    ensemble_cal.fit(y_true, y_prob)

    print(f"Best method: {ensemble_cal.best_method}")

    calibrated_ensemble = ensemble_cal.calibrate(y_prob)

    print(f"\nOriginal prob mean: {y_prob.mean():.4f}")
    print(f"Calibrated prob mean: {calibrated_ensemble.mean():.4f}")
    print(f"True positive rate: {y_true.mean():.4f}")
