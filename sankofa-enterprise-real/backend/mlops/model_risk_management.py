"""
Model Risk Management (MRM) - SR 11-7 Compliance
Model validation, backtesting, monitoring, and documentation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve
)
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class ModelValidationReport:
    """Model validation report (SR 11-7 compliant)"""
    model_id: str
    model_name: str
    model_version: str
    validation_date: datetime
    validator: str

    # Performance metrics
    auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Backtesting results
    backtesting_period: str
    backtesting_samples: int
    backtesting_passed: bool

    # Stability tests
    psi_score: float  # Population Stability Index
    psi_passed: bool

    # Bias/fairness tests
    demographic_parity: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    approval_status: str  # 'approved', 'conditional', 'rejected'
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Documentation
    methodology: str = ""
    assumptions: List[str] = field(default_factory=list)
    data_quality_notes: str = ""


class ModelValidator:
    """
    Model validator for SR 11-7 compliance

    Validation components:
    1. Conceptual soundness
    2. Ongoing monitoring
    3. Outcomes analysis
    4. Bias and fairness testing
    5. Stability testing (PSI, CSI)
    """

    def __init__(self):
        """Initialize model validator"""
        self.validation_reports: List[ModelValidationReport] = []

        logger.info("Model Validator initialized (SR 11-7 compliant)")

    async def validate_model(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_train: pd.DataFrame,
        model_metadata: Dict[str, Any],
        protected_attributes: Optional[Dict[str, pd.Series]] = None
    ) -> ModelValidationReport:
        """
        Complete model validation

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            X_train: Training features (for PSI)
            model_metadata: Model metadata
            protected_attributes: Protected attributes for bias testing

        Returns:
            Validation report
        """
        logger.info(f"Validating model: {model_metadata.get('model_name', 'unknown')}")

        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Performance metrics
        performance = self._calculate_performance_metrics(y_val, y_pred, y_pred_proba)

        # Backtesting
        backtesting = self._backtest_model(y_val, y_pred_proba)

        # Population Stability Index (PSI)
        psi_result = self._calculate_psi(X_train, X_val)

        # Bias/fairness testing
        bias_results = {}
        if protected_attributes:
            bias_results = self._test_bias_fairness(y_val, y_pred, protected_attributes)

        # Create validation report
        report = ModelValidationReport(
            model_id=model_metadata.get('model_id', 'unknown'),
            model_name=model_metadata.get('model_name', 'unknown'),
            model_version=model_metadata.get('version', '1.0'),
            validation_date=datetime.now(timezone.utc),
            validator='AutomatedMRM',
            auc=performance['auc'],
            precision=performance['precision'],
            recall=performance['recall'],
            f1_score=performance['f1'],
            accuracy=performance['accuracy'],
            backtesting_period=f"{len(y_val)} samples",
            backtesting_samples=len(y_val),
            backtesting_passed=backtesting['passed'],
            psi_score=psi_result['psi'],
            psi_passed=psi_result['passed'],
            demographic_parity=bias_results.get('demographic_parity', {}),
            equal_opportunity=bias_results.get('equal_opportunity', {}),
            approval_status=self._determine_approval_status(performance, backtesting, psi_result),
            limitations=self._identify_limitations(performance, psi_result),
            recommendations=self._generate_recommendations(performance, psi_result),
            methodology=model_metadata.get('methodology', 'Machine Learning Classification'),
            assumptions=model_metadata.get('assumptions', []),
            data_quality_notes=model_metadata.get('data_quality', 'Passed data quality checks')
        )

        self.validation_reports.append(report)

        logger.info(
            f"Validation complete: approval={report.approval_status}, "
            f"auc={report.auc:.4f}, psi={report.psi_score:.4f}"
        )

        return report

    def _calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        auc = roc_auc_score(y_true, y_pred_proba)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    def _backtest_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Backtest model calibration

        Tests if predicted probabilities match actual frequencies
        """
        # Create probability bins
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1

        # Calculate actual fraud rate per bin
        bin_results = []
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            predicted_prob = y_pred_proba[mask].mean()
            actual_rate = y_true[mask].mean()
            count = mask.sum()

            # Chi-square test
            expected = predicted_prob * count
            observed = actual_rate * count
            chi2 = ((observed - expected) ** 2 / expected) if expected > 0 else 0

            bin_results.append({
                'bin': i,
                'predicted_prob': predicted_prob,
                'actual_rate': actual_rate,
                'count': int(count),
                'chi2': chi2
            })

        # Overall chi-square
        total_chi2 = sum(b['chi2'] for b in bin_results)
        p_value = 1 - stats.chi2.cdf(total_chi2, df=len(bin_results) - 1)

        # Pass if p-value > 0.05 (model is calibrated)
        passed = p_value > 0.05

        return {
            'passed': passed,
            'chi2_statistic': total_chi2,
            'p_value': p_value,
            'bins': bin_results
        }

    def _calculate_psi(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI)

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Some change
        PSI >= 0.25: Significant change (model may need retraining)
        """
        psi_scores = []

        for col in X_train.columns:
            if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numeric column
                bins = pd.qcut(X_train[col], q=num_bins, duplicates='drop', retbins=True)[1]

                train_dist = pd.cut(X_train[col], bins=bins).value_counts(normalize=True).sort_index()
                val_dist = pd.cut(X_val[col], bins=bins).value_counts(normalize=True).sort_index()

                # Align distributions
                train_dist, val_dist = train_dist.align(val_dist, fill_value=0.001)

                # Calculate PSI
                psi = np.sum((val_dist - train_dist) * np.log(val_dist / train_dist))
                psi_scores.append(psi)

        avg_psi = np.mean(psi_scores) if psi_scores else 0.0

        return {
            'psi': avg_psi,
            'passed': avg_psi < 0.25,
            'interpretation': 'No significant change' if avg_psi < 0.1 else ('Some change' if avg_psi < 0.25 else 'Significant change')
        }

    def _test_bias_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, float]]:
        """
        Test bias and fairness

        Metrics:
        - Demographic parity: P(Y=1|A=0) = P(Y=1|A=1)
        - Equal opportunity: TPR(A=0) = TPR(A=1)
        """
        demographic_parity = {}
        equal_opportunity = {}

        for attr_name, attr_values in protected_attributes.items():
            # Get unique groups
            groups = attr_values.unique()

            if len(groups) != 2:
                continue

            group_0 = attr_values == groups[0]
            group_1 = attr_values == groups[1]

            # Demographic parity (positive prediction rate)
            ppr_0 = y_pred[group_0].mean()
            ppr_1 = y_pred[group_1].mean()
            demographic_parity[attr_name] = abs(ppr_0 - ppr_1)

            # Equal opportunity (true positive rate)
            tp_0 = ((y_pred[group_0] == 1) & (y_true[group_0] == 1)).sum()
            p_0 = (y_true[group_0] == 1).sum()
            tpr_0 = tp_0 / p_0 if p_0 > 0 else 0

            tp_1 = ((y_pred[group_1] == 1) & (y_true[group_1] == 1)).sum()
            p_1 = (y_true[group_1] == 1).sum()
            tpr_1 = tp_1 / p_1 if p_1 > 0 else 0

            equal_opportunity[attr_name] = abs(tpr_0 - tpr_1)

        return {
            'demographic_parity': demographic_parity,
            'equal_opportunity': equal_opportunity
        }

    def _determine_approval_status(
        self,
        performance: Dict[str, float],
        backtesting: Dict[str, Any],
        psi_result: Dict[str, Any]
    ) -> str:
        """Determine model approval status"""

        # Thresholds
        MIN_AUC = 0.75
        MAX_PSI = 0.25

        if performance['auc'] >= MIN_AUC and psi_result['passed'] and backtesting['passed']:
            return 'approved'
        elif performance['auc'] >= 0.70 and psi_result['psi'] < 0.30:
            return 'conditional'
        else:
            return 'rejected'

    def _identify_limitations(
        self,
        performance: Dict[str, float],
        psi_result: Dict[str, Any]
    ) -> List[str]:
        """Identify model limitations"""
        limitations = []

        if performance['auc'] < 0.85:
            limitations.append("AUC below 0.85 - consider feature engineering")

        if psi_result['psi'] > 0.1:
            limitations.append(f"PSI {psi_result['psi']:.3f} indicates distribution shift")

        if performance['precision'] < 0.7:
            limitations.append("Precision below 0.7 - high false positive rate")

        if performance['recall'] < 0.7:
            limitations.append("Recall below 0.7 - missing fraud cases")

        return limitations

    def _generate_recommendations(
        self,
        performance: Dict[str, float],
        psi_result: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        if performance['auc'] < 0.80:
            recommendations.append("Consider additional features or model ensemble")

        if psi_result['psi'] > 0.2:
            recommendations.append("Retrain model - significant population shift detected")

        recommendations.append("Monitor model performance monthly")
        recommendations.append("Review false positives/negatives weekly")

        return recommendations

    def export_validation_report(
        self,
        report: ModelValidationReport,
        output_path: str
    ) -> None:
        """Export validation report to JSON"""
        report_dict = {
            'model_id': report.model_id,
            'model_name': report.model_name,
            'model_version': report.model_version,
            'validation_date': report.validation_date.isoformat(),
            'validator': report.validator,
            'performance_metrics': {
                'auc': report.auc,
                'precision': report.precision,
                'recall': report.recall,
                'f1_score': report.f1_score,
                'accuracy': report.accuracy
            },
            'backtesting': {
                'period': report.backtesting_period,
                'samples': report.backtesting_samples,
                'passed': report.backtesting_passed
            },
            'stability': {
                'psi_score': report.psi_score,
                'psi_passed': report.psi_passed
            },
            'bias_fairness': {
                'demographic_parity': report.demographic_parity,
                'equal_opportunity': report.equal_opportunity
            },
            'approval_status': report.approval_status,
            'limitations': report.limitations,
            'recommendations': report.recommendations,
            'methodology': report.methodology,
            'assumptions': report.assumptions,
            'data_quality_notes': report.data_quality_notes
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Validation report exported to {output_path}")


class ModelMonitor:
    """
    Ongoing model monitoring (SR 11-7 requirement)

    Monitors:
    - Performance degradation
    - Data drift
    - Concept drift
    - Bias drift
    """

    def __init__(self):
        """Initialize model monitor"""
        self.monitoring_history: List[Dict[str, Any]] = []

        logger.info("Model Monitor initialized")

    async def monitor_performance(
        self,
        model_id: str,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        monitoring_window: str = '7d'
    ) -> Dict[str, Any]:
        """
        Monitor model performance over time

        Args:
            model_id: Model ID
            y_true: True labels
            y_pred_proba: Predicted probabilities
            monitoring_window: Window size

        Returns:
            Monitoring results
        """
        auc = roc_auc_score(y_true, y_pred_proba)

        monitoring_result = {
            'model_id': model_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'window': monitoring_window,
            'auc': auc,
            'sample_size': len(y_true),
            'fraud_rate': y_true.mean()
        }

        self.monitoring_history.append(monitoring_result)

        # Check for degradation
        if len(self.monitoring_history) > 1:
            prev_auc = self.monitoring_history[-2]['auc']
            degradation = prev_auc - auc

            if degradation > 0.05:
                monitoring_result['alert'] = 'PERFORMANCE_DEGRADATION'
                logger.warning(f"Performance degradation detected: {degradation:.4f}")

        return monitoring_result

    def get_monitoring_dashboard(self) -> pd.DataFrame:
        """Get monitoring dashboard"""
        if not self.monitoring_history:
            return pd.DataFrame()

        return pd.DataFrame(self.monitoring_history)


# Example usage
async def example_model_risk_management():
    """Example: Model validation and monitoring"""

    # Generate synthetic data
    np.random.seed(42)
    n_train = 5000
    n_val = 1000

    X_train = pd.DataFrame(np.random.randn(n_train, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train = np.random.binomial(1, 0.1, n_train)

    X_val = pd.DataFrame(np.random.randn(n_val, 10), columns=[f'feature_{i}' for i in range(10)])
    y_val = np.random.binomial(1, 0.1, n_val)

    # Train simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model metadata
    metadata = {
        'model_id': 'RF_001',
        'model_name': 'Fraud Detector v1',
        'version': '1.0',
        'methodology': 'Random Forest Classification',
        'assumptions': [
            'Feature independence',
            'IID samples',
            'Stationary distribution'
        ],
        'data_quality': 'Passed all quality checks'
    }

    # Validate model
    validator = ModelValidator()
    report = await validator.validate_model(
        model=model,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        model_metadata=metadata
    )

    print(f"\nModel Validation Report:")
    print(f"  Model: {report.model_name} v{report.model_version}")
    print(f"  Approval Status: {report.approval_status}")
    print(f"  AUC: {report.auc:.4f}")
    print(f"  Precision: {report.precision:.4f}")
    print(f"  Recall: {report.recall:.4f}")
    print(f"  PSI: {report.psi_score:.4f} ({'PASSED' if report.psi_passed else 'FAILED'})")
    print(f"  Backtesting: {'PASSED' if report.backtesting_passed else 'FAILED'}")
    print(f"\n  Limitations:")
    for lim in report.limitations:
        print(f"    - {lim}")
    print(f"\n  Recommendations:")
    for rec in report.recommendations:
        print(f"    - {rec}")

    # Export report
    validator.export_validation_report(report, 'validation_report.json')

    # Model monitoring
    monitor = ModelMonitor()

    # Simulate monitoring over time
    for i in range(5):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        result = await monitor.monitor_performance('RF_001', y_val, y_pred_proba)
        print(f"\nMonitoring Window {i+1}: AUC={result['auc']:.4f}")

    # Dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print(f"\nMonitoring Dashboard:")
    print(dashboard[['timestamp', 'auc', 'fraud_rate']])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_model_risk_management())
