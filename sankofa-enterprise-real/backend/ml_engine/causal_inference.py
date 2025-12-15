"""
Causal Inference Framework - DoWhy/CausalML for impact analysis
Understand causal effects of fraud rules and A/B test results
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    from dowhy import CausalModel
    import dowhy.datasets
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    CausalModel = None  # Placeholder for type hints
    logging.warning("DoWhy not available. Install with: pip install dowhy")

try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.metrics import plot_gain, auuc_score
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    logging.warning("CausalML not available. Install with: pip install causalml")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class CausalEffect:
    """Represents a causal effect estimate"""
    treatment: str
    outcome: str
    ate: float  # Average Treatment Effect
    ate_se: float  # Standard error
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    is_significant: bool


class CausalImpactAnalyzer:
    """
    Causal impact analysis using DoWhy

    Analyzes causal effects of:
    - Fraud rules (e.g., does velocity rule reduce fraud?)
    - MFA challenges (does SMS OTP reduce fraud vs no challenge?)
    - Risk thresholds (does lowering threshold reduce fraud loss?)

    Methods:
    - Propensity score matching
    - Inverse propensity weighting
    - Doubly robust estimation
    - Regression discontinuity
    """

    def __init__(self):
        """Initialize causal analyzer"""
        if not DOWHY_AVAILABLE:
            raise ImportError("DoWhy not installed")

        self.causal_models: Dict[str, CausalModel] = {}
        self.results: List[CausalEffect] = []

        logger.info("Causal Impact Analyzer initialized")

    def analyze_rule_impact(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: List[str],
        method: str = 'backdoor.propensity_score_matching'
    ) -> CausalEffect:
        """
        Analyze causal impact of a fraud rule

        Args:
            data: Transaction data
            treatment_col: Treatment column (e.g., 'rule_triggered')
            outcome_col: Outcome column (e.g., 'is_fraud')
            confounders: Confounding variables
            method: Causal inference method

        Returns:
            Causal effect estimate

        Example:
            # Does velocity rule reduce fraud?
            effect = analyzer.analyze_rule_impact(
                data=transactions,
                treatment_col='velocity_rule_triggered',
                outcome_col='is_fraud',
                confounders=['amount', 'hour', 'channel']
            )
        """
        logger.info(
            f"Analyzing causal impact: treatment={treatment_col}, "
            f"outcome={outcome_col}, confounders={len(confounders)}"
        )

        # Create causal graph
        causal_graph = self._create_causal_graph(
            treatment=treatment_col,
            outcome=outcome_col,
            confounders=confounders
        )

        # Create causal model
        model = CausalModel(
            data=data,
            treatment=treatment_col,
            outcome=outcome_col,
            graph=causal_graph
        )

        # Identify causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate causal effect
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            confidence_intervals=True
        )

        # Extract results
        ate = estimate.value
        ate_se = estimate.get_standard_error() if hasattr(estimate, 'get_standard_error') else 0.0
        ci_lower, ci_upper = estimate.get_confidence_intervals() if hasattr(estimate, 'get_confidence_intervals') else (ate - 1.96 * ate_se, ate + 1.96 * ate_se)

        # Calculate p-value
        z_score = ate / ate_se if ate_se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        causal_effect = CausalEffect(
            treatment=treatment_col,
            outcome=outcome_col,
            ate=ate,
            ate_se=ate_se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=method,
            is_significant=(p_value < 0.05)
        )

        self.results.append(causal_effect)

        logger.info(
            f"Causal effect estimated: ATE={ate:.4f}, "
            f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
            f"p={p_value:.4f}, significant={causal_effect.is_significant}"
        )

        return causal_effect

    def _create_causal_graph(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> str:
        """
        Create causal graph in GML format

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounding variables

        Returns:
            Causal graph string
        """
        # Simple DAG: Confounders -> Treatment -> Outcome
        edges = []

        # Confounders affect both treatment and outcome
        for confounder in confounders:
            edges.append(f"{confounder} -> {treatment}")
            edges.append(f"{confounder} -> {outcome}")

        # Treatment affects outcome
        edges.append(f"{treatment} -> {outcome}")

        graph = "digraph { " + "; ".join(edges) + "; }"

        return graph

    def run_sensitivity_analysis(
        self,
        model: CausalModel,
        estimate: Any
    ) -> Dict[str, Any]:
        """
        Run sensitivity analysis (robustness check)

        Args:
            model: Causal model
            estimate: Causal estimate

        Returns:
            Sensitivity analysis results
        """
        try:
            refutation = model.refute_estimate(
                estimate,
                method_name="random_common_cause",
                num_simulations=100
            )

            return {
                'method': 'random_common_cause',
                'new_effect': refutation.new_effect,
                'refutation_result': str(refutation)
            }

        except Exception as e:
            logger.warning(f"Sensitivity analysis failed: {e}")
            return {'error': str(e)}


class UpliftModeling:
    """
    Uplift modeling for personalized fraud prevention

    Estimates heterogeneous treatment effects:
    - Which customers benefit most from MFA?
    - Which transactions need manual review?
    - Optimal threshold per customer segment

    Uses meta-learners:
    - S-Learner (single model)
    - T-Learner (two models)
    - X-Learner (cross-fit)
    """

    def __init__(self, method: str = 'T-Learner'):
        """
        Args:
            method: Uplift method ('S-Learner', 'T-Learner', 'X-Learner')
        """
        if not CAUSALML_AVAILABLE:
            raise ImportError("CausalML not installed")

        self.method = method
        self.model = None

        logger.info(f"Uplift Modeling initialized: method={method}")

    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Fit uplift model

        Args:
            X: Features
            treatment: Treatment assignment (0/1)
            y: Outcome (0/1)
        """
        logger.info(f"Fitting {self.method} uplift model...")

        if self.method == 'T-Learner':
            # T-Learner: Separate models for treatment and control
            self.model = XGBTRegressor()
        elif self.method == 'S-Learner':
            # S-Learner: Single model with treatment as feature
            self.model = LRSRegressor()
        else:
            # X-Learner (cross-fit)
            self.model = XGBTRegressor()

        # Fit model
        self.model.fit(X.values, treatment, y)

        logger.info("Uplift model fitted")

    def predict_uplift(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict uplift (treatment effect) for each sample

        Args:
            X: Features

        Returns:
            Uplift scores (positive = treatment beneficial)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        uplift = self.model.predict(X.values)

        return uplift.flatten()

    def get_top_responders(
        self,
        X: pd.DataFrame,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Get top responders (highest uplift)

        Args:
            X: Features
            top_n: Number of top responders

        Returns:
            Top responders dataframe
        """
        uplift = self.predict_uplift(X)

        X_copy = X.copy()
        X_copy['uplift'] = uplift

        top_responders = X_copy.nlargest(top_n, 'uplift')

        return top_responders


class ABTestCausalAnalyzer:
    """
    Causal analysis for A/B tests

    Features:
    - CUPED (Controlled-experiment Using Pre-Experiment Data)
    - Variance reduction using pre-treatment covariates
    - Heterogeneous treatment effect analysis
    - Multiple testing correction
    """

    def __init__(self):
        """Initialize A/B test analyzer"""
        self.test_results: List[Dict[str, Any]] = []

        logger.info("A/B Test Causal Analyzer initialized")

    def analyze_ab_test(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        pre_treatment_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze A/B test with causal inference

        Args:
            data: Experiment data
            treatment_col: Treatment assignment (0=control, 1=treatment)
            outcome_col: Outcome metric
            pre_treatment_col: Pre-treatment outcome (for CUPED)

        Returns:
            Test results
        """
        logger.info(f"Analyzing A/B test: treatment={treatment_col}, outcome={outcome_col}")

        # Split into treatment and control
        treatment = data[data[treatment_col] == 1][outcome_col]
        control = data[data[treatment_col] == 0][outcome_col]

        # Basic stats
        mean_treatment = treatment.mean()
        mean_control = control.mean()
        ate = mean_treatment - mean_control

        # Standard error
        se = np.sqrt(treatment.var() / len(treatment) + control.var() / len(control))

        # T-test
        t_stat, p_value = stats.ttest_ind(treatment, control)

        # Confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        results = {
            'treatment_mean': mean_treatment,
            'control_mean': mean_control,
            'ate': ate,
            'ate_relative': ate / mean_control if mean_control != 0 else 0,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05,
            'sample_size_treatment': len(treatment),
            'sample_size_control': len(control)
        }

        # CUPED variance reduction
        if pre_treatment_col and pre_treatment_col in data.columns:
            cuped_results = self._apply_cuped(
                data,
                treatment_col,
                outcome_col,
                pre_treatment_col
            )
            results['cuped'] = cuped_results

        self.test_results.append(results)

        logger.info(
            f"A/B test results: ATE={ate:.4f} ({ate/mean_control*100:.1f}%), "
            f"p={p_value:.4f}, significant={results['is_significant']}"
        )

        return results

    def _apply_cuped(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        pre_treatment_col: str
    ) -> Dict[str, Any]:
        """
        Apply CUPED for variance reduction

        CUPED formula:
        Y_cuped = Y - θ * (X_pre - E[X_pre])

        where θ = Cov(Y, X_pre) / Var(X_pre)
        """
        # Calculate theta
        cov = data[[outcome_col, pre_treatment_col]].cov().iloc[0, 1]
        var_pre = data[pre_treatment_col].var()
        theta = cov / var_pre if var_pre > 0 else 0

        # Adjust outcome
        mean_pre = data[pre_treatment_col].mean()
        data['outcome_cuped'] = data[outcome_col] - theta * (data[pre_treatment_col] - mean_pre)

        # Re-calculate ATE with CUPED
        treatment_cuped = data[data[treatment_col] == 1]['outcome_cuped']
        control_cuped = data[data[treatment_col] == 0]['outcome_cuped']

        ate_cuped = treatment_cuped.mean() - control_cuped.mean()
        se_cuped = np.sqrt(
            treatment_cuped.var() / len(treatment_cuped) +
            control_cuped.var() / len(control_cuped)
        )

        # Variance reduction
        se_original = np.sqrt(
            data[data[treatment_col] == 1][outcome_col].var() / len(treatment_cuped) +
            data[data[treatment_col] == 0][outcome_col].var() / len(control_cuped)
        )
        variance_reduction = 1 - (se_cuped / se_original) ** 2

        return {
            'ate_cuped': ate_cuped,
            'se_cuped': se_cuped,
            'theta': theta,
            'variance_reduction': variance_reduction
        }


# Example usage
async def example_causal_inference():
    """Example: Causal inference for fraud rules"""

    if not DOWHY_AVAILABLE:
        print("DoWhy not installed. Install with: pip install dowhy")
        return

    # Generate synthetic data
    np.random.seed(42)
    n = 5000

    # Confounders
    amount = np.random.exponential(500, n)
    hour = np.random.randint(0, 24, n)
    channel_pix = np.random.binomial(1, 0.5, n)

    # Treatment (velocity rule triggered)
    # Higher amounts and night hours more likely to trigger
    trigger_prob = 1 / (1 + np.exp(-(amount / 1000 + (hour > 22) * 2 - 2)))
    velocity_rule = np.random.binomial(1, trigger_prob)

    # Outcome (fraud)
    # Fraud more likely with high amounts, night, and if rule triggered (causal effect)
    fraud_prob = 1 / (1 + np.exp(-(
        amount / 2000 +
        (hour > 22) * 0.5 +
        velocity_rule * (-0.8) - 1  # Rule REDUCES fraud by 0.8 logit
    )))
    is_fraud = np.random.binomial(1, fraud_prob)

    # Create dataframe
    data = pd.DataFrame({
        'amount': amount,
        'hour': hour,
        'channel_pix': channel_pix,
        'velocity_rule_triggered': velocity_rule,
        'is_fraud': is_fraud
    })

    print(f"Generated {len(data)} transactions")
    print(f"  Fraud rate: {is_fraud.mean():.2%}")
    print(f"  Rule triggered: {velocity_rule.mean():.2%}")
    print(f"  Fraud rate (rule triggered): {data[data['velocity_rule_triggered']==1]['is_fraud'].mean():.2%}")
    print(f"  Fraud rate (rule not triggered): {data[data['velocity_rule_triggered']==0]['is_fraud'].mean():.2%}")

    # Causal analysis
    analyzer = CausalImpactAnalyzer()

    effect = analyzer.analyze_rule_impact(
        data=data,
        treatment_col='velocity_rule_triggered',
        outcome_col='is_fraud',
        confounders=['amount', 'hour', 'channel_pix']
    )

    print(f"\nCausal Effect of Velocity Rule:")
    print(f"  ATE: {effect.ate:.4f}")
    print(f"  95% CI: [{effect.confidence_interval[0]:.4f}, {effect.confidence_interval[1]:.4f}]")
    print(f"  P-value: {effect.p_value:.4f}")
    print(f"  Significant: {effect.is_significant}")
    print(f"  Interpretation: Velocity rule {'REDUCES' if effect.ate < 0 else 'INCREASES'} fraud by {abs(effect.ate)*100:.1f}pp")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_causal_inference())


# Aliases para compatibilidade
CausalInferenceEngine = CausalImpactAnalyzer
CausalAnalyzer = CausalImpactAnalyzer
UpliftModel = UpliftModeling
