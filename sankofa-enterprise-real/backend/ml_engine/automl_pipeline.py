"""
AutoML Pipeline - Automated model selection and hyperparameter tuning
H2O AutoML integration for fraud detection optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pickle

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    logging.warning("H2O not available. Install with: pip install h2o")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

logger = logging.getLogger(__name__)


class H2OAutoMLFraudDetector:
    """
    AutoML-powered fraud detection using H2O

    Features:
    - Automated model selection (GBM, RF, XGBoost, Deep Learning, GLM, Stacked Ensembles)
    - Automated hyperparameter tuning
    - Automated feature engineering
    - Leaderboard comparison
    - Production model export
    """

    def __init__(
        self,
        max_runtime_secs: int = 3600,  # 1 hour
        max_models: int = 20,
        balance_classes: bool = True,
        seed: int = 42
    ):
        """
        Args:
            max_runtime_secs: Max runtime for AutoML
            max_models: Max number of models to train
            balance_classes: Balance fraud/legitimate classes
            seed: Random seed
        """
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.balance_classes = balance_classes
        self.seed = seed

        # H2O instance
        self.h2o_initialized = False
        self.automl: Optional[H2OAutoML] = None
        self.best_model = None

        # Training metadata
        self.training_metadata: Dict[str, Any] = {}
        self.leaderboard: Optional[pd.DataFrame] = None

        logger.info(
            f"H2O AutoML Fraud Detector initialized: "
            f"max_runtime={max_runtime_secs}s, max_models={max_models}"
        )

    def _initialize_h2o(self) -> None:
        """Initialize H2O cluster"""
        if not H2O_AVAILABLE:
            raise ImportError("H2O not installed")

        if not self.h2o_initialized:
            try:
                h2o.init(max_mem_size='8G', nthreads=-1)
                self.h2o_initialized = True
                logger.info("H2O cluster initialized")
            except Exception as e:
                logger.error(f"Failed to initialize H2O: {e}")
                raise

    async def train_automl(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train AutoML models

        Args:
            X: Feature dataframe
            y: Target series (0=legitimate, 1=fraud)
            validation_split: Validation set size

        Returns:
            Training results
        """
        start_time = datetime.now(timezone.utc)

        # Initialize H2O
        self._initialize_h2o()

        # Prepare data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.seed, stratify=y
        )

        # Convert to H2OFrame
        train_df = pd.concat([X_train, y_train.rename('is_fraud')], axis=1)
        val_df = pd.concat([X_val, y_val.rename('is_fraud')], axis=1)

        train_h2o = h2o.H2OFrame(train_df)
        val_h2o = h2o.H2OFrame(val_df)

        # Set target as factor (categorical)
        train_h2o['is_fraud'] = train_h2o['is_fraud'].asfactor()
        val_h2o['is_fraud'] = val_h2o['is_fraud'].asfactor()

        # Feature names
        feature_names = list(X.columns)
        target_name = 'is_fraud'

        logger.info(
            f"Training AutoML: samples={len(train_df)}, features={len(feature_names)}, "
            f"fraud_rate={y_train.mean():.2%}"
        )

        # Initialize AutoML
        self.automl = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            max_models=self.max_models,
            seed=self.seed,
            balance_classes=self.balance_classes,
            sort_metric='AUC',  # Sort leaderboard by AUC
            exclude_algos=['StackedEnsemble'],  # Exclude for faster training
            verbosity='info'
        )

        # Train
        logger.info("Starting AutoML training...")
        self.automl.train(
            x=feature_names,
            y=target_name,
            training_frame=train_h2o,
            validation_frame=val_h2o
        )

        # Get best model
        self.best_model = self.automl.leader

        # Get leaderboard
        self.leaderboard = self.automl.leaderboard.as_data_frame()

        # Evaluate on validation set
        val_preds = self.best_model.predict(val_h2o)
        val_preds_df = val_preds.as_data_frame()

        y_pred_proba = val_preds_df['p1'].values
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        auc_pr = np.trapz(recall, precision)

        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Store metadata
        self.training_metadata = {
            'training_time_secs': training_time,
            'num_models_trained': len(self.leaderboard),
            'best_model_id': self.best_model.model_id,
            'best_model_type': self.best_model.algo,
            'validation_auc': auc,
            'validation_f1': f1,
            'validation_auc_pr': auc_pr,
            'feature_count': len(feature_names),
            'training_samples': len(train_df),
            'fraud_rate': float(y_train.mean())
        }

        logger.info(
            f"AutoML training complete: best_model={self.best_model.algo}, "
            f"auc={auc:.4f}, f1={f1:.4f}, time={training_time:.0f}s"
        )

        return self.training_metadata

    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability

        Args:
            X: Feature dataframe

        Returns:
            Fraud probabilities
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call train_automl() first.")

        # Convert to H2OFrame
        X_h2o = h2o.H2OFrame(X)

        # Predict
        preds = self.best_model.predict(X_h2o)
        preds_df = preds.as_data_frame()

        # Return fraud probability
        return preds_df['p1'].values

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from best model

        Returns:
            Feature importance dataframe
        """
        if self.best_model is None:
            raise ValueError("No model trained")

        varimp = self.best_model.varimp(use_pandas=True)
        return varimp

    def get_leaderboard(self) -> pd.DataFrame:
        """Get AutoML leaderboard"""
        if self.leaderboard is None:
            raise ValueError("No models trained")
        return self.leaderboard

    def save_best_model(self, path: str) -> None:
        """
        Save best model to disk

        Args:
            path: Save path
        """
        if self.best_model is None:
            raise ValueError("No model trained")

        model_path = h2o.save_model(model=self.best_model, path=path, force=True)
        logger.info(f"Best model saved to {model_path}")

        # Save metadata
        metadata_path = Path(path) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2, default=str)

    def load_model(self, path: str) -> None:
        """
        Load saved model

        Args:
            path: Model path
        """
        self._initialize_h2o()

        self.best_model = h2o.load_model(path)
        logger.info(f"Model loaded from {path}")

        # Load metadata
        metadata_path = Path(path).parent / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)

    def shutdown_h2o(self) -> None:
        """Shutdown H2O cluster"""
        if self.h2o_initialized:
            h2o.cluster().shutdown()
            self.h2o_initialized = False
            logger.info("H2O cluster shutdown")


class AutoFeatureEngineering:
    """
    Automated feature engineering for fraud detection

    Features generated:
    - Time-based: hour, day_of_week, is_weekend, is_night
    - Amount-based: log_amount, amount_bins, amount_zscore
    - Velocity: transactions per hour/day
    - Aggregations: avg/max/min by customer/merchant
    - Ratios: amount_to_avg_ratio, deviation_from_normal
    """

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automated feature engineering

        Args:
            df: Raw transaction dataframe

        Returns:
            Dataframe with engineered features
        """
        df = df.copy()

        # Time features
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)

        # Amount features
        if 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'])
            df['amount_rounded'] = (df['amount'] % 100 == 0).astype(int)

            # Amount bins
            df['amount_bin'] = pd.cut(
                df['amount'],
                bins=[0, 100, 500, 1000, 5000, np.inf],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)

        # Customer aggregations
        if 'customer_id' in df.columns and 'amount' in df.columns:
            customer_agg = df.groupby('customer_id')['amount'].agg(['mean', 'max', 'std', 'count'])
            customer_agg.columns = [f'customer_{col}' for col in customer_agg.columns]
            df = df.merge(customer_agg, left_on='customer_id', right_index=True, how='left')

            # Amount to customer average ratio
            df['amount_to_customer_avg'] = df['amount'] / (df['customer_mean'] + 1)

        # Merchant aggregations
        if 'merchant_id' in df.columns and 'amount' in df.columns:
            merchant_agg = df.groupby('merchant_id')['amount'].agg(['mean', 'std'])
            merchant_agg.columns = [f'merchant_{col}' for col in merchant_agg.columns]
            df = df.merge(merchant_agg, left_on='merchant_id', right_index=True, how='left')

        # Channel encoding
        if 'channel' in df.columns:
            df['channel_pix'] = (df['channel'] == 'PIX').astype(int)
            df['channel_credit'] = (df['channel'] == 'credit_card').astype(int)
            df['channel_debit'] = (df['channel'] == 'debit_card').astype(int)

        # Fill NaN
        df = df.fillna(0)

        logger.info(f"Feature engineering complete: {len(df.columns)} features")

        return df

    @staticmethod
    def select_features(
        df: pd.DataFrame,
        target: pd.Series,
        max_features: int = 50
    ) -> List[str]:
        """
        Automated feature selection using mutual information

        Args:
            df: Feature dataframe
            target: Target series
            max_features: Max number of features to select

        Returns:
            Selected feature names
        """
        from sklearn.feature_selection import mutual_info_classif

        # Calculate mutual information
        mi_scores = mutual_info_classif(df, target, random_state=42)

        # Create dataframe
        mi_df = pd.DataFrame({
            'feature': df.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        # Select top features
        selected_features = mi_df.head(max_features)['feature'].tolist()

        logger.info(
            f"Feature selection complete: {len(selected_features)}/{len(df.columns)} features selected"
        )

        return selected_features


class AutoMLPipeline:
    """
    End-to-end AutoML pipeline for fraud detection

    Pipeline:
    1. Data loading and validation
    2. Automated feature engineering
    3. Feature selection
    4. AutoML training (H2O)
    5. Model evaluation
    6. Production deployment
    """

    def __init__(
        self,
        max_runtime_secs: int = 3600,
        max_models: int = 20,
        max_features: int = 50
    ):
        """
        Args:
            max_runtime_secs: Max AutoML runtime
            max_models: Max models to train
            max_features: Max features to use
        """
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.max_features = max_features

        # Components
        self.automl = H2OAutoMLFraudDetector(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models
        )
        self.feature_engineer = AutoFeatureEngineering()

        # Selected features
        self.selected_features: List[str] = []

        logger.info("AutoML Pipeline initialized")

    async def train_pipeline(
        self,
        transactions: pd.DataFrame,
        target_column: str = 'is_fraud'
    ) -> Dict[str, Any]:
        """
        Train complete AutoML pipeline

        Args:
            transactions: Transaction dataframe
            target_column: Target column name

        Returns:
            Training results
        """
        logger.info(f"Starting AutoML pipeline: {len(transactions)} transactions")

        # Extract target
        y = transactions[target_column]
        X_raw = transactions.drop(columns=[target_column])

        # Feature engineering
        logger.info("Step 1: Feature engineering...")
        X_engineered = self.feature_engineer.engineer_features(X_raw)

        # Select numeric columns only
        X_numeric = X_engineered.select_dtypes(include=[np.number])

        # Feature selection
        logger.info("Step 2: Feature selection...")
        self.selected_features = self.feature_engineer.select_features(
            X_numeric,
            y,
            max_features=self.max_features
        )

        X_selected = X_numeric[self.selected_features]

        # Train AutoML
        logger.info("Step 3: AutoML training...")
        training_results = await self.automl.train_automl(X_selected, y)

        # Feature importance
        logger.info("Step 4: Feature importance analysis...")
        feature_importance = self.automl.get_feature_importance()

        # Combine results
        results = {
            **training_results,
            'selected_features': self.selected_features,
            'top_features': feature_importance.head(10).to_dict('records')
        }

        logger.info(f"AutoML pipeline complete: AUC={results['validation_auc']:.4f}")

        return results

    async def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict using trained pipeline

        Args:
            transactions: Transaction dataframe

        Returns:
            Fraud probabilities
        """
        # Feature engineering
        X_engineered = self.feature_engineer.engineer_features(transactions)

        # Select features
        X_selected = X_engineered[self.selected_features]

        # Predict
        return await self.automl.predict(X_selected)

    def save_pipeline(self, path: str) -> None:
        """Save complete pipeline"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save AutoML model
        self.automl.save_best_model(str(save_path / 'h2o_model'))

        # Save selected features
        with open(save_path / 'selected_features.json', 'w') as f:
            json.dump(self.selected_features, f, indent=2)

        logger.info(f"Pipeline saved to {path}")

    def shutdown(self) -> None:
        """Shutdown H2O cluster"""
        self.automl.shutdown_h2o()


# Example usage
async def example_automl_pipeline():
    """Example: AutoML pipeline for fraud detection"""

    if not H2O_AVAILABLE:
        print("H2O not installed. Install with: pip install h2o")
        return

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000

    data = {
        'customer_id': [f'CUST_{i % 1000}' for i in range(n_samples)],
        'merchant_id': [f'MERCH_{i % 200}' for i in range(n_samples)],
        'amount': np.random.exponential(500, n_samples),
        'channel': np.random.choice(['PIX', 'credit_card', 'debit_card'], n_samples),
        'created_at': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    }

    df = pd.DataFrame(data)

    logger.info(f"Generated {len(df)} transactions, fraud_rate={df['is_fraud'].mean():.2%}")

    # Create pipeline
    pipeline = AutoMLPipeline(
        max_runtime_secs=300,  # 5 minutes for demo
        max_models=10,
        max_features=30
    )

    # Train
    results = await pipeline.train_pipeline(df)

    print(f"\nAutoML Results:")
    print(f"  Best Model: {results['best_model_type']}")
    print(f"  Validation AUC: {results['validation_auc']:.4f}")
    print(f"  Validation F1: {results['validation_f1']:.4f}")
    print(f"  Training Time: {results['training_time_secs']:.0f}s")
    print(f"  Models Trained: {results['num_models_trained']}")

    print(f"\nTop Features:")
    for feat in results['top_features'][:5]:
        print(f"  - {feat['variable']}: {feat['relative_importance']:.4f}")

    # Predict on new data
    new_transactions = df.head(10).drop(columns=['is_fraud'])
    predictions = await pipeline.predict(new_transactions)

    print(f"\nPredictions (first 10):")
    for i, prob in enumerate(predictions[:10]):
        print(f"  Transaction {i}: {prob:.4f}")

    # Shutdown
    pipeline.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_automl_pipeline())
