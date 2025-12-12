"""
ONNX Model Serving - Ultra-low latency inference (<5ms P95)
Converts ML models to ONNX format for optimized serving
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import pickle
import json

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False
    logging.warning("skl2onnx not available. Install with: pip install skl2onnx")

logger = logging.getLogger(__name__)


class ONNXModelConverter:
    """
    Converts scikit-learn and XGBoost models to ONNX format

    Benefits:
    - 3-10x faster inference vs native Python
    - Hardware acceleration (CPU vectorization, GPU optional)
    - Language-agnostic deployment
    - Smaller model size
    """

    @staticmethod
    def convert_sklearn_model(
        model: Any,
        input_shape: Tuple[int, ...],
        model_name: str = "fraud_model",
        target_opset: int = 15
    ) -> bytes:
        """
        Convert scikit-learn model to ONNX

        Args:
            model: Trained scikit-learn model
            input_shape: Input shape (n_features,)
            model_name: Model name
            target_opset: ONNX opset version

        Returns:
            ONNX model as bytes
        """
        if not SKL2ONNX_AVAILABLE:
            raise ImportError("skl2onnx not installed")

        # Define initial types
        initial_type = [('float_input', FloatTensorType([None, input_shape[0]]))]

        # Convert
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=target_opset,
            options={
                'zipmap': False,  # Disable zipmap for faster inference
                'nocl': True      # No class labels output
            }
        )

        logger.info(f"Converted {model_name} to ONNX (opset={target_opset})")

        return onnx_model.SerializeToString()

    @staticmethod
    def convert_xgboost_model(
        model: Any,
        input_shape: Tuple[int, ...],
        model_name: str = "xgb_model"
    ) -> bytes:
        """
        Convert XGBoost model to ONNX

        Args:
            model: Trained XGBoost model
            input_shape: Input shape
            model_name: Model name

        Returns:
            ONNX model as bytes
        """
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("onnxmltools not installed")

        initial_type = [('float_input', FloatTensorType([None, input_shape[0]]))]

        onnx_model = convert_xgboost(model, initial_types=initial_type)

        logger.info(f"Converted XGBoost {model_name} to ONNX")

        return onnx_model.SerializeToString()

    @staticmethod
    def save_onnx_model(onnx_bytes: bytes, output_path: str) -> None:
        """Save ONNX model to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(onnx_bytes)

        logger.info(f"ONNX model saved to {output_path}")


class ONNXInferenceSession:
    """
    High-performance ONNX inference session

    Features:
    - Session pooling for concurrency
    - Batch inference support
    - Graph optimization
    - Execution provider selection (CPU/CUDA)
    """

    def __init__(
        self,
        model_path: str,
        execution_providers: Optional[List[str]] = None,
        session_options: Optional[Any] = None
    ):
        """
        Args:
            model_path: Path to ONNX model file
            execution_providers: ['CPUExecutionProvider'] or ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session_options: ONNX Runtime session options
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")

        self.model_path = model_path

        # Default to CPU
        if execution_providers is None:
            execution_providers = ['CPUExecutionProvider']

        # Session options for optimization
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4  # Parallel ops within graph
            session_options.inter_op_num_threads = 2  # Parallel graphs

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=execution_providers
        )

        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            f"ONNX session created: model={model_path}, "
            f"providers={execution_providers}, "
            f"input={self.input_name}, outputs={self.output_names}"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference (synchronous)

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Predictions
        """
        # Ensure float32
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: features}
        )

        # Return primary output (probabilities or scores)
        return outputs[0]

    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference (async - uses thread pool)

        Args:
            features: Input features

        Returns:
            Predictions
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, features)

    def predict_batch(
        self,
        features_batch: List[np.ndarray],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Batch inference with optimal batch size

        Args:
            features_batch: List of feature arrays
            batch_size: Batch size for inference

        Returns:
            List of predictions
        """
        results = []

        for i in range(0, len(features_batch), batch_size):
            batch = features_batch[i:i + batch_size]

            # Stack into single array
            batch_array = np.vstack(batch)

            # Predict
            preds = self.predict(batch_array)

            results.append(preds)

        # Concatenate all results
        return np.vstack(results) if results else np.array([])


class ONNXFraudModelServing:
    """
    Production ONNX model serving for fraud detection

    Architecture:
    - Primary model: XGBoost fraud classifier (ONNX)
    - Fallback: Scikit-learn rules-based (ONNX)
    - Feature preprocessing: Vectorized NumPy
    - Latency target: <5ms P95
    """

    def __init__(
        self,
        primary_model_path: str,
        fallback_model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            primary_model_path: Path to primary ONNX model
            fallback_model_path: Path to fallback ONNX model
            feature_names: Expected feature names
            threshold: Decision threshold
        """
        # Load primary model
        self.primary_session = ONNXInferenceSession(primary_model_path)

        # Load fallback model
        self.fallback_session = None
        if fallback_model_path and Path(fallback_model_path).exists():
            self.fallback_session = ONNXInferenceSession(fallback_model_path)

        self.feature_names = feature_names or []
        self.threshold = threshold

        # Metrics
        self.inference_count = 0
        self.total_latency_ms = 0.0
        self.p95_latency_ms = 0.0

        logger.info(
            f"ONNX Fraud Model Serving initialized: "
            f"primary={primary_model_path}, threshold={threshold}"
        )

    async def predict_fraud(
        self,
        transaction: Dict[str, Any],
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Predict fraud using ONNX model

        Args:
            transaction: Transaction data
            features: Pre-computed features (optional)

        Returns:
            Prediction result
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Extract or use provided features
            if features is None:
                features = self._extract_features(transaction)

            # Convert to numpy array
            feature_array = self._features_to_array(features)

            # Predict (async to not block)
            probs = await self.primary_session.predict_async(feature_array)

            # Extract fraud probability
            fraud_probability = float(probs[0][1]) if probs.ndim > 1 else float(probs[0])

            # Decision
            is_fraud = fraud_probability >= self.threshold

            # Calculate latency
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Update metrics
            self._update_metrics(latency_ms)

            result = {
                'transaction_id': transaction.get('id', 'unknown'),
                'is_fraud': is_fraud,
                'fraud_probability': fraud_probability,
                'risk_score': fraud_probability,
                'model_type': 'onnx_primary',
                'latency_ms': latency_ms,
                'threshold': self.threshold,
                'features_used': len(features)
            }

            logger.debug(
                f"ONNX prediction: txn={result['transaction_id']}, "
                f"fraud_prob={fraud_probability:.4f}, latency={latency_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"ONNX prediction error: {e}")

            # Fallback to secondary model
            if self.fallback_session:
                return await self._predict_with_fallback(transaction, features)
            else:
                raise

    async def _predict_with_fallback(
        self,
        transaction: Dict[str, Any],
        features: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Fallback prediction using secondary model"""
        try:
            feature_array = self._features_to_array(features or self._extract_features(transaction))
            probs = await self.fallback_session.predict_async(feature_array)
            fraud_probability = float(probs[0][1]) if probs.ndim > 1 else float(probs[0])

            return {
                'transaction_id': transaction.get('id', 'unknown'),
                'is_fraud': fraud_probability >= self.threshold,
                'fraud_probability': fraud_probability,
                'risk_score': fraud_probability,
                'model_type': 'onnx_fallback',
                'threshold': self.threshold
            }
        except Exception as e:
            logger.error(f"Fallback prediction also failed: {e}")
            raise

    def _extract_features(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from transaction

        Args:
            transaction: Transaction data

        Returns:
            Feature dict
        """
        # Simplified feature extraction
        # In production, use feature_engineering.py
        features = {
            'amount': float(transaction.get('amount', 0)),
            'hour': datetime.fromisoformat(transaction.get('created_at', datetime.now(timezone.utc).isoformat())).hour,
            'is_weekend': datetime.fromisoformat(transaction.get('created_at', datetime.now(timezone.utc).isoformat())).weekday() >= 5,
            'channel_pix': 1.0 if transaction.get('channel') == 'PIX' else 0.0,
            'channel_credit': 1.0 if transaction.get('channel') == 'credit_card' else 0.0,
        }

        return features

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dict to numpy array

        Args:
            features: Feature dict

        Returns:
            Feature array (1, n_features)
        """
        # Order features consistently
        if self.feature_names:
            feature_values = [features.get(name, 0.0) for name in self.feature_names]
        else:
            feature_values = list(features.values())

        # Convert to numpy array
        array = np.array([feature_values], dtype=np.float32)

        return array

    def _update_metrics(self, latency_ms: float) -> None:
        """Update inference metrics"""
        self.inference_count += 1
        self.total_latency_ms += latency_ms

        # Simple P95 approximation (in production use proper quantile tracking)
        if latency_ms > self.p95_latency_ms:
            self.p95_latency_ms = self.p95_latency_ms * 0.95 + latency_ms * 0.05

    def get_metrics(self) -> Dict[str, Any]:
        """Get serving metrics"""
        avg_latency = (
            self.total_latency_ms / self.inference_count
            if self.inference_count > 0
            else 0.0
        )

        return {
            'inference_count': self.inference_count,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': self.p95_latency_ms,
            'target_latency_ms': 5.0,
            'latency_sla_met': self.p95_latency_ms < 5.0
        }


# Example: Convert existing model to ONNX
async def convert_fraud_model_to_onnx():
    """Example: Convert existing fraud model to ONNX"""

    # 1. Load existing model (example)
    from sklearn.ensemble import RandomForestClassifier

    # Train simple model
    X_train = np.random.rand(1000, 50).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000)

    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    logger.info("Trained example model")

    # 2. Convert to ONNX
    converter = ONNXModelConverter()
    onnx_bytes = converter.convert_sklearn_model(
        model=model,
        input_shape=(50,),
        model_name="fraud_rf"
    )

    # 3. Save ONNX model
    output_path = "models/onnx/fraud_rf.onnx"
    converter.save_onnx_model(onnx_bytes, output_path)

    logger.info(f"Model converted and saved to {output_path}")


# Example: Serving with ONNX
async def example_onnx_serving():
    """Example: Serve fraud predictions with ONNX"""

    # Assume model already converted
    model_path = "models/onnx/fraud_rf.onnx"

    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}. Run convert_fraud_model_to_onnx() first.")
        return

    # Create serving instance
    serving = ONNXFraudModelServing(
        primary_model_path=model_path,
        threshold=0.5
    )

    # Simulate transaction
    transaction = {
        'id': 'TXN_001',
        'amount': 1000.0,
        'channel': 'PIX',
        'created_at': datetime.now(timezone.utc).isoformat()
    }

    # Predict
    result = await serving.predict_fraud(transaction)

    print(f"Prediction: {result}")
    print(f"Metrics: {serving.get_metrics()}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    asyncio.run(convert_fraud_model_to_onnx())
    asyncio.run(example_onnx_serving())
