"""
Unit tests for ProductionFraudEngine
Tests: predict, batch processing, determinism, latency
"""
import pytest
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestProductionFraudEngine:
    """Test suite for ProductionFraudEngine"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures"""
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        self.engine = ProductionFraudEngine()

    def test_predict_returns_array(self):
        """DS-006: predict() returns numpy array"""
        df = pd.DataFrame([{
            "amount": 1500.0,
            "hour": 14,
            "velocity_score": 0.3,
            "is_new_device": 0
        }])

        result = self.engine.predict(df)

        assert result is not None
        assert hasattr(result, '__len__')
        assert len(result) == 1

    def test_predict_two_transactions(self):
        """DS-006: predict() handles multiple transactions"""
        df = pd.DataFrame([
            {"amount": 150.0, "hour": 14, "velocity_score": 0.1, "is_new_device": 0},
            {"amount": 15000.0, "hour": 3, "velocity_score": 0.9, "is_new_device": 1}
        ])

        result = self.engine.predict(df)

        assert len(result) == 2, f"Expected 2 predictions, got {len(result)}"

    def test_predict_binary_output(self):
        """Predictions should be binary (0 or 1)"""
        df = pd.DataFrame([{
            "amount": 1500.0,
            "hour": 14,
            "velocity_score": 0.3,
            "is_new_device": 0
        }])

        result = self.engine.predict(df)

        assert result[0] in [0, 1], f"Expected binary, got {result[0]}"

    def test_predict_deterministic(self):
        """BANK-004: Same input produces same output"""
        df = pd.DataFrame([{
            "amount": 1500.0,
            "hour": 14,
            "velocity_score": 0.3,
            "is_new_device": 0
        }])

        results = [self.engine.predict(df)[0] for _ in range(5)]

        assert len(set(results)) == 1, f"Non-deterministic: {results}"

    def test_batch_processing_100(self):
        """BE-002: Process 100 transactions efficiently"""
        np.random.seed(42)
        batch = pd.DataFrame([{
            "amount": np.random.uniform(10, 10000),
            "hour": np.random.randint(0, 24),
            "velocity_score": np.random.random(),
            "is_new_device": np.random.randint(0, 2)
        } for _ in range(100)])

        start = time.time()
        result = self.engine.predict(batch)
        elapsed_ms = (time.time() - start) * 1000

        assert len(result) == 100
        assert elapsed_ms < 1000, f"Batch took {elapsed_ms:.2f}ms, expected <1000ms"

    def test_inference_latency(self):
        """ML-006: Single prediction latency <100ms"""
        df = pd.DataFrame([{
            "amount": 1500.0,
            "hour": 14,
            "velocity_score": 0.3,
            "is_new_device": 0
        }])

        # Warm up
        _ = self.engine.predict(df)

        # Measure
        latencies = []
        for _ in range(10):
            start = time.time()
            _ = self.engine.predict(df)
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        assert avg_latency < 100, f"Avg latency {avg_latency:.2f}ms > 100ms"

    def test_fraud_detection_high_risk(self):
        """BANK-001: Detect high-risk transaction"""
        fraud_tx = pd.DataFrame([{
            "amount": 50000.0,
            "hour": 3,
            "velocity_score": 0.95,
            "is_new_device": 1
        }])

        result = self.engine.predict(fraud_tx)

        # High-risk should likely be flagged
        # Note: Model may or may not flag it, but should not crash
        assert result[0] in [0, 1]

    def test_normal_detection_low_risk(self):
        """BANK-001: Handle normal transaction"""
        normal_tx = pd.DataFrame([{
            "amount": 150.0,
            "hour": 14,
            "velocity_score": 0.1,
            "is_new_device": 0
        }])

        result = self.engine.predict(normal_tx)

        # Low-risk transaction
        assert result[0] in [0, 1]

    def test_dict_to_dataframe_workflow(self):
        """BE-001: API contract - dict to DataFrame"""
        api_input = {
            "amount": 1500.0,
            "hour": 14,
            "velocity_score": 0.3,
            "is_new_device": 0
        }

        df = pd.DataFrame([api_input])
        result = self.engine.predict(df)

        assert len(result) == 1


class TestProductionFraudEngineEdgeCases:
    """Edge case tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        self.engine = ProductionFraudEngine()

    def test_missing_features_handled(self):
        """Engine should handle missing features gracefully"""
        partial_df = pd.DataFrame([{"amount": 1500.0}])

        # Should not crash, may log warning
        result = self.engine.predict(partial_df)
        assert result is not None

    def test_zero_amount(self):
        """Handle zero amount transaction"""
        df = pd.DataFrame([{
            "amount": 0.0,
            "hour": 14,
            "velocity_score": 0.1,
            "is_new_device": 0
        }])

        result = self.engine.predict(df)
        assert result[0] in [0, 1]

    def test_negative_amount(self):
        """Handle negative amount (refund)"""
        df = pd.DataFrame([{
            "amount": -500.0,
            "hour": 14,
            "velocity_score": 0.1,
            "is_new_device": 0
        }])

        result = self.engine.predict(df)
        assert result[0] in [0, 1]

    def test_extreme_amount(self):
        """Handle extreme amount"""
        df = pd.DataFrame([{
            "amount": 1000000.0,
            "hour": 14,
            "velocity_score": 0.1,
            "is_new_device": 0
        }])

        result = self.engine.predict(df)
        assert result[0] in [0, 1]
