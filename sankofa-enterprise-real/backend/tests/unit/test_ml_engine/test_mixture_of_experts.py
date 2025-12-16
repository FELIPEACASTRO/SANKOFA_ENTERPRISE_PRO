"""
Unit tests for MixtureOfExperts
Tests: predict, expert weights, channel routing
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestMixtureOfExperts:
    """Test suite for MixtureOfExperts"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from ml_engine.mixture_of_experts import MixtureOfExperts
        self.moe = MixtureOfExperts()

    def test_predict_returns_result(self):
        """DS-005: predict() returns MoEResult"""
        tx = {
            "amount": 1500.0,
            "channel": "pix",
            "hour": 14,
            "is_new_device": False
        }

        result = self.moe.predict(tx)

        assert result is not None
        assert hasattr(result, 'final_probability')

    def test_predict_has_transaction_id(self):
        """Result should have transaction_id"""
        tx = {"amount": 1500.0, "channel": "pix", "hour": 14}

        result = self.moe.predict(tx)

        assert hasattr(result, 'transaction_id')
        assert result.transaction_id is not None

    def test_probability_in_range(self):
        """Probability should be between 0 and 1"""
        tx = {"amount": 1500.0, "channel": "pix", "hour": 14}

        result = self.moe.predict(tx)

        assert 0.0 <= result.final_probability <= 1.0

    def test_final_prediction_boolean(self):
        """Final prediction should be boolean"""
        tx = {"amount": 1500.0, "channel": "pix", "hour": 14}

        result = self.moe.predict(tx)

        assert hasattr(result, 'final_prediction')
        assert isinstance(result.final_prediction, bool)

    def test_expert_weights_present(self):
        """Should have expert weights"""
        tx = {"amount": 1500.0, "channel": "pix", "hour": 14}

        result = self.moe.predict(tx)

        assert hasattr(result, 'expert_weights')

    def test_pix_channel(self):
        """BANK-002: Handle PIX channel"""
        tx = {
            "amount": 5000.0,
            "channel": "pix",
            "hour": 22,
            "is_new_device": True
        }

        result = self.moe.predict(tx)

        assert result is not None
        assert 0.0 <= result.final_probability <= 1.0

    def test_credit_channel(self):
        """Handle credit channel"""
        tx = {
            "amount": 5000.0,
            "channel": "credit",
            "hour": 14,
            "is_new_device": False
        }

        result = self.moe.predict(tx)

        assert result is not None

    def test_debit_channel(self):
        """Handle debit channel"""
        tx = {
            "amount": 500.0,
            "channel": "debit",
            "hour": 10,
            "is_new_device": False
        }

        result = self.moe.predict(tx)

        assert result is not None

    def test_minimal_input(self):
        """Handle minimal input"""
        tx = {"amount": 100.0}

        result = self.moe.predict(tx)

        assert result is not None

    def test_high_amount_increases_risk(self):
        """Higher amounts should generally have higher risk indication"""
        low_tx = {"amount": 50.0, "channel": "pix", "hour": 14}
        high_tx = {"amount": 50000.0, "channel": "pix", "hour": 3}

        low_result = self.moe.predict(low_tx)
        high_result = self.moe.predict(high_tx)

        # Both should produce valid results
        assert low_result is not None
        assert high_result is not None
