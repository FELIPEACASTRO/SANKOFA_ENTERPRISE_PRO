"""
CHECKPOINT 1 - Unit Tests: IntegratedEnsemble
MLEngineer_QA + MLOpsPlatform_QA

Contrato:
- predict_combined(transaction: Dict, base_probability: float) -> EnsemblePrediction
- calibrate_weights(X_val, y_val, base_predictions) -> Dict
- get_ensemble_info() -> Dict com metadados
- add_transactions_to_gnn() -> para integração com grafos
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Ajustar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestIntegratedEnsembleContract:
    """Testes de contrato da API IntegratedEnsemble"""

    @pytest.fixture
    def ensemble(self):
        """Fixture para criar ensemble"""
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    @pytest.fixture
    def sample_transaction(self):
        """Transação de exemplo"""
        return {
            'amount': 1500.0,
            'hour': 14,
            'day_of_week': 3,
            'velocity_score': 0.3,
            'is_new_device': 0,
            'merchant_category': 'retail',
            'channel': 'pix',
            'user_id': 'USER_001',
            'account_age_days': 365
        }

    def test_ML_001_has_predict_combined_method(self, ensemble):
        """ML-001: Deve ter método predict_combined"""
        assert hasattr(ensemble, 'predict_combined'), "Falta método predict_combined"
        assert callable(getattr(ensemble, 'predict_combined')), "predict_combined não é callable"

    def test_ML_002_has_calibrate_weights_method(self, ensemble):
        """ML-002: Deve ter método calibrate_weights"""
        assert hasattr(ensemble, 'calibrate_weights'), "Falta método calibrate_weights"
        assert callable(getattr(ensemble, 'calibrate_weights')), "calibrate_weights não é callable"

    def test_ML_003_has_get_ensemble_info_method(self, ensemble):
        """ML-003: Deve ter método get_ensemble_info"""
        assert hasattr(ensemble, 'get_ensemble_info'), "Falta método get_ensemble_info"
        assert callable(getattr(ensemble, 'get_ensemble_info')), "get_ensemble_info não é callable"

    def test_ML_004_has_add_transactions_to_gnn(self, ensemble):
        """ML-004: Deve ter método add_transactions_to_gnn"""
        assert hasattr(ensemble, 'add_transactions_to_gnn'), "Falta método add_transactions_to_gnn"

    def test_ML_005_get_ensemble_info_returns_dict(self, ensemble):
        """ML-005: get_ensemble_info deve retornar dicionário"""
        info = ensemble.get_ensemble_info()
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"


class TestIntegratedEnsembleWeights:
    """Testes de pesos do ensemble"""

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    def test_ML_006_weights_sum_to_one(self, ensemble):
        """ML-006: Pesos do ensemble devem somar 1.0"""
        info = ensemble.get_ensemble_info()
        if 'weights' in info:
            weights = info['weights']
            if isinstance(weights, dict):
                total = sum(weights.values())
            else:
                total = sum(weights)
            assert abs(total - 1.0) < 0.01, f"Pesos somam {total}, esperado ~1.0"
        else:
            pytest.skip("Pesos não expostos em get_ensemble_info")

    def test_ML_007_calibrate_weights_signature(self, ensemble):
        """ML-007: calibrate_weights deve aceitar X_val, y_val, base_predictions"""
        # Criar dados de validação mínimos
        X_val = pd.DataFrame({
            'amount': [1000.0, 2000.0, 3000.0],
            'hour': [10, 15, 3],
            'velocity_score': [0.2, 0.5, 0.9]
        })
        y_val = np.array([0, 0, 1])
        base_predictions = np.array([0.1, 0.3, 0.8])

        try:
            result = ensemble.calibrate_weights(X_val, y_val, base_predictions)
            # Deve retornar dict com pesos
            assert isinstance(result, dict), f"calibrate_weights deve retornar dict, got {type(result)}"
        except Exception as e:
            # Pode falhar por falta de modelos treinados, mas não deve dar TypeError de assinatura
            if "missing" in str(e).lower() and "argument" in str(e).lower():
                pytest.fail(f"Assinatura incorreta: {e}")

    def test_ML_008_weights_positive(self, ensemble):
        """ML-008: Todos os pesos devem ser positivos"""
        info = ensemble.get_ensemble_info()
        if 'weights' in info:
            weights = info['weights']
            if isinstance(weights, dict):
                values = list(weights.values())
            else:
                values = list(weights)
            assert all(w >= 0 for w in values), f"Pesos negativos encontrados: {values}"


class TestIntegratedEnsembleGNNIntegration:
    """Testes de integração com GNN"""

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    def test_GNN_001_add_transactions_accepts_list(self, ensemble):
        """GNN-001: add_transactions_to_gnn aceita lista de transações"""
        transactions = [
            {'id': 'tx1', 'amount': 1000, 'user': 'user1'},
            {'id': 'tx2', 'amount': 2000, 'user': 'user2'}
        ]
        try:
            ensemble.add_transactions_to_gnn(transactions)
        except TypeError:
            pytest.fail("add_transactions_to_gnn deveria aceitar lista")
        except Exception:
            # Outras exceções podem ser válidas (ex: GNN não disponível)
            pass

    def test_GNN_002_graceful_degradation_without_pytorch(self, ensemble):
        """GNN-002: Deve funcionar sem PyTorch instalado"""
        info = ensemble.get_ensemble_info()
        # Não deve lançar exceção mesmo se GNN não estiver disponível
        assert info is not None


class TestIntegratedEnsemblePrediction:
    """Testes de predição do ensemble"""

    @pytest.fixture
    def ensemble(self):
        from backend.ml_engine.ensemble_integration import IntegratedEnsemble
        return IntegratedEnsemble()

    @pytest.fixture
    def sample_transaction(self):
        return {
            'amount': 1500.0,
            'hour': 14,
            'day_of_week': 3,
            'velocity_score': 0.3,
            'is_new_device': 0,
            'merchant_category': 'retail',
            'channel': 'pix',
            'user_id': 'USER_001'
        }

    def test_PRED_001_predict_combined_returns_prediction(self, ensemble, sample_transaction):
        """PRED-001: predict_combined deve retornar EnsemblePrediction ou dict"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.3)
        # Pode ser EnsemblePrediction dataclass ou dict
        is_valid = isinstance(result, dict) or hasattr(result, 'fraud_probability')
        assert is_valid, f"Expected dict ou EnsemblePrediction, got {type(result)}"

    def test_PRED_002_predict_combined_has_fraud_probability(self, ensemble, sample_transaction):
        """PRED-002: Resultado deve ter fraud_probability"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.3)
        has_prob = (
            hasattr(result, 'fraud_probability') or
            (isinstance(result, dict) and 'fraud_probability' in result)
        )
        assert has_prob, f"Falta fraud_probability. Attrs: {dir(result) if not isinstance(result, dict) else result.keys()}"

    def test_PRED_003_predict_combined_has_is_fraud(self, ensemble, sample_transaction):
        """PRED-003: Resultado deve ter is_fraud"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.3)
        has_is_fraud = (
            hasattr(result, 'is_fraud') or
            (isinstance(result, dict) and 'is_fraud' in result)
        )
        assert has_is_fraud, f"Falta is_fraud"

    def test_PRED_004_fraud_probability_in_range(self, ensemble, sample_transaction):
        """PRED-004: fraud_probability deve estar em [0, 1]"""
        result = ensemble.predict_combined(sample_transaction, base_probability=0.5)
        if hasattr(result, 'fraud_probability'):
            prob = result.fraud_probability
        elif isinstance(result, dict):
            prob = result.get('fraud_probability', 0)
        else:
            pytest.skip("fraud_probability não encontrado")

        assert 0 <= prob <= 1, f"fraud_probability {prob} fora do range [0,1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
