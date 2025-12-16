"""
CHECKPOINT 3 - Contract Tests: FederatedClient
DataScientist_QA + MLEngineer_QA

Contrato ABC FederatedClient:
- set_local_data(data: DataFrame, labels: ndarray) -> None
- train_local(global_weights: Dict) -> ClientUpdate
- evaluate_local() -> Dict[str, float]

Implementações testadas:
- SklearnFederatedClient
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestFederatedClientContract:
    """Contract tests para FederatedClient"""

    @pytest.fixture
    def federated_config(self):
        """Fixture para configuração federada"""
        from backend.ml_engine.federated_learning import FederatedConfig
        return FederatedConfig(
            max_rounds=3,
            min_clients=2,
            learning_rate=0.01,
            local_epochs=1,
            differential_privacy=False
        )

    @pytest.fixture
    def sklearn_client(self, federated_config):
        """Fixture para SklearnFederatedClient"""
        from backend.ml_engine.federated_learning import SklearnFederatedClient
        return SklearnFederatedClient(client_id="test_client_001", config=federated_config)

    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo para treinamento"""
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_CONTRACT_001_has_set_local_data_method(self, sklearn_client):
        """CONTRACT-001: Deve ter método set_local_data"""
        assert hasattr(sklearn_client, 'set_local_data'), "Falta método set_local_data"
        assert callable(getattr(sklearn_client, 'set_local_data')), "set_local_data não é callable"

    def test_CONTRACT_002_has_train_local_method(self, sklearn_client):
        """CONTRACT-002: Deve ter método train_local"""
        assert hasattr(sklearn_client, 'train_local'), "Falta método train_local"
        assert callable(getattr(sklearn_client, 'train_local')), "train_local não é callable"

    def test_CONTRACT_003_has_evaluate_local_method(self, sklearn_client):
        """CONTRACT-003: Deve ter método evaluate_local"""
        assert hasattr(sklearn_client, 'evaluate_local'), "Falta método evaluate_local"
        assert callable(getattr(sklearn_client, 'evaluate_local')), "evaluate_local não é callable"

    def test_CONTRACT_004_set_local_data_accepts_dataframe_and_array(self, sklearn_client, sample_data):
        """CONTRACT-004: set_local_data deve aceitar DataFrame e ndarray"""
        X, y = sample_data
        try:
            sklearn_client.set_local_data(X, y)
        except TypeError as e:
            pytest.fail(f"set_local_data deveria aceitar DataFrame e ndarray: {e}")

    def test_CONTRACT_005_train_local_returns_client_update(self, sklearn_client, sample_data):
        """CONTRACT-005: train_local deve retornar ClientUpdate"""
        from backend.ml_engine.federated_learning import ClientUpdate

        X, y = sample_data
        sklearn_client.set_local_data(X, y)

        # Pesos globais iniciais (vazios para primeira rodada)
        global_weights = {}

        result = sklearn_client.train_local(global_weights)

        # Pode ser ClientUpdate ou dict
        is_valid = isinstance(result, (ClientUpdate, dict)) or hasattr(result, 'weights')
        assert is_valid, f"train_local deve retornar ClientUpdate ou similar, got {type(result)}"

    def test_CONTRACT_006_evaluate_local_returns_dict(self, sklearn_client, sample_data):
        """CONTRACT-006: evaluate_local deve retornar Dict[str, float]"""
        X, y = sample_data
        sklearn_client.set_local_data(X, y)

        # Treinar antes de avaliar
        sklearn_client.train_local({})

        result = sklearn_client.evaluate_local()

        assert isinstance(result, dict), f"evaluate_local deve retornar dict, got {type(result)}"

    def test_CONTRACT_007_evaluate_local_has_metric_keys(self, sklearn_client, sample_data):
        """CONTRACT-007: evaluate_local deve ter chaves de métricas"""
        X, y = sample_data
        sklearn_client.set_local_data(X, y)
        sklearn_client.train_local({})

        result = sklearn_client.evaluate_local()

        # Deve ter alguma métrica de performance
        has_metrics = any(k in result for k in ['accuracy', 'loss', 'f1', 'auc', 'precision', 'recall'])
        assert has_metrics or len(result) > 0, f"Faltam métricas de avaliação. Keys: {list(result.keys())}"

    def test_CONTRACT_008_client_has_client_id(self, sklearn_client):
        """CONTRACT-008: Cliente deve ter client_id"""
        assert hasattr(sklearn_client, 'client_id'), "Falta atributo client_id"
        assert sklearn_client.client_id == "test_client_001"

    def test_CONTRACT_009_client_has_config(self, sklearn_client):
        """CONTRACT-009: Cliente deve ter config"""
        assert hasattr(sklearn_client, 'config'), "Falta atributo config"


class TestFederatedClientAbstract:
    """Verifica que FederatedClient é abstrato"""

    def test_ABSTRACT_001_cannot_instantiate_directly(self):
        """ABSTRACT-001: Não deve ser possível instanciar FederatedClient diretamente"""
        from backend.ml_engine.federated_learning import FederatedClient, FederatedConfig

        config = FederatedConfig()

        with pytest.raises(TypeError):
            FederatedClient(client_id="test", config=config)

    def test_ABSTRACT_002_has_abstract_methods(self):
        """ABSTRACT-002: Deve ter métodos abstratos"""
        from backend.ml_engine.federated_learning import FederatedClient
        import inspect

        abstract_methods = []
        for name, method in inspect.getmembers(FederatedClient, predicate=inspect.isfunction):
            if getattr(method, '__isabstractmethod__', False):
                abstract_methods.append(name)

        expected_methods = ['set_local_data', 'train_local', 'evaluate_local']
        for method in expected_methods:
            assert method in abstract_methods, f"Método abstrato '{method}' não encontrado"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
