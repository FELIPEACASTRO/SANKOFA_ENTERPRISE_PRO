"""
CHECKPOINT 1 - Unit Tests: ProductionFraudEngine
DataScientist_QA + MLEngineer_QA

Contrato:
- predict(DataFrame) -> np.ndarray[int] (0 ou 1)
- predict_proba(DataFrame) -> np.ndarray[float] (0.0 a 1.0)
- fit(DataFrame, Series) -> self
- Latência < 200ms por transação
- Determinismo: mesma entrada = mesma saída
"""
import pytest
import numpy as np
import pandas as pd
import time
import sys
import os

# Ajustar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestProductionFraudEngineContract:
    """Testes de contrato da API"""

    @pytest.fixture
    def engine(self):
        """Fixture para criar engine"""
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    @pytest.fixture
    def sample_df(self):
        """DataFrame de exemplo com features mínimas"""
        return pd.DataFrame({
            'amount': [1500.0, 2500.0, 500.0],
            'hour': [14, 3, 10],
            'day_of_week': [3, 1, 5],
            'velocity_score': [0.3, 0.8, 0.1],
            'is_new_device': [0, 1, 0],
            'merchant_category': ['retail', 'online', 'retail'],
            'channel': ['pix', 'pix', 'ted']
        })

    def test_DS_001_predict_accepts_dataframe(self, engine, sample_df):
        """DS-001: predict() deve aceitar DataFrame"""
        result = engine.predict(sample_df)
        assert result is not None, "predict() retornou None"
        assert hasattr(result, '__len__'), "Resultado não é iterável"

    def test_DS_002_predict_returns_binary_array(self, engine, sample_df):
        """DS-002: predict() deve retornar array com valores 0 ou 1"""
        result = engine.predict(sample_df)
        result_array = np.array(result)
        unique_values = set(result_array.flatten())
        assert unique_values.issubset({0, 1}), f"Valores não binários: {unique_values}"

    def test_DS_003_predict_output_length_matches_input(self, engine, sample_df):
        """DS-003: Quantidade de predições == quantidade de inputs"""
        result = engine.predict(sample_df)
        assert len(result) == len(sample_df), f"Expected {len(sample_df)}, got {len(result)}"

    def test_DS_004_predict_proba_returns_float_array(self, engine, sample_df):
        """DS-004: predict_proba() deve retornar floats entre 0 e 1"""
        if not hasattr(engine, 'predict_proba'):
            pytest.skip("predict_proba não implementado")
        result = engine.predict_proba(sample_df)
        result_array = np.array(result).flatten()
        assert all(0 <= v <= 1 for v in result_array), "Probabilidades fora do range [0,1]"

    def test_DS_005_predict_handles_single_row(self, engine):
        """DS-005: predict() deve funcionar com uma única linha"""
        single_row = pd.DataFrame({
            'amount': [1000.0],
            'hour': [12],
            'day_of_week': [2],
            'velocity_score': [0.5],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })
        result = engine.predict(single_row)
        assert len(result) == 1, "Deveria retornar exatamente 1 predição"


class TestProductionFraudEnginePerformance:
    """Testes de performance"""

    @pytest.fixture
    def engine(self):
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    def test_ML_001_latency_under_200ms_single(self, engine):
        """ML-001: Latência < 200ms para transação única"""
        single_df = pd.DataFrame({
            'amount': [1500.0],
            'hour': [14],
            'day_of_week': [3],
            'velocity_score': [0.3],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })

        start = time.perf_counter()
        _ = engine.predict(single_df)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Latência {elapsed_ms:.2f}ms > 200ms"

    def test_ML_002_batch_100_under_2000ms(self, engine):
        """ML-002: Batch de 100 transações em < 2000ms"""
        batch_df = pd.DataFrame({
            'amount': np.random.uniform(100, 10000, 100),
            'hour': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'velocity_score': np.random.uniform(0, 1, 100),
            'is_new_device': np.random.randint(0, 2, 100),
            'merchant_category': np.random.choice(['retail', 'online', 'food'], 100),
            'channel': np.random.choice(['pix', 'ted', 'card'], 100)
        })

        start = time.perf_counter()
        result = engine.predict(batch_df)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(result) == 100, f"Expected 100, got {len(result)}"
        assert elapsed_ms < 2000, f"Batch latency {elapsed_ms:.2f}ms > 2000ms"

    def test_ML_003_p99_latency_under_500ms(self, engine):
        """ML-003: P99 latência < 500ms em 20 execuções"""
        single_df = pd.DataFrame({
            'amount': [1500.0],
            'hour': [14],
            'day_of_week': [3],
            'velocity_score': [0.3],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            _ = engine.predict(single_df)
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = np.percentile(latencies, 99)
        assert p99 < 500, f"P99 latency {p99:.2f}ms > 500ms"


class TestProductionFraudEngineDeterminism:
    """Testes de determinismo"""

    @pytest.fixture
    def engine(self):
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    def test_BANK_001_deterministic_predictions(self, engine):
        """BANK-001: Mesma entrada = mesma saída"""
        fixed_df = pd.DataFrame({
            'amount': [1500.0],
            'hour': [14],
            'day_of_week': [3],
            'velocity_score': [0.3],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })

        results = [engine.predict(fixed_df)[0] for _ in range(5)]
        assert len(set(results)) == 1, f"Resultados não determinísticos: {results}"

    def test_BANK_002_consistent_batch_vs_individual(self, engine):
        """BANK-002: Predição em batch == predições individuais"""
        df1 = pd.DataFrame({
            'amount': [1500.0],
            'hour': [14],
            'day_of_week': [3],
            'velocity_score': [0.3],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })
        df2 = pd.DataFrame({
            'amount': [5000.0],
            'hour': [3],
            'day_of_week': [1],
            'velocity_score': [0.9],
            'is_new_device': [1],
            'merchant_category': ['online'],
            'channel': ['pix']
        })

        # Individual
        pred1 = engine.predict(df1)[0]
        pred2 = engine.predict(df2)[0]

        # Batch
        batch_df = pd.concat([df1, df2], ignore_index=True)
        batch_preds = engine.predict(batch_df)

        assert batch_preds[0] == pred1, f"Batch[0]={batch_preds[0]} != Individual={pred1}"
        assert batch_preds[1] == pred2, f"Batch[1]={batch_preds[1]} != Individual={pred2}"


class TestProductionFraudEngineEdgeCases:
    """Testes de casos extremos"""

    @pytest.fixture
    def engine(self):
        from backend.ml_engine.production_fraud_engine import ProductionFraudEngine
        return ProductionFraudEngine()

    def test_EDGE_001_high_amount_transaction(self, engine):
        """EDGE-001: Transação de alto valor"""
        high_value_df = pd.DataFrame({
            'amount': [999999.99],
            'hour': [3],
            'day_of_week': [0],
            'velocity_score': [0.95],
            'is_new_device': [1],
            'merchant_category': ['online'],
            'channel': ['pix']
        })
        result = engine.predict(high_value_df)
        assert result[0] in [0, 1], f"Resultado inválido: {result[0]}"

    def test_EDGE_002_zero_amount_transaction(self, engine):
        """EDGE-002: Transação com valor zero"""
        zero_df = pd.DataFrame({
            'amount': [0.0],
            'hour': [12],
            'day_of_week': [3],
            'velocity_score': [0.1],
            'is_new_device': [0],
            'merchant_category': ['retail'],
            'channel': ['pix']
        })
        result = engine.predict(zero_df)
        assert result[0] in [0, 1], f"Resultado inválido: {result[0]}"

    def test_EDGE_003_midnight_transaction(self, engine):
        """EDGE-003: Transação à meia-noite"""
        midnight_df = pd.DataFrame({
            'amount': [5000.0],
            'hour': [0],
            'day_of_week': [6],
            'velocity_score': [0.7],
            'is_new_device': [1],
            'merchant_category': ['online'],
            'channel': ['pix']
        })
        result = engine.predict(midnight_df)
        assert result[0] in [0, 1], f"Resultado inválido: {result[0]}"

    def test_EDGE_004_empty_dataframe_handling(self, engine):
        """EDGE-004: DataFrame vazio deve ser tratado"""
        empty_df = pd.DataFrame(columns=[
            'amount', 'hour', 'day_of_week', 'velocity_score',
            'is_new_device', 'merchant_category', 'channel'
        ])

        try:
            result = engine.predict(empty_df)
            assert len(result) == 0, "DataFrame vazio deveria retornar array vazio"
        except (ValueError, IndexError) as e:
            # Aceitável lançar exceção para entrada vazia
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
