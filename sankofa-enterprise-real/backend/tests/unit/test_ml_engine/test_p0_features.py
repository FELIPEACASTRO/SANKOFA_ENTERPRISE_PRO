"""
Testes para modulos P0 - Features de Contexto Externo, FinOps e MCC
MODO MILITAR 10000x
"""
import pytest
from datetime import date, datetime, timedelta
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestExternalContextFeatures:
    """Testes para external_context.py"""

    @pytest.fixture
    def context_generator(self):
        from backend.ml_engine.feature_engineering.external_context import ExternalContextGenerator
        return ExternalContextGenerator()

    def test_P0_CTX_001_generator_exists(self, context_generator):
        """P0-CTX-001: Gerador de contexto externo existe"""
        assert context_generator is not None
        assert hasattr(context_generator, 'generate')

    def test_P0_CTX_002_generate_returns_features(self, context_generator):
        """P0-CTX-002: generate() retorna ExternalContextFeatures"""
        from backend.ml_engine.feature_engineering.external_context import ExternalContextFeatures
        result = context_generator.generate(datetime(2025, 12, 25))
        assert isinstance(result, ExternalContextFeatures)

    def test_P0_CTX_003_christmas_is_holiday(self, context_generator):
        """P0-CTX-003: Natal e identificado como feriado"""
        result = context_generator.generate(datetime(2025, 12, 25))
        assert result.is_holiday is True
        assert result.holiday_name == "Natal"

    def test_P0_CTX_004_independence_day_is_holiday(self, context_generator):
        """P0-CTX-004: 7 de setembro e feriado"""
        result = context_generator.generate(datetime(2025, 9, 7))
        assert result.is_holiday is True

    def test_P0_CTX_005_normal_day_not_holiday(self, context_generator):
        """P0-CTX-005: Dia normal nao e feriado"""
        result = context_generator.generate(datetime(2025, 6, 18))  # Quarta normal
        assert result.is_holiday is False

    def test_P0_CTX_006_black_friday_detected(self, context_generator):
        """P0-CTX-006: Black Friday detectada como evento comercial"""
        # Black Friday 2025 = 28 de novembro
        result = context_generator.generate(datetime(2025, 11, 28))
        assert result.is_commercial_event is True
        assert result.commercial_event_name == "Black Friday"
        assert result.event_fraud_multiplier >= 2.0

    def test_P0_CTX_007_salary_period_detected(self, context_generator):
        """P0-CTX-007: Periodo de salario (dias 1-5) detectado"""
        result = context_generator.generate(datetime(2025, 7, 3))
        assert result.is_salary_period is True

    def test_P0_CTX_008_end_of_month_detected(self, context_generator):
        """P0-CTX-008: Fim do mes detectado"""
        result = context_generator.generate(datetime(2025, 7, 30))
        assert result.is_end_of_month is True

    def test_P0_CTX_009_risk_multiplier_calculated(self, context_generator):
        """P0-CTX-009: Risk multiplier e calculado"""
        result = context_generator.generate(datetime(2025, 12, 25))
        assert result.context_risk_multiplier >= 1.0

    def test_P0_CTX_010_to_dict_works(self, context_generator):
        """P0-CTX-010: to_dict() funciona"""
        result = context_generator.generate(datetime(2025, 7, 15))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "is_holiday" in d
        assert "context_risk_multiplier" in d

    def test_P0_CTX_011_to_feature_array_works(self, context_generator):
        """P0-CTX-011: to_feature_array() retorna numpy array"""
        result = context_generator.generate(datetime(2025, 7, 15))
        arr = result.to_feature_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) > 0

    def test_P0_CTX_012_batch_generation_works(self, context_generator):
        """P0-CTX-012: Batch generation funciona"""
        dates = [
            datetime(2025, 7, 15),
            datetime(2025, 12, 25),
            datetime(2025, 11, 28),
        ]
        results = context_generator.generate_batch(dates)
        assert len(results) == 3

    def test_P0_CTX_013_carnaval_detected(self, context_generator):
        """P0-CTX-013: Carnaval (feriado movel) detectado"""
        # Carnaval 2025 = 4 de marco (47 dias antes da Pascoa)
        result = context_generator.generate(datetime(2025, 3, 4))
        assert result.is_holiday is True
        assert "Carnaval" in result.holiday_name


class TestFinOpsTelemetry:
    """Testes para finops_telemetry.py"""

    @pytest.fixture
    def telemetry(self):
        from backend.ml_engine.feature_engineering.finops_telemetry import FinOpsTelemetry
        t = FinOpsTelemetry()
        t.reset()
        return t

    def test_P0_FIN_001_telemetry_exists(self, telemetry):
        """P0-FIN-001: Telemetria FinOps existe"""
        assert telemetry is not None
        assert hasattr(telemetry, 'record')

    def test_P0_FIN_002_record_returns_metric(self, telemetry):
        """P0-FIN-002: record() retorna CostMetric"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType, CostMetric
        )
        metric = telemetry.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0,
        )
        assert isinstance(metric, CostMetric)

    def test_P0_FIN_003_cost_calculated(self, telemetry):
        """P0-FIN-003: Custo e calculado"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType
        )
        metric = telemetry.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0,
        )
        assert metric.cost_usd > 0

    def test_P0_FIN_004_cache_hit_reduces_cost(self, telemetry):
        """P0-FIN-004: Cache hit reduz custo"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType
        )
        metric_no_cache = telemetry.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0,
            cache_hit=False,
        )
        metric_cache = telemetry.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0,
            cache_hit=True,
        )
        assert metric_cache.cost_usd < metric_no_cache.cost_usd

    def test_P0_FIN_005_get_summary_works(self, telemetry):
        """P0-FIN-005: get_summary() funciona"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType, CostSummary
        )
        telemetry.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0,
        )
        summary = telemetry.get_summary()
        assert isinstance(summary, CostSummary)
        assert summary.total_operations >= 1

    def test_P0_FIN_006_cost_per_prediction_works(self, telemetry):
        """P0-FIN-006: get_cost_per_prediction() funciona"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType
        )
        for _ in range(5):
            telemetry.record(
                service=ServiceType.ML_INFERENCE,
                operation=OperationType.PREDICT,
                duration_ms=50.0,
            )
        cost = telemetry.get_cost_per_prediction()
        assert cost > 0

    def test_P0_FIN_007_cache_hit_rate_works(self, telemetry):
        """P0-FIN-007: get_cache_hit_rate() funciona"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType
        )
        telemetry.record(ServiceType.ML_INFERENCE, OperationType.PREDICT, 50.0, cache_hit=True)
        telemetry.record(ServiceType.ML_INFERENCE, OperationType.PREDICT, 50.0, cache_hit=False)
        rate = telemetry.get_cache_hit_rate()
        assert 0 <= rate <= 1
        assert rate == 0.5

    def test_P0_FIN_008_cost_breakdown_works(self, telemetry):
        """P0-FIN-008: get_cost_breakdown() funciona"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            ServiceType, OperationType
        )
        telemetry.record(ServiceType.ML_INFERENCE, OperationType.PREDICT, 50.0)
        telemetry.record(ServiceType.FEATURE_ENGINEERING, OperationType.FEATURE_COMPUTE, 30.0)
        breakdown = telemetry.get_cost_breakdown()
        assert "total_cost_usd" in breakdown
        assert "cost_by_service" in breakdown

    def test_P0_FIN_009_tracker_context_manager_works(self):
        """P0-FIN-009: FinOpsTracker context manager funciona"""
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            track_cost, ServiceType, OperationType, get_finops_telemetry
        )
        get_finops_telemetry().reset()

        with track_cost(ServiceType.ML_INFERENCE, OperationType.PREDICT) as tracker:
            # Simular trabalho
            _ = sum(range(1000))

        total = get_finops_telemetry().get_total_cost()
        assert total > 0


class TestMCCHierarchy:
    """Testes para mcc_hierarchy.py"""

    @pytest.fixture
    def mcc_generator(self):
        from backend.ml_engine.feature_engineering.mcc_hierarchy import MCCFeatureGenerator
        return MCCFeatureGenerator()

    def test_P0_MCC_001_generator_exists(self, mcc_generator):
        """P0-MCC-001: Gerador MCC existe"""
        assert mcc_generator is not None
        assert hasattr(mcc_generator, 'generate')

    def test_P0_MCC_002_generate_returns_features(self, mcc_generator):
        """P0-MCC-002: generate() retorna MCCFeatures"""
        from backend.ml_engine.feature_engineering.mcc_hierarchy import MCCFeatures
        result = mcc_generator.generate("5411")  # Grocery
        assert isinstance(result, MCCFeatures)

    def test_P0_MCC_003_grocery_low_risk(self, mcc_generator):
        """P0-MCC-003: Supermercado (5411) e baixo risco"""
        result = mcc_generator.generate("5411")
        assert result.risk_tier == "very_low"
        assert result.risk_score < 0.3

    def test_P0_MCC_004_gambling_high_risk(self, mcc_generator):
        """P0-MCC-004: Jogos (7995) e alto risco"""
        result = mcc_generator.generate("7995")
        assert result.risk_tier == "very_high"
        assert result.is_gambling is True
        assert result.is_high_risk_category is True

    def test_P0_MCC_005_money_transfer_detected(self, mcc_generator):
        """P0-MCC-005: Money transfer (4829) detectado"""
        result = mcc_generator.generate("4829")
        assert result.is_money_transfer is True
        assert result.is_high_risk_category is True

    def test_P0_MCC_006_unknown_mcc_handled(self, mcc_generator):
        """P0-MCC-006: MCC desconhecido tem fallback"""
        result = mcc_generator.generate("9999")
        assert result.category == "unknown"
        assert result.risk_tier == "medium"

    def test_P0_MCC_007_digital_goods_detected(self, mcc_generator):
        """P0-MCC-007: Digital goods (5816) detectado"""
        result = mcc_generator.generate("5816")
        assert result.is_digital is True

    def test_P0_MCC_008_to_dict_works(self, mcc_generator):
        """P0-MCC-008: to_dict() funciona"""
        result = mcc_generator.generate("5411")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "mcc_code" in d
        assert "mcc_risk_score" in d

    def test_P0_MCC_009_to_feature_array_works(self, mcc_generator):
        """P0-MCC-009: to_feature_array() retorna numpy array"""
        result = mcc_generator.generate("5411")
        arr = result.to_feature_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) > 0

    def test_P0_MCC_010_batch_generation_works(self, mcc_generator):
        """P0-MCC-010: Batch generation funciona"""
        mccs = ["5411", "7995", "4829", "9999"]
        results = mcc_generator.generate_batch(mccs)
        assert len(results) == 4

    def test_P0_MCC_011_risk_score_in_range(self, mcc_generator):
        """P0-MCC-011: Risk score em [0, 1]"""
        mccs = ["5411", "7995", "4829", "5732", "8011"]
        for mcc in mccs:
            result = mcc_generator.generate(mcc)
            assert 0 <= result.risk_score <= 1

    def test_P0_MCC_012_fraud_rate_baseline_positive(self, mcc_generator):
        """P0-MCC-012: Fraud rate baseline e positivo"""
        result = mcc_generator.generate("7995")
        assert result.fraud_rate_baseline > 0


class TestP0Integration:
    """Testes de integracao dos modulos P0"""

    def test_P0_INT_001_all_modules_import(self):
        """P0-INT-001: Todos os modulos P0 importam"""
        from backend.ml_engine.feature_engineering.external_context import (
            ExternalContextGenerator, get_context_generator
        )
        from backend.ml_engine.feature_engineering.finops_telemetry import (
            FinOpsTelemetry, get_finops_telemetry
        )
        from backend.ml_engine.feature_engineering.mcc_hierarchy import (
            MCCFeatureGenerator, get_mcc_generator
        )
        assert ExternalContextGenerator is not None
        assert FinOpsTelemetry is not None
        assert MCCFeatureGenerator is not None

    def test_P0_INT_002_singletons_work(self):
        """P0-INT-002: Singletons funcionam"""
        from backend.ml_engine.feature_engineering.external_context import get_context_generator
        from backend.ml_engine.feature_engineering.finops_telemetry import get_finops_telemetry
        from backend.ml_engine.feature_engineering.mcc_hierarchy import get_mcc_generator

        ctx1 = get_context_generator()
        ctx2 = get_context_generator()
        assert ctx1 is ctx2

        fin1 = get_finops_telemetry()
        fin2 = get_finops_telemetry()
        assert fin1 is fin2

        mcc1 = get_mcc_generator()
        mcc2 = get_mcc_generator()
        assert mcc1 is mcc2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
