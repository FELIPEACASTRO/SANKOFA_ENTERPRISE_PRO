"""
TRIPLE CHECK 1000x - Validacao rigorosa dos modulos P0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta
import numpy as np

def test_external_context():
    """Validacao completa de external_context.py"""
    print("=" * 60)
    print("TRIPLE CHECK: external_context.py")
    print("=" * 60)

    from ml_engine.feature_engineering.external_context import (
        HolidayType, EventRiskLevel, Holiday, CommercialEvent,
        ExternalContextFeatures, BrazilianHolidayCalendar,
        CommercialEventCalendar, ExternalContextGenerator, get_context_generator
    )

    errors = []

    # 1. ENUM VALIDATION
    print("\n[1] Validando Enums...")
    try:
        assert HolidayType.NATIONAL.value == "national"
        assert HolidayType.STATE.value == "state"
        assert EventRiskLevel.CRITICAL.value == "critical"
        print("    Enums: OK")
    except Exception as e:
        errors.append(f"Enums: {e}")

    # 2. DATACLASS VALIDATION
    print("\n[2] Validando Dataclasses...")
    try:
        h = Holiday(name="Test", date=date(2025, 1, 1), holiday_type=HolidayType.NATIONAL)
        assert h.applies_to_state("SP") == True

        ecf = ExternalContextFeatures()
        d = ecf.to_dict()
        assert isinstance(d, dict)
        arr = ecf.to_feature_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 14
        print("    Dataclasses: OK")
    except Exception as e:
        errors.append(f"Dataclasses: {e}")

    # 3. CALENDAR VALIDATION
    print("\n[3] Validando Calendarios...")
    cal = BrazilianHolidayCalendar()
    try:
        # Easter 2025
        easter = cal._calculate_easter(2025)
        assert easter == date(2025, 4, 20), f"Easter wrong: {easter}"

        # Carnaval 2025 = Easter - 47 dias = 4 de marco
        carnaval = cal.is_holiday(date(2025, 3, 4))
        assert carnaval is not None, "Carnaval nao detectado"
        assert "Carnaval" in carnaval.name

        # Natal
        natal = cal.is_holiday(date(2025, 12, 25))
        assert natal is not None
        assert natal.name == "Natal"

        print("    Calendario feriados: OK")
    except Exception as e:
        errors.append(f"Calendario: {e}")

    evt_cal = CommercialEventCalendar()
    try:
        bf = evt_cal._get_black_friday(2025)
        assert bf == date(2025, 11, 28), f"BF wrong: {bf}"

        events = evt_cal.get_events_for_year(2025)
        assert len(events) >= 8

        bf_event = evt_cal.get_active_event(date(2025, 11, 28))
        assert bf_event is not None
        assert bf_event.name == "Black Friday"

        print("    Calendario eventos: OK")
    except Exception as e:
        errors.append(f"Eventos: {e}")

    # 4. GENERATOR VALIDATION
    print("\n[4] Validando Generator...")
    gen = get_context_generator()
    try:
        # Natal
        r = gen.generate(datetime(2025, 12, 25))
        assert r.is_holiday == True
        assert r.holiday_name == "Natal"

        # Black Friday
        r = gen.generate(datetime(2025, 11, 28))
        assert r.is_commercial_event == True
        assert r.event_fraud_multiplier >= 2.0

        # Salary period
        r = gen.generate(datetime(2025, 7, 3))
        assert r.is_salary_period == True

        # End of month
        r = gen.generate(datetime(2025, 7, 30))
        assert r.is_end_of_month == True

        # Batch
        results = gen.generate_batch([datetime(2025, 1, 1), datetime(2025, 12, 25)])
        assert len(results) == 2

        print("    Generator: OK")
    except Exception as e:
        errors.append(f"Generator: {e}")

    # 5. SINGLETON
    print("\n[5] Validando Singleton...")
    try:
        g1 = get_context_generator()
        g2 = get_context_generator()
        assert g1 is g2
        print("    Singleton: OK")
    except Exception as e:
        errors.append(f"Singleton: {e}")

    return errors


def test_finops_telemetry():
    """Validacao completa de finops_telemetry.py"""
    print("\n" + "=" * 60)
    print("TRIPLE CHECK: finops_telemetry.py")
    print("=" * 60)

    from ml_engine.feature_engineering.finops_telemetry import (
        ServiceType, OperationType, CostMetric, CostSummary, CostModel,
        FinOpsTelemetry, FinOpsTracker, get_finops_telemetry, track_cost
    )

    errors = []

    # 1. ENUMS
    print("\n[1] Validando Enums...")
    try:
        assert ServiceType.ML_INFERENCE.value == "ml_inference"
        assert OperationType.PREDICT.value == "predict"
        print("    Enums: OK")
    except Exception as e:
        errors.append(f"Enums: {e}")

    # 2. COST MODEL
    print("\n[2] Validando CostModel...")
    try:
        cost = CostModel.calculate_cost(
            ServiceType.ML_INFERENCE,
            OperationType.PREDICT,
            duration_ms=50.0
        )
        assert cost > 0

        cost_cache = CostModel.calculate_cost(
            ServiceType.ML_INFERENCE,
            OperationType.PREDICT,
            duration_ms=50.0,
            cache_hit=True
        )
        assert cost_cache < cost  # Cache deve reduzir custo

        print("    CostModel: OK")
    except Exception as e:
        errors.append(f"CostModel: {e}")

    # 3. TELEMETRY
    print("\n[3] Validando FinOpsTelemetry...")
    try:
        tel = FinOpsTelemetry()
        tel.reset()

        # Record
        metric = tel.record(
            service=ServiceType.ML_INFERENCE,
            operation=OperationType.PREDICT,
            duration_ms=50.0
        )
        assert isinstance(metric, CostMetric)
        assert metric.cost_usd > 0

        # Summary
        summary = tel.get_summary()
        assert isinstance(summary, CostSummary)
        assert summary.total_operations >= 1

        # Cost per prediction
        cost_per = tel.get_cost_per_prediction()
        assert cost_per > 0

        # Cache hit rate (with mixed hits)
        tel.reset()
        tel.record(ServiceType.ML_INFERENCE, OperationType.PREDICT, 50.0, cache_hit=True)
        tel.record(ServiceType.ML_INFERENCE, OperationType.PREDICT, 50.0, cache_hit=False)
        rate = tel.get_cache_hit_rate()
        assert rate == 0.5

        # Cost breakdown (NO DEADLOCK TEST)
        breakdown = tel.get_cost_breakdown()
        assert "total_cost_usd" in breakdown
        assert "cost_by_service" in breakdown

        print("    FinOpsTelemetry: OK")
    except Exception as e:
        errors.append(f"FinOpsTelemetry: {e}")

    # 4. TRACKER
    print("\n[4] Validando FinOpsTracker...")
    try:
        tel = get_finops_telemetry()
        tel.reset()

        with track_cost(ServiceType.ML_INFERENCE, OperationType.PREDICT) as tracker:
            _ = sum(range(1000))

        assert tel.get_total_cost() > 0
        print("    FinOpsTracker: OK")
    except Exception as e:
        errors.append(f"FinOpsTracker: {e}")

    # 5. SINGLETON
    print("\n[5] Validando Singleton...")
    try:
        t1 = get_finops_telemetry()
        t2 = get_finops_telemetry()
        assert t1 is t2
        print("    Singleton: OK")
    except Exception as e:
        errors.append(f"Singleton: {e}")

    return errors


def test_mcc_hierarchy():
    """Validacao completa de mcc_hierarchy.py"""
    print("\n" + "=" * 60)
    print("TRIPLE CHECK: mcc_hierarchy.py")
    print("=" * 60)

    from ml_engine.feature_engineering.mcc_hierarchy import (
        MCCCode, MCCFeatures, MCCHierarchy, MCCFeatureGenerator, get_mcc_generator
    )

    errors = []

    # 1. HIERARCHY
    print("\n[1] Validando MCCHierarchy...")
    try:
        from ml_engine.feature_engineering.mcc_hierarchy import MCCCategory, MCCRiskTier
        hierarchy = MCCHierarchy()

        # Grocery
        mcc = hierarchy.get_mcc("5411")
        assert mcc is not None, "5411 not found"
        assert mcc.category == MCCCategory.RETAIL_STORES, f"5411 category wrong: {mcc.category}"
        assert mcc.risk_tier == MCCRiskTier.VERY_LOW, f"5411 risk_tier wrong: {mcc.risk_tier}"

        # Gambling
        mcc = hierarchy.get_mcc("7995")
        assert mcc is not None, "7995 not found"
        assert mcc.category == MCCCategory.GAMBLING, f"7995 category wrong: {mcc.category}"
        assert mcc.risk_tier == MCCRiskTier.VERY_HIGH, f"7995 risk_tier wrong: {mcc.risk_tier}"

        # Money transfer
        mcc = hierarchy.get_mcc("4829")
        assert mcc is not None, "4829 not found"
        assert mcc.category == MCCCategory.MONEY_TRANSFER, f"4829 category wrong: {mcc.category}"

        # Unknown - get_mcc returns None for unknown
        mcc = hierarchy.get_mcc("9999")
        assert mcc is None, "9999 should return None from get_mcc"

        # Use get_mcc_or_default for unknown
        mcc = hierarchy.get_mcc_or_default("9999")
        assert mcc is not None, "9999 should return default"
        assert mcc.category == MCCCategory.UNKNOWN, f"9999 category wrong: {mcc.category}"

        print("    MCCHierarchy: OK")
    except Exception as e:
        errors.append(f"MCCHierarchy: {e}")

    # 2. GENERATOR
    print("\n[2] Validando MCCFeatureGenerator...")
    try:
        gen = get_mcc_generator()

        # Generate
        features = gen.generate("5411")
        assert isinstance(features, MCCFeatures)
        assert features.risk_tier == "very_low"
        assert 0 <= features.risk_score <= 1

        # High risk
        features = gen.generate("7995")
        assert features.is_gambling == True
        assert features.is_high_risk_category == True

        # to_dict
        d = features.to_dict()
        assert isinstance(d, dict)
        assert "mcc_code" in d

        # to_feature_array
        arr = features.to_feature_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) > 0

        # Batch
        results = gen.generate_batch(["5411", "7995", "4829"])
        assert len(results) == 3

        print("    MCCFeatureGenerator: OK")
    except Exception as e:
        errors.append(f"MCCFeatureGenerator: {e}")

    # 3. RISK SCORE VALIDATION
    print("\n[3] Validando Risk Scores...")
    try:
        gen = get_mcc_generator()
        test_mccs = ["5411", "7995", "4829", "5732", "8011", "9999"]
        for mcc in test_mccs:
            features = gen.generate(mcc)
            assert 0 <= features.risk_score <= 1, f"MCC {mcc} risk_score fora do range: {features.risk_score}"
        print("    Risk scores em [0,1]: OK")
    except Exception as e:
        errors.append(f"Risk scores: {e}")

    # 4. SINGLETON
    print("\n[4] Validando Singleton...")
    try:
        g1 = get_mcc_generator()
        g2 = get_mcc_generator()
        assert g1 is g2
        print("    Singleton: OK")
    except Exception as e:
        errors.append(f"Singleton: {e}")

    return errors


def test_integration():
    """Teste de integracao entre modulos"""
    print("\n" + "=" * 60)
    print("TRIPLE CHECK: INTEGRACAO")
    print("=" * 60)

    errors = []

    print("\n[1] Testando uso combinado...")
    try:
        from ml_engine.feature_engineering.external_context import get_context_generator
        from ml_engine.feature_engineering.finops_telemetry import track_cost, ServiceType, OperationType, get_finops_telemetry
        from ml_engine.feature_engineering.mcc_hierarchy import get_mcc_generator

        # Reset telemetry
        get_finops_telemetry().reset()

        # Simular predicao
        ctx_gen = get_context_generator()
        mcc_gen = get_mcc_generator()

        with track_cost(ServiceType.FEATURE_ENGINEERING, OperationType.FEATURE_COMPUTE):
            ctx = ctx_gen.generate(datetime(2025, 11, 28))  # Black Friday
            mcc = mcc_gen.generate("7995")  # Gambling

        # Verificar contexto
        assert ctx.is_commercial_event == True
        assert ctx.event_fraud_multiplier >= 2.0

        # Verificar MCC
        assert mcc.is_gambling == True

        # Verificar telemetria
        tel = get_finops_telemetry()
        assert tel.get_total_cost() > 0

        print("    Uso combinado: OK")
    except Exception as e:
        errors.append(f"Integracao: {e}")

    print("\n[2] Testando arrays para ML...")
    try:
        ctx = get_context_generator().generate(datetime(2025, 12, 25))
        mcc = get_mcc_generator().generate("5411")

        ctx_arr = ctx.to_feature_array()
        mcc_arr = mcc.to_feature_array()

        # Combinar features
        combined = np.concatenate([ctx_arr, mcc_arr])
        assert len(combined) == len(ctx_arr) + len(mcc_arr)
        assert combined.dtype in [np.float32, np.float64]

        print("    Arrays ML: OK")
    except Exception as e:
        errors.append(f"Arrays ML: {e}")

    return errors


def main():
    print("\n" + "=" * 60)
    print("  TRIPLE CHECK 1000x - VALIDACAO RIGOROSA P0")
    print("=" * 60)

    all_errors = []

    # Test each module
    all_errors.extend(test_external_context())
    all_errors.extend(test_finops_telemetry())
    all_errors.extend(test_mcc_hierarchy())
    all_errors.extend(test_integration())

    # Final report
    print("\n" + "=" * 60)
    print("  RELATORIO FINAL")
    print("=" * 60)

    if all_errors:
        print(f"\nERROS ENCONTRADOS: {len(all_errors)}")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    else:
        print("\nTODAS AS VALIDACOES PASSARAM!")
        print("\nModulos validados:")
        print("  - external_context.py: OK")
        print("  - finops_telemetry.py: OK")
        print("  - mcc_hierarchy.py: OK")
        print("  - Integracao: OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
