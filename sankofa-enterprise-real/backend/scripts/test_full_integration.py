"""
Script de teste completo para verificar todas as integracoes do sistema.
Testa: Aliases, Modelos, Ensemble, MuleDetector, HardRulesEngine
"""
import sys
import os
import json
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_aliases():
    """Teste 1: Verificar todos os aliases"""
    logger.info("=" * 60)
    logger.info("TESTE 1: VERIFICANDO ALIASES")
    logger.info("=" * 60)

    results = []

    try:
        from ml_engine.mixture_of_experts import MixtureOfExpertsDetector, MixtureOfExperts
        assert MixtureOfExpertsDetector is MixtureOfExperts
        results.append(("mixture_of_experts", "OK"))
    except Exception as e:
        results.append(("mixture_of_experts", f"FAIL: {e}"))

    try:
        from ml_engine.multi_armed_bandits import MultiArmedBanditOptimizer, StepUpMFAOptimizer
        assert MultiArmedBanditOptimizer is StepUpMFAOptimizer
        results.append(("multi_armed_bandits", "OK"))
    except Exception as e:
        results.append(("multi_armed_bandits", f"FAIL: {e}"))

    try:
        from ml_engine.feature_engineering import FeatureEngineering, FeatureGenerator
        assert FeatureEngineering is FeatureGenerator
        results.append(("feature_engineering", "OK"))
    except Exception as e:
        results.append(("feature_engineering", f"FAIL: {e}"))

    try:
        from ml_engine.causal_inference import CausalInferenceEngine, CausalImpactAnalyzer
        assert CausalInferenceEngine is CausalImpactAnalyzer
        results.append(("causal_inference", "OK"))
    except Exception as e:
        results.append(("causal_inference", f"FAIL: {e}"))

    try:
        from ml_engine.llm.fraud_llm import FraudLLM, FraudLLMAnalyzer
        assert FraudLLM is FraudLLMAnalyzer
        results.append(("fraud_llm", "OK"))
    except Exception as e:
        results.append(("fraud_llm", f"FAIL: {e}"))

    ok_count = sum(1 for _, status in results if status == "OK")
    for name, status in results:
        logger.info(f"  {name}: {status}")

    logger.info(f"  TOTAL: {ok_count}/{len(results)} OK")
    return ok_count == len(results)


def test_trained_models():
    """Teste 2: Verificar modelos treinados"""
    logger.info("=" * 60)
    logger.info("TESTE 2: VERIFICANDO MODELOS TREINADOS")
    logger.info("=" * 60)

    from pathlib import Path
    import pickle

    model_path = Path(__file__).parent.parent / "models" / "production"
    results = []

    # Verificar CatBoost
    catboost_path = model_path / "catboost_fraud_model.cbm"
    if catboost_path.exists():
        results.append(("CatBoost", "OK", f"{catboost_path.stat().st_size / 1024:.1f} KB"))
    else:
        results.append(("CatBoost", "NOT FOUND", ""))

    # Verificar Autoencoder
    autoencoder_path = model_path / "autoencoder_model.pkl"
    if autoencoder_path.exists():
        results.append(("Autoencoder", "OK", f"{autoencoder_path.stat().st_size / 1024:.1f} KB"))
    else:
        results.append(("Autoencoder", "NOT FOUND", ""))

    # Verificar BiLSTM
    bilstm_path = model_path / "bilstm_model.pkl"
    if bilstm_path.exists():
        results.append(("BiLSTM", "OK", f"{bilstm_path.stat().st_size / 1024:.1f} KB"))
    else:
        results.append(("BiLSTM", "NOT FOUND", ""))

    # Verificar Ensemble Config
    ensemble_config_path = model_path / "ensemble_config.json"
    if ensemble_config_path.exists():
        results.append(("Ensemble Config", "OK", f"{ensemble_config_path.stat().st_size / 1024:.1f} KB"))
    else:
        results.append(("Ensemble Config", "NOT FOUND", ""))

    ok_count = sum(1 for _, status, _ in results if status == "OK")
    for name, status, size in results:
        logger.info(f"  {name}: {status} {size}")

    logger.info(f"  TOTAL: {ok_count}/{len(results)} OK")
    return ok_count >= 3  # At least 3 models


def test_ensemble_integration():
    """Teste 3: Verificar IntegratedEnsemble"""
    logger.info("=" * 60)
    logger.info("TESTE 3: VERIFICANDO INTEGRATED ENSEMBLE")
    logger.info("=" * 60)

    try:
        from ml_engine.ensemble_integration import get_integrated_ensemble

        ensemble = get_integrated_ensemble()
        info = ensemble.get_ensemble_info()

        logger.info(f"  Version: {info['version']}")
        logger.info(f"  Weights: {info['weights']}")
        logger.info(f"  CatBoost disponivel: {info['catboost_available']}")
        logger.info(f"  CatBoost treinado: {info['catboost_trained']}")
        logger.info(f"  GNN disponivel: {info['gnn_available']}")
        logger.info(f"  MuleDetector disponivel: {info['mule_detector_available']}")
        logger.info(f"  HardRulesEngine disponivel: {info['hard_rules_available']}")

        return True
    except Exception as e:
        logger.error(f"  ERRO: {e}")
        return False


def test_prediction():
    """Teste 4: Testar predicao completa"""
    logger.info("=" * 60)
    logger.info("TESTE 4: TESTANDO PREDICAO COMPLETA")
    logger.info("=" * 60)

    try:
        from ml_engine.ensemble_integration import get_integrated_ensemble

        ensemble = get_integrated_ensemble()

        # Transacao de teste
        test_transaction = {
            "transaction_id": "TEST_001",
            "customer_id": "CUST_12345",
            "amount": 5000.00,
            "hour": 3,  # Horario suspeito
            "day_of_week": 6,
            "is_weekend": 1,
            "is_night": 1,
            "latitude": -23.5505,
            "longitude": -46.6333,
            "is_new_device": True,
            "is_new_recipient": True,
            "velocity_score": 0.8,
            "transaction_count_24h": 10,
            "channel": "PIX",
            "merchant_category": "transferencia_pf"
        }

        # Fazer predicao
        result = ensemble.predict_combined(test_transaction, base_probability=0.6)

        logger.info(f"  Transacao: {test_transaction['transaction_id']}")
        logger.info(f"  Fraud Probability: {result.fraud_probability:.4f}")
        logger.info(f"  Is Fraud: {result.is_fraud}")
        logger.info(f"  Model Contributions: {result.model_contributions}")
        logger.info(f"  Mule Score: {result.mule_score:.4f}")
        logger.info(f"  Mule Detected: {result.mule_detected}")
        logger.info(f"  Hard Rules Triggered: {result.hard_rules_triggered}")
        logger.info(f"  Processing Time: {result.processing_time_ms:.2f}ms")

        return True
    except Exception as e:
        logger.error(f"  ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mule_detector():
    """Teste 5: Testar MuleDetector diretamente"""
    logger.info("=" * 60)
    logger.info("TESTE 5: TESTANDO MULEDETECTOR")
    logger.info("=" * 60)

    try:
        from ml_engine.mule_detection import create_mule_detector

        detector = create_mule_detector()

        # Dados de conta suspeita
        account_data = {
            "account_id": "ACC_SUSPECT_001",
            "created_at": "2023-01-01T00:00:00"  # Conta antiga
        }

        # Historico de transacoes suspeitas
        transaction_history = [
            {"amount": 10000, "timestamp": "2024-12-01T03:00:00", "channel": "PIX"},
            {"amount": 15000, "timestamp": "2024-12-01T03:05:00", "channel": "PIX"},
            {"amount": 20000, "timestamp": "2024-12-01T03:10:00", "channel": "PIX"},
            {"amount": 25000, "timestamp": "2024-12-01T03:15:00", "channel": "PIX"},
            {"amount": 30000, "timestamp": "2024-12-01T03:20:00", "channel": "PIX"},
        ]

        result = detector.detect(
            account_id="ACC_SUSPECT_001",
            account_data=account_data,
            transaction_history=transaction_history
        )

        logger.info(f"  Account ID: {result.account_id}")
        logger.info(f"  Is Mule: {result.is_mule}")
        logger.info(f"  Mule Probability: {result.mule_probability:.4f}")
        logger.info(f"  Risk Level: {result.risk_level.value}")
        logger.info(f"  Mule Type: {result.mule_type.value}")
        logger.info(f"  Confidence: {result.confidence:.4f}")
        logger.info(f"  Indicators: {len(result.indicators)}")

        return True
    except Exception as e:
        logger.error(f"  ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoencoder():
    """Teste 6: Testar Autoencoder"""
    logger.info("=" * 60)
    logger.info("TESTE 6: TESTANDO AUTOENCODER")
    logger.info("=" * 60)

    try:
        from pathlib import Path
        import pickle

        model_path = Path(__file__).parent.parent / "models" / "production" / "autoencoder_model.pkl"

        if not model_path.exists():
            logger.warning("  Modelo nao encontrado - pulando teste")
            return True

        with open(model_path, 'rb') as f:
            autoencoder = pickle.load(f)

        # Transacao de teste
        test_transaction = {
            "amount": 50000,
            "log_amount": 10.82,
            "hour": 3,
            "day_of_week": 6,
            "is_weekend": 1,
            "is_night": 1,
            "latitude": -23.5505,
            "longitude": -46.6333,
            "is_new_device": 1,
            "is_new_recipient": 1,
            "velocity_score": 0.9,
            "transaction_count_24h": 15,
            "avg_amount_30d": 500,
            "std_amount_30d": 100,
        }

        result = autoencoder.detect_anomaly(test_transaction)

        logger.info(f"  Is Anomaly: {result.is_anomaly}")
        logger.info(f"  Anomaly Score: {result.anomaly_score:.4f}")
        logger.info(f"  Reconstruction Error: {result.reconstruction_error:.4f}")
        logger.info(f"  Confidence: {result.confidence:.4f}")

        return True
    except Exception as e:
        logger.error(f"  ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes"""
    logger.info("=" * 60)
    logger.info("INICIANDO TESTES DE INTEGRACAO COMPLETA")
    logger.info("=" * 60)

    results = []

    # Executar testes
    results.append(("Aliases", test_aliases()))
    results.append(("Modelos Treinados", test_trained_models()))
    results.append(("Ensemble Integration", test_ensemble_integration()))
    results.append(("Predicao Completa", test_prediction()))
    results.append(("MuleDetector", test_mule_detector()))
    results.append(("Autoencoder", test_autoencoder()))

    # Resumo final
    logger.info("=" * 60)
    logger.info("RESUMO DOS TESTES")
    logger.info("=" * 60)

    passed = 0
    failed = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        else:
            failed += 1
        logger.info(f"  {name}: {status}")

    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed} PASS, {failed} FAIL")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
