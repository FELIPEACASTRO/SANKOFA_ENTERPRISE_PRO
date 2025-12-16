"""
=================================================================
FASE 2: ML ENGINEER - VALIDAÇÃO DE PIPELINES PRODUTIVOS
=================================================================
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("AUDITORIA ML ENGINEER - VALIDACAO DE PIPELINES PRODUTIVOS")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")

# 1. TESTE DE PREDICAO END-TO-END
print("\n" + "=" * 60)
print("1. TESTE DE PREDICAO END-TO-END")
print("=" * 60)

try:
    from ml_engine.production_fraud_engine import ProductionFraudEngine

    engine = ProductionFraudEngine()

    # Transacao normal
    normal_txn = {
        "amount": 150.00,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0,
        "is_new_device": 0,
        "is_new_recipient": 0,
        "velocity_score": 0.1,
        "transaction_count_24h": 2,
        "customer_id": "CUST_001"
    }

    # Transacao suspeita
    fraud_txn = {
        "amount": 15000.00,
        "hour": 3,
        "day_of_week": 6,
        "is_weekend": 1,
        "is_night": 1,
        "is_new_device": 1,
        "is_new_recipient": 1,
        "velocity_score": 0.9,
        "transaction_count_24h": 15,
        "customer_id": "CUST_002"
    }

    print("\n  Transacao Normal:")
    result_normal = engine.predict(normal_txn)
    print(f"    Score: {result_normal.get('fraud_probability', result_normal.get('score', 'N/A'))}")
    print(f"    Decisao: {result_normal.get('decision', result_normal.get('is_fraud', 'N/A'))}")

    print("\n  Transacao Suspeita:")
    result_fraud = engine.predict(fraud_txn)
    print(f"    Score: {result_fraud.get('fraud_probability', result_fraud.get('score', 'N/A'))}")
    print(f"    Decisao: {result_fraud.get('decision', result_fraud.get('is_fraud', 'N/A'))}")

    # Validar que fraude tem score maior
    score_normal = result_normal.get('fraud_probability', result_normal.get('score', 0))
    score_fraud = result_fraud.get('fraud_probability', result_fraud.get('score', 0))

    if isinstance(score_normal, (int, float)) and isinstance(score_fraud, (int, float)):
        if score_fraud > score_normal:
            print("\n  [OK] VALIDACAO: Fraude tem score maior que normal - CORRETO")
        else:
            print("\n  [WARN] VALIDACAO: Fraude deveria ter score maior - VERIFICAR!")

    print("\n  [OK] ProductionFraudEngine: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] ProductionFraudEngine: ERRO - {e}")
    import traceback
    traceback.print_exc()

# 2. TESTE DO ENSEMBLE INTEGRATION
print("\n" + "=" * 60)
print("2. TESTE DO ENSEMBLE INTEGRATION")
print("=" * 60)

try:
    from ml_engine.ensemble_integration import IntegratedEnsemble

    ensemble = IntegratedEnsemble()

    test_data = {
        "amount": 5000.00,
        "hour": 2,
        "day_of_week": 0,
        "is_weekend": 0,
        "is_night": 1,
        "is_new_device": 1,
        "is_new_recipient": 1,
        "velocity_score": 0.7,
        "customer_id": "TEST_001"
    }

    result = ensemble.predict(test_data)
    print(f"  Resultado tipo: {type(result).__name__}")
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    elif isinstance(result, dict):
        for key, value in result.items():
            print(f"    {key}: {value}")

    print("\n  [OK] IntegratedEnsemble: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] IntegratedEnsemble: ERRO - {e}")

# 3. TESTE DO AUTOENCODER
print("\n" + "=" * 60)
print("3. TESTE DO AUTOENCODER ANOMALY DETECTOR")
print("=" * 60)

try:
    from ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
    import pickle
    from pathlib import Path

    # Carregar modelo treinado
    model_path = Path(__file__).parent.parent / "models" / "production" / "autoencoder_model.pkl"
    with open(model_path, 'rb') as f:
        autoencoder = pickle.load(f)

    # Testar deteccao
    normal_data = {
        "amount": 200.0,
        "hour": 10,
        "day_of_week": 3,
        "is_weekend": 0,
        "is_night": 0,
        "velocity_score": 0.1,
        "transaction_count_24h": 1
    }

    anomaly_data = {
        "amount": 50000.0,
        "hour": 3,
        "day_of_week": 6,
        "is_weekend": 1,
        "is_night": 1,
        "velocity_score": 0.95,
        "transaction_count_24h": 20
    }

    result_normal = autoencoder.detect_anomaly(normal_data)
    result_anomaly = autoencoder.detect_anomaly(anomaly_data)

    print(f"  Normal: is_anomaly={result_normal.is_anomaly}, score={result_normal.anomaly_score:.4f}")
    print(f"  Anomalia: is_anomaly={result_anomaly.is_anomaly}, score={result_anomaly.anomaly_score:.4f}")

    print("\n  [OK] AutoencoderAnomalyDetector: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] AutoencoderAnomalyDetector: ERRO - {e}")

# 4. TESTE DO BILSTM
print("\n" + "=" * 60)
print("4. TESTE DO BiLSTM SEQUENCE ANALYZER")
print("=" * 60)

try:
    from ml_engine.bilstm_sequence_analyzer import BiLSTMSequenceAnalyzer
    import pickle
    from pathlib import Path

    model_path = Path(__file__).parent.parent / "models" / "production" / "bilstm_model.pkl"
    with open(model_path, 'rb') as f:
        bilstm = pickle.load(f)

    # Adicionar algumas transacoes para um cliente
    customer_id = "TEST_BILSTM_001"

    txns = [
        {"amount": 100, "hour": 10, "day_of_week": 1},
        {"amount": 150, "hour": 12, "day_of_week": 1},
        {"amount": 200, "hour": 14, "day_of_week": 1},
        {"amount": 50000, "hour": 3, "day_of_week": 6},  # Anomalia
    ]

    for txn in txns:
        bilstm.sequence_buffer.add_transaction(customer_id, txn)

    result = bilstm.analyze_sequence(customer_id, txns[-1])
    print(f"  Sequencia suspeita: {result.is_suspicious_sequence}")
    print(f"  Risk score: {result.sequence_risk_score:.4f}")

    print("\n  [OK] BiLSTMSequenceAnalyzer: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] BiLSTMSequenceAnalyzer: ERRO - {e}")

# 5. TESTE DO CATBOOST
print("\n" + "=" * 60)
print("5. TESTE DO CATBOOST MODEL")
print("=" * 60)

try:
    from ml_engine.catboost_model import CatBoostFraudModel

    catboost = CatBoostFraudModel()

    test_features = {
        "amount": 1000.0,
        "latitude": -23.5505,
        "longitude": -46.6333,
        "hour": 14,
        "day_of_week": 3,
        "velocity_score": 0.3,
        "is_new_device": 0,
        "location_risk_score": 0.2
    }

    result = catboost.predict(test_features)
    print(f"  Resultado: {result}")

    print("\n  [OK] CatBoostFraudModel: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] CatBoostFraudModel: ERRO - {e}")

# 6. TESTE DO FEATURE GENERATOR
print("\n" + "=" * 60)
print("6. TESTE DO FEATURE GENERATOR")
print("=" * 60)

try:
    from ml_engine.feature_engineering import FeatureGenerator

    fg = FeatureGenerator()

    txn = {
        "amount": 500.0,
        "timestamp": datetime.now().isoformat(),
        "customer_id": "CUST_FE_001",
        "merchant_id": "MERCH_001"
    }

    features = fg.generate_features(txn)
    print(f"  Features geradas: {len(features)} campos")
    print(f"  Algumas features: {list(features.keys())[:5]}")

    print("\n  [OK] FeatureGenerator: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] FeatureGenerator: ERRO - {e}")

# 7. TESTE DO MULE DETECTOR
print("\n" + "=" * 60)
print("7. TESTE DO MULE DETECTOR")
print("=" * 60)

try:
    from ml_engine.mule_detection.mule_detector import MuleDetector

    mule = MuleDetector()

    account = {
        "account_id": "ACC_MULE_001",
        "dormancy_days": 180,
        "turnover_ratio": 0.95,
        "in_degree": 50,
        "out_degree": 2,
        "total_received": 100000,
        "total_sent": 95000
    }

    result = mule.analyze(account)
    print(f"  Is mule: {result.get('is_mule', 'N/A')}")
    print(f"  Mule score: {result.get('mule_score', 'N/A')}")

    print("\n  [OK] MuleDetector: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] MuleDetector: ERRO - {e}")

# 8. TESTE DO MIXTURE OF EXPERTS
print("\n" + "=" * 60)
print("8. TESTE DO MIXTURE OF EXPERTS")
print("=" * 60)

try:
    from ml_engine.mixture_of_experts import MixtureOfExperts

    moe = MixtureOfExperts()

    txn = {
        "amount": 2500.0,
        "channel": "PIX",
        "hour": 15,
        "customer_id": "CUST_MOE_001"
    }

    result = moe.predict(txn)
    print(f"  Resultado: {result}")

    print("\n  [OK] MixtureOfExperts: FUNCIONANDO")

except Exception as e:
    print(f"\n  [FAIL] MixtureOfExperts: ERRO - {e}")

print("\n" + "=" * 80)
print("AUDITORIA ML ENGINEER CONCLUIDA")
print("=" * 80)
