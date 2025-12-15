"""
Testa a integração do CatBoost com o ensemble
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np


def test_catboost_integration():
    """Testa se o CatBoost está integrado e funcionando"""

    print("=" * 60)
    print("TESTE DE INTEGRAÇÃO DO CATBOOST")
    print("=" * 60)

    # 1. Testar carregamento do modelo
    print("\n1. Testando carregamento do CatBoost...")
    from ml_engine.catboost_model import CatBoostFraudModel, CATBOOST_AVAILABLE

    if not CATBOOST_AVAILABLE:
        print("   ERRO: CatBoost não está instalado!")
        return False

    model_path = backend_path / "models" / "production" / "catboost_fraud_model.cbm"
    if not model_path.exists():
        print(f"   ERRO: Modelo não encontrado em {model_path}")
        return False

    model = CatBoostFraudModel({})
    model.load(str(model_path))
    print(f"   OK - CatBoost carregado (trained: {model.is_trained})")
    print(f"   Features: {len(model.feature_names)}")

    # 2. Testar predição direta
    print("\n2. Testando predição direta do CatBoost...")
    test_transaction = pd.DataFrame([{
        "amount": 5000.0,
        "latitude": -23.5505,
        "longitude": -46.6333,
        "location_risk_score": 0.8,
        "merchant_category": "crypto_exchange",
        "hour": 2,
        "velocity_score": 0.7,
        "is_new_device": 1,
        "canal": "pix",
    }])

    predictions = model.predict(test_transaction)
    print(f"   OK - Predição: fraud_probability={predictions[0].fraud_probability:.4f}")
    print(f"   is_fraud: {predictions[0].is_fraud}")

    # 3. Testar IntegratedEnsemble
    print("\n3. Testando IntegratedEnsemble...")
    from ml_engine.ensemble_integration import IntegratedEnsemble, get_integrated_ensemble

    ensemble = get_integrated_ensemble()
    info = ensemble.get_ensemble_info()

    print(f"   OK - Ensemble inicializado")
    print(f"   CatBoost disponível: {info['catboost_available']}")
    print(f"   CatBoost treinado: {info['catboost_trained']}")
    print(f"   Pesos: {info['weights']}")

    # 4. Testar predição combinada
    print("\n4. Testando predição combinada...")
    txn_dict = {
        "transaction_id": "TEST001",
        "customer_id": "CUST001",
        "amount": 8000.0,
        "latitude": -25.5163,  # Foz do Iguaçu (fronteira)
        "longitude": -54.5854,
        "location_risk_score": 0.9,
        "merchant_category": "crypto_exchange",
        "hour": 3,
        "velocity_score": 0.8,
        "is_new_device": 1,
        "is_new_receiver": 1,
        "canal": "pix",
        "estado": "PR",
        "pais": "BR",
    }

    base_probability = 0.3  # Probabilidade do modelo base
    result = ensemble.predict_combined(txn_dict, base_probability)

    print(f"   OK - Predição combinada:")
    print(f"      Base probability: {base_probability:.4f}")
    print(f"      CatBoost probability: {result.catboost_probability:.4f}")
    print(f"      Combined probability: {result.fraud_probability:.4f}")
    print(f"      is_fraud: {result.is_fraud}")
    print(f"      Contributions: {result.model_contributions}")

    # 5. Testar via production_fraud_engine
    print("\n5. Testando via ProductionFraudEngine...")
    from ml_engine.production_fraud_engine import get_fraud_engine

    fraud_engine = get_fraud_engine()
    if not fraud_engine.is_trained:
        print("   WARN -  FraudEngine não treinado, treinando...")
        fraud_engine.train_with_api_features()

    df = pd.DataFrame([{
        "amount": 10000.0,
        "hour": 2,
        "channel": "pix",
        "latitude": -25.5163,
        "longitude": -54.5854,
        "location_risk_score": 0.95,
        "merchant_category": "crypto_exchange",
        "velocity_score": 0.9,
        "is_new_device": 1,
    }])

    predictions = fraud_engine.predict_detailed(df)
    pred = predictions[0]

    print(f"   OK - Predição via FraudEngine:")
    print(f"      risk_score: {pred.risk_score:.4f}")
    print(f"      risk_level: {pred.risk_level}")
    print(f"      is_fraud: {pred.is_fraud}")

    print("\n" + "=" * 60)
    print("TODOS OS TESTES PASSARAM! OK -")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_catboost_integration()
    sys.exit(0 if success else 1)
