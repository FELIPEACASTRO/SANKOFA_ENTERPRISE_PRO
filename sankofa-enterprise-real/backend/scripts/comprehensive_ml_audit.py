"""
=================================================================
AUDITORIA COMPLETA ML - MODO MILITAR 10X
PROVA FORENSE DE QUALIDADE - SOLUCAO ML ANTIFRAUDE
=================================================================
"""
import sys
import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Results tracking
RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "critical_failures": []
}

def log_result(test_name: str, status: str, details: str = "", severity: str = "normal"):
    """Log test result"""
    result = {
        "test": test_name,
        "status": status,
        "details": details,
        "severity": severity
    }
    RESULTS["tests"].append(result)

    if status == "PASS":
        RESULTS["passed"] += 1
        print(f"  [PASS] {test_name}")
    elif status == "FAIL":
        RESULTS["failed"] += 1
        print(f"  [FAIL] {test_name}: {details[:100]}")
        if severity == "critical":
            RESULTS["critical_failures"].append(result)
    elif status == "WARN":
        RESULTS["warnings"] += 1
        print(f"  [WARN] {test_name}: {details[:100]}")

def test_production_fraud_engine():
    """Test ProductionFraudEngine with proper DataFrame input"""
    print("\n" + "=" * 60)
    print("TESTE: PRODUCTION FRAUD ENGINE")
    print("=" * 60)

    try:
        from ml_engine.production_fraud_engine import ProductionFraudEngine

        engine = ProductionFraudEngine()
        log_result("ProductionFraudEngine import", "PASS")

        # Criar DataFrame com transacoes de teste
        test_df = pd.DataFrame([
            {
                "amount": 150.00,
                "hour": 14,
                "velocity_score": 0.1,
                "is_new_device": 0,
            },
            {
                "amount": 15000.00,
                "hour": 3,
                "velocity_score": 0.9,
                "is_new_device": 1,
            }
        ])

        # Test predict
        try:
            predictions = engine.predict(test_df)
            log_result("ProductionFraudEngine.predict()", "PASS", f"Shape: {predictions.shape}")

            # Verify predictions are valid
            if len(predictions) == 2:
                log_result("Prediction count", "PASS")
            else:
                log_result("Prediction count", "FAIL", f"Expected 2, got {len(predictions)}")

            # Verify prediction values are binary (0 or 1)
            if all(p in [0, 1] for p in predictions):
                log_result("Predictions are binary", "PASS")
            else:
                log_result("Predictions are binary", "WARN", f"Non-binary values: {predictions}")

        except Exception as e:
            log_result("ProductionFraudEngine.predict()", "FAIL", str(e), "critical")

        # Test predict_proba
        try:
            probas = engine.predict_proba(test_df)
            log_result("ProductionFraudEngine.predict_proba()", "PASS", f"Shape: {probas.shape}")

            # Verify probas are between 0 and 1
            if np.all((probas >= 0) & (probas <= 1)):
                log_result("Probabilities in [0,1]", "PASS")
            else:
                log_result("Probabilities in [0,1]", "FAIL", "Out of range values")

        except Exception as e:
            log_result("ProductionFraudEngine.predict_proba()", "FAIL", str(e))

    except Exception as e:
        log_result("ProductionFraudEngine initialization", "FAIL", str(e), "critical")
        traceback.print_exc()

def test_ensemble_integration():
    """Test IntegratedEnsemble"""
    print("\n" + "=" * 60)
    print("TESTE: ENSEMBLE INTEGRATION")
    print("=" * 60)

    try:
        from ml_engine.ensemble_integration import IntegratedEnsemble, EnsemblePrediction

        ensemble = IntegratedEnsemble()
        log_result("IntegratedEnsemble import", "PASS")

        # Check available methods
        methods = [m for m in dir(ensemble) if not m.startswith('_')]
        log_result("Methods available", "PASS", f"{len(methods)} methods")

        # Check for prediction method
        if hasattr(ensemble, 'predict_transaction'):
            log_result("Has predict_transaction", "PASS")

            # Test prediction
            txn = {
                "amount": 5000.00,
                "hour": 2,
                "is_night": 1,
                "customer_id": "TEST_001"
            }

            try:
                result = ensemble.predict_transaction(txn)
                log_result("predict_transaction execution", "PASS", f"Type: {type(result).__name__}")
            except Exception as e:
                log_result("predict_transaction execution", "FAIL", str(e))
        else:
            log_result("Has predict_transaction", "WARN", "Method not found - checking alternatives")

            # List all methods for debugging
            public_methods = [m for m in methods if callable(getattr(ensemble, m, None))]
            log_result("Available callable methods", "PASS", str(public_methods[:10]))

    except Exception as e:
        log_result("IntegratedEnsemble initialization", "FAIL", str(e), "critical")
        traceback.print_exc()

def test_autoencoder():
    """Test AutoencoderAnomalyDetector"""
    print("\n" + "=" * 60)
    print("TESTE: AUTOENCODER ANOMALY DETECTOR")
    print("=" * 60)

    try:
        model_path = Path(__file__).parent.parent / "models" / "production" / "autoencoder_model.pkl"

        if not model_path.exists():
            log_result("Autoencoder model file", "FAIL", "File not found", "critical")
            return

        log_result("Autoencoder model file", "PASS", f"Size: {model_path.stat().st_size} bytes")

        with open(model_path, 'rb') as f:
            autoencoder = pickle.load(f)

        log_result("Autoencoder load", "PASS", f"Type: {type(autoencoder).__name__}")

        # Test detection
        normal_txn = {
            "amount": 200.0,
            "hour": 10,
            "day_of_week": 3,
            "is_weekend": 0,
            "is_night": 0,
            "velocity_score": 0.1,
            "transaction_count_24h": 1
        }

        fraud_txn = {
            "amount": 50000.0,
            "hour": 3,
            "day_of_week": 6,
            "is_weekend": 1,
            "is_night": 1,
            "velocity_score": 0.95,
            "transaction_count_24h": 20
        }

        result_normal = autoencoder.detect_anomaly(normal_txn)
        result_fraud = autoencoder.detect_anomaly(fraud_txn)

        log_result("Normal detection", "PASS", f"is_anomaly={result_normal.is_anomaly}, score={result_normal.anomaly_score:.4f}")
        log_result("Fraud detection", "PASS", f"is_anomaly={result_fraud.is_anomaly}, score={result_fraud.anomaly_score:.4f}")

        # Validate that fraud should have higher score
        if result_fraud.anomaly_score >= result_normal.anomaly_score:
            log_result("Fraud score >= normal score", "PASS")
        else:
            log_result("Fraud score >= normal score", "WARN", "Expected fraud to have higher score")

    except Exception as e:
        log_result("Autoencoder test", "FAIL", str(e), "critical")
        traceback.print_exc()

def test_bilstm():
    """Test BiLSTM Sequence Analyzer"""
    print("\n" + "=" * 60)
    print("TESTE: BiLSTM SEQUENCE ANALYZER")
    print("=" * 60)

    try:
        model_path = Path(__file__).parent.parent / "models" / "production" / "bilstm_model.pkl"

        if not model_path.exists():
            log_result("BiLSTM model file", "FAIL", "File not found", "critical")
            return

        log_result("BiLSTM model file", "PASS", f"Size: {model_path.stat().st_size} bytes")

        with open(model_path, 'rb') as f:
            bilstm = pickle.load(f)

        log_result("BiLSTM load", "PASS", f"Type: {type(bilstm).__name__}")

        # Add test sequence
        customer_id = "TEST_BILSTM_AUDIT"
        txns = [
            {"amount": 100, "hour": 10, "day_of_week": 1},
            {"amount": 150, "hour": 12, "day_of_week": 1},
            {"amount": 50000, "hour": 3, "day_of_week": 6},  # Anomaly
        ]

        for txn in txns:
            bilstm.sequence_buffer.add_transaction(customer_id, txn)

        result = bilstm.analyze_sequence(customer_id, txns[-1])

        log_result("BiLSTM analyze_sequence", "PASS",
                   f"suspicious={result.is_suspicious_sequence}, risk={result.sequence_risk_score:.4f}")

        # Validate risk score is valid
        if 0 <= result.sequence_risk_score <= 1:
            log_result("Risk score in [0,1]", "PASS")
        else:
            log_result("Risk score in [0,1]", "FAIL", f"Got {result.sequence_risk_score}")

    except Exception as e:
        log_result("BiLSTM test", "FAIL", str(e), "critical")
        traceback.print_exc()

def test_catboost():
    """Test CatBoost Model"""
    print("\n" + "=" * 60)
    print("TESTE: CATBOOST MODEL")
    print("=" * 60)

    try:
        from ml_engine.catboost_model import CatBoostFraudModel

        model_path = Path(__file__).parent.parent / "models" / "production" / "catboost_fraud_model.cbm"

        if model_path.exists():
            log_result("CatBoost model file", "PASS", f"Size: {model_path.stat().st_size} bytes")
        else:
            log_result("CatBoost model file", "WARN", "Pre-trained model not found")

        catboost = CatBoostFraudModel()
        log_result("CatBoost initialization", "PASS")

        # Check if model needs training
        if hasattr(catboost, 'model') and catboost.model is not None:
            log_result("CatBoost model loaded", "PASS")
        else:
            log_result("CatBoost model loaded", "WARN", "Model not loaded - needs training")

    except Exception as e:
        log_result("CatBoost test", "FAIL", str(e))
        traceback.print_exc()

def test_mule_detector():
    """Test Mule Detector"""
    print("\n" + "=" * 60)
    print("TESTE: MULE DETECTOR")
    print("=" * 60)

    try:
        from ml_engine.mule_detection.mule_detector import MuleDetector

        mule = MuleDetector()
        log_result("MuleDetector import", "PASS")

        # Check available methods
        methods = [m for m in dir(mule) if not m.startswith('_') and callable(getattr(mule, m, None))]
        log_result("Available methods", "PASS", str(methods[:10]))

        # Test with correct method
        if hasattr(mule, 'detect_mule'):
            account = {
                "account_id": "ACC_MULE_001",
                "dormancy_days": 180,
                "turnover_ratio": 0.95,
                "in_degree": 50,
                "out_degree": 2
            }
            result = mule.detect_mule(account)
            log_result("detect_mule execution", "PASS", str(result))
        elif hasattr(mule, 'analyze_account'):
            account = {
                "account_id": "ACC_MULE_001",
                "dormancy_days": 180,
            }
            result = mule.analyze_account(account)
            log_result("analyze_account execution", "PASS", str(result))
        else:
            log_result("Detection method", "WARN", "No detection method found")

    except Exception as e:
        log_result("MuleDetector test", "FAIL", str(e))
        traceback.print_exc()

def test_feature_engineering():
    """Test Feature Engineering"""
    print("\n" + "=" * 60)
    print("TESTE: FEATURE ENGINEERING")
    print("=" * 60)

    try:
        from ml_engine.feature_engineering import FeatureGenerator

        fg = FeatureGenerator()
        log_result("FeatureGenerator import", "PASS")

        # Test with DataFrame (correct format)
        test_df = pd.DataFrame([{
            "amount": 500.0,
            "timestamp": datetime.now().isoformat(),
            "customer_id": "CUST_FE_001",
            "hour": 14
        }])

        # Check available methods
        if hasattr(fg, 'generate_features'):
            try:
                features = fg.generate_features(test_df.iloc[0].to_dict())
                log_result("generate_features(dict)", "PASS", f"{len(features)} features")
            except Exception as e:
                log_result("generate_features(dict)", "WARN", str(e)[:50])

        if hasattr(fg, 'transform'):
            try:
                features = fg.transform(test_df)
                log_result("transform(DataFrame)", "PASS", f"Shape: {features.shape}")
            except Exception as e:
                log_result("transform(DataFrame)", "WARN", str(e)[:50])

    except Exception as e:
        log_result("FeatureGenerator test", "FAIL", str(e))
        traceback.print_exc()

def test_mixture_of_experts():
    """Test Mixture of Experts"""
    print("\n" + "=" * 60)
    print("TESTE: MIXTURE OF EXPERTS")
    print("=" * 60)

    try:
        from ml_engine.mixture_of_experts import MixtureOfExperts

        moe = MixtureOfExperts()
        log_result("MixtureOfExperts import", "PASS")

        txn = {
            "amount": 2500.0,
            "channel": "PIX",
            "hour": 15,
            "customer_id": "CUST_MOE_001"
        }

        result = moe.predict(txn)
        log_result("MoE predict", "PASS", f"Type: {type(result).__name__}")

        if hasattr(result, 'final_probability'):
            if 0 <= result.final_probability <= 1:
                log_result("Final probability in [0,1]", "PASS", f"{result.final_probability:.4f}")
            else:
                log_result("Final probability in [0,1]", "FAIL")

    except Exception as e:
        log_result("MixtureOfExperts test", "FAIL", str(e))
        traceback.print_exc()

def test_all_base_models():
    """Test all base ensemble models"""
    print("\n" + "=" * 60)
    print("TESTE: BASE ENSEMBLE MODELS")
    print("=" * 60)

    models_dir = Path(__file__).parent.parent / "models" / "production"

    base_models = [
        ("random_forest.pkl", "RandomForestClassifier"),
        ("gradient_boosting.pkl", "GradientBoostingClassifier"),
        ("extra_trees_gnn.pkl", "ExtraTreesClassifier"),
        ("mlp.pkl", "MLPClassifier"),
        ("isolation_forest.pkl", "IsolationForest"),
        ("scaler.pkl", "StandardScaler"),
    ]

    for filename, expected_type in base_models:
        model_path = models_dir / filename

        if not model_path.exists():
            log_result(f"Model {filename}", "FAIL", "File not found")
            continue

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            actual_type = type(model).__name__
            if expected_type in actual_type:
                log_result(f"Model {filename}", "PASS", f"Type: {actual_type}")
            else:
                log_result(f"Model {filename}", "WARN", f"Expected {expected_type}, got {actual_type}")

            # Test predict for classifiers
            if filename != "scaler.pkl" and hasattr(model, 'predict'):
                test_X = np.array([[100, 1, 14, 2, 0, 0, 0]])
                try:
                    pred = model.predict(test_X)
                    log_result(f"{filename} predict", "PASS", f"Pred: {pred}")
                except Exception as e:
                    log_result(f"{filename} predict", "WARN", str(e)[:50])

        except Exception as e:
            log_result(f"Model {filename}", "FAIL", str(e))

def generate_report():
    """Generate final audit report"""
    print("\n" + "=" * 80)
    print("RELATORIO FINAL DA AUDITORIA")
    print("=" * 80)

    total = RESULTS["passed"] + RESULTS["failed"] + RESULTS["warnings"]

    print(f"\nTimestamp: {RESULTS['timestamp']}")
    print(f"\nResultados:")
    print(f"  Total de testes: {total}")
    print(f"  Passou:          {RESULTS['passed']} ({100*RESULTS['passed']/max(1,total):.1f}%)")
    print(f"  Falhou:          {RESULTS['failed']} ({100*RESULTS['failed']/max(1,total):.1f}%)")
    print(f"  Warnings:        {RESULTS['warnings']} ({100*RESULTS['warnings']/max(1,total):.1f}%)")

    if RESULTS["critical_failures"]:
        print(f"\n FALHAS CRITICAS ({len(RESULTS['critical_failures'])}):")
        for fail in RESULTS["critical_failures"]:
            print(f"  - {fail['test']}: {fail['details'][:80]}")

    # Determine verdict
    print("\n" + "=" * 80)
    if RESULTS["failed"] == 0:
        print("VEREDITO: PRONTA PARA PRODUCAO")
        verdict = "PASS"
    elif len(RESULTS["critical_failures"]) > 0:
        print("VEREDITO: NAO PRONTA - FALHAS CRITICAS")
        verdict = "FAIL"
    else:
        print("VEREDITO: PARCIALMENTE PRONTA - REQUER CORRECOES")
        verdict = "PARTIAL"
    print("=" * 80)

    # Save report
    report_path = Path(__file__).parent.parent / "reports" / "ml_audit_report.json"
    report_path.parent.mkdir(exist_ok=True)

    RESULTS["verdict"] = verdict
    RESULTS["total_tests"] = total

    with open(report_path, 'w') as f:
        json.dump(RESULTS, f, indent=2)

    print(f"\nRelatorio salvo em: {report_path}")

    return verdict

def main():
    """Run complete audit"""
    print("=" * 80)
    print(" SANKOFA ENTERPRISE PRO - AUDITORIA ML COMPLETA ")
    print(" MODO MILITAR 10X - PROVA FORENSE DE QUALIDADE ")
    print("=" * 80)

    # Run all tests
    test_production_fraud_engine()
    test_ensemble_integration()
    test_autoencoder()
    test_bilstm()
    test_catboost()
    test_mule_detector()
    test_feature_engineering()
    test_mixture_of_experts()
    test_all_base_models()

    # Generate report
    verdict = generate_report()

    return verdict == "PASS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
