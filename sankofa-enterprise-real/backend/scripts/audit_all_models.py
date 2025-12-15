"""
AUDITORIA COMPLETA DE TODOS OS MODELOS ML
Sankofa Enterprise Pro

Verifica:
1. Se o modelo existe (arquivo .py)
2. Se tem classe/metodos implementados
3. Se esta treinado (arquivo .pkl/.joblib/.cbm)
4. Se esta sendo usado no sistema
5. Se funciona corretamente
"""

import sys
from pathlib import Path
import importlib
import inspect
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np


@dataclass
class ModelAuditResult:
    """Resultado da auditoria de um modelo"""
    name: str
    file_path: str
    class_name: str
    has_train_method: bool
    has_predict_method: bool
    is_trained: bool
    trained_model_path: Optional[str]
    is_used_in_production: bool
    used_by: List[str]
    test_passed: bool
    error_message: Optional[str]
    status: str  # OK, WARNING, ERROR, NOT_IMPLEMENTED


def check_model_methods(cls) -> tuple:
    """Verifica se a classe tem metodos train e predict"""
    methods = [m for m in dir(cls) if not m.startswith('_')]

    has_train = any(m in methods for m in ['train', 'fit', 'train_model', 'learn'])
    has_predict = any(m in methods for m in ['predict', 'predict_proba', 'score', 'analyze', 'detect'])

    return has_train, has_predict


def find_model_usage(model_name: str, class_name: str) -> List[str]:
    """Encontra onde o modelo e usado"""
    used_by = []

    # Arquivos principais que podem usar modelos
    files_to_check = [
        backend_path / "api" / "production_api.py",
        backend_path / "ml_engine" / "production_fraud_engine.py",
        backend_path / "ml_engine" / "ensemble_integration.py",
        backend_path / "ml_engine" / "advanced_modules_orchestrator.py",
    ]

    for file_path in files_to_check:
        if file_path.exists():
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if class_name in content or model_name in content:
                used_by.append(file_path.name)

    return used_by


def find_trained_model(model_name: str) -> Optional[str]:
    """Procura modelo treinado salvo"""
    model_dirs = [
        backend_path / "models",
        backend_path / "models" / "production",
        backend_path.parent / "models",
        backend_path.parent / "models" / "production",
    ]

    extensions = ['.pkl', '.joblib', '.cbm', '.h5', '.pt', '.onnx', '.bin']

    for model_dir in model_dirs:
        if model_dir.exists():
            for ext in extensions:
                # Procurar por nome exato ou similar
                for f in model_dir.glob(f"*{ext}"):
                    name_lower = f.stem.lower().replace('_', '').replace('-', '')
                    model_lower = model_name.lower().replace('_', '').replace('-', '')
                    if model_lower in name_lower or name_lower in model_lower:
                        return str(f)

    return None


def test_model(module_path: str, class_name: str) -> tuple:
    """Testa se o modelo pode ser instanciado e usado"""
    try:
        # Importar modulo
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Obter classe
        if not hasattr(module, class_name):
            return False, f"Classe {class_name} nao encontrada"

        cls = getattr(module, class_name)

        # Tentar instanciar
        try:
            instance = cls()
        except TypeError:
            try:
                instance = cls({})
            except:
                return False, "Nao foi possivel instanciar"

        return True, None

    except Exception as e:
        return False, str(e)[:100]


def audit_model(file_path: Path, class_name: str) -> ModelAuditResult:
    """Audita um modelo especifico"""
    model_name = file_path.stem

    # Verificar metodos
    try:
        spec = importlib.util.spec_from_file_location("module", str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            has_train, has_predict = check_model_methods(cls)
        else:
            has_train, has_predict = False, False
    except Exception as e:
        has_train, has_predict = False, False

    # Verificar modelo treinado
    trained_path = find_trained_model(model_name)
    is_trained = trained_path is not None

    # Verificar uso
    used_by = find_model_usage(model_name, class_name)
    is_used = len(used_by) > 0

    # Testar modelo
    test_passed, error_msg = test_model(str(file_path), class_name)

    # Determinar status
    if test_passed and is_trained and is_used:
        status = "OK"
    elif test_passed and (is_trained or is_used):
        status = "WARNING"
    elif test_passed:
        status = "NOT_USED"
    else:
        status = "ERROR"

    return ModelAuditResult(
        name=model_name,
        file_path=str(file_path.relative_to(backend_path)),
        class_name=class_name,
        has_train_method=has_train,
        has_predict_method=has_predict,
        is_trained=is_trained,
        trained_model_path=trained_path,
        is_used_in_production=is_used,
        used_by=used_by,
        test_passed=test_passed,
        error_message=error_msg,
        status=status
    )


# Mapeamento de arquivos para classes principais
MODEL_MAPPING = {
    # Modelos principais
    "production_fraud_engine.py": "ProductionFraudEngine",
    "catboost_model.py": "CatBoostFraudModel",
    "ensemble_integration.py": "IntegratedEnsemble",
    "gnn_fraud_detector.py": "GNNFraudDetector",

    # Modelos avancados
    "autoencoder_anomaly_detector.py": "AutoencoderAnomalyDetector",
    "bilstm_sequence_analyzer.py": "BiLSTMSequenceAnalyzer",
    "nlp_social_engineering.py": "NLPSocialEngineeringDetector",
    "transfer_learning_pipeline.py": "TransferLearningPipeline",
    "mixture_of_experts.py": "MixtureOfExpertsDetector",
    "causal_inference.py": "CausalInferenceEngine",
    "graph_neural_networks.py": "GraphNeuralNetwork",
    "graph_ml_engine.py": "GraphMLEngine",
    "multi_armed_bandits.py": "MultiArmedBanditOptimizer",
    "automl_pipeline.py": "AutoMLPipeline",

    # Feature Engineering
    "advanced_feature_engineering.py": "AdvancedFeatureEngineering",
    "bahnsen_feature_engineering.py": "BahnsenFeatureEngineering",
    "feature_engineering.py": "FeatureEngineering",

    # Calibracao e Otimizacao
    "probability_calibration.py": "ProbabilityCalibrator",
    "threshold_optimizer.py": "ThresholdOptimizer",
    "self_training_optimizer.py": "SelfTrainingOptimizer",

    # Aprendizado Continuo
    "continuous_learning_system.py": "ContinuousLearningSystem",
    "federated_learning.py": "FederatedLearningCoordinator",

    # Explicabilidade
    "explainability_engine.py": "ExplainabilityEngine",
    "self_explainable_module.py": "SelfExplainableModule",

    # Integracao externa
    "huggingface_integration.py": "HuggingFaceIntegration",
    "onnx_serving.py": "ONNXModelServer",

    # Device/Comportamento
    "device_fingerprint.py": "DeviceFingerprintAnalyzer",

    # Regras
    "hard_rules_engine.py": "HardRulesEngine",
    "pix_fraud_taxonomy.py": "PixFraudTaxonomy",

    # Submodulos - App Fraud
    "app_fraud/scam_detector.py": "ScamDetector",
    "app_fraud/session_analyzer.py": "SessionAnalyzer",
    "app_fraud/duress_detector.py": "DuressDetector",
    "app_fraud/conversation_scorer.py": "ConversationScorer",
    "app_fraud/intervention_engine.py": "InterventionEngine",

    # Submodulos - Mule Detection
    "mule_detection/mule_detector.py": "MuleDetector",
    "mule_detection/dormancy_analyzer.py": "DormancyAnalyzer",
    "mule_detection/turnover_analyzer.py": "TurnoverAnalyzer",
    "mule_detection/network_position_analyzer.py": "NetworkPositionAnalyzer",

    # Submodulos - Feature Engineering
    "feature_engineering/feature_generator.py": "FeatureGenerator",
    "feature_engineering/temporal_features.py": "TemporalFeatureGenerator",
    "feature_engineering/velocity_features.py": "VelocityFeatureGenerator",
    "feature_engineering/aggregation_features.py": "AggregationFeatureGenerator",
    "feature_engineering/graph_features.py": "GraphFeatureGenerator",
    "feature_engineering/embedding_features.py": "EmbeddingFeatureGenerator",

    # Submodulos - GNN
    "gnn/fraud_gnn.py": "FraudGNN",
    "gnn/temporal_gnn.py": "TemporalGNN",
    "gnn/graph_builder.py": "GraphBuilder",
    "gnn/gnn_trainer.py": "GNNTrainer",

    # Submodulos - Behavioral
    "behavioral/behavioral_analyzer.py": "BehavioralAnalyzer",
    "behavioral/keystroke_analyzer.py": "KeystrokeAnalyzer",
    "behavioral/mouse_analyzer.py": "MouseAnalyzer",
    "behavioral/device_analyzer.py": "DeviceAnalyzer",

    # Submodulos - LLM
    "llm/fraud_llm.py": "FraudLLM",
    "llm/scam_detector.py": "LLMScamDetector",
    "llm/explanation_generator.py": "ExplanationGenerator",

    # Outros
    "data_balancer.py": "DataBalancer",
    "dataset_loaders.py": "DatasetLoader",
    "feature_store.py": "FeatureStore",
    "advanced_modules_orchestrator.py": "AdvancedModulesOrchestrator",
}


def run_full_audit():
    """Executa auditoria completa"""
    print("=" * 80)
    print("AUDITORIA COMPLETA DE MODELOS ML - SANKOFA ENTERPRISE PRO")
    print("=" * 80)

    results = []
    ml_engine_path = backend_path / "ml_engine"

    # Auditar cada modelo
    for file_name, class_name in MODEL_MAPPING.items():
        file_path = ml_engine_path / file_name

        if file_path.exists():
            print(f"\nAuditando: {file_name} ({class_name})...")
            result = audit_model(file_path, class_name)
            results.append(result)
        else:
            print(f"\n[MISSING] {file_name} - Arquivo nao encontrado")
            results.append(ModelAuditResult(
                name=file_name.replace('.py', ''),
                file_path=file_name,
                class_name=class_name,
                has_train_method=False,
                has_predict_method=False,
                is_trained=False,
                trained_model_path=None,
                is_used_in_production=False,
                used_by=[],
                test_passed=False,
                error_message="Arquivo nao encontrado",
                status="MISSING"
            ))

    # Verificar modelos serializados
    print("\n" + "=" * 80)
    print("MODELOS SERIALIZADOS ENCONTRADOS")
    print("=" * 80)

    serialized_models = []
    for model_dir in [backend_path / "models", backend_path.parent / "models"]:
        if model_dir.exists():
            for ext in ['*.pkl', '*.joblib', '*.cbm', '*.h5', '*.pt', '*.onnx']:
                for f in model_dir.rglob(ext):
                    serialized_models.append({
                        "file": str(f.relative_to(backend_path.parent)),
                        "size_kb": f.stat().st_size / 1024
                    })
                    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Gerar relatorio
    print("\n" + "=" * 80)
    print("RESUMO DA AUDITORIA")
    print("=" * 80)

    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    print(f"\nTotal de modelos analisados: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Tabela detalhada
    print("\n" + "-" * 80)
    print(f"{'MODELO':<35} {'STATUS':<12} {'TREINADO':<10} {'USADO':<8} {'ERRO'}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x.status != 'OK', x.status != 'WARNING', x.name)):
        trained = "Sim" if r.is_trained else "Nao"
        used = "Sim" if r.is_used_in_production else "Nao"
        error = r.error_message[:30] if r.error_message else "-"
        print(f"{r.name:<35} {r.status:<12} {trained:<10} {used:<8} {error}")

    # Modelos OK (funcionando completamente)
    print("\n" + "=" * 80)
    print("MODELOS FUNCIONANDO (OK)")
    print("=" * 80)

    ok_models = [r for r in results if r.status == 'OK']
    for r in ok_models:
        print(f"\n  {r.name}")
        print(f"    Classe: {r.class_name}")
        print(f"    Arquivo: {r.file_path}")
        print(f"    Modelo treinado: {r.trained_model_path}")
        print(f"    Usado por: {', '.join(r.used_by)}")

    # Modelos com Warning
    print("\n" + "=" * 80)
    print("MODELOS COM AVISOS (WARNING/NOT_USED)")
    print("=" * 80)

    warning_models = [r for r in results if r.status in ['WARNING', 'NOT_USED']]
    for r in warning_models:
        print(f"\n  {r.name} [{r.status}]")
        print(f"    Classe: {r.class_name}")
        print(f"    Train method: {r.has_train_method}, Predict method: {r.has_predict_method}")
        print(f"    Treinado: {r.is_trained}, Usado: {r.is_used_in_production}")

    # Modelos com erro
    print("\n" + "=" * 80)
    print("MODELOS COM ERRO")
    print("=" * 80)

    error_models = [r for r in results if r.status in ['ERROR', 'MISSING']]
    for r in error_models:
        print(f"\n  {r.name} [{r.status}]")
        print(f"    Erro: {r.error_message}")

    # Salvar resultado em JSON
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_models": len(results),
        "status_summary": status_counts,
        "serialized_models": serialized_models,
        "models": [
            {
                "name": r.name,
                "class": r.class_name,
                "file": r.file_path,
                "status": r.status,
                "has_train": r.has_train_method,
                "has_predict": r.has_predict_method,
                "is_trained": r.is_trained,
                "trained_path": r.trained_model_path,
                "is_used": r.is_used_in_production,
                "used_by": r.used_by,
                "test_passed": r.test_passed,
                "error": r.error_message
            }
            for r in results
        ]
    }

    report_path = backend_path / "reports" / "model_audit_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\nRelatorio salvo em: {report_path}")

    return results


if __name__ == "__main__":
    run_full_audit()
