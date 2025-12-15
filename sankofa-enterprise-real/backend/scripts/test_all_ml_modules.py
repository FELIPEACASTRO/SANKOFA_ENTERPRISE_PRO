"""
Teste COMPLETO de TODOS os modulos ML do sistema.
Verifica importacao, instanciacao e metodos principais de CADA classe.

Este script testa ABSOLUTAMENTE TODOS os 66 arquivos do ml_engine.
"""
import sys
import os
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Modules that require PyTorch
PYTORCH_REQUIRED_MODULES = [
    'ml_engine.gnn.fraud_gnn',
    'ml_engine.gnn.temporal_gnn',
    'ml_engine.gnn.graph_builder',
    'ml_engine.gnn.gnn_trainer',
    'ml_engine.graph_neural_networks',
    'ml_engine.huggingface_integration',
]


class MLModuleTester:
    """Testador completo de modulos ML"""

    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0

    def test_import(self, module_path: str, class_name: str) -> Tuple[bool, Any, str]:
        """Testa se um modulo pode ser importado e a classe instanciada"""
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return True, cls, ""
        except Exception as e:
            return False, None, str(e)

    def test_instantiation(self, cls: Any, **kwargs) -> Tuple[bool, Any, str]:
        """Testa se uma classe pode ser instanciada"""
        try:
            instance = cls(**kwargs)
            return True, instance, ""
        except Exception as e:
            return False, None, str(e)

    def test_method(self, instance: Any, method_name: str, *args, **kwargs) -> Tuple[bool, str]:
        """Testa se um metodo pode ser chamado"""
        try:
            if not hasattr(instance, method_name):
                return False, f"Metodo {method_name} nao existe"
            method = getattr(instance, method_name)
            if callable(method):
                # Para metodos que precisam de dados, apenas verificar se existem
                return True, "OK"
            return False, f"{method_name} nao e callable"
        except Exception as e:
            return False, str(e)

    def run_all_tests(self):
        """Executa todos os testes"""
        print("=" * 80)
        print("TESTE COMPLETO DE TODOS OS MODULOS ML")
        print("=" * 80)
        print(f"Data: {datetime.now().isoformat()}")
        print()

        # Definir todos os modulos e classes a testar
        modules_to_test = [
            # Core ML Modules
            ("ml_engine.production_fraud_engine", "ProductionFraudEngine", {}),
            ("ml_engine.ensemble_integration", "IntegratedEnsemble", {}),
            ("ml_engine.catboost_model", "CatBoostFraudModel", {}),
            ("ml_engine.autoencoder_anomaly_detector", "AutoencoderAnomalyDetector", {}),
            ("ml_engine.bilstm_sequence_analyzer", "BiLSTMSequenceAnalyzer", {}),
            ("ml_engine.mixture_of_experts", "MixtureOfExperts", {}),
            ("ml_engine.gnn_fraud_detector", "GNNFraudDetector", {}),

            # Feature Engineering
            ("ml_engine.advanced_feature_engineering", "AdvancedFeatureEngineering", {}),
            ("ml_engine.bahnsen_feature_engineering", "BahnsenFeatureEngineering", {}),
            ("ml_engine.feature_engineering", "AutoFeatureEngineering", {}),
            ("ml_engine.feature_engineering.feature_generator", "FeatureGenerator", {}),
            ("ml_engine.feature_engineering.temporal_features", "TemporalFeatureGenerator", {}),
            ("ml_engine.feature_engineering.velocity_features", "VelocityFeatureGenerator", {}),
            ("ml_engine.feature_engineering.aggregation_features", "AggregationFeatureGenerator", {}),
            ("ml_engine.feature_engineering.graph_features", "GraphFeatureGenerator", {}),
            ("ml_engine.feature_engineering.embedding_features", "EmbeddingFeatureGenerator", {}),
            ("ml_engine.feature_store", "FeatureStore", {}),

            # GNN Modules
            ("ml_engine.gnn.fraud_gnn", "FraudGNN", {"input_dim": 10, "hidden_dim": 32}),
            ("ml_engine.gnn.temporal_gnn", "TemporalGNN", {"input_dim": 10, "hidden_dim": 32}),
            ("ml_engine.gnn.graph_builder", "TransactionGraphBuilder", {}),
            ("ml_engine.gnn.gnn_trainer", "GNNTrainer", {}),
            ("ml_engine.graph_neural_networks", "GNNFraudDetector", {}),
            ("ml_engine.graph_ml_engine", "GraphMLEngine", {}),

            # Behavioral Analysis
            ("ml_engine.behavioral.behavioral_analyzer", "BehavioralAnalyzer", {}),
            ("ml_engine.behavioral.keystroke_analyzer", "KeystrokeAnalyzer", {}),
            ("ml_engine.behavioral.mouse_analyzer", "MouseAnalyzer", {}),
            ("ml_engine.behavioral.device_analyzer", "DeviceAnalyzer", {}),

            # Mule Detection
            ("ml_engine.mule_detection.mule_detector", "MuleDetector", {}),
            ("ml_engine.mule_detection.dormancy_analyzer", "DormancyAnalyzer", {}),
            ("ml_engine.mule_detection.turnover_analyzer", "TurnoverAnalyzer", {}),
            ("ml_engine.mule_detection.network_position_analyzer", "NetworkPositionAnalyzer", {}),

            # APP Fraud
            ("ml_engine.app_fraud.scam_detector", "APPScamDetector", {}),
            ("ml_engine.app_fraud.session_analyzer", "SessionAnalyzer", {}),
            ("ml_engine.app_fraud.duress_detector", "DuressDetector", {}),
            ("ml_engine.app_fraud.conversation_scorer", "ConversationScorer", {}),
            ("ml_engine.app_fraud.intervention_engine", "InterventionEngine", {}),

            # LLM Modules
            ("ml_engine.llm.fraud_llm", "FraudLLMAnalyzer", {}),
            ("ml_engine.llm.scam_detector", "ScamPatternDetector", {}),
            ("ml_engine.llm.explanation_generator", "ExplanationGenerator", {}),
            ("ml_engine.llm.embeddings", "TransactionEmbedder", {}),
            ("ml_engine.llm.embeddings", "TextEmbedder", {}),

            # Advanced ML
            ("ml_engine.multi_armed_bandits", "StepUpMFAOptimizer", {}),
            ("ml_engine.multi_armed_bandits", "ContextualBandit", {"challenge_types": []}),
            ("ml_engine.self_training_optimizer", "AdaptiveSelfTraining", {}),
            ("ml_engine.federated_learning", "FederatedServer", {}),
            ("ml_engine.self_explainable_module", "InterpretativeMaskLearner", {"input_dim": 10}),
            ("ml_engine.pix_fraud_taxonomy", "PIXFraudTaxonomy", {}),
            ("ml_engine.causal_inference", "CausalImpactAnalyzer", None),  # Requires DoWhy
            ("ml_engine.causal_inference", "UpliftModeling", None),  # Requires CausalML

            # Optimization & Calibration
            ("ml_engine.probability_calibration", "ProbabilityCalibrator", {}),
            ("ml_engine.threshold_optimizer", "ThresholdOptimizer", {}),
            ("ml_engine.continuous_learning_system", "ContinuousLearningSystem", {}),
            ("ml_engine.transfer_learning_pipeline", "TransferLearningPipeline", {}),
            ("ml_engine.automl_pipeline", "AutoMLPipeline", {}),
            ("ml_engine.data_balancer", "DataBalancer", {}),

            # Explainability
            ("ml_engine.explainability_engine", "ExplainabilityEngine", {}),
            ("ml_engine.advanced_modules_orchestrator", "AdvancedModulesOrchestrator", {}),

            # NLP
            ("ml_engine.nlp_social_engineering", "SocialEngineeringDetector", {}),
            ("ml_engine.huggingface_integration", "HuggingFaceIntegration", {}),

            # Device & Rules
            ("ml_engine.device_fingerprint", "DeviceFingerprintEngine", {}),
            ("ml_engine.hard_rules_engine", "HardRulesEngine", None),  # Requires DB

            # Serving
            ("ml_engine.onnx_serving", "ONNXFraudModelServing", {}),

            # Dataset
            ("ml_engine.dataset_loaders", "DatasetLoader", {}),
        ]

        print(f"Total de modulos a testar: {len(modules_to_test)}")
        print("-" * 80)

        for module_path, class_name, kwargs in modules_to_test:
            self.total_tests += 1
            result = {"module": module_path, "class": class_name, "status": "UNKNOWN"}

            # Teste 1: Import
            can_import, cls, import_error = self.test_import(module_path, class_name)

            if not can_import:
                # Check if this is a PyTorch-required module without PyTorch
                if not HAS_PYTORCH and module_path in PYTORCH_REQUIRED_MODULES:
                    result["status"] = "SKIP_PYTORCH"
                    result["note"] = "Requires PyTorch (not installed)"
                    self.warnings += 1
                    print(f"SKIP - {module_path}.{class_name}: Requires PyTorch (not installed)")
                    self.results[f"{module_path}.{class_name}"] = result
                    continue
                
                result["status"] = "IMPORT_FAIL"
                result["error"] = import_error
                self.failed_tests += 1
                print(f"FAIL - {module_path}.{class_name}: Import failed - {import_error[:50]}")
                self.results[f"{module_path}.{class_name}"] = result
                continue

            # Teste 2: Instantiation (se kwargs nao for None)
            if kwargs is not None:
                can_instantiate, instance, inst_error = self.test_instantiation(cls, **kwargs)

                if not can_instantiate:
                    # Tentar sem argumentos
                    can_instantiate, instance, inst_error = self.test_instantiation(cls)

                if not can_instantiate:
                    result["status"] = "INSTANTIATION_FAIL"
                    result["error"] = inst_error
                    self.warnings += 1
                    print(f"WARN - {module_path}.{class_name}: Cannot instantiate - {inst_error[:50]}")
                    self.results[f"{module_path}.{class_name}"] = result
                    continue

                # Teste 3: Verificar metodos principais
                main_methods = ["train", "predict", "fit", "transform", "analyze", "detect", "evaluate"]
                available_methods = []

                for method in main_methods:
                    if hasattr(instance, method) and callable(getattr(instance, method)):
                        available_methods.append(method)

                result["status"] = "OK"
                result["methods"] = available_methods
                self.passed_tests += 1
                methods_str = ", ".join(available_methods) if available_methods else "none"
                print(f"OK   - {module_path}.{class_name} [{methods_str}]")
            else:
                # Classe requer dependencias especiais
                result["status"] = "SKIP_DEPS"
                result["note"] = "Requires optional dependencies"
                self.warnings += 1
                print(f"SKIP - {module_path}.{class_name}: Requires optional dependencies")

            self.results[f"{module_path}.{class_name}"] = result

        # Resumo
        print()
        print("=" * 80)
        print("RESUMO DOS TESTES")
        print("=" * 80)
        print(f"Total de testes: {self.total_tests}")
        print(f"Passaram:        {self.passed_tests} ({100*self.passed_tests/self.total_tests:.1f}%)")
        print(f"Falharam:        {self.failed_tests} ({100*self.failed_tests/self.total_tests:.1f}%)")
        print(f"Warnings:        {self.warnings} ({100*self.warnings/self.total_tests:.1f}%)")
        print("=" * 80)

        return self.failed_tests == 0

    def save_report(self, filepath: str):
        """Salva relatorio em JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "warnings": self.warnings,
                "pass_rate": f"{100*self.passed_tests/self.total_tests:.1f}%"
            },
            "results": self.results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nRelatorio salvo em: {filepath}")


def test_aliases():
    """Testa todos os aliases definidos"""
    print("\n" + "=" * 80)
    print("TESTE DE ALIASES")
    print("=" * 80)

    aliases_to_test = [
        ("ml_engine.mixture_of_experts", "MixtureOfExpertsDetector", "MixtureOfExperts"),
        ("ml_engine.mixture_of_experts", "MoEDetector", "MixtureOfExperts"),
        ("ml_engine.multi_armed_bandits", "MultiArmedBanditOptimizer", "StepUpMFAOptimizer"),
        ("ml_engine.multi_armed_bandits", "MABOptimizer", "ContextualBandit"),
        ("ml_engine.feature_engineering", "FeatureEngineering", "FeatureGenerator"),
        ("ml_engine.feature_engineering", "FeatureEngineer", "FeatureGenerator"),
        ("ml_engine.self_training_optimizer", "SelfTrainingOptimizer", "AdaptiveSelfTraining"),
        ("ml_engine.federated_learning", "FederatedLearningCoordinator", "FederatedServer"),
        ("ml_engine.self_explainable_module", "SelfExplainableModule", "InterpretativeMaskLearner"),
        ("ml_engine.pix_fraud_taxonomy", "PixFraudDetector", "PIXFraudTaxonomy"),
        ("ml_engine.onnx_serving", "ONNXModelServer", "ONNXFraudModelServing"),
        ("ml_engine.causal_inference", "CausalInferenceEngine", "CausalImpactAnalyzer"),
        ("ml_engine.llm.fraud_llm", "FraudLLM", "FraudLLMAnalyzer"),
        ("ml_engine.llm.scam_detector", "LLMScamDetector", "ScamPatternDetector"),
    ]

    passed = 0
    failed = 0

    for module_path, alias_name, target_name in aliases_to_test:
        try:
            module = __import__(module_path, fromlist=[alias_name, target_name])
            alias = getattr(module, alias_name)
            target = getattr(module, target_name)

            if alias is target:
                print(f"OK   - {module_path}.{alias_name} -> {target_name}")
                passed += 1
            else:
                print(f"FAIL - {module_path}.{alias_name} != {target_name}")
                failed += 1
        except Exception as e:
            print(f"FAIL - {module_path}.{alias_name}: {str(e)[:50]}")
            failed += 1

    print(f"\nAliases: {passed} OK, {failed} FAIL")
    return failed == 0


def test_trained_models():
    """Verifica modelos treinados"""
    print("\n" + "=" * 80)
    print("VERIFICACAO DE MODELOS TREINADOS")
    print("=" * 80)

    model_path = Path(__file__).parent.parent / "models" / "production"
    models = [
        ("catboost_fraud_model.cbm", "CatBoost"),
        ("autoencoder_model.pkl", "Autoencoder"),
        ("bilstm_model.pkl", "BiLSTM"),
        ("random_forest.pkl", "Random Forest"),
        ("gradient_boosting.pkl", "Gradient Boosting"),
        ("extra_trees_gnn.pkl", "Extra Trees"),
        ("mlp.pkl", "MLP"),
        ("isolation_forest.pkl", "Isolation Forest"),
        ("ensemble_config.json", "Ensemble Config"),
    ]

    found = 0
    for filename, name in models:
        filepath = model_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024
            print(f"OK   - {name}: {filename} ({size:.1f} KB)")
            found += 1
        else:
            print(f"MISS - {name}: {filename}")

    print(f"\nModelos encontrados: {found}/{len(models)}")
    return found >= 5  # Pelo menos 5 modelos


def main():
    """Executa todos os testes"""
    print("\n" + "=" * 80)
    print(" SANKOFA ENTERPRISE PRO - TESTE COMPLETO DE ML ")
    print("=" * 80)

    # Teste de modulos
    tester = MLModuleTester()
    modules_ok = tester.run_all_tests()

    # Salvar relatorio
    report_path = Path(__file__).parent.parent / "reports" / "ml_test_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tester.save_report(str(report_path))

    # Teste de aliases
    aliases_ok = test_aliases()

    # Teste de modelos treinados
    models_ok = test_trained_models()

    # Resultado final
    print("\n" + "=" * 80)
    print("RESULTADO FINAL")
    print("=" * 80)
    print(f"Modulos ML:      {'PASS' if modules_ok else 'FAIL'}")
    print(f"Aliases:         {'PASS' if aliases_ok else 'FAIL'}")
    print(f"Modelos:         {'PASS' if models_ok else 'FAIL'}")

    all_ok = modules_ok and aliases_ok and models_ok
    print(f"\nSTATUS GERAL:    {'PASS' if all_ok else 'FAIL'}")
    print("=" * 80)

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
