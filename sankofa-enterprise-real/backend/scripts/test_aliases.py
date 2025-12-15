"""
Test script to verify all class aliases are working correctly.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_aliases():
    """Test all added aliases"""
    results = []

    # Test 1: mixture_of_experts.py
    try:
        from ml_engine.mixture_of_experts import MixtureOfExpertsDetector, MoEDetector, MixtureOfExperts
        assert MixtureOfExpertsDetector is MixtureOfExperts
        assert MoEDetector is MixtureOfExperts
        results.append(("mixture_of_experts", "OK", "MixtureOfExpertsDetector, MoEDetector"))
    except Exception as e:
        results.append(("mixture_of_experts", "FAIL", str(e)))

    # Test 2: multi_armed_bandits.py
    try:
        from ml_engine.multi_armed_bandits import MultiArmedBanditOptimizer, MABOptimizer, StepUpMFAOptimizer, ContextualBandit
        assert MultiArmedBanditOptimizer is StepUpMFAOptimizer
        assert MABOptimizer is ContextualBandit
        results.append(("multi_armed_bandits", "OK", "MultiArmedBanditOptimizer, MABOptimizer"))
    except Exception as e:
        results.append(("multi_armed_bandits", "FAIL", str(e)))

    # Test 3: feature_engineering (package)
    try:
        from ml_engine.feature_engineering import FeatureEngineering, FeatureEngineer, FeatureGenerator
        assert FeatureEngineering is FeatureGenerator
        assert FeatureEngineer is FeatureGenerator
        results.append(("feature_engineering", "OK", "FeatureEngineering, FeatureEngineer -> FeatureGenerator"))
    except Exception as e:
        results.append(("feature_engineering", "FAIL", str(e)))

    # Test 4: self_training_optimizer.py
    try:
        from ml_engine.self_training_optimizer import SelfTrainingOptimizer, AdaptiveSelfTraining
        assert SelfTrainingOptimizer is AdaptiveSelfTraining
        results.append(("self_training_optimizer", "OK", "SelfTrainingOptimizer"))
    except Exception as e:
        results.append(("self_training_optimizer", "FAIL", str(e)))

    # Test 5: federated_learning.py
    try:
        from ml_engine.federated_learning import FederatedLearningCoordinator, FederatedServer
        assert FederatedLearningCoordinator is FederatedServer
        results.append(("federated_learning", "OK", "FederatedLearningCoordinator"))
    except Exception as e:
        results.append(("federated_learning", "FAIL", str(e)))

    # Test 6: self_explainable_module.py
    try:
        from ml_engine.self_explainable_module import SelfExplainableModule, InterpretativeMaskLearner
        assert SelfExplainableModule is InterpretativeMaskLearner
        results.append(("self_explainable_module", "OK", "SelfExplainableModule"))
    except Exception as e:
        results.append(("self_explainable_module", "FAIL", str(e)))

    # Test 7: pix_fraud_taxonomy.py
    try:
        from ml_engine.pix_fraud_taxonomy import PixFraudDetector, PIXFraudTaxonomy
        assert PixFraudDetector is PIXFraudTaxonomy
        results.append(("pix_fraud_taxonomy", "OK", "PixFraudDetector"))
    except Exception as e:
        results.append(("pix_fraud_taxonomy", "FAIL", str(e)))

    # Test 8: onnx_serving.py
    try:
        from ml_engine.onnx_serving import ONNXModelServer, ONNXFraudModelServing
        assert ONNXModelServer is ONNXFraudModelServing
        results.append(("onnx_serving", "OK", "ONNXModelServer"))
    except Exception as e:
        results.append(("onnx_serving", "FAIL", str(e)))

    # Test 9: causal_inference.py
    try:
        from ml_engine.causal_inference import CausalInferenceEngine, CausalImpactAnalyzer
        assert CausalInferenceEngine is CausalImpactAnalyzer
        results.append(("causal_inference", "OK", "CausalInferenceEngine"))
    except Exception as e:
        results.append(("causal_inference", "FAIL", str(e)))

    # Test 10: llm/fraud_llm.py
    try:
        from ml_engine.llm.fraud_llm import FraudLLM, FraudLLMAnalyzer
        assert FraudLLM is FraudLLMAnalyzer
        results.append(("llm/fraud_llm", "OK", "FraudLLM"))
    except Exception as e:
        results.append(("llm/fraud_llm", "FAIL", str(e)))

    # Test 11: llm/scam_detector.py
    try:
        from ml_engine.llm.scam_detector import LLMScamDetector, ScamPatternDetector
        assert LLMScamDetector is ScamPatternDetector
        results.append(("llm/scam_detector", "OK", "LLMScamDetector"))
    except Exception as e:
        results.append(("llm/scam_detector", "FAIL", str(e)))

    # Print results
    print("\n" + "=" * 60)
    print("ALIAS VERIFICATION RESULTS")
    print("=" * 60)

    ok_count = 0
    fail_count = 0

    for module, status, details in results:
        if status == "OK":
            ok_count += 1
            print(f"OK   - {module}: {details}")
        else:
            fail_count += 1
            print(f"FAIL - {module}: {details}")

    print("=" * 60)
    print(f"TOTAL: {ok_count} OK, {fail_count} FAIL")
    print("=" * 60)

    return fail_count == 0


if __name__ == "__main__":
    success = test_aliases()
    sys.exit(0 if success else 1)
