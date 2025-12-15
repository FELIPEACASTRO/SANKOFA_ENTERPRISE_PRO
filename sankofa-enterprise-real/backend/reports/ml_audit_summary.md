# Relatório de Auditoria - ML Engine

**Data:** 2025-12-15T18:04:22.483758
**Diretório:** `c:\Users\davis\Workspace\SANKOFA_ENTERPRISE_PRO\sankofa-enterprise-real\backend\ml_engine`

## Sumário Executivo

- **Total de arquivos analisados:** 66
- **Total de classes:** 190
- **Total de métodos:** 872
- **Métodos incompletos:** 8
- **Arquivos com problemas:** 2

## Bibliotecas Opcionais Detectadas

Total de imports opcionais únicos: 42

- `cache.prediction_cache`
- `catboost`
- `causalml.inference.meta`
- `causalml.metrics`
- `datasets`
- `dowhy`
- `dowhy.datasets`
- `featuretools`
- `h2o`
- `h2o.automl`
- `networkx`
- `onnxmltools.convert`
- `onnxmltools.convert.common.data_types`
- `onnxruntime`
- `scipy.stats`
- `sentence_transformers`
- `shap`
- `skl2onnx`
- `skl2onnx.common.data_types`
- `sklearn.decomposition`
- `sklearn.ensemble`
- `sklearn.inspection`
- `sklearn.linear_model`
- `sklearn.metrics`
- `sklearn.neural_network`
- `sklearn.preprocessing`
- `tensorflow`
- `tensorflow.keras`
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `torch.optim`
- `torch.utils.data`
- `torch_geometric`
- `torch_geometric.data`
- `torch_geometric.datasets`
- `torch_geometric.loader`
- `torch_geometric.nn`
- `torch_geometric.utils`
- `transformers`
- `tsfresh`
- `tsfresh.feature_extraction`

## Arquivos com Problemas

### explanation_generator.py

- **Pode importar:** True
- **Problemas:**
  - Contém 1 TODOs

### production_fraud_engine.py

- **Pode importar:** True
- **Problemas:**
  - Contém 1 TODOs

## Métodos Incompletos

### device_fingerprint.py - DeviceFingerprintPersistence

- `save_device` (stub/NotImplementedError)
- `get_device` (stub/NotImplementedError)
- `get_user_devices` (stub/NotImplementedError)
- `get_device_users` (stub/NotImplementedError)
- `update_device` (stub/NotImplementedError)

### federated_learning.py - FederatedClient

- `set_local_data` (stub/NotImplementedError)
- `train_local` (stub/NotImplementedError)
- `evaluate_local` (stub/NotImplementedError)

## Classes com Funcionalidade ML

Total de classes ML: 27

| Arquivo | Classe | train | predict | fit | analyze | detect |
|---------|--------|-------|---------|-----|---------|--------|
| autoencoder_anomaly_detector.py | AutoencoderAnomalyDetector | ✗ | ✗ | ✓ | ✗ | ✗ |
| behavioral_analyzer.py | BehavioralAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| catboost_model.py | CatBoostFraudModel | ✓ | ✓ | ✗ | ✗ | ✗ |
| causal_inference.py | UpliftModeling | ✗ | ✗ | ✓ | ✗ | ✗ |
| device_analyzer.py | DeviceAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| device_fingerprint.py | DeviceFingerprintAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| dormancy_analyzer.py | DormancyAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| duress_detector.py | DuressDetector | ✗ | ✗ | ✗ | ✓ | ✗ |
| federated_learning.py | FederatedFraudDetection | ✓ | ✓ | ✗ | ✗ | ✗ |
| federated_learning.py | FederatedServer | ✓ | ✗ | ✗ | ✗ | ✗ |
| gnn_trainer.py | GNNTrainer | ✓ | ✓ | ✗ | ✗ | ✗ |
| keystroke_analyzer.py | KeystrokeAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| mixture_of_experts.py | Expert | ✗ | ✓ | ✓ | ✗ | ✗ |
| mixture_of_experts.py | GatingNetwork | ✗ | ✗ | ✓ | ✗ | ✗ |
| mixture_of_experts.py | MixtureOfExperts | ✗ | ✓ | ✓ | ✗ | ✗ |
| mouse_analyzer.py | MouseAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| mule_detector.py | MuleDetector | ✗ | ✗ | ✗ | ✗ | ✓ |
| network_position_analyzer.py | NetworkPositionAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
| onnx_serving.py | ONNXInferenceSession | ✗ | ✓ | ✗ | ✗ | ✗ |
| probability_calibration.py | EnsembleCalibrator | ✗ | ✗ | ✓ | ✗ | ✗ |
| probability_calibration.py | ProbabilityCalibrator | ✗ | ✗ | ✓ | ✗ | ✗ |
| production_fraud_engine.py | ProductionFraudEngine | ✓ | ✓ | ✓ | ✗ | ✗ |
| scam_detector.py | ScamPatternDetector | ✗ | ✗ | ✗ | ✗ | ✓ |
| self_explainable_module.py | InterpretativeMaskLearner | ✗ | ✗ | ✓ | ✗ | ✗ |
| self_training_optimizer.py | AdaptiveSelfTraining | ✗ | ✓ | ✓ | ✗ | ✗ |
| self_training_optimizer.py | SelfTrainingClassifier | ✗ | ✓ | ✓ | ✗ | ✗ |
| turnover_analyzer.py | TurnoverAnalyzer | ✗ | ✗ | ✗ | ✓ | ✗ |
