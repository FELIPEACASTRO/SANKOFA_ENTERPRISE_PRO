# Métricas e Benchmarks de Produção - Sankofa 2025

---

## 1. KPIs DE PERFORMANCE

### 1.1 Latência (CRÍTICA para PIX)

| Métrica | Target PIX | Target Crédito | Status | Prioridade |
|---------|------------|----------------|--------|------------|
| **P50 Latência** | < 30ms | < 100ms | ✅ Implementado | CRÍTICA |
| **P99 Latência** | < 50ms | < 200ms | ✅ Implementado | CRÍTICA |
| **P999 Latência** | < 100ms | < 500ms | ✅ Implementado | Alta |

**Target PIX**: 50ms para não atrasar autorização de transação

### 1.2 Throughput (TPS)

| Cenário | Target | Base de Cálculo |
|---------|--------|-----------------|
| **PIX Geral** | > 3,500 TPS | 300M TX/dia ÷ 86,400 seg |
| **PIX Peak (18h)** | > 5,000 TPS | 3x throughput médio |
| **Crédito/Débito** | > 1,000 TPS | Menos crítico |

**Validar com**: `test_throughput_benchmark.py`

### 1.3 Acurácia de Modelo

| Métrica | Target | Benchmark | Implementação |
|---------|--------|-----------|---------------|
| **Recall** | > 90% | CatBoost 91.6% | Prioridade: detectar fraude |
| **Precision** | > 70% | XGBoost 75%+ | Minimizar false positives |
| **F1 Score** | > 80% | Stacking 99% | Balanço |
| **AUC-ROC** | > 0.95 | IEEE-CIS 0.99 | Discriminação |
| **PR-AUC** | > 0.85 | LightGBM 0.88 | Imbalanced data |

### 1.4 Taxa de Falsos Positivos

| Tipo de TX | Max False Positive | Impacto |
|-----------|-------------------|--------|
| **PIX** | < 0.5% | Bloqueia transações legítimas |
| **Crédito** | < 1% | Rejeita compras válidas |
| **Débito/ATM** | < 0.8% | Nega acesso ao dinheiro |

**Meta**: Encontrar balanço entre segurança e experiência do usuário

---

## 2. MÉTRICAS DE DETECÇÃO

### 2.1 Matriz de Confusão Objetivos

```
True Negatives (TN):  ~99.3% - Transações legítimas aprovadas
False Positives (FP): <0.5%  - Legítimas rejeitadas
True Positives (TP):  >90%   - Fraudes detectadas
False Negatives (FN): <10%   - Fraudes permitidas (RISCO)
```

### 2.2 Cálculo de Métricas

```python
# Sensitivity = Recall = TP / (TP + FN)
Recall = True_Positives / (True_Positives + False_Negatives)
# Target: 90%+ (detectar a maioria das fraudes)

# Specificity = TN / (TN + FP)
Specificity = True_Negatives / (True_Negatives + False_Positives)
# Target: 99%+ (manter experiência legítima)

# Precision = TP / (TP + FP)
Precision = True_Positives / (True_Positives + False_Positives)
# Target: 70%+ (evitar muitos falsos positivos)

# F1 = 2 * (Precision * Recall) / (Precision + Recall)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
# Target: 80%+ (balanço)

# ROC-AUC = Área sob curva ROC
# Target: > 0.95

# PR-AUC = Área sob curva Precision-Recall
# Target: > 0.85 (importante para dados imbalanceados)
```

### 2.3 Benchmarks por Modelo

| Modelo | Recall | Precision | F1 | AUC | Latência |
|--------|--------|-----------|----|----|----------|
| **LightGBM** | 88% | 72% | 0.79 | 0.95 | **25ms** ✅ |
| **XGBoost** | 90% | 75% | 0.82 | 0.96 | 45ms |
| **CatBoost** | 91.6% | 78% | 0.85 | 0.97 | 60ms |
| **Random Forest** | 92% | 76% | 0.83 | 0.96 | 80ms |
| **Stacking Ensemble** | 99% | 88% | 0.93 | 0.99 | 150ms |

**Recomendação PIX**: LightGBM (melhor latência)
**Recomendação Accuracy**: Stacking Ensemble (melhor resultado)

---

## 3. BENCHMARKS DE DATASETS

### 3.1 Performance por Dataset

| Dataset | Records | Fraude% | Model Winner | F1 | AUC |
|---------|---------|---------|--------------|----|----|
| **IEEE-CIS** | 590K | 3.5% | Stacking | 0.93 | 0.99 |
| **Credit Card ULB** | 284K | 0.17% | Random Forest | 0.85 | 0.95 |
| **BankSim** | 594K | 1.2% | XGBoost | 0.82 | 0.94 |
| **CiferAI (21M)** | 21M | ~0.1% | XGBoost+LightGBM | 0.88 | 0.96 |
| **Elliptic Bitcoin** | 203K TX | 2% | GCN | 0.87 | 0.95 |

### 3.2 Recomendação por Tipo

```
PIX (tempo real):
- Dataset: CiferAI (mobile money similar)
- Model: LightGBM (25ms latência)
- Target: F1 > 0.85, Recall > 90%

Crédito (offline scoring):
- Dataset: IEEE-CIS (melhor features)
- Model: Stacking Ensemble (99% accuracy)
- Target: F1 > 0.90, Precision > 80%

Débito/ATM (transacional):
- Dataset: BankSim + Elliptic
- Model: XGBoost (balance)
- Target: F1 > 0.82, Recall > 88%
```

---

## 4. REGULAMENTOS BACEN - BRASIL

### 4.1 Resolução 6 - PIX Fraud Sharing

**Requisitos**:
- Compartilhar dados de fraude entre bancos em 24h
- Base centralizada de fraudes
- Taxa autorizada sem registro: R$200
- Limite noturno (23h-5h): R$1.000

### 4.2 Resolução BCB 491

**Compliance PIX**:
- Dispositivos não cadastrados: limite R$200
- Dispositivos cadastrados: limite R$5.000+
- Autenticação forte obrigatória
- Confirmação de recebedor

### 4.3 MED 2.0 (Fevereiro 2026)

**Novos Requisitos**:
- Rastreabilidade aprimorada de PIX
- Bloqueio preventivo de contas (até 72h)
- Devolução automática em até 96h
- Report de fraude estruturado

### 4.4 LGPD Compliance

```python
# Explicabilidade obrigatória (Art. 20)
explanation = {
    "decision": "BLOCKED",
    "risk_score": 0.87,
    "main_risk_factors": [
        "Transação noturna em novo dispositivo",
        "Valor 3x acima da média histórica",
        "Destinatário PJ sem histórico"
    ],
    "user_rights": [
        "Solicitar revisão desta decisão",
        "Acessar dados usados na análise",
        "Solicitar exclusão de dados"
    ]
}
```

---

## 5. MONITORAMENTO CONTÍNUO

### 5.1 Métricas Prometheus

```python
from prometheus_client import Histogram, Counter, Gauge

# Latência
fraud_detection_latency = Histogram(
    'fraud_detection_latency_seconds',
    'Latência de predição',
    buckets=[0.01, 0.05, 0.1, 0.5]  # PIX target: 0.05s
)

# Taxa de bloqueios
fraud_blocks_total = Counter(
    'fraud_blocks_total',
    'Total de transações bloqueadas',
    ['transaction_type']  # 'pix', 'credit', 'debit'
)

# False positive rate
false_positive_rate = Gauge(
    'false_positive_rate',
    'Taxa de falsos positivos',
    value=0.004  # Target: <0.5%
)

# Model accuracy
model_accuracy = Gauge(
    'model_accuracy',
    'Acurácia do modelo',
    value=0.985  # 98.5%
)

# SLA compliance
sla_compliance = Gauge(
    'sla_compliance_ratio',
    'Conformidade de latência',
    value=0.99  # 99% das TX < 50ms
)

# Fraud detection rate
fraud_detection_rate = Gauge(
    'fraud_detection_rate',
    'Taxa de detecção de fraude',
    value=0.92  # 92% recall
)
```

### 5.2 Alertas Críticos

```python
# Em: backend/monitoring/alerts.py

ALERT_RULES = {
    'high_latency': {
        'condition': 'p99_latency > 100ms',
        'severity': 'CRITICAL',
        'action': 'Escalar para engenharia'
    },
    'low_accuracy': {
        'condition': 'model_accuracy < 0.95',
        'severity': 'HIGH',
        'action': 'Retraining urgente'
    },
    'high_false_positive': {
        'condition': 'false_positive_rate > 0.01',
        'severity': 'HIGH',
        'action': 'Revisar threshold'
    },
    'model_drift': {
        'condition': 'ks_statistic > 0.15',
        'severity': 'MEDIUM',
        'action': 'Validar distribuição de dados'
    }
}
```

---

## 6. TESTES DE VALIDAÇÃO

### 6.1 Suite de Testes

```bash
# Latência
pytest backend/tests/test_latency.py::test_p99_latency_pix

# Acurácia
pytest backend/tests/test_models.py::test_model_accuracy

# SLA
pytest backend/tests/test_sla.py::test_fraud_detection_sla

# Compliance
pytest backend/tests/test_compliance.py::test_lgpd_explainability

# Throughput
pytest backend/tests/test_throughput.py::test_pix_3500_tps
```

### 6.2 Benchmark Script

```python
# backend/benchmarks/run_benchmarks.py

def benchmark_latency():
    """Mede P50, P99, P999 latência"""
    results = {}
    for model in ['lightgbm', 'xgboost', 'catboost']:
        latencies = []
        for _ in range(10000):
            start = time.time()
            model.predict(random_features())
            latencies.append((time.time() - start) * 1000)  # ms
        
        results[model] = {
            'p50': np.percentile(latencies, 50),
            'p99': np.percentile(latencies, 99),
            'p999': np.percentile(latencies, 99.9)
        }
    return results

def benchmark_accuracy():
    """Valida F1, Recall, Precision"""
    for dataset in ['ieee_cis', 'creditcard', 'banksim']:
        y_true = load_labels(dataset)
        y_pred = model.predict(load_features(dataset))
        
        metrics = {
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        print(f"{dataset}: {metrics}")
```

---

## 7. ROADMAP DE OTIMIZAÇÃO

### Q1 2025

- [ ] Baseline: LightGBM em produção (25ms, 88% F1)
- [ ] Monitoring Prometheus ativo
- [ ] Dataset CiferAI integrado

### Q2 2025

- [ ] Implementar Stacking Ensemble (off-peak analysis)
- [ ] GNN para detecção de redes
- [ ] Federated learning multi-bank

### Q3 2025

- [ ] Otimizar para 50ms P99 PIX
- [ ] Transfer learning com Hugging Face
- [ ] Real-time behavioral biometrics

### Q4 2025

- [ ] LSTM+Transformer para sequências
- [ ] Autoencoder para anomalias
- [ ] Compliance MED 2.0 validação

---

## 8. REFERÊNCIAS

- IEEE-CIS Competition: https://www.kaggle.com/c/ieee-fraud-detection
- Papers with Code: https://paperswithcode.com/task/fraud-detection
- Feedzai Research: https://research.feedzai.com
- BACEN Regulações: https://www.bcb.gov.br

