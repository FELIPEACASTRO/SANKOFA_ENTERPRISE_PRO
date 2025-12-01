# Análise Técnica Completa: Datasets para Fraud Detection
## Sankofa Enterprise Pro - Perspectiva Multidisciplinar

**Data:** 01/12/2025  
**Equipe:** Cientista de Dados | ML Engineers | Fraud Specialists  
**Total Recursos Analisados:** 104+ fontes verificadas

---

# PARTE 1: VISÃO DO CIENTISTA DE DADOS

## 1.1 Análise Estatística dos Datasets Disponíveis

### Distribuição de Volume
| Dataset | Registros | Fraudes | Taxa Fraude | Desbalanceamento |
|---------|-----------|---------|-------------|------------------|
| PaySim | 6,362,620 | ~8,200 | 0.13% | 769:1 |
| FCA-UK APP | 15,000,000 | 61,000 | 0.41% | 244:1 |
| Nigerian Financial | 5,000,000 | ~50,000 | ~1% | 99:1 |
| Feedzai BAF | 6,000,000 | ~66,000 | 1.1% | 90:1 |
| IEEE-CIS | 590,540 | ~20,600 | 3.5% | 28:1 |
| MLG-ULB | 284,807 | 492 | 0.17% | 577:1 |
| PIX Kaggle | 10,000 | 100 | 1% | 99:1 |

### Implicações para Modelagem
```
PROBLEMA CRÍTICO: Extreme Class Imbalance
- Média de desbalanceamento: ~200:1
- Técnicas necessárias: SMOTE, ADASYN, Cost-Sensitive Learning
- Métricas recomendadas: AUPRC > AUC-ROC (mais sensível ao desbalanceamento)
```

## 1.2 Análise de Features Disponíveis

### Features Categóricas (Cross-Dataset)
| Feature | Datasets Presentes | Cardinalidade Típica |
|---------|-------------------|---------------------|
| transaction_type | PaySim, Nigerian, PIX | 5-8 valores |
| merchant_category | Nigerian, IEEE-CIS | 20-100 valores |
| payment_channel | Nigerian, FCA | 4-6 valores |
| device_type | IEEE-CIS, Nigerian | 3-5 valores |
| location | Nigerian, PIX | Alta (cidades) |

### Features Numéricas (Cross-Dataset)
| Feature | Datasets Presentes | Range Típico | Distribuição |
|---------|-------------------|--------------|--------------|
| amount | TODOS | 0 - 10M+ | Log-normal |
| time_since_last | Nigerian, Feedzai | 0 - 10000h | Exponencial |
| velocity_score | Nigerian | 0 - 100 | Bimodal |
| balance_before | PaySim | 0 - 10M+ | Heavy-tail |
| balance_after | PaySim | 0 - 10M+ | Heavy-tail |

### Features Derivadas (Engenharia Crítica)
Baseado em **Bahnsen et al. 2016** (paper referência):
```python
# 1. Agregação Temporal (24h, 72h, 168h janelas)
txn_count_window = COUNT(transactions WHERE time < window_hours)
txn_sum_window = SUM(amount WHERE time < window_hours)
txn_avg_window = AVG(amount WHERE time < window_hours)

# 2. Features Periódicas (Von Mises Distribution)
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
day_sin = sin(2 * pi * day_of_week / 7)
day_cos = cos(2 * pi * day_of_week / 7)

# 3. Desvio do Comportamento
spending_deviation = (amount - user_avg) / user_std
frequency_deviation = (txn_count_24h - user_avg_freq) / user_std_freq
```

## 1.3 Qualidade dos Dados

### Análise de Missing Values
| Dataset | Missing Rate | Tratamento Recomendado |
|---------|-------------|----------------------|
| IEEE-CIS | 30-50% em features device | Imputation + flag |
| Nigerian | <1% | Drop ou median |
| PaySim | 0% | N/A (sintético) |
| FCA-UK | <5% | Domain-specific |

### Análise de Data Leakage
```
ALERTA CRÍTICO - Verificar:
1. Features com informação futura (ex: label_timestamp antes de transaction)
2. Features derivadas do label (ex: "is_flagged" em PaySim)
3. Features administrativas (ex: EndToEndId em PIX)
```

---

# PARTE 2: VISÃO DA EQUIPE DE MACHINE LEARNING

## 2.1 Arquiteturas de Modelos Disponíveis

### Modelos Tradicionais (Tabular)
| Modelo | Dataset Benchmark | AUC | F1 | Referência |
|--------|------------------|-----|-----|-----------|
| XGBoost | IEEE-CIS | 0.965 | 0.85 | Amazon FDB |
| LightGBM | Feedzai BAF | 0.95 | 0.83 | NeurIPS 2022 |
| CatBoost | Nigerian | 0.92 | 0.78 | HuggingFace |
| Random Forest | MLG-ULB | 0.98 | 0.82 | Baseline |

### Graph Neural Networks (Estado da Arte)
| Modelo | Arquitetura | Dataset | AUC | F1 | Paper |
|--------|-------------|---------|-----|-----|-------|
| **HOGRL** | High-Order GNN | YelpChi | 0.9808 | 0.86 | IJCAI 2024 |
| **Grad** | Diffusion GNN | YelpChi | 0.9908 | - | WWW 2025 |
| **RGTAN** | Risk-aware Graph | Amazon | 0.9750 | 0.92 | TKDE 2025 |
| **GTAN** | Semi-supervised | S-FFSD | 0.8286 | 0.73 | AAAI 2023 |
| **FraudGT** | Graph Transformer | AML | - | +17.8% | ACM 2024 |
| **DGA-GNN** | Dynamic Grouping | Multi | +16% | - | AAAI 2024 |

### Transformers para Transações
| Modelo | Abordagem | Destaque |
|--------|-----------|----------|
| PTP | Payment Transaction Pre-training | Embedding de sequências |
| Generative Pretraining | Transformer encoding | arXiv:2312.14406 |

## 2.2 Transfer Learning Opportunities

### Estratégias Validadas
```
1. FINE-TUNING CROSS-DOMAIN
   - Treinar em PaySim (6M) → Fine-tune em PIX (10K)
   - Esperado: +15-25% F1 vs treino do zero
   
2. DOMAIN ADAPTATION
   - Nigerian (5M) → Brazilian PIX
   - Features compartilhadas: velocity, channel, time
   
3. GNN PRE-TRAINING
   - Usar embeddings do AI4Risk em dados proprietários
   - Checkpoints disponíveis: STAN, GTAN, HOGRL
```

### Modelos Pré-treinados Disponíveis
| Fonte | Modelo | Checkpoint | Formato |
|-------|--------|-----------|---------|
| AI4Risk | STAN | stan_3d_ckpt | PyTorch |
| AI4Risk | GTAN | gtan_ckpt | PyTorch |
| NVIDIA | GNN+XGB | NGC Container | Docker |
| Hugging Face | VAE-GAN | kmasiak/FraudDetection | HF Hub |

## 2.3 Pipeline de ML Recomendado

```
┌─────────────────────────────────────────────────────────────────┐
│                    SANKOFA ML PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [1] DATA INGESTION                                             │
│      └── PaySim (6M) + Nigerian (5M) + PIX (10K)               │
│                                                                 │
│  [2] FEATURE ENGINEERING (Bahnsen et al.)                       │
│      ├── Agregação temporal (24h, 72h, 168h)                   │
│      ├── Features periódicas (von Mises)                       │
│      └── Desvio comportamental                                  │
│                                                                 │
│  [3] MODEL TRAINING                                             │
│      ├── Stage 1: XGBoost baseline                             │
│      ├── Stage 2: Stacking ensemble                            │
│      └── Stage 3: GNN (GTAN/RGTAN) para grafos                 │
│                                                                 │
│  [4] CALIBRATION                                                │
│      ├── Platt scaling                                          │
│      └── Isotonic regression                                    │
│                                                                 │
│  [5] DEPLOYMENT                                                 │
│      ├── NVIDIA Triton (batch)                                 │
│      └── Real-time API (<50ms)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2.4 Benchmark Framework

### Amazon FDB (Fraud Dataset Benchmark)
```python
from fdb.datasets import FraudDatasetBenchmark

# Datasets padronizados
BENCHMARK_DATASETS = [
    'ieeecis',    # CNP Fraud - 590K
    'ccfraud',    # Card Fraud - 284K  
    'fraudecom',  # E-commerce - 150K
    'sparknov',   # Simulated - 1.3M
    'twitterbot', # Bot Attack - 37K
    'malurl',     # Malicious URL - 650K
    'fakejob',    # Job Scam - 17K
    'vehicleloan',# Credit Risk - 233K
    'ipblock',    # IP Blacklist - 215K
]

# Métricas padronizadas
METRICS = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision@k', 'Recall@k']
```

---

# PARTE 3: VISÃO DOS ESPECIALISTAS EM FRAUDE BANCÁRIA

## 3.1 Taxonomia de Fraudes (Brasil/PIX)

### Classificação por Vetor de Ataque (arXiv:2511.20902)

| Categoria | Tipo de Fraude | Prevalência | Detectabilidade ML |
|-----------|---------------|-------------|-------------------|
| **Engenharia Social** | Mão Fantasma | Alta | Média |
| | Central Falsa | Alta | Alta |
| | Golpe WhatsApp | Muito Alta | Alta |
| **Manipulação Técnica** | QR Code Adulterado | Média | Alta |
| | App Malicioso | Baixa | Baixa |
| **Coação Física** | Sequestro Relâmpago | Média | Baixa* |
| | Roubo com Violência | Média | Baixa* |
| **Fraude Documental** | Conta Laranja | Alta | Alta |
| | Identidade Falsa | Média | Alta |

*Requer features contextuais adicionais (horário, localização, velocidade)

### Features de Detecção por Tipo de Fraude

```
1. MÃO FANTASMA (Ghost Hand)
   Features críticas:
   - remote_access_detected: boolean
   - session_anomaly_score: float
   - device_fingerprint_change: boolean
   - mouse_movement_pattern: string (human vs bot)
   
2. QR CODE ADULTERADO
   Features críticas:
   - qr_source_verified: boolean
   - recipient_first_transaction: boolean
   - merchant_category_mismatch: boolean
   - donation_stream_context: boolean

3. SEQUESTRO RELÂMPAGO
   Features críticas:
   - is_night_transaction: boolean (22h-6h)
   - location_unusual: boolean
   - multiple_rapid_transactions: boolean
   - amount_pattern_change: float
   - device_movement_anomaly: boolean
```

## 3.2 Regras de Negócio Críticas (BACEN/LGPD)

### Limites Regulatórios PIX
| Período | Limite Padrão | Limite Reduzido (Noite) |
|---------|--------------|------------------------|
| Diurno (6h-20h) | R$ 1.000 | N/A |
| Noturno (20h-6h) | R$ 1.000 | R$ 200-500 |
| Mensal | Sem limite padrão | Customizável |

### MED (Mecanismo Especial de Devolução)
```
VERSÃO ATUAL (MED 1.0):
- Prazo: 7 dias para contestação
- Bloqueio: Apenas conta destinatária

VERSÃO 2.0 (Fevereiro 2026):
- Rastreamento até 5 camadas de contas
- Bloqueio automático em cascata
- Sistema GRAF (grafos direcionados acíclicos)
```

### Features de Compliance
```python
COMPLIANCE_FEATURES = {
    'lgpd': {
        'data_minimization': True,
        'purpose_limitation': True,
        'consent_tracking': True,
        'right_to_explanation': True,  # SHAP values
    },
    'bacen': {
        'med_eligibility': True,
        'night_limit_check': True,
        'pix_key_validation': True,
        'suspicious_account_flag': True,
    },
    'pci_dss': {
        'data_masking': True,
        'audit_trail': True,
        'encryption_at_rest': True,
        'encryption_in_transit': True,
    }
}
```

## 3.3 Métricas de Negócio

### KPIs de Fraud Prevention
| Métrica | Benchmark Mercado | Target Sankofa |
|---------|------------------|----------------|
| Fraud Detection Rate | 85-95% | >95% |
| False Positive Rate | 1-5% | <2% |
| Time to Detect | <1 min | <50ms |
| Recovery Rate (MED) | 30-50% | >60% |
| Customer Friction Score | 3-5% | <3% |

### Custo de Erros
```
FALSE NEGATIVE (fraude não detectada):
- Custo médio: R$ 2.500 por transação
- Impacto reputacional: Alto
- Risco regulatório: Multas BACEN

FALSE POSITIVE (bloqueio indevido):
- Custo médio: R$ 50 (atendimento)
- Impacto reputacional: Médio
- Churn risk: 5-10% por incidente
```

---

# PARTE 4: RECOMENDAÇÕES CONSOLIDADAS

## 4.1 Datasets Prioritários para Sankofa

| Prioridade | Dataset | Justificativa | Uso |
|------------|---------|--------------|-----|
| 1 | **Nigerian Financial** | 5M transações, features ricas | Transfer Learning base |
| 2 | **PaySim** | 6M transações, volume | Stress test, scaling |
| 3 | **Feedzai BAF** | Fairness testing, bias | Compliance, LGPD |
| 4 | **FCA-UK APP** | APP Fraud patterns | Regras PIX |
| 5 | **AI4Risk Models** | GNN pré-treinados | Upgrade arquitetura |

## 4.2 Features a Implementar (Ordem de Prioridade)

### Fase 1 (Semana 1-2)
```python
PHASE_1_FEATURES = [
    'txn_count_last_1h',
    'txn_count_last_24h', 
    'txn_amount_sum_24h',
    'time_since_last_txn',
    'spending_deviation_score',
    'channel_risk_score',
    'is_night_transaction',
    'is_weekend',
    'is_first_recipient',
]
```

### Fase 2 (Semana 3-4)
```python
PHASE_2_FEATURES = [
    'velocity_score',
    'geo_anomaly_score',
    'device_fingerprint_score',
    'merchant_fraud_rate',
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'recipient_account_age',
    'sender_account_age',
]
```

### Fase 3 (Mês 2)
```python
PHASE_3_FEATURES = [
    'graph_embedding_user',
    'graph_embedding_merchant', 
    'community_fraud_rate',
    'temporal_pattern_score',
    'cross_channel_velocity',
    'whatsapp_contact_verified',
    'pix_key_type_risk',
]
```

## 4.3 Arquitetura de Produção Recomendada

```
┌─────────────────────────────────────────────────────────────────┐
│                    SANKOFA PRODUCTION STACK                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1: REAL-TIME SCORING (<50ms)                            │
│  ├── Hard Rules Engine (BACEN limits, blacklists)              │
│  ├── XGBoost Fast Scorer (tabular features)                    │
│  └── Risk Decision API                                          │
│                                                                 │
│  LAYER 2: NEAR-REAL-TIME (<5min)                               │
│  ├── GNN Scorer (GTAN/RGTAN)                                   │
│  ├── Behavioral Analysis                                        │
│  └── Alert Escalation                                           │
│                                                                 │
│  LAYER 3: BATCH ANALYTICS (daily)                              │
│  ├── Model Retraining Pipeline                                 │
│  ├── Drift Detection                                            │
│  ├── Performance Monitoring                                     │
│  └── Compliance Reports                                         │
│                                                                 │
│  OBSERVABILITY                                                  │
│  ├── SHAP Explainability                                       │
│  ├── Fairness Metrics                                          │
│  └── Audit Logs                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# APÊNDICE: REFERÊNCIAS VERIFICADAS

## Datasets Verificados (URLs Funcionais)
- [x] Kaggle MLG-ULB: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- [x] Kaggle PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
- [x] Kaggle PIX: https://www.kaggle.com/datasets/juniorbueno/pix-banking-transaction
- [x] Kaggle Feedzai BAF: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- [x] HuggingFace Nigerian: https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset
- [x] HuggingFace DIFrauD: https://huggingface.co/datasets/redasers/difraud
- [x] GitHub AI4Risk: https://github.com/AI4Risk/antifraud
- [x] GitHub Amazon FDB: https://github.com/amazon-science/fraud-dataset-benchmark
- [x] GitHub NVIDIA: https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection
- [x] GitHub IBM AMLSim: https://github.com/IBM/AMLSim
- [x] GitHub Google Fraudfinder: https://github.com/GoogleCloudPlatform/fraudfinder
- [x] GitHub Feedzai BAF: https://github.com/feedzai/bank-account-fraud
- [x] GitHub FraudGT: https://github.com/junhongmit/FraudGT
- [x] GitHub Graph Papers: https://github.com/safe-graph/graph-fraud-detection-papers
- [x] FCA-UK APP: https://www.fca.org.uk/firms/digital-sandbox/authorised-push-payment-synthetic-data
- [x] arXiv PIX Taxonomy: https://arxiv.org/abs/2511.20902

## Papers Acadêmicos Chave
1. Feedzai BAF (NeurIPS 2022): arXiv:2211.13358
2. Amazon FDB (2022): arXiv:2208.14417
3. Feature Engineering (Bahnsen 2016): albahnsen.github.io
4. PIX Fraud Taxonomy (2025): arXiv:2511.20902
5. RGTAN (TKDE 2025): IEEE 10.1109/TKDE.2025.3543887
6. HOGRL (IJCAI 2024): arxiv.org/pdf/2503.01556
7. Grad WWW 2025: ACM 10.1145/3696410.3714520

---

*Relatório gerado: 01/12/2025*  
*Versão: 2.0 - Análise Multidisciplinar Completa*
