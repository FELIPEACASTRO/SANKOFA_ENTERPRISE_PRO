# Catálogo Completo de Recursos Globais - Detecção de Fraude 2025
## Datasets, Modelos, Plataformas Cloud, Vendors e Arquiteturas State-of-the-Art

---

# SUMÁRIO EXECUTIVO

**Última Atualização**: Novembro 2025

Este documento consolida pesquisa extensiva em repositórios globais, papers acadêmicos e plataformas enterprise para detecção de fraude em transações PIX, Débito e Crédito.

**Descobertas Principais:**
- **21+ datasets** mapeados para treinamento de modelos
- **10+ modelos pré-treinados** disponíveis (Hugging Face, NVIDIA NGC)
- **4 plataformas cloud** com soluções completas (AWS, GCP, NVIDIA, Azure)
- **6 vendors enterprise** analisados (Feedzai, FICO, SAS, BioCatch, SEON, Fingerprint.com)
- **15+ papers arXiv 2024-2025** com arquiteturas state-of-the-art

---

# PARTE 1: DATASETS COMPLETOS

## 1.1 Tabela Mestre de Datasets

| Dataset | Registros | Features | Fraude % | Tipo | Fonte | Link |
|---------|-----------|----------|----------|------|-------|------|
| **CiferAI/Cifer-Fraud-Detection** | 21M | 14+ | ~0.1% | Mobile Money | Hugging Face | [Link](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) |
| **IEEE-CIS Fraud Detection** | 590K | 394 | 3.5% | Card-not-present | Kaggle | [Link](https://www.kaggle.com/c/ieee-fraud-detection) |
| **Credit Card Fraud (ULB)** | 284K | 31 | 0.17% | Crédito EU | Kaggle/OpenML | [Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Bank Account Fraud (NeurIPS)** | 6M | 32 | Variável | Contábil | Feedzai | [Link](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) |
| **PaySim** | 6.3M | 10 | 0.13% | Mobile (PIX proxy) | GitHub | [Link](https://github.com/EdgarLopezPhD/PaySim) |
| **Elliptic Bitcoin** | 203K | 166 | 2% | Blockchain | Kaggle | [Link](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| **Elliptic++** | 822K wallets | 56 | Multi | Blockchain | GitHub | [Link](https://github.com/git-disl/EllipticPlusPlus) |
| **Credit Card Default (UCI)** | 30K | 24 | Variável | Default Taiwan | UCI | [Link](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) |
| **Credit Approval (Statlog)** | 690 | 14 | Binary | Aplicações | UCI | [Link](https://archive.ics.uci.edu/dataset/27/credit+approval) |
| **BankSim** | 594K | 10+ | 1.2% | Simulado | Open | [Link](https://www.kaggle.com/datasets/ealaxi/banksim1) |

## 1.2 Amazon Fraud Dataset Benchmark (FDB)

**Framework unificado com 9 datasets:**

```python
pip install fraud-dataset-benchmark

from fraud_dataset_benchmark import FraudDatasetBenchmark

# Datasets disponíveis
datasets = [
    'ieeecis',           # Credit card
    'european_cc',       # European transactions
    'ecommerce',         # E-commerce fraud
    'vehicle_loan',      # Credit risk (233K, 21.7% default)
    'malicious_ip',      # Bot/traffic
    'job_postings',      # Content moderation
    'simulated_cc',      # Synthetic (Sparkov)
]

# Carregar com splits padronizados
fdb = FraudDatasetBenchmark(dataset_key='ieeecis')
train_data = fdb.train
test_data = fdb.test
```

**GitHub**: https://github.com/amazon-science/fraud-dataset-benchmark

## 1.3 Datasets Específicos para PIX

**IMPORTANTE: Não existe dataset público de PIX do BACEN**

**Alternativas recomendadas:**

| Dataset | Por que usar | Adaptação necessária |
|---------|-------------|---------------------|
| **CiferAI (21M)** | Mobile money similar a PIX | Adicionar features BCB 491 |
| **PaySim (6.3M)** | Transferências P2P | Adicionar limites noturnos |
| **Dados sintéticos CTGAN** | Pode modelar PIX real | Treinar com dados próprios |

**Features PIX obrigatórias:**
```python
PIX_FEATURES = {
    'regulatory': [
        'device_registered',      # BCB 491: limite R$200 sem cadastro
        'nocturnal_limit',        # R$1.000 (23h-5h)
        'pix_key_type',           # CPF, CNPJ, email, telefone, aleatória
        'recipient_is_pj',        # 2/3 das fraudes vão para PJ
    ],
    'velocity': [
        'pix_count_1h', 'pix_count_24h',
        'distinct_recipients_24h',
        'pix_to_pj_24h',
    ]
}
```

---

# PARTE 2: MODELOS PRÉ-TREINADOS

## 2.1 Hugging Face Models

| Modelo | Arquitetura | Acurácia | Caso de Uso | Link |
|--------|-------------|----------|-------------|------|
| **CiferAI/cifer-fraud-detection-k1-a** | Binary Classifier | 99.93% | General fraud | [Link](https://huggingface.co/CiferAI/cifer-fraud-detection-k1-a) |
| **keras-io/imbalanced_classification** | DNN | 99.82% recall | Credit card | [Link](https://huggingface.co/keras-io/imbalanced_classification) |
| **kmasiak/FraudDetection** | VAE-GAN | - | Anomalias | [Link](https://huggingface.co/kmasiak/FraudDetection) |
| **Bilic/Mistral-7B-LLM-Fraud-Detection** | Mistral-7B LLM | - | Transcripts | [Link](https://huggingface.co/Bilic/Mistral-7B-LLM-Fraud-Detection) |
| **Bilic/NeuralChat-finetuned-for-fraud-detection** | Intel NeuralChat | - | Conversas | [Link](https://huggingface.co/Bilic/NeuralChat-finetuned-for-fraud-detection) |
| **saifhmb/fraud-detection-model** | Gaussian NB | - | J.P. Morgan | [Link](https://huggingface.co/saifhmb/fraud-detection-model) |
| **vaibhav07112004/fraud-detection-models** | Ensemble 11 models | 95.7% | Flink | [Link](https://huggingface.co/vaibhav07112004/fraud-detection-models) |

## 2.2 NVIDIA NGC - Financial Fraud Training

**Container Docker para treinar GNN + XGBoost:**

```bash
# Login NGC
docker login nvcr.io

# Pull container
docker pull nvcr.io/nvidia/cugraph/financial-fraud-training:latest

# Executar
docker run --gpus all -it nvcr.io/nvidia/cugraph/financial-fraud-training:latest
```

**Características:**
- GNN gera embeddings de transações
- XGBoost para classificação final
- Shapley values para explainability
- Triton Inference Server para produção
- **39x speedup** em preprocessing
- **5.63x speedup** em treinamento

**GitHub**: https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection

---

# PARTE 3: PLATAFORMAS CLOUD

## 3.1 AWS SageMaker

**Status Amazon Fraud Detector**: Não aceita novos clientes (Nov 2025)

**Solução recomendada: SageMaker + DGL (Deep Graph Library)**

```python
# Fraud Detection com GNN
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

# Treinar modelo GNN
estimator = Estimator(
    image_uri='sagemaker-graph-fraud-detection',
    role=get_execution_role(),
    instance_count=1,
    instance_type='ml.p3.2xlarge'
)

# Deploy
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

**Recursos:**
- **JumpStart**: Templates prontos
- **Feature Store**: Armazenamento de features
- **Federated Learning**: Flower framework (Jul 2025)
- **Clarify**: Explainability

**GitHub**: https://github.com/awslabs/sagemaker-graph-fraud-detection

## 3.2 Google Cloud

**AML AI - Anti-Money Laundering:**
- 2-4x mais detecções (HSBC case study)
- 60% redução em alerts

**BigQuery ML:**
```sql
CREATE OR REPLACE MODEL fraud.detection_model
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['is_fraud'],
  auto_class_weights=TRUE
) AS
SELECT * FROM fraud.transactions
WHERE split = 'train'
```

**Swift Partnership (H1 2025):**
- Federated learning com 12 instituições
- Sem compartilhar dados raw
- Detecção cross-bank

**GitHub**: https://github.com/GoogleCloudPlatform/fraudfinder

## 3.3 NVIDIA Blueprint

**Stack completo:**
1. **cuGraph**: Construção de grafo GPU-accelerated
2. **GNN**: Embeddings de transações
3. **XGBoost**: Classificação final
4. **Triton**: Serving real-time
5. **Morpheus**: Monitoring

**Performance:**
- 39x speedup preprocessing
- Real-time inference (ms)
- Shapley values para compliance

---

# PARTE 4: VENDORS ENTERPRISE

## 4.1 Comparação de Mercado

| Vendor | Foco | Diferencial | Clientes |
|--------|------|-------------|----------|
| **Feedzai** | AI-native | RiskOps, adaptativo | Fintechs |
| **FICO** | Consortium | 10K+ instituições | Bancos tradicionais |
| **SAS** | Enterprise | Analytics completo | Governo |
| **BioCatch** | Behavioral | 3,000+ sinais | 280+ FIs |
| **SEON** | Multi-signal | 900+ sinais | Volume |
| **Fingerprint.com** | Device ID | 98% accuracy | Developers |

## 4.2 BioCatch - Behavioral Biometrics

**Métricas 2025:**
- **ARR**: $160M+ (Q2 2025)
- **Usuários protegidos**: 555 milhões
- **Sessões/mês**: 16.1 bilhões
- **Fraude prevenida Q3 2025**: $60M+
- **Pagamentos processados**: 180M+ ($330B+)

**Sinais analisados:**
- Keystroke dynamics (velocidade, ritmo, pressão)
- Mouse movements (fluência, padrões)
- Touch gestures (swipe, pressão)
- Navigation flow (hesitação, copy-paste)
- RAT detection (acesso remoto)

**Trust Network (Nov 2024):**
- Primeira rede de compartilhamento inter-banco
- Austrália: 85%+ da população bancária
- Padrões de fraude em tempo real

## 4.3 SEON vs Fingerprint.com

| Feature | SEON | Fingerprint.com |
|---------|------|-----------------|
| **Accuracy** | Dynamic scoring | **98%** sustentado |
| **Signals** | 900+ first-party | Device fingerprint |
| **Detection** | VPN, GPS spoof, emulators | Incognito, bots, anti-detect |
| **Pricing** | Custom | $99/mês (Pro) |
| **Best for** | Fraud + AML completo | Device ID puro |

---

# PARTE 5: ARQUITETURAS STATE-OF-THE-ART (arXiv 2024-2025)

## 5.1 Papers Principais

| Paper | arXiv | Arquitetura | Performance | Data |
|-------|-------|-------------|-------------|------|
| **RAGFormer** | 2402.17472 | GNN + Transformer | SOTA | Feb 2025 |
| **BRIGHT** | 2205.13084 | Two-Stage Graph | 75% latência reduzida | 2022/2025 |
| **Hybrid MoE** | 2504.03750 | RNN + Transformer + AE | 98.7% accuracy | Apr 2025 |
| **Transformer-GAN** | 2509.19032 | Transformer + GAN | Class imbalance | Sep 2025 |
| **FraudGT** | ACM DL | Graph Transformer | Edge attention | 2024 |
| **STA-GT** | 2307.05121 | Spatial-Temporal GT | Temporal encoding | Jul 2023 |

## 5.2 RAGFormer - Melhor Combinação (2025)

**Insight principal:**
> "GNN e Transformer features são **quase ortogonais**:
> - **GNN** → topologia (relações multi-hop)
> - **Transformer** → semântica (atributos de nó)"

**Arquitetura:**
```
Input Graph
    ↓
┌──────────────────────────────────┐
│ Semantic Encoder (Transformer)   │ → Self-attention
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ Topology Encoder (Relation GNN)  │ → Message passing
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│ Attention Fusion Module          │ → Merge both
└──────────────────────────────────┘
    ↓
Fraud Prediction
```

## 5.3 BRIGHT - Real-Time GNN

**Problema**: GNN latency (centenas de ms)

**Solução:**
1. **Two-Stage Graph**: Separa histórico vs real-time
2. **Lambda Neural Network**: Batch embeddings + real-time prediction

**Resultados:**
- **>75% redução P99 latência**
- **7.8x inference mais rápido**
- **>2% melhoria precision**

## 5.4 Hybrid MoE - Mixture of Experts

**Componentes:**
- RNNs: Captura sequências
- Transformers: Features high-order
- Autoencoders: Anomalias

**Performance:**
- **98.7% accuracy**
- **94.3% precision**

---

# PARTE 6: MÉTRICAS GLOBAIS DE FRAUDE

## 6.1 Estatísticas Globais 2024-2025

| Métrica | Valor | Fonte |
|---------|-------|-------|
| **Global card fraud** | $33.45B (2022) | Nilson Report |
| **EU/EEA fraud total** | €4.3B (2022) | ECB/EBA |
| **Card fraud rate (EU)** | 0.031% | ECB/EBA 2024 |
| **Cross-border fraud** | 71% de card fraud | ECB/EBA |
| **Average fraud value (card)** | €80 | ECB/EBA |
| **Average fraud value (transfer)** | €2,252 | ECB/EBA |
| **PIX Brazil (9 meses)** | 28M fraudes | BACEN |
| **PIX fraud rate** | 0.007% | BACEN |
| **US consumer losses** | $8.8B (2022) | FTC |
| **UK unauthorized** | £708.7M (2023) | UK Finance |

## 6.2 BIS/ECB Key Findings

**Tendências 2024-2025:**
- QR-code fraud: Manipulação IBAN, faturas falsas
- AI-enabled fraud: Social engineering escalado
- ATM skimming: Aumento significativo
- Instant payment fraud: Alto risco (janela curta de recuperação)

**Strong Customer Authentication (SCA):**
- Reduziu fraude significativamente em EU
- 10x mais fraude quando destinatário fora EEA

## 6.3 ACFE/SAS Technology Report 2024

**Adoção de tecnologia:**
- 90%+ organizações usam data analytics
- 18% usam AI/ML atualmente
- 32% planejam implementar em 2 anos
- 82% esperam usar GenAI até 2025

**Impacto real:**
- Um banco reduziu alerts em 40%
- Melhorou detecção em 35%

---

# PARTE 7: REGULAMENTAÇÃO

## 7.1 BACEN (Brasil)

**BCB 491 - Limites PIX:**
- Dispositivo não cadastrado: R$200
- Limite noturno (23h-5h): R$1.000

**Resolução 6 - Compartilhamento:**
- Dados de fraude entre bancos em 24h
- Base centralizada

**MED 2.0 (Fevereiro 2026):**
- Rastreabilidade aprimorada
- Bloqueio preventivo até 72h
- Devolução em até 96h
- Self-service dispute (Outubro 2025)

**CPF Blocking (Dezembro 2025):**
- Usuários podem bloquear CPF de novas contas

## 7.2 EU - PSD2 / AI Act

**PSD2:**
- SCA obrigatório
- Reporting de fraude

**AI Act (Agosto 2024):**
- Requisitos de transparência
- Explainability obrigatória

## 7.3 LGPD (Brasil)

**Art. 20 - Direito à Explicação:**
```python
lgpd_explanation = {
    "decision": "BLOCKED",
    "risk_score": 0.87,
    "factors": [
        "Transação noturna",
        "Novo dispositivo",
        "Valor acima da média"
    ],
    "user_rights": [
        "Solicitar revisão",
        "Acessar dados",
        "Solicitar exclusão"
    ]
}
```

---

# PARTE 8: IMPLEMENTAÇÃO PRÁTICA

## 8.1 Stack Recomendado para PIX

**Modelo**: LightGBM (latência 25ms)
**Dataset**: CiferAI (21M) + features BCB 491
**Backup**: Stacking Ensemble para análise offline

```python
# Pipeline completo
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Carregar dados
df = load_cifer_dataset()

# Adicionar features PIX
df['device_registered'] = ...
df['nocturnal_limit_exceeded'] = ...
df['recipient_is_pj'] = ...

# Treinar
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('is_fraud', axis=1), 
    df['is_fraud'], 
    test_size=0.2,
    stratify=df['is_fraud']
)

model = LGBMClassifier(
    is_unbalance=True,
    learning_rate=0.1,
    num_leaves=64,
    max_bin=63  # Reduz latência
)

model.fit(X_train, y_train)
```

## 8.2 Integração BioCatch-style

```python
# Behavioral biometrics features
BEHAVIORAL_FEATURES = {
    'typing': [
        'typing_speed_wpm',
        'keystroke_rhythm_std',
        'error_rate',
        'backspace_frequency'
    ],
    'mouse': [
        'mouse_speed_avg',
        'click_patterns',
        'scroll_behavior'
    ],
    'session': [
        'session_duration_seconds',
        'pages_visited',
        'idle_time_total'
    ]
}

def collect_behavioral_data(session):
    """Coleta dados comportamentais durante sessão"""
    return {
        'typing_speed': calculate_typing_speed(session.keystrokes),
        'mouse_fluency': calculate_mouse_fluency(session.mouse_events),
        'session_anomaly_score': detect_session_anomalies(session)
    }
```

## 8.3 Device Fingerprinting

```python
# SEON-style device signals
DEVICE_SIGNALS = {
    'hardware': ['device_model', 'os_version', 'screen_resolution'],
    'software': ['browser_type', 'timezone', 'language'],
    'network': ['ip_address', 'is_vpn', 'is_proxy', 'is_tor'],
    'risk': ['is_emulator', 'is_rooted', 'gps_spoofing']
}

def create_device_fingerprint(request):
    """Gera fingerprint único do dispositivo"""
    signals = collect_signals(request)
    return hash_signals(signals), calculate_risk_score(signals)
```

---

# PARTE 9: ROADMAP DE INTEGRAÇÃO

## Fase 1: Baseline (Semanas 1-2)

- [ ] Baixar dataset CiferAI (21M)
- [ ] Implementar features PIX (BCB 491)
- [ ] Treinar LightGBM baseline
- [ ] Deploy endpoint com latência < 50ms

## Fase 2: GNN (Semanas 3-4)

- [ ] Carregar Elliptic++ dataset
- [ ] Implementar GNN com PyTorch Geometric
- [ ] Detectar redes de fraude (mule accounts)
- [ ] Integrar embeddings com XGBoost

## Fase 3: Behavioral (Semanas 5-6)

- [ ] Coletar dados comportamentais (typing, mouse)
- [ ] Implementar scoring BioCatch-style
- [ ] Integrar device fingerprinting
- [ ] Adicionar ao ensemble

## Fase 4: Federated (Semanas 7-8)

- [ ] Configurar Flower framework
- [ ] Simular treinamento multi-banco
- [ ] Validar privacy (differential privacy)
- [ ] Compliance LGPD

## Fase 5: Production (Semanas 9-10)

- [ ] Monitoring Prometheus
- [ ] SLA validation (< 50ms PIX)
- [ ] Explainability (SHAP/LIME)
- [ ] Deploy final

---

# PARTE 10: REFERÊNCIAS COMPLETAS

## Datasets
- IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
- CiferAI: https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF
- Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- FDB (Amazon): https://github.com/amazon-science/fraud-dataset-benchmark
- OpenML 1597: https://www.openml.org/d/1597

## Papers
- RAGFormer: https://arxiv.org/abs/2402.17472
- BRIGHT: https://arxiv.org/abs/2205.13084
- Hybrid MoE: https://arxiv.org/abs/2504.03750
- PIX Taxonomy: https://arxiv.org/abs/2511.20902

## Plataformas
- AWS SageMaker: https://github.com/awslabs/sagemaker-graph-fraud-detection
- NVIDIA Blueprint: https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection
- Google FraudFinder: https://github.com/GoogleCloudPlatform/fraudfinder

## Vendors
- BioCatch: https://www.biocatch.com
- SEON: https://seon.io
- Fingerprint.com: https://fingerprint.com
- Feedzai: https://www.feedzai.com

## Regulamentação
- BACEN: https://www.bcb.gov.br
- ECB/EBA 2024: https://www.eba.europa.eu/sites/default/files/2024-08/465e3044-4773-4e9d-8ca8-b1cd031295fc/EBA_ECB%202024%20Report%20on%20Payment%20Fraud.pdf
- BIS Digital Fraud: https://www.bis.org/bcbs/publ/d558.pdf

---

*Documento gerado em: Novembro 2025*
*Versão: 3.0 (Completa e Expandida)*
*Total de recursos mapeados: 50+*
