# Hub de Recursos - Detecção de Fraude PIX/Débito/Crédito
## Datasets Globais, Modelos Hugging Face e Métricas de Produção

---

## 1. DATASETS HUGGING FACE

### 1.1 Datasets com Código Pronto

| Dataset | Transações | Features | Tipo | Link |
|---------|------------|----------|------|------|
| **CiferAI/Cifer-Fraud-Detection** | 21M | 14+ | Móvel (proxy PIX) | https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF |
| **keras-io/imbalanced_classification** | 284K | 30 | Crédito | https://huggingface.co/datasets/liberatoratif/Credit-card-fraud-detection |
| **kmasiak/FraudDetection** | 532K | 9 | Contábil | https://huggingface.co/datasets/kmasiak/FraudDetection |
| **amitkedia/Financial-Fraud-Dataset** | 170 | Múltiplos | SEC filings | https://huggingface.co/datasets/amitkedia/Financial-Fraud-Dataset |

### 1.2 Dataset CiferAI (RECOMENDADO PARA PIX)

```python
from datasets import load_dataset

# Carrega dataset 21M transações (mobile money - similar ao PIX)
dataset = load_dataset(
    "CiferAI/Cifer-Fraud-Detection-Dataset-AF",
    data_files="Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv"
)

# Para treinamento federado (4 partições)
train_splits = []
for i in range(1, 5):
    part = load_dataset(
        "CiferAI/Cifer-Fraud-Detection-Dataset-AF",
        data_files=f"Cifer-Fraud-Detection-Dataset-AF-part-{i}-14.csv"
    )
    train_splits.append(part)

# Características do dataset:
# - 99.93% accuracy em benchmark real-world
# - Federated learning ready
# - Padrões similares ao PIX (transferências instantâneas P2P)
# - Altamente desbalanceado (fraude ~0.1%)
```

### 1.3 Integração com Sankofa

```python
# Em: backend/ml_engine/data_loader.py

def load_huggingface_dataset(dataset_name="CiferAI/Cifer-Fraud-Detection-Dataset-AF"):
    """
    Carrega dataset do Hugging Face e prepara para treinamento
    """
    from datasets import load_dataset
    import pandas as pd
    
    dataset = load_dataset(dataset_name)
    df = dataset['train'].to_pandas()
    
    # Mapeamento de features para Sankofa
    feature_mapping = {
        'amount': 'amount',
        'time': 'timestamp',
        'merchant': 'merchant_id',
        'customer': 'customer_id',
        'fraud': 'is_fraud'
    }
    
    df_mapped = df.rename(columns=feature_mapping)
    return df_mapped
```

---

## 2. MODELOS PRÉ-TREINADOS HUGGING FACE

### 2.1 Modelos Disponíveis

| Modelo | Base | Caso de Uso | Acurácia | Link |
|--------|------|-----------|----------|------|
| **CiferAI/cifer-fraud-detection-k1-a** | Binary | General fraud | 99.93% | https://huggingface.co/CiferAI/cifer-fraud-detection-k1-a |
| **keras-io/imbalanced_classification** | DNN | Credit card | 99.82% fraud recall | https://huggingface.co/keras-io/imbalanced_classification |
| **kmasiak/FraudDetection** | VAE-GAN | Anomalias | - | https://huggingface.co/kmasiak/FraudDetection |
| **Mistral-7B-LLM-Fraud-Detection** | Mistral-7B | Análise transcripts | - | https://huggingface.co/Bilic/Mistral-7B-LLM-Fraud-Detection |

### 2.2 Carregar Modelo Pré-treinado

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Usar modelo CiferAI pre-treinado
model = AutoModelForSequenceClassification.from_pretrained(
    "CiferAI/cifer-fraud-detection-k1-a"
)

# Ou usar modelo Keras com imbalanced data handling
model = AutoModelForSequenceClassification.from_pretrained(
    "keras-io/imbalanced_classification"
)

# Usar para predição
def predict_fraud(input_features, model):
    """
    Predição em tempo real usando modelo pré-treinado
    """
    inputs = tokenizer(str(input_features), return_tensors="pt")
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=1)
    fraud_probability = probabilities[0, 1].item()
    return fraud_probability > 0.5, fraud_probability
```

---

## 3. DATASETS STANFORD SNAP (Redes de Fraude)

### 3.1 Bitcoin Alpha/OTC Trust Network

```
URL: http://snap.stanford.edu/data/soc-sign-bitcoinalpha.html

Características:
- Rede de confiança assinada (Who-trusts-whom)
- Escala: -10 (desconfiança total) a +10 (confiança total)
- Use case: Detecção de risco com base em reputação
- Formato: Rede dirigida com pesos
```

### 3.2 Elliptic Dataset (Blockch

ain Fraud)

**Principal dataset de fraude financeira com estrutura de rede**

```
URL: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Especificações:
- Transações: 203,769 nós
- Arestas: 234,355 (fluxos de Bitcoin)
- Time steps: 49 (a cada 3 horas)
- Valor total: ~$6 bilhões

Classes:
- Ilícitas: 4,545 (2%)
- Lícitas: 42,019 (21%)
- Unlabeled: 157,205 (77%)

Features:
- 166 features por transação (anonimizadas)
- Estrutura temporal (DAG - Directed Acyclic Graph)
```

### 3.3 Elliptic++ (Extended)

```
URL: https://github.com/git-disl/EllipticPlusPlus

Expansão do Elliptic:
- Transações: 203K
- Wallets/Addresses: 822K com 56 features cada
- Interações temporais: 1.27M

4 Tipos de Grafos:
1. Transaction-to-transaction (fluxo de dinheiro)
2. Address-to-address (interação)
3. Bipartite address-transaction
4. User entity graph (endereços agrupados)

Arquivos:
- txs_features.csv (features de transação)
- txs_classes.csv (labels)
- txs_edgelist.csv (grafo de transação)
- wallets_features.csv (features de carteira)
- wallets_classes.csv (labels de carteira)
```

### 3.4 Código de Integração - GNN para Sankofa

```python
# Em: backend/ml_engine/graph_integration.py

import pandas as pd
import networkx as nx
from torch_geometric.data import Graph

def load_elliptic_dataset():
    """
    Carrega dataset Elliptic e prepara como grafo para GNN
    """
    # Features
    features = pd.read_csv('elliptic_txs_features.csv', header=None)
    # Coluna 0: txId, Coluna 1: timestep, Colunas 2-166: features
    
    # Labels
    labels = pd.read_csv('elliptic_txs_classes.csv')
    # txId, class (1=illicit, 2=licit, unknown)
    
    # Grafo
    edges = pd.read_csv('elliptic_txs_edgelist.csv')
    
    # Merge
    df = features.merge(labels, left_on=0, right_on='txId', how='left')
    df['class'] = df['class'].fillna('unknown')
    
    # Criar PyG Graph
    edge_index = torch.LongTensor([
        edges['txId1'].values,
        edges['txId2'].values
    ])
    
    node_features = torch.FloatTensor(features.iloc[:, 2:].values)
    node_labels = torch.LongTensor(
        df['class'].map({'1': 1, '2': 0, 'unknown': -1}).values
    )
    
    return Graph(x=node_features, edge_index=edge_index, y=node_labels)
```

---

## 4. PAPERS WITH CODE - BENCHMARKS

### 4.1 Datasets Consolidados

| Dataset | Records | Fraude% | Melhor Modelo | AUC |
|---------|---------|---------|--------------|-----|
| **IEEE-CIS** | 590K | 3.5% | Stacking Ensemble | 0.99 |
| **Credit Card ULB** | 284K | 0.17% | Random Forest | 0.95+ |
| **BankSim** | 594K | 1.2% | XGBoost | 0.92 |
| **FDB Benchmark** | Múltiplos | Variável | AutoML | Varia |

### 4.2 Modelos SOTA (State-of-the-Art)

```python
# Stacking Ensemble (99% accuracy)
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

stacked = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=400)),
        ('lgb', LGBMClassifier(n_estimators=400)),
        ('cat', CatBoostClassifier(iterations=400, verbose=0))
    ],
    final_estimator=XGBClassifier(n_estimators=100),
    cv=5
)

# Treinar
stacked.fit(X_train, y_train)

# Performance: 99% accuracy, 0.99 AUC
y_pred_proba = stacked.predict_proba(X_test)
```

### 4.3 Awesome Fraud Detection Papers

**Repositório com 100+ papers curados:**
https://github.com/benedekrozemberczki/awesome-fraud-detection-papers

Organizado por:
- Ano de publicação
- Conferência (KDD, SIGIR, WWW, AAAI)
- Técnica (GNN, RNN, Ensemble)

### 4.4 Fraud Dataset Benchmark (FDB - Amazon)

```python
# Instalação
pip install fraud-dataset-benchmark

# Uso
from fdb import load_dataset

# Carregar dataset
X_train, y_train, X_test, y_test = load_dataset('ieeecis')

# Treinar XGBoost com imbalance handling
from xgboost import XGBClassifier
model = XGBClassifier(
    scale_pos_weight=sum(y_train==0)/sum(y_train==1),
    eval_metric='auc'
)

# Avaliar
from sklearn.metrics import roc_auc_score
predictions = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, predictions)
```

---

## 5. RECURSOS GOVERNAMENTAIS & REGULATÓRIOS

### 5.1 World Bank Fast Payments Report (PIX)

**URL**: https://fastpayments.worldbank.org/sites/default/files/2023-10/Fraud%20in%20Fast%20Payments_Final.pdf

**Foco**: Fraude em sistemas de pagamento instantâneo (Brasil PIX, Índia UPI)

**Métricas Chave**:
- Brasil: 28 milhões de fraudes reportadas (Jan-Set 2025)
- Taxa oficial PIX: 0.007% (7 fraudes por 100K transações)
- Limite autorizado sem registro: R$200
- Limite noturno (23h-5h): R$1.000

### 5.2 EBA/ECB Report 2024

**URL**: https://www.eba.europa.eu/sites/default/files/2024-08/465e3044-4773-4e9d-8ca8-b1cd031295fc/EBA_ECB%202024%20Report%20on%20Payment%20Fraud.pdf

**Estatísticas EU/EEA**:
- Fraude em cartões: €633M (H1 2023)
- Transações online: crescimento de fraude

### 5.3 Métricas Globais de Referência

| Métrica | Valor | Fonte |
|---------|-------|-------|
| **Global card fraud** | $33.45B (2022) | Nilson Report |
| **US card fraud** | $12B | US Regulators |
| **UK unauthorized** | £708.7M (2023) | UK Finance |
| **Fraud rate** | 0.05% (~5/10K TX) | Global average |
| **PIX Brazil** | 28M fraudes (2025) | BACEN |

---

## 6. INTEGRAÇÃO SANKOFA - CHECKLIST

### 6.1 Adicionar Datasets Hugging Face

```bash
# Em: backend/ml_engine/data_loader.py

# 1. Instalar library
pip install datasets transformers

# 2. Carregar dataset
from datasets import load_dataset

dataset = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF")

# 3. Converter para format Sankofa
# Mapear features para schema do banco
```

### 6.2 Integrar Modelos Pré-treinados

```python
# Em: backend/api/production_api.py

from transformers import AutoModelForSequenceClassification

# Carregar modelo pré-treinado
fraud_model = AutoModelForSequenceClassification.from_pretrained(
    "CiferAI/cifer-fraud-detection-k1-a"
)

# Usar em endpoint de predição
@app.route('/api/fraud-prediction', methods=['POST'])
def predict_fraud_huggingface():
    data = request.json
    prediction = fraud_model.predict(data['features'])
    return {'fraud_probability': prediction}
```

### 6.3 Implementar GNN com Stanford SNAP

```python
# Em: backend/ml_engine/gnn_fraud_detector.py

# Carregar Elliptic dataset
from torch_geometric.datasets import EllipticBitcoinDataset

dataset = EllipticBitcoinDataset()
graph_data = dataset[0]

# Usar em modelo GNN
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(166, 128)
        self.conv2 = GCNConv(128, 64)
        self.classifier = torch.nn.Linear(64, 2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x)
```

### 6.4 Métricas de Produção

```python
# Em: backend/metrics/prometheus_metrics.py

from prometheus_client import Histogram, Counter, Gauge

# Latência
fraud_detection_latency = Histogram(
    'fraud_detection_latency_seconds',
    'Latência de detecção de fraude',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]  # PIX target: <50ms = 0.05s
)

# Acurácia
model_accuracy = Gauge(
    'model_accuracy_gauge',
    'Acurácia do modelo em tempo real'
)

# Recall (detectar fraudes)
fraud_recall = Gauge(
    'fraud_recall_gauge',
    'Recall: % de fraudes detectadas',
    value=0.90  # Target: >90%
)

# False positive rate
false_positive_rate = Gauge(
    'false_positive_rate',
    'Taxa de falsos positivos',
    value=0.005  # Target: <0.5%
)
```

---

## 7. ROADMAP DE INTEGRAÇÃO

### Fase 1: Dados (Semana 1-2)

- [ ] Baixar dataset CiferAI (21M registros)
- [ ] Importar para banco PostgreSQL
- [ ] Criar scripts de preprocessing
- [ ] Validar schema com features.py

### Fase 2: Modelos (Semana 2-3)

- [ ] Carregar modelos Hugging Face pré-treinados
- [ ] Implementar endpoint de predição
- [ ] Integrar Stacking Ensemble (99% accuracy)
- [ ] Testes com IEEE-CIS dataset

### Fase 3: GNN (Semana 3-4)

- [ ] Baixar Elliptic++ dataset
- [ ] Implementar GNN com PyTorch Geometric
- [ ] Integrar com fraud_engine.py
- [ ] Testar detecção de redes de fraude

### Fase 4: Federado (Semana 4-5)

- [ ] Configurar Flower para aprendizado federado
- [ ] Implementar multi-bank training
- [ ] Testes de privacidade
- [ ] Compliance com LGPD

### Fase 5: Produção (Semana 5-6)

- [ ] Performance testing (50ms latência PIX)
- [ ] Monitoring com Prometheus
- [ ] SLA monitoring
- [ ] Deployment containerizado

---

## 8. RECURSOS ADICIONAIS

### Buscar Mais Datasets

- **Google Dataset Search**: https://datasetsearch.research.google.com
- **UCI ML Repository**: https://archive.ics.uci.edu
- **AWS Open Data**: https://registry.opendata.aws
- **OpenML**: https://www.openml.org

### Comunidades

- **Papers with Code**: https://paperswithcode.com/task/fraud-detection
- **Kaggle Competitions**: https://www.kaggle.com/competitions?search=fraud
- **ArXiv**: https://arxiv.org (buscar: fraud detection 2025)

### Blogs & Articles

- **Elliptic Blog**: https://www.elliptic.co/blog
- **Feedzai**: https://www.feedzai.com/blog
- **BioCatch**: https://www.biocatch.com

---

**Próximas ações**: Implemente as Fases 1-2 deste roadmap para adicionar datasets e modelos Hugging Face ao Sankofa!
