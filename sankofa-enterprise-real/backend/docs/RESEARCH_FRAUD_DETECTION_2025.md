# Pesquisa Avançada: Detecção de Fraude Bancária 2025

## Sumário Executivo

Este documento consolida pesquisa extensiva sobre datasets, features, transfer learning e melhores práticas para detecção de fraude em transações PIX, Débito e Crédito no contexto brasileiro.

---

## 1. Datasets Disponíveis

### 1.1 Datasets Públicos Principais

| Dataset | Transações | Features | Taxa Fraude | Uso Recomendado |
|---------|------------|----------|-------------|-----------------|
| **Kaggle Credit Card Fraud** | 284K | 30 (V1-V28 + Time + Amount) | 0.17% | Baseline, prototipagem |
| **IEEE-CIS Fraud Detection** | 590K | 394 (Transaction + Identity) | 3.5% | Produção, competições |
| **PaySim** | 6.3M | 10 | 0.13% | Proxy para PIX (mobile money) |
| **Bank Account Fraud (NeurIPS)** | 1M | 30+ | Variável | Benchmark acadêmico |

### 1.2 IEEE-CIS Dataset - Detalhes

**Features de Transação:**
- `TransactionID`: ID único
- `TransactionDT`: Timedelta em segundos
- `TransactionAmt`: Valor em USD
- `ProductCD`: Código do produto (W, C, H, R, S)
- `card1-card6`: Informações do cartão
- `addr1, addr2`: Região/país de cobrança
- `P_emaildomain, R_emaildomain`: Domínios de email
- `C1-C14`: Features de contagem (anônimas)
- `D1-D15`: Features de timedelta (anônimas)
- `M1-M9`: Features de match (anônimas)
- `V1-V339`: 339 features Vesta (alto missing)

**Features de Identidade:**
- `id_01-id_11`: Ratings numéricos (device, IP, proxy)
- `id_12-id_38`: Categóricas (OS, browser, resolução)
- `DeviceType`: Desktop vs Mobile
- `DeviceInfo`: Modelo do dispositivo

### 1.3 PaySim como Proxy para PIX

PaySim simula transações de mobile money, tornando-o útil como proxy para PIX:
- Transações instantâneas P2P
- Padrões de comportamento similares
- Tipos: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER

### 1.4 Dados Sintéticos

**Técnicas recomendadas:**
- **CTGAN**: Generative Adversarial Network para dados tabulares
- **SMOTE-ENN**: Oversampling + cleaning
- **FinDiff**: Modelos de difusão para dados financeiros
- **TimeGAN**: Para séries temporais

---

## 2. Features para Detecção de Fraude

### 2.1 Features Essenciais por Tipo de Transação

#### PIX (Específico Brasil)
```python
PIX_FEATURES = {
    'regulatory': [
        'device_registered',      # BCB 491: dispositivo cadastrado
        'pix_key_type',           # CPF, CNPJ, email, telefone, aleatória
        'recipient_is_pj',        # 2/3 das fraudes vão para PJ
        'qr_code_used',           # QR dinâmico vs estático
        'first_pix_to_recipient', # Primeira transação para destinatário
    ],
    'velocity': [
        'pix_count_1h',
        'pix_count_24h',
        'pix_amount_24h',
        'distinct_recipients_24h',
    ],
    'behavioral': [
        'night_transaction',      # 23h-5h
        'weekend_transaction',
        'device_age_hours',
        'session_duration_seconds',
    ],
    'limits': [
        'amount_vs_nocturnal_limit',  # Limite noturno R$1.000
        'amount_vs_daily_limit',
        'unregistered_device_limit',  # R$200 para dispositivos não cadastrados
    ]
}
```

#### Crédito
```python
CREDIT_FEATURES = {
    'transaction': [
        'amount_normalized',
        'merchant_category_code',
        'entry_mode',             # Chip, stripe, contactless, online
        'is_international',
        'is_recurring',
    ],
    'card': [
        'card_age_days',
        'card_type',              # Credit, debit, prepaid
        'issuer_country',
        'is_chip_enabled',
    ],
    'velocity': [
        'tx_count_1h',
        'tx_count_24h',
        'distinct_merchants_24h',
        'amount_7d_rolling',
    ],
    'behavioral': [
        'avg_amount_30d',
        'std_amount_30d',
        'amount_zscore',
        'typical_hour_deviation',
    ]
}
```

#### Débito
```python
DEBIT_FEATURES = {
    'transaction': [
        'amount',
        'terminal_type',          # ATM, POS, online
        'is_contactless',
        'pin_verified',
    ],
    'account': [
        'account_age_days',
        'balance_ratio',          # amount / balance
        'overdraft_used',
    ],
    'velocity': [
        'withdrawals_24h',
        'distinct_atms_24h',
        'pos_tx_24h',
    ]
}
```

### 2.2 Feature Engineering Avançado

#### Aggregations por UID
```python
def create_uid_features(df):
    """
    UID = card1 + addr1 + D1 (identificador único de cliente)
    Técnica vencedora do IEEE-CIS Kaggle
    """
    df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    
    aggregations = {
        'amount': ['mean', 'std', 'min', 'max', 'count'],
        'C1': ['mean', 'nunique'],
        'D1': ['mean', 'std'],
    }
    
    for col, aggs in aggregations.items():
        for agg in aggs:
            feature_name = f'{col}_uid_{agg}'
            df[feature_name] = df.groupby('uid')[col].transform(agg)
    
    return df
```

#### Temporal Features
```python
def create_temporal_features(df, timestamp_col='timestamp'):
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df
```

#### Velocity Features
```python
def create_velocity_features(df, entity_col='customer_id', timestamp_col='timestamp'):
    df = df.sort_values([entity_col, timestamp_col])
    
    # Time since last transaction
    df['time_since_last'] = df.groupby(entity_col)[timestamp_col].diff().dt.total_seconds()
    
    # Rolling counts (1h, 24h windows)
    for window in ['1H', '24H']:
        df[f'tx_count_{window}'] = df.groupby(entity_col).rolling(
            window, on=timestamp_col, closed='left'
        ).count().reset_index(drop=True)
    
    return df
```

---

## 3. Transfer Learning e Modelos Pré-treinados

### 3.1 Técnicas de Transfer Learning

#### Autoencoder Pre-training
```python
from tensorflow import keras

def create_fraud_autoencoder(input_dim, encoding_dim=32):
    """
    Pré-treina autoencoder em dados não-rotulados
    Encoder é usado como feature extractor
    """
    input_layer = keras.layers.Input(shape=(input_dim,))
    
    # Encoder
    encoded = keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.Dense(64, activation='relu')(encoded)
    encoded = keras.layers.Dense(encoding_dim, activation='relu', name='embedding')(encoded)
    
    # Decoder
    decoded = keras.layers.Dense(64, activation='relu')(encoded)
    decoded = keras.layers.Dense(128, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = keras.Model(input_layer, decoded)
    encoder = keras.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder
```

#### GNN Embeddings (NVIDIA Blueprint)
```python
# Conceitual - baseado em NVIDIA Morpheus pipeline
class GNNEmbedding:
    """
    Graph Neural Network para embeddings de transações
    Captura relacionamentos entre entidades
    """
    def __init__(self):
        self.node_types = ['customer', 'device', 'ip', 'merchant', 'receiver']
        self.edge_types = ['transacts', 'uses_device', 'from_ip', 'at_merchant']
    
    def create_embeddings(self, transactions_df):
        """
        Cria embeddings baseados em:
        1. Vizinhança do nó (1-hop, 2-hop)
        2. Padrões temporais
        3. Features de centralidade
        """
        # Implementação com PyTorch Geometric ou DGL
        pass
```

### 3.2 Federated Learning (Multi-Instituição)

#### Arquitetura Google Cloud + SWIFT
```
┌─────────────────────────────────────────────────┐
│         FEDERATED LEARNING WORKFLOW              │
└─────────────────────────────────────────────────┘

1. LOCAL MODEL TRAINING
   ├─ Cada banco treina em dados proprietários
   ├─ Dados nunca saem da instituição
   └─ Computação local de gradientes

2. SECURE MODEL AGGREGATION  
   ├─ Updates encriptados enviados ao servidor central
   ├─ Uso de FedAvg ou FedProx
   └─ Protocolos de agregação segura

3. GLOBAL MODEL DISTRIBUTION
   ├─ Modelo agregado compartilhado com participantes
   ├─ Detecção de fraude melhorada para todos
   └─ Ciclo de aprendizado contínuo

4. PRIVACY ENHANCEMENTS
   ├─ Differential Privacy (injeção de ruído)
   ├─ Secure Multi-Party Computation (SMPC)
   └─ Homomorphic Encryption
```

#### Implementação com Flower
```python
import flwr as fl

class BankClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
        return loss, len(self.x_train), {"accuracy": accuracy}

# Servidor
strategy = fl.server.strategy.FedAvg(
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy
)
```

---

## 4. Comparação de Modelos (Benchmarks 2025)

### 4.1 Performance Summary

| Modelo | F1 Score | AUC | Latência | Melhor Para |
|--------|----------|-----|----------|-------------|
| **CatBoost** | 0.9161 | 0.94+ | Média | Accuracy, features categóricas |
| **LightGBM** | 0.80 | 0.95 | **Mais Baixa** | Real-time, alta frequência |
| **XGBoost** | 0.80 | 0.94+ | Alta | Flexibilidade, tuning |
| **Stacking Ensemble** | 0.99 | 0.99 | - | Máxima accuracy |

### 4.2 Recomendações por Caso de Uso

**PIX (50ms latência requerida):**
- Primary: LightGBM (25-30% menor latência)
- Fallback: CatBoost com otimizações

**Crédito (batch processing OK):**
- Primary: Stacking Ensemble (XGB + LGB + CAT)
- Secondary: CatBoost standalone

**Débito (tempo real):**
- Primary: LightGBM
- Secondary: XGBoost com early stopping

### 4.3 Configurações Otimizadas

#### CatBoost Otimizado
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_columns,
    eval_metric='AUC',
    od_type='Iter',
    od_wait=20,
    task_type='GPU',  # Se disponível
    random_seed=42
)
```

#### LightGBM Otimizado para Latência
```python
import lightgbm as lgb

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 64,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'device': 'gpu',
    'max_bin': 63,  # Reduz latência
}
```

#### Stacking Ensemble
```python
from sklearn.ensemble import StackingClassifier

stacked_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=400)),
        ('lgb', LGBMClassifier(n_estimators=400)),
        ('cat', CatBoostClassifier(iterations=400, verbose=0))
    ],
    final_estimator=XGBClassifier(n_estimators=100),
    cv=5
)
```

---

## 5. Padrões de Fraude PIX (Taxonomia 2025)

### 5.1 Metodologias de Ataque (arXiv:2511.20902)

**Categoria 1: Engenharia Social**
- Golpe do falso funcionário de banco
- Falso sequestro
- Golpe do PIX errado
- Romance scam

**Categoria 2: Técnicas Híbridas (IA-Amplified)**
- Deepfake para autenticação biométrica
- Voice cloning para confirmação telefônica
- Phishing personalizado com LLMs

**Categoria 3: Exploração Técnica**
- Malware de interceptação (RAT)
- SIM swap
- Exploração de APIs
- Manipulação de QR codes

### 5.2 Indicadores de Risco PIX

```python
PIX_RISK_INDICATORS = {
    'high_risk': [
        'first_pix_to_pj',           # Primeira vez enviando para PJ
        'night_high_value',           # Valor alto (>R$2k) à noite
        'new_device',                 # Dispositivo não cadastrado
        'velocity_spike',             # Muitas transações em curto período
        'recipient_high_fraud_rate',  # Destinatário com histórico
    ],
    'medium_risk': [
        'unusual_amount',             # Valor fora do padrão
        'different_location',         # IP/localização diferente
        'weekend_transaction',        # Fim de semana
    ],
    'regulatory': [
        'exceeds_nocturnal_limit',    # Excede limite noturno R$1k
        'unregistered_device_limit',  # Excede limite R$200
    ]
}
```

---

## 6. Conformidade Regulatória

### 6.1 BACEN (Banco Central do Brasil)

**Normativa BCB 491:**
- Cadastro obrigatório de dispositivos
- Limite de R$200 para dispositivos não cadastrados
- Limite noturno de R$1.000 (personalizável)

**Resolução 6:**
- Compartilhamento de dados de fraude entre instituições
- Base de dados centralizada de fraudes
- Prazo de 24h para comunicação

**MED 2.0 (Fevereiro 2026):**
- Rastreabilidade aprimorada de PIX
- Bloqueio preventivo obrigatório
- Devolução em até 96h

### 6.2 LGPD (Lei Geral de Proteção de Dados)

**Artigo 20 - Explainability:**
```python
def generate_lgpd_explanation(prediction, features):
    """
    Gera explicação LGPD-compliant para decisão de fraude
    """
    explanation = {
        'decision': 'BLOCKED' if prediction.is_fraud else 'APPROVED',
        'risk_score': prediction.risk_score,
        'main_factors': [
            f"Valor da transação: R$ {features.amount:.2f}",
            f"Horário: {'suspeito (noturno)' if features.is_night else 'normal'}",
            f"Dispositivo: {'novo' if features.is_new_device else 'conhecido'}",
        ],
        'user_rights': [
            'Você pode solicitar revisão desta decisão',
            'Você pode acessar os dados usados na análise',
            'Você pode solicitar exclusão dos dados'
        ],
        'contact': 'ouvidoria@banco.com.br'
    }
    return explanation
```

### 6.3 PCI DSS

**Requisitos implementados:**
- Mascaramento de dados sensíveis (CPF, cartão)
- Logs estruturados sem dados pessoais
- Criptografia AES-256 para dados em repouso
- TLS 1.3 para dados em trânsito

---

## 7. Métricas de Produção

### 7.1 KPIs Principais

| Métrica | Target | Descrição |
|---------|--------|-----------|
| **Latência P50** | < 30ms | Mediana de tempo de resposta |
| **Latência P99** | < 50ms | Percentil 99 (Bradesco standard) |
| **TPS** | > 3,500 | Transações por segundo |
| **Recall** | > 90% | Detectar 90%+ das fraudes |
| **Precision** | > 70% | Minimizar falsos positivos |
| **F1 Score** | > 80% | Balanço precision/recall |
| **False Positive Rate** | < 0.5% | Transações legítimas bloqueadas |

### 7.2 Monitoramento

```python
PROMETHEUS_METRICS = {
    'fraud_detection_latency_seconds': 'Histogram - tempo de predição',
    'fraud_predictions_total': 'Counter - predições por tipo (fraud/legit)',
    'fraud_risk_score': 'Histogram - distribuição de scores',
    'model_accuracy_gauge': 'Gauge - accuracy em tempo real',
    'sla_compliance_ratio': 'Gauge - % de requests < 50ms',
}
```

---

## 8. Referências

### Papers Acadêmicos
1. arXiv:2511.20902 - "A Taxonomy of Pix Fraud in Brazil" (Nov 2025)
2. MDPI - "Secure and Transparent Banking: Explainable AI-Driven Federated Learning" (Mar 2025)
3. Nature - "Bank data protection with adaptive federated learning and WGAN" (Jul 2025)
4. MDPI - "FinGraphFL: Financial Graph-Based Federated Learning" (Apr 2025)

### Datasets
- Kaggle IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
- PaySim: https://github.com/EdgarLopezPhD/PaySim
- Bank Account Fraud: NeurIPS 2022

### Frameworks
- TensorFlow Federated: https://www.tensorflow.org/federated
- Flower: https://flower.dev/
- NVIDIA Morpheus: https://developer.nvidia.com/morpheus-cybersecurity
- CatBoost: https://catboost.ai/
- LightGBM: https://lightgbm.readthedocs.io/

---

*Documento gerado em: Novembro 2025*
*Versão: 1.0*
*Autor: Sankofa Enterprise Pro Team*
