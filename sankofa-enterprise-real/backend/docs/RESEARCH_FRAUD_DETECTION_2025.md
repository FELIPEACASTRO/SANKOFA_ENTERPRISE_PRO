# Pesquisa Avançada: Detecção de Fraude Bancária 2025
## Datasets, Features e Transfer Learning para PIX, Débito e Crédito

---

## Sumário Executivo

Este documento consolida pesquisa extensiva sobre datasets, features, transfer learning e melhores práticas para detecção de fraude em transações PIX, Débito e Crédito no contexto brasileiro.

**Última Atualização**: Novembro 2025

---

# PARTE 1: DATASETS

## 1.1 Catálogo Completo de Datasets

### Datasets Públicos Principais

| Dataset | Transações | Features | Taxa Fraude | Formato | Uso Recomendado |
|---------|------------|----------|-------------|---------|-----------------|
| **IEEE-CIS Fraud Detection** | 590K | 394 | 3.5% | CSV | Produção, competições |
| **Credit Card Fraud 2023** | 550K+ | 30 | 0.17% | CSV | Baseline atualizado |
| **Kaggle Credit Card (ULB)** | 284K | 30 | 0.17% | CSV | Prototipagem |
| **Bank Account Fraud (NeurIPS)** | 6M | 32 | Variável | Parquet | Fairness, bias testing |
| **PaySim** | 6.3M | 10 | 0.13% | CSV | Proxy para PIX |
| **Bank Transaction Fraud** | 20K | 12 | Binário | CSV | Testes rápidos |
| **UK Financial Fraud** | Variável | 15+ | Variável | CSV | Fintech UK |

### IEEE-CIS Dataset - Descrição Completa

**Features de Transação (Transaction.csv):**
```
TransactionID      - ID único
TransactionDT      - Timedelta em segundos desde referência
TransactionAmt     - Valor em USD
ProductCD          - Código produto (W, C, H, R, S)
card1-card6        - Info do cartão (tipo, categoria, banco, país)
addr1, addr2       - Região/país de cobrança
P_emaildomain      - Domínio email comprador
R_emaildomain      - Domínio email destinatário
C1-C14             - Features de contagem (anônimas)
D1-D15             - Features de timedelta (anônimas)
M1-M9              - Features de match (anônimas)
V1-V339            - 339 features Vesta engenharia (alto missing)
```

**Features de Identidade (Identity.csv):**
```
id_01-id_11        - Ratings numéricos (device, IP, proxy)
id_12-id_38        - Categóricas (OS, browser, resolução)
DeviceType         - Desktop vs Mobile
DeviceInfo         - Modelo do dispositivo
```

**Técnica Vencedora (UID):**
```python
# Identificador único de cliente (Kaggle Top 5%)
df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['D1'].astype(str)
```

### Bank Account Fraud (BAF) - NeurIPS 2022

**Criador**: Feedzai Research
**Técnica**: CTGAN + Differential Privacy

| Característica | Valor |
|----------------|-------|
| Total Records | 6M (1M por variante) |
| Variantes | 6 (1 base + 5 com bias) |
| Features | 32 atributos |
| Protected Attrs | age, income, employment_status |
| Split Temporal | 6 meses treino / 2 meses teste |

**Download:**
- Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- GitHub: https://github.com/feedzai/bank-account-fraud

### PaySim - Proxy para PIX

Simula transações de mobile money, útil como proxy para PIX:
- Transações instantâneas P2P
- Tipos: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- Padrões comportamentais similares ao PIX

### Amazon Fraud Dataset Benchmark (FDB)

Framework unificado para benchmarks:
- Loaders padronizados
- Train/test splits
- Métricas de avaliação
- Suporte múltiplos tipos de fraude

GitHub: https://github.com/amazon-science/fraud-dataset-benchmark

---

## 1.2 Dados para PIX (Brasil)

### Situação Atual

**NÃO existe dataset público de PIX do BACEN**. Alternativas:

1. **PaySim como Proxy** - Mobile money similar ao PIX
2. **Dados Sintéticos** - CTGAN, SMOTE, VAE
3. **Parcerias** - Instituições financeiras brasileiras

### Features PIX Recomendadas

Baseado em regulamentação BACEN:

```python
PIX_FEATURES = {
    'regulatory': [
        'device_registered',      # BCB 491: dispositivo cadastrado
        'pix_key_type',           # CPF, CNPJ, email, telefone, aleatória
        'recipient_is_pj',        # 2/3 das fraudes vão para PJ
        'qr_code_used',           # QR dinâmico vs estático
        'first_pix_to_recipient', # Primeira transação para destinatário
        'mule_account_indicator', # Indicador conta laranja
    ],
    'dict_features': [
        'dict_fraud_marker',      # Marcador de fraude no DICT
        'dict_registration_date', # Data registro chave PIX
        'cpf_status',             # Status CPF na Receita Federal
    ],
    'limits': [
        'amount_vs_nocturnal_limit',  # Limite noturno R$1.000
        'amount_vs_daily_limit',       # Limite diário
        'unregistered_device_limit',   # R$200 para dispositivos não cadastrados
    ]
}
```

### Estatísticas PIX 2025

- **28 milhões de fraudes** reportadas (Jan-Set 2025)
- **Taxa oficial**: 0.007% (7 fraudes por 100K transações)
- **Resolução 6**: Compartilhamento de dados de fraude entre instituições
- **MED 2.0** (Fev 2026): Rastreabilidade aprimorada

---

# PARTE 2: FEATURES

## 2.1 Features por Tipo de Transação

### CRÉDITO

```python
CREDIT_FEATURES = {
    'transaction': [
        'amount', 'amount_log', 'amount_normalized',
        'merchant_category_code', 'merchant_reputation',
        'entry_mode',             # Chip, stripe, contactless, online
        'is_international', 'is_recurring', 'is_card_present',
    ],
    'card': [
        'card_age_days', 'card_type', 'card_issuer',
        'issuer_country', 'is_chip_enabled',
        'card_present_vs_not_present',
    ],
    'velocity': [
        'tx_count_1h', 'tx_count_24h', 'tx_count_7d',
        'distinct_merchants_24h', 'distinct_countries_24h',
        'amount_7d_rolling', 'amount_30d_rolling',
    ],
    'behavioral': [
        'avg_amount_30d', 'std_amount_30d',
        'amount_zscore', 'typical_hour_deviation',
        'typical_merchant_category',
    ],
    'device': [
        'device_fingerprint', 'browser_type', 'os_version',
        'ip_address', 'ip_reputation', 'is_vpn', 'is_proxy',
    ]
}
```

### DÉBITO / ATM / POS

```python
DEBIT_FEATURES = {
    'transaction': [
        'amount', 'terminal_type',  # ATM, POS, online
        'is_contactless', 'pin_verified',
        'terminal_id', 'terminal_risk_score',  # CPP detection
    ],
    'account': [
        'account_age_days', 'balance_ratio',  # amount / balance
        'overdraft_used', 'account_type',
    ],
    'atm_specific': [
        'atm_location_type',      # Indoor bank vs outdoor
        'skimmer_risk_score', 'keypad_anomaly',
        'cash_withdrawal_pattern',
    ],
    'pos_specific': [
        'pos_terminal_id', 'compromised_terminal_flag',
        'merchant_reputation', 'transaction_speed',
    ],
    'velocity': [
        'withdrawals_24h', 'distinct_atms_24h',
        'pos_tx_24h', 'distinct_merchants_24h',
    ]
}
```

### PIX (Brasil)

```python
PIX_FEATURES = {
    'transaction': [
        'amount', 'amount_log', 'pix_key_type',
        'qr_code_type',           # Dinâmico vs estático
        'recipient_type',         # PF vs PJ (2/3 fraudes vão para PJ)
        'first_pix_to_recipient',
    ],
    'regulatory': [
        'device_registered',      # BCB 491
        'nocturnal_limit_check',  # R$1.000 23h-5h
        'unregistered_device_limit',  # R$200
    ],
    'velocity': [
        'pix_count_1h', 'pix_count_24h',
        'pix_amount_24h', 'distinct_recipients_24h',
        'pix_to_pj_24h',         # Transações para PJ
    ],
    'behavioral': [
        'night_transaction',      # 23h-5h
        'weekend_transaction',
        'device_age_hours', 'session_duration',
        'typical_amount_deviation',
    ],
    'risk_indicators': [
        'dict_fraud_marker',      # Marcador no DICT
        'mule_account_score',     # Score conta laranja
        'recipient_risk_score',
    ]
}
```

## 2.2 Feature Engineering Avançado

### Aggregations por UID

```python
def create_uid_features(df, uid_cols=['card1', 'addr1', 'D1']):
    """
    Cria UID e agregações (Técnica Top 5% Kaggle IEEE-CIS)
    """
    df['uid'] = df[uid_cols].astype(str).agg('_'.join, axis=1)
    
    aggregations = {
        'amount': ['mean', 'std', 'min', 'max', 'count', 'sum'],
        'C1': ['mean', 'nunique'],
        'C13': ['mean', 'sum'],
        'D1': ['mean', 'std'],
    }
    
    for col, aggs in aggregations.items():
        for agg in aggs:
            feature_name = f'{col}_uid_{agg}'
            df[feature_name] = df.groupby('uid')[col].transform(agg)
    
    return df
```

### Temporal Features (Cyclical Encoding)

```python
def create_temporal_features(df, timestamp_col='timestamp'):
    """
    Features temporais com encoding cíclico
    """
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    
    # Categorias
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_business_hours'] = (
        (df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5)
    ).astype(int)
    
    # Cyclical encoding (preserva continuidade)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df
```

### Velocity Features (Rolling Windows)

```python
def create_velocity_features(df, entity_col='customer_id', timestamp_col='timestamp'):
    """
    Features de velocidade com janelas deslizantes
    """
    df = df.sort_values([entity_col, timestamp_col])
    
    # Tempo desde última transação
    df['time_since_last'] = df.groupby(entity_col)[timestamp_col].diff().dt.total_seconds()
    df['time_since_last_hours'] = df['time_since_last'] / 3600
    
    # Contagens em janelas
    for window in ['1H', '6H', '24H', '7D']:
        df[f'tx_count_{window}'] = df.groupby(entity_col).apply(
            lambda x: x.rolling(window, on=timestamp_col, closed='left').count()
        ).reset_index(drop=True)
        
        df[f'tx_amount_{window}'] = df.groupby(entity_col).apply(
            lambda x: x['amount'].rolling(window, on=timestamp_col, closed='left').sum()
        ).reset_index(drop=True)
    
    # Destinatários únicos
    df['distinct_recipients_24h'] = df.groupby(entity_col)['recipient_id'].transform(
        lambda x: x.rolling('24H', on=timestamp_col).apply(lambda y: y.nunique())
    )
    
    return df
```

### Behavioral Features (Z-Score)

```python
def create_behavioral_features(df, entity_col='customer_id'):
    """
    Features comportamentais com desvio do padrão
    """
    # Média e desvio histórico
    df['avg_amount_30d'] = df.groupby(entity_col)['amount'].transform(
        lambda x: x.rolling('30D').mean()
    )
    df['std_amount_30d'] = df.groupby(entity_col)['amount'].transform(
        lambda x: x.rolling('30D').std()
    )
    
    # Z-Score (desvio do padrão)
    df['amount_zscore'] = (
        (df['amount'] - df['avg_amount_30d']) / (df['std_amount_30d'] + 1e-6)
    )
    
    # Hora típica
    df['typical_hour'] = df.groupby(entity_col)['hour'].transform('median')
    df['hour_deviation'] = abs(df['hour'] - df['typical_hour'])
    
    # Merchant típico
    df['typical_merchant'] = df.groupby(entity_col)['merchant_category'].transform(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    )
    df['is_unusual_merchant'] = (df['merchant_category'] != df['typical_merchant']).astype(int)
    
    return df
```

### Device Fingerprinting Features

```python
DEVICE_FINGERPRINT_FEATURES = {
    'hardware': [
        'device_model', 'os_version', 'screen_resolution',
        'battery_level', 'storage_capacity', 'sensor_data',
        'phone_orientation', 'gps_location',
    ],
    'software': [
        'browser_type', 'browser_version', 'installed_apps',
        'language_settings', 'timezone', 'user_agent',
    ],
    'network': [
        'ip_address', 'ip_reputation', 'is_proxy', 'is_vpn',
        'is_tor', 'wifi_vs_mobile', 'connection_type',
    ],
    'derived': [
        'device_id', 'device_age_days', 'is_new_device',
        'devices_per_user', 'users_per_device',
        'is_emulator', 'is_rooted', 'is_jailbroken',
    ]
}
```

---

# PARTE 3: TRANSFER LEARNING

## 3.1 Técnicas de Transfer Learning

### Autoencoder Pre-training

```python
from tensorflow import keras

def create_fraud_autoencoder(input_dim, encoding_dim=32):
    """
    Pré-treina autoencoder em dados não-rotulados
    Encoder usado como feature extractor
    """
    input_layer = keras.layers.Input(shape=(input_dim,))
    
    # Encoder
    encoded = keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = keras.layers.BatchNormalization()(encoded)
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

# Uso para detecção de anomalias
def detect_fraud_autoencoder(encoder, decoder, X, threshold):
    """
    Detecta fraude por reconstruction error
    """
    reconstructed = decoder.predict(encoder.predict(X))
    mse = np.mean((X - reconstructed) ** 2, axis=1)
    return mse > threshold
```

### FraudTransformer (HSBC 2025)

```python
class FraudTransformer(nn.Module):
    """
    GPT-style architecture com time encoders
    Desenvolvido pelo HSBC Payment Fraud Group
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Time encoding (rotational/sinusoidal)
        self.time_encoder = SinusoidalTimeEncoding(d_model)
        self.absolute_time_encoder = CalendarEncoding(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, transactions, timestamps, calendar_features):
        # Embeddings com codificação temporal
        x = self.time_encoder(transactions, timestamps)
        x = x + self.absolute_time_encoder(calendar_features)
        
        # Transformer
        out = self.transformer(x)
        
        # Classificação no último token
        return self.classifier(out[:, -1, :])
```

### LSTM com Attention

```python
class FraudLSTMAttention(nn.Module):
    """
    LSTM com mecanismo de atenção para sequências de transações
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        
    def forward(self, x):
        # LSTM bidirectional
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)
```

### GNN Embeddings (NVIDIA Blueprint)

```python
# Configuração NVIDIA Financial Fraud Training Container
gnn_config = {
    "paths": {
        "data_dir": "/data/transactions",
        "model_output": "/models/fraud_detector"
    },
    "models": {
        "kind": "GNN_XGBOOST",
        "gnn_params": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "aggregation": "mean"
        },
        "xgboost_params": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 400
        }
    },
    "graph": {
        "node_types": ["customer", "merchant", "device", "ip"],
        "edge_types": ["transacts", "uses_device", "from_ip"]
    }
}
```

### Federated Transfer Learning (FED-SPFD)

```python
import flwr as fl

class BankFederatedClient(fl.client.NumPyClient):
    """
    Cliente federado para treinamento multi-banco
    Com transfer learning entre domínios
    """
    def __init__(self, model, local_data, pretrained_weights=None):
        self.model = model
        self.local_data = local_data
        
        # Transfer learning: inicializa com pesos pré-treinados
        if pretrained_weights:
            self.model.set_weights(pretrained_weights)
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Fine-tuning local
        self.model.fit(
            self.local_data['X'], 
            self.local_data['y'],
            epochs=config.get('local_epochs', 5),
            batch_size=32,
            verbose=0
        )
        
        return self.model.get_weights(), len(self.local_data['X']), {}
```

### Tabular Embeddings com DistilBERT

```python
from arize.pandas.embeddings import EmbeddingGenerator

def create_tabular_embeddings(df):
    """
    Converte features tabulares em embeddings com DistilBERT
    """
    # Converte row para texto
    def row_to_text(row):
        return f"state is {row['state']}, merchant is {row['merchant']}, FICO is {row['fico_score']}, amount is {row['amount']}"
    
    df['text_representation'] = df.apply(row_to_text, axis=1)
    
    # Gera embeddings
    generator = EmbeddingGenerator(model_name="distilbert-base-uncased")
    embeddings = generator.generate_embedding(df, text_cols=['text_representation'])
    
    return embeddings
```

## 3.2 Datasets para Transfer Learning

| Dataset | Arquitetura | Uso |
|---------|-------------|-----|
| **TalkingData** | 1D-CNN (FINet) | Click/ad fraud → Financial fraud |
| **ImageNet** | ResNet/DenseNet (DCNNTr) | Feature extraction |
| **Bank Account Fraud** | MLP Autoencoder | Domain adaptation |
| **IEEE-CIS** | GNN embeddings | Transaction network |

## 3.3 Modelos Pré-treinados

| Modelo | Base | Training Data | Use Case |
|--------|------|---------------|----------|
| FraudTransformer | GPT | HSBC transactions | Credit card |
| BERT4Fraud | BERT | Financial corpora | Multi-subtype |
| FINet | 1D-CNN | TalkingData | Ad/click fraud |
| NVIDIA GNN | GNN + XGBoost | Graph networks | Real-time |
| DistilBERT | DistilBERT | General text | Tabular embeddings |

---

# PARTE 4: GERAÇÃO DE DADOS SINTÉTICOS

## 4.1 Métodos de Geração

### CTGAN (Conditional Tabular GAN)

```python
from ctgan import CTGAN

def generate_synthetic_fraud(real_data, n_samples=5000):
    """
    Gera dados sintéticos de fraude com CTGAN
    """
    # Especifica colunas categóricas
    discrete_cols = ['transaction_type', 'merchant_category', 'device_type', 'fraud_flag']
    
    # Treina CTGAN
    ctgan = CTGAN(epochs=80, batch_size=500)
    ctgan.fit(real_data, discrete_cols)
    
    # Gera amostras sintéticas
    synthetic_data = ctgan.sample(n_samples)
    
    return synthetic_data
```

### TVAE (Tabular VAE)

```python
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

def generate_tvae_data(real_data, n_samples=5000):
    """
    Gera dados com Tabular VAE (SDV)
    """
    # Auto-detecta metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Treina TVAE
    tvae = TVAESynthesizer(metadata, epochs=300)
    tvae.fit(real_data)
    
    # Gera amostras
    synthetic = tvae.sample(n_samples)
    
    return synthetic
```

### SMOTE + CTGAN Híbrido

```python
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN

def hybrid_oversampling(X, y, target_ratio=0.5):
    """
    Combina SMOTE (rápido) + CTGAN (realístico)
    """
    # Fase 1: SMOTE para aumento rápido
    smote = SMOTE(sampling_strategy=0.2, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Fase 2: CTGAN para dados mais realísticos
    fraud_data = X_smote[y_smote == 1]
    
    ctgan = CTGAN(epochs=50)
    ctgan.fit(fraud_data)
    
    n_additional = int(len(y_smote) * target_ratio) - fraud_data.shape[0]
    synthetic_fraud = ctgan.sample(n_additional)
    
    # Combina
    X_final = pd.concat([X_smote, synthetic_fraud])
    y_final = np.concatenate([y_smote, np.ones(n_additional)])
    
    return X_final, y_final
```

### Transformer-Enhanced GAN (2025)

```python
# ArXiv 2509.19032 - Cutting-edge para extreme imbalance
class TransformerGAN:
    """
    Hybrid GAN + Transformer encoder block
    Self-attention captura interações entre features
    Supera SMOTE, CTGAN, TVAE em datasets extremamente desbalanceados
    """
    pass  # Implementação baseada no paper
```

## 4.2 Comparação de Métodos

| Método | Privacy | Utility (AUC) | Speed | Best For |
|--------|---------|---------------|-------|----------|
| SMOTE | Alta (pior) | Média | Muito rápido | Prototipagem |
| CTGAN | Moderada | Alta | Lento | Produção |
| TVAE | Moderada | Alta | Lento | Latent space |
| Transformer-GAN | - | Muito alta | Muito lento | Extreme imbalance |
| LLaMA 3.3 | Muito alta | Altíssima | Variável | Não recomendado |

---

# PARTE 5: MODELOS E BENCHMARKS

## 5.1 Comparação de Modelos (2025)

| Modelo | F1 Score | AUC | Latência | Melhor Para |
|--------|----------|-----|----------|-------------|
| **CatBoost** | 0.9161 | 0.94+ | Média | Accuracy, categóricas |
| **LightGBM** | 0.80 | 0.95 | **Mais baixa** | Real-time, PIX |
| **XGBoost** | 0.80 | 0.94+ | Alta | Flexibilidade |
| **Random Forest** | 0.99 | 0.99 | Média | Imbalanced data |
| **Stacking Ensemble** | 0.99 | 0.99 | Alta | Máxima accuracy |

## 5.2 Configurações Otimizadas

### CatBoost (Melhor Accuracy)

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
    random_seed=42,
    auto_class_weights='Balanced'  # Handle imbalance
)
```

### LightGBM (Melhor Latência - PIX)

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
    'max_bin': 63,           # Reduz latência
    'is_unbalance': True,    # Handle imbalance
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=400)
```

### Stacking Ensemble (Máxima Accuracy)

```python
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

stacked_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=400, use_label_encoder=False, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=400, verbose=-1)),
        ('cat', CatBoostClassifier(iterations=400, verbose=0))
    ],
    final_estimator=XGBClassifier(n_estimators=100, use_label_encoder=False),
    cv=5,
    stack_method='predict_proba'
)
```

## 5.3 GNN + XGBoost (NVIDIA Blueprint)

```python
# Pipeline GNN + XGBoost
# 1. GNN gera embeddings de 64 dimensões
# 2. Embeddings + features tabulares → XGBoost
# 3. Reduz feature engineering em 59.2%
# 4. Mantém 92.6% performance vs features manuais

class GNNXGBoostPipeline:
    def __init__(self):
        self.gnn = GraphAttentionNetwork(hidden_dim=128)
        self.xgb = XGBClassifier(n_estimators=400)
    
    def fit(self, graph_data, tabular_features, labels):
        # Fase 1: Treina GNN e extrai embeddings
        embeddings = self.gnn.fit_transform(graph_data)
        
        # Fase 2: Combina embeddings + features
        combined = np.hstack([embeddings, tabular_features])
        
        # Fase 3: Treina XGBoost
        self.xgb.fit(combined, labels)
    
    def predict(self, graph_data, tabular_features):
        embeddings = self.gnn.transform(graph_data)
        combined = np.hstack([embeddings, tabular_features])
        return self.xgb.predict_proba(combined)[:, 1]
```

---

# PARTE 6: DETECÇÃO EM TEMPO REAL

## 6.1 Velocity Checks

```python
class VelocityChecker:
    """
    Sistema de verificação de velocidade em tempo real
    """
    def __init__(self, redis_client):
        self.redis = redis_client
        self.rules = {
            'tx_count_1h': 10,        # Max 10 transações/hora
            'tx_amount_1h': 5000,     # Max R$5000/hora
            'distinct_recipients_24h': 5,  # Max 5 destinatários/dia
            'failed_attempts_1h': 3,  # Max 3 tentativas falhas/hora
        }
    
    def check(self, customer_id, transaction):
        violations = []
        
        # Verifica cada regra
        for rule, threshold in self.rules.items():
            current_value = self.get_current_value(customer_id, rule)
            if current_value >= threshold:
                violations.append(f"{rule}: {current_value} >= {threshold}")
        
        return len(violations) == 0, violations
```

## 6.2 Behavioral Biometrics

```python
BEHAVIORAL_BIOMETRICS_FEATURES = {
    'typing': [
        'typing_speed', 'typing_rhythm', 'keystroke_pressure',
        'keystroke_dynamics', 'error_rate',
    ],
    'mouse': [
        'mouse_speed', 'mouse_acceleration', 'click_patterns',
        'navigation_paths', 'scroll_behavior',
    ],
    'touch': [
        'touch_pressure', 'swipe_gestures', 'tap_location',
        'tap_speed', 'multi_touch_patterns',
    ],
    'session': [
        'session_duration', 'page_navigation_sequence',
        'interaction_frequency', 'idle_time_patterns',
    ]
}
```

## 6.3 Device Fingerprinting

```python
class DeviceFingerprint:
    """
    Fingerprinting de dispositivo para detecção de fraude
    """
    def collect_signals(self, request):
        return {
            'hardware': {
                'device_model': self.get_device_model(),
                'os_version': self.get_os_version(),
                'screen_resolution': self.get_screen_resolution(),
                'battery_level': self.get_battery_level(),
            },
            'software': {
                'browser_type': self.get_browser_type(),
                'browser_version': self.get_browser_version(),
                'timezone': self.get_timezone(),
                'language': self.get_language(),
            },
            'network': {
                'ip_address': request.remote_addr,
                'is_vpn': self.detect_vpn(),
                'is_proxy': self.detect_proxy(),
                'is_tor': self.detect_tor(),
            },
            'risk_signals': {
                'is_emulator': self.detect_emulator(),
                'is_rooted': self.detect_rooted(),
                'gps_spoofing': self.detect_gps_spoofing(),
            }
        }
```

---

# PARTE 7: PADRÕES DE FRAUDE PIX

## 7.1 Taxonomia de Fraudes PIX (arXiv:2511.20902)

### Categoria 1: Engenharia Social
- Golpe do falso funcionário de banco
- Falso sequestro
- Golpe do PIX errado
- Romance scam
- Golpe do boleto falso

### Categoria 2: Técnicas Híbridas (IA-Amplified)
- Deepfake para autenticação biométrica
- Voice cloning para confirmação telefônica
- Phishing personalizado com LLMs
- Chatbots maliciosos

### Categoria 3: Exploração Técnica
- Malware de interceptação (RAT)
- SIM swap
- Exploração de APIs
- Manipulação de QR codes
- Account takeover

## 7.2 Indicadores de Risco PIX

```python
PIX_RISK_INDICATORS = {
    'high_risk': [
        ('first_pix_to_pj', 0.8),           # Primeira vez para PJ
        ('night_high_value', 0.9),           # >R$2k à noite
        ('new_device', 0.7),                 # Dispositivo não cadastrado
        ('velocity_spike', 0.85),            # Muitas TX em curto período
        ('recipient_high_fraud_rate', 0.95), # Destinatário com histórico
    ],
    'medium_risk': [
        ('unusual_amount', 0.5),             # Valor fora do padrão
        ('different_location', 0.4),         # IP/localização diferente
        ('weekend_transaction', 0.3),        # Fim de semana
        ('qr_code_static', 0.4),             # QR estático vs dinâmico
    ],
    'regulatory': [
        ('exceeds_nocturnal_limit', 1.0),    # Excede limite noturno R$1k
        ('unregistered_device_limit', 1.0),  # Excede limite R$200
    ]
}
```

---

# PARTE 8: CONFORMIDADE REGULATÓRIA

## 8.1 BACEN (Brasil)

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

## 8.2 LGPD

```python
def generate_lgpd_explanation(prediction, features):
    """
    Gera explicação LGPD-compliant (Art. 20)
    """
    return {
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
```

## 8.3 PCI DSS

- Mascaramento de dados sensíveis (CPF, cartão)
- Logs estruturados sem dados pessoais
- Criptografia AES-256 para dados em repouso
- TLS 1.3 para dados em trânsito

---

# PARTE 9: MÉTRICAS DE PRODUÇÃO

## 9.1 KPIs Principais

| Métrica | Target PIX | Target Crédito | Descrição |
|---------|------------|----------------|-----------|
| **Latência P50** | < 30ms | < 100ms | Mediana |
| **Latência P99** | < 50ms | < 200ms | Percentil 99 |
| **TPS** | > 3,500 | > 1,000 | Transações/segundo |
| **Recall** | > 90% | > 85% | Detectar fraudes |
| **Precision** | > 70% | > 80% | Minimizar FP |
| **F1 Score** | > 80% | > 82% | Balanço |
| **False Positive Rate** | < 0.5% | < 1% | TX legítimas bloqueadas |

## 9.2 Monitoramento

```python
PROMETHEUS_METRICS = {
    'fraud_detection_latency_seconds': 'Histogram - tempo de predição',
    'fraud_predictions_total': 'Counter - predições por tipo',
    'fraud_risk_score': 'Histogram - distribuição de scores',
    'model_accuracy_gauge': 'Gauge - accuracy em tempo real',
    'sla_compliance_ratio': 'Gauge - % de requests < threshold',
    'velocity_violations_total': 'Counter - violações de velocidade',
    'device_fingerprint_anomalies': 'Counter - anomalias de device',
}
```

---

# PARTE 10: REFERÊNCIAS

## Papers Acadêmicos
1. arXiv:2511.20902 - "A Taxonomy of Pix Fraud in Brazil" (Nov 2025)
2. arXiv:2509.23712 - "FraudTransformer: Time-Aware GPT" (Oct 2025)
3. arXiv:2411.05815 - "GNNs for Financial Fraud Detection: A Review" (Nov 2024)
4. arXiv:2509.19032 - "Transformer-Enhanced GAN Oversampling" (Sep 2025)
5. NeurIPS 2022 - "Bank Account Fraud Dataset" (Feedzai)
6. MDPI 2025 - "Secure Banking: Explainable AI-Driven FL"

## Datasets
- IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection
- BAF (NeurIPS): https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- PaySim: https://github.com/EdgarLopezPhD/PaySim
- Amazon FDB: https://github.com/amazon-science/fraud-dataset-benchmark

## Frameworks
- TensorFlow Federated: https://www.tensorflow.org/federated
- Flower: https://flower.dev/
- NVIDIA Morpheus: https://developer.nvidia.com/morpheus-cybersecurity
- CTGAN/SDV: https://sdv.dev/
- CatBoost: https://catboost.ai/
- LightGBM: https://lightgbm.readthedocs.io/

## Vendors
- BioCatch (Behavioral Biometrics): https://www.biocatch.com
- Feedzai (ML Platform): https://www.feedzai.com
- SEON (Device Fingerprint): https://seon.io
- Fingerprint.com: https://fingerprint.com

---

*Documento gerado em: Novembro 2025*
*Versão: 2.0 (Expandida)*
*Autor: Sankofa Enterprise Pro Team*
