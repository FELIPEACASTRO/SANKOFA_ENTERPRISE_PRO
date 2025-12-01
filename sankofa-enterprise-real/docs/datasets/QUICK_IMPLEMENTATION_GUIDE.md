# Guia Rápido de Implementação
## Integração de Datasets no Sankofa Enterprise Pro

---

## PRIORIDADE 1: Features Imediatas (Esta Semana)

### 1. Velocity Features (Nigerian Dataset Pattern)
```python
# features/velocity.py
def calculate_velocity_features(transaction, user_history):
    """Calcula features de velocidade baseadas no Nigerian Financial Dataset"""
    
    features = {
        'txn_count_last_1h': count_transactions(user_history, hours=1),
        'txn_count_last_24h': count_transactions(user_history, hours=24),
        'total_amount_last_1h': sum_amounts(user_history, hours=1),
        'time_since_last_txn': get_time_since_last(user_history),
        'avg_gap_between_txns': calculate_avg_gap(user_history),
        'spending_deviation_score': calculate_spending_deviation(
            transaction.amount, 
            user_history.avg_amount, 
            user_history.std_amount
        ),
    }
    return features
```

### 2. Channel Risk Score (Nigerian + FCA Pattern)
```python
# features/channel_risk.py
CHANNEL_RISK_SCORES = {
    'USSD': 0.8,       # Alto risco - comum em fraudes
    'Mobile App': 0.6, # Médio-alto
    'Card': 0.4,       # Médio
    'Bank Transfer': 0.3, # Médio-baixo
    'POS': 0.2,        # Baixo
}

def get_channel_risk(payment_channel):
    return CHANNEL_RISK_SCORES.get(payment_channel, 0.5)
```

### 3. Regras PIX Taxonomy (arXiv:2511.20902)
```python
# rules/pix_taxonomy.py
PIX_FRAUD_RULES = {
    'QR_CODE_TAMPERED': {
        'indicators': ['broadcast_donation', 'ngo_stream', 'qr_mismatch'],
        'risk_weight': 0.9
    },
    'GHOST_HAND': {
        'indicators': ['remote_access', 'fear_inducing_call', 'bank_impersonation'],
        'risk_weight': 0.95
    },
    'WRONG_PIX': {
        'indicators': ['request_return', 'first_contact', 'urgent_tone'],
        'risk_weight': 0.7
    },
    'FAKE_RECEIPT': {
        'indicators': ['screenshot_shared', 'no_notification', 'pressure_delivery'],
        'risk_weight': 0.8
    },
    'WHATSAPP_CLONE': {
        'indicators': ['profile_change_recent', 'urgent_money_request', 'family_contact'],
        'risk_weight': 0.85
    }
}
```

---

## PRIORIDADE 2: Transfer Learning (Próximas 2 Semanas)

### Modelo Base com PaySim
```python
# ml/paysim_transfer.py
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def train_base_model_paysim():
    """Treina modelo base com 6M transações PaySim"""
    
    # Carregar PaySim
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    
    # Features relevantes para PIX
    feature_mapping = {
        'step': 'hora_transacao',
        'type': 'tipo_transacao', 
        'amount': 'valor',
        'oldbalanceOrg': 'saldo_anterior',
        'newbalanceOrig': 'saldo_posterior',
    }
    
    # Filtrar apenas TRANSFER e CASH_OUT (similar ao PIX)
    df_pix_like = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    
    # Treinar modelo base
    X = df_pix_like[['amount', 'oldbalanceOrg', 'newbalanceOrig']]
    y = df_pix_like['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    return model
```

### Fine-tune com Feedzai BAF
```python
# ml/feedzai_finetune.py
def finetune_with_feedzai(base_model, feedzai_data):
    """Fine-tune para fairness e bias"""
    
    # Avaliar bias em grupos protegidos
    protected_groups = ['customer_age', 'employment_status', 'income']
    
    for group in protected_groups:
        evaluate_fairness(base_model, feedzai_data, group)
    
    # Ajustar thresholds por grupo se necessário
    calibrated_model = calibrate_for_fairness(base_model, feedzai_data)
    
    return calibrated_model
```

---

## PRIORIDADE 3: GNN Integration (Mês 2)

### AI4Risk GTAN Model
```python
# ml/gnn_integration.py
"""
Integração com AI4Risk GTAN para detecção via grafos
Requer: PyTorch, DGL, CUDA
"""

# 1. Construir grafo de transações
# 2. Aplicar GTAN para embeddings
# 3. Combinar com features tradicionais

GTAN_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 0.001,
}

# Performance esperada (YelpChi benchmark):
# AUC: 0.9241, F1: 0.7988
```

---

## MÉTRICAS DE AVALIAÇÃO

### Benchmark com Amazon FDB
```python
# evaluation/fdb_benchmark.py
from fdb.datasets import FraudDatasetBenchmark

DATASETS_TO_BENCHMARK = ['ieeecis', 'ccfraud', 'fraudecom']

def run_full_benchmark(model):
    results = {}
    for key in DATASETS_TO_BENCHMARK:
        fdb = FraudDatasetBenchmark(key=key)
        
        # Treinar e avaliar
        model.fit(fdb.train)
        predictions = model.predict(fdb.test)
        
        results[key] = {
            'auc_roc': calculate_auc(fdb.test_labels, predictions),
            'auc_pr': calculate_auc_pr(fdb.test_labels, predictions),
            'f1': calculate_f1(fdb.test_labels, predictions),
        }
    
    return results

# Targets (baseado em resultados publicados):
# ieeecis: AUC > 0.95
# ccfraud: AUC > 0.99
# fraudecom: AUC > 0.90
```

---

## CHECKLIST DE IMPLEMENTAÇÃO

### Semana 1
- [ ] Implementar velocity_features.py
- [ ] Implementar channel_risk.py
- [ ] Adicionar regras PIX taxonomy
- [ ] Testar com dados existentes

### Semana 2
- [ ] Download PaySim dataset
- [ ] Treinar modelo base
- [ ] Validar métricas baseline

### Semana 3-4
- [ ] Download Feedzai BAF
- [ ] Fine-tune para fairness
- [ ] Documentar bias analysis

### Mês 2
- [ ] Setup AI4Risk environment
- [ ] Implementar GNN pipeline
- [ ] Benchmark completo FDB

---

*Guia atualizado: 01/12/2025*
