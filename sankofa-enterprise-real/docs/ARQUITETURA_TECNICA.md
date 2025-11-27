# Arquitetura Técnica - Sankofa Enterprise Pro v11.0
## Documentação Técnica Detalhada

**Versão:** 11.0  
**Última Atualização:** 27 de Novembro de 2025  
**Status:** Desenvolvimento/Staging - 45 Testes Passando

---

## Estado de Implementação

| Componente | Implementado | Testado | Integrado na API |
|------------|--------------|---------|------------------|
| Flask API (50+ endpoints) | ✅ | ✅ | ✅ |
| React Dashboard (9 páginas) | ✅ | ✅ | ✅ |
| ML Stacking (RF+GB+LR) | ✅ | ✅ | ✅ |
| PostgreSQL (Neon) | ✅ | ✅ | ✅ |
| Explainability Engine | ✅ | ✅ | ⚠️ Módulo separado |
| Probability Calibration | ✅ | ✅ | ⚠️ Módulo separado |
| Location Entropy Features | ✅ | ✅ | ⚠️ Módulo separado |
| Self-Training Optimizer | ✅ | ✅ | ⚠️ Módulo separado |
| Redis Cache | ⚠️ | ⚠️ | Fallback in-memory |

---

## 1. Visão Geral da Arquitetura

### 1.1 Diagrama de Alto Nível (Implementado)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SANKOFA ENTERPRISE PRO v11.0                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│   │   FRONTEND   │     │   BACKEND    │     │   DATABASE   │                │
│   │   React/Vite │────▶│   Flask API  │────▶│  PostgreSQL  │                │
│   │   Port 5000  │     │   Port 8445  │     │   (Neon)     │                │
│   └──────────────┘     └──────────────┘     └──────────────┘                │
│          │                    │                    │                         │
│          │                    ▼                    │                         │
│          │             ┌──────────────┐           │                         │
│          │             │   ML ENGINE  │           │                         │
│          │             │  Stacking:   │           │                         │
│          │             │  RF + GB + LR│           │                         │
│          │             └──────────────┘           │                         │
│          │                    │                    │                         │
│          │                    ▼                    │                         │
│          │             ┌──────────────┐           │                         │
│          │             │    CACHE     │           │                         │
│          └────────────▶│  In-Memory   │◀──────────┘                         │
│                        │  (Principal) │                                      │
│                        └──────────────┘                                      │
│                                                                              │
│   Módulos Adicionais (não integrados na API principal):                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│   │ Explainability│  │ Probability │  │ Self-Training│                     │
│   │    Engine    │  │ Calibration │  │  Optimizer   │                     │
│   └──────────────┘  └──────────────┘  └──────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Stack Tecnológico Atual

| Camada | Tecnologia | Versão |
|--------|------------|--------|
| **Frontend** | React + Vite | 18+ / 5+ |
| **UI Components** | shadcn/ui + TailwindCSS | - |
| **Backend** | Flask + Flask-CORS | 3.0.0 |
| **Autenticação** | Flask-JWT-Extended | 4.6.0 |
| **Rate Limiting** | Flask-Limiter | - |
| **ML Framework** | scikit-learn | 1.5.2+ |
| **Gradient Boosting** | XGBoost, LightGBM | 2.1.2+, 4.5.0+ |
| **Explicabilidade** | SHAP | - |
| **Data Processing** | Pandas, NumPy | 2.2.3+, 1.26.4+ |
| **Database** | PostgreSQL (Neon) | 13+ |
| **ORM** | SQLAlchemy | - |
| **Cache** | In-Memory (Redis fallback) | - |
| **Logging** | Structured JSON | - |

---

## 2. Estrutura de Diretórios

```
sankofa-enterprise-real/
├── backend/
│   ├── api/
│   │   └── production_api.py        # API principal (30+ endpoints)
│   │
│   ├── ml_engine/
│   │   ├── production_fraud_engine.py    # Motor ML principal
│   │   ├── advanced_feature_engineering.py # 47+ features
│   │   ├── explainability_engine.py      # SHAP values
│   │   ├── probability_calibration.py    # Calibração isotônica/sigmoid
│   │   └── self_training_optimizer.py    # Semi-supervised learning
│   │
│   ├── mlops/
│   │   ├── ab_testing_manager.py         # Testes A/B
│   │   ├── canary_deployment_manager.py  # Deploy gradual
│   │   ├── drift_detector.py             # Detecção de drift
│   │   └── model_lifecycle_manager.py    # Versionamento
│   │
│   ├── cache/
│   │   └── redis_cache_system.py         # Sistema de cache
│   │
│   ├── security/
│   │   └── enterprise_security_system.py # Segurança
│   │
│   ├── compliance/
│   │   └── compliance_manager.py         # LGPD, BACEN, PCI
│   │
│   ├── utils/
│   │   ├── structured_logging.py         # Logs JSON
│   │   └── error_handling.py             # Tratamento de erros
│   │
│   ├── tests/
│   │   ├── test_improvements.py          # 20 testes ML
│   │   └── test_e2e.py                   # 25 testes E2E
│   │
│   └── data/
│       └── metrics_state.json            # Estado persistido
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Transactions.tsx
│   │   │   ├── Calibration.tsx
│   │   │   ├── Investigation.tsx
│   │   │   ├── ManualReview.tsx
│   │   │   ├── Monitoring.tsx
│   │   │   ├── Reports.tsx
│   │   │   ├── Metrics.tsx
│   │   │   └── Alerts.tsx
│   │   │
│   │   ├── components/
│   │   │   └── ui/                       # shadcn components
│   │   │
│   │   └── lib/
│   │       └── api.ts                    # API client
│   │
│   └── vite.config.ts
│
├── docs/
│   ├── DOCUMENTACAO_FUNCIONAL.md
│   ├── ARQUITETURA_TECNICA.md
│   ├── MANUAL_USUARIO.md
│   └── USE_A_CABECA_SANKOFA.md
│
└── models/
    └── fraud_ensemble_v*.joblib          # Modelos serializados
```

---

## 3. Backend API

### 3.1 Configuração do Flask

```python
# production_api.py

app = Flask(__name__)
CORS(app, origins=["*"])

# Rate Limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per minute", "50000 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET', generate_key())
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
```

### 3.2 Endpoints Principais

| Endpoint | Método | Descrição | Rate Limit |
|----------|--------|-----------|------------|
| `/api/health` | GET | Health check | - |
| `/api/fraud/predict` | POST | Predição em tempo real | 1000/min |
| `/api/fraud/batch` | POST | Predição em lote | 100/min |
| `/api/transactions` | GET | Listar transações | 500/min |
| `/api/model/metrics` | GET | Métricas do modelo | 500/min |
| `/api/feedback` | POST | Feedback do analista | 500/min |
| `/api/dashboard/summary` | GET | Resumo do dashboard | 500/min |
| `/api/dashboard/kpis` | GET | KPIs em tempo real | 500/min |
| `/api/dashboard/timeseries` | GET | Dados temporais | 500/min |
| `/api/dashboard/channels` | GET | Estatísticas por canal | 500/min |
| `/api/dashboard/alerts` | GET | Alertas recentes | 500/min |
| `/api/dashboard/model-status` | GET | Status dos modelos | 500/min |
| `/api/manual-review` | GET | Fila de revisão | 500/min |
| `/api/alerts` | GET | Lista de alertas | 500/min |

### 3.3 Estrutura de Resposta Padrão

```python
# Sucesso
{
    "success": True,
    "data": {...},
    "timestamp": "2025-11-27T14:30:00.000Z"
}

# Erro
{
    "success": False,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Campo obrigatório ausente",
        "details": {...}
    },
    "timestamp": "2025-11-27T14:30:00.000Z"
}
```

### 3.4 Tratamento de Erros

```python
class ErrorCategory(Enum):
    VALIDATION = "validation"
    DATABASE = "database"
    ML_MODEL = "ml_model"
    SECURITY = "security"
    NETWORK = "network"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

---

## 4. Motor de Machine Learning

### 4.1 Arquitetura do Ensemble (Implementado)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STACKING ENSEMBLE (Atual)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT: Transação                                                           │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────────────────────────────────────────────────────┐         │
│   │              FEATURE ENGINEERING (Básico)                       │         │
│   │   Features: Temporal, Valor, Geográficas básicas                │         │
│   │   (Location Entropy disponível em módulo separado)              │         │
│   └───────────────────────────────────────────────────────────────┘         │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────────────────────────────────────────────────────┐         │
│   │              BASE MODELS (sklearn StackingClassifier)           │         │
│   │                                                                  │         │
│   │  ┌─────────────┐  ┌─────────────┐                               │         │
│   │  │   Random    │  │  Gradient   │                               │         │
│   │  │   Forest    │  │  Boosting   │                               │         │
│   │  │ n=100,d=15  │  │ n=100,d=8   │                               │         │
│   │  └──────┬──────┘  └──────┬──────┘                               │         │
│   │         │                │                                       │         │
│   └─────────│────────────────│───────────────────────────────────────┘        │
│             │                │                                               │
│             ▼                ▼                                               │
│   ┌───────────────────────────────────────────────────────────────┐         │
│   │              META-MODEL (Final Estimator)                       │         │
│   │   Logistic Regression                                           │         │
│   │   - Combina predições dos base models                           │         │
│   │   - Class weights balanced                                      │         │
│   └───────────────────────────────────────────────────────────────┘         │
│             │                                                                │
│             ▼                                                                │
│         OUTPUT: FraudPrediction                                              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│   MÓDULOS DISPONÍVEIS (não integrados na API principal):                    │
│                                                                              │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│   │  EXPLAINABILITY  │  │   CALIBRATION    │  │  SELF-TRAINING   │         │
│   │     ENGINE       │  │    (Isotonic/    │  │    OPTIMIZER     │         │
│   │  (SHAP values)   │  │    Sigmoid)      │  │  (Pseudo-label)  │         │
│   │   ✅ Testado     │  │   ✅ Testado     │  │   ✅ Testado     │         │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Feature Engineering (47+ Features)

#### Temporais (5 features)
```python
features['hour'] = df['timestamp'].dt.hour
features['day_of_week'] = df['timestamp'].dt.dayofweek
features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 6)).astype(int)
features['is_business_hours'] = features['hour'].between(9, 18).astype(int)
```

#### Location Entropy (11 features) - NOVO
```python
def calculate_location_entropy(locations):
    """Calcula entropia de Shannon para diversidade de localizações"""
    if len(locations) <= 1:
        return 0.0
    counter = Counter(locations)
    probs = [count / len(locations) for count in counter.values()]
    return -sum(p * log2(p) for p in probs if p > 0)

features['location_entropy'] = calculate_location_entropy(user_locations)
features['unique_locations'] = len(set(user_locations))
features['location_diversity_score'] = unique_locations / total_transactions
```

#### Transaction Patterns (NOVO)
```python
features['amount_zscore'] = (amount - mean) / std
features['is_outlier'] = (features['amount_zscore'].abs() > 3).astype(int)
features['hour_pattern_deviation'] = calculate_hour_deviation(user_history)
```

### 4.3 Probability Calibration

```python
class EnsembleCalibrator:
    """Calibrador com seleção automática isotônica/sigmoid"""
    
    def __init__(self):
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.sigmoid = _SigmoidCalibration()
        
    def fit(self, y_prob, y_true):
        # Treina ambos e seleciona melhor ECE
        self.isotonic.fit(y_prob, y_true)
        self.sigmoid.fit(y_prob, y_true)
        
        ece_iso = self._calculate_ece(self.isotonic.predict(y_prob), y_true)
        ece_sig = self._calculate_ece(self.sigmoid.predict(y_prob), y_true)
        
        self.best_method = 'isotonic' if ece_iso < ece_sig else 'sigmoid'
```

### 4.4 Explainability Engine (SHAP)

```python
class ExplainabilityEngine:
    """Motor de explicabilidade usando SHAP"""
    
    def explain_prediction(self, model, X, feature_names):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        explanation = {
            'shap_values': dict(zip(feature_names, shap_values[0])),
            'top_features': self._get_top_features(shap_values, feature_names, n=5),
            'text_explanation': self._generate_text_explanation(shap_values, feature_names)
        }
        return explanation
    
    def _generate_text_explanation(self, shap_values, feature_names):
        """Gera explicação em texto para compliance LGPD"""
        top_positive = self._get_top_features(shap_values, feature_names, n=3, positive=True)
        
        explanations = []
        for feature, value in top_positive:
            if feature == 'is_night':
                explanations.append("Transação realizada em horário noturno")
            elif feature == 'amount_zscore':
                explanations.append("Valor significativamente diferente do padrão")
            # ... mais mapeamentos
        
        return explanations
```

### 4.5 Self-Training Optimizer

```python
class SelfTrainingOptimizer:
    """Semi-supervised learning com pseudo-labeling"""
    
    def __init__(self, confidence_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        
    def optimize(self, model, X_labeled, y_labeled, X_unlabeled):
        # 1. Predição nos dados não rotulados
        proba = model.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        
        # 2. Seleciona amostras de alta confiança
        high_confidence_mask = max_proba >= self.confidence_threshold
        pseudo_labels = np.argmax(proba[high_confidence_mask], axis=1)
        
        # 3. Combina com dados rotulados
        X_combined = np.vstack([X_labeled, X_unlabeled[high_confidence_mask]])
        y_combined = np.hstack([y_labeled, pseudo_labels])
        
        # 4. Retreina modelo
        model.fit(X_combined, y_combined)
        return model
```

---

## 5. Banco de Dados

### 5.1 Schema PostgreSQL

```sql
-- Tabela principal de transações
CREATE TABLE transactions (
    id VARCHAR PRIMARY KEY,
    amount DECIMAL(15,2) NOT NULL,
    channel VARCHAR(50),
    location VARCHAR(100),
    cpf VARCHAR(14),
    timestamp TIMESTAMP DEFAULT NOW(),
    fraud_score DECIMAL(5,2),
    is_fraud BOOLEAN DEFAULT FALSE,
    decision VARCHAR(20),
    risk_factors JSONB,
    shap_values JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de alertas
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR REFERENCES transactions(id),
    type VARCHAR(50),
    severity VARCHAR(20),
    status VARCHAR(20) DEFAULT 'NEW',
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

-- Tabela de audit log
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id VARCHAR,
    user_id VARCHAR,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de métricas
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,4),
    labels JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Índices para performance
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_cpf ON transactions(cpf);
CREATE INDEX idx_transactions_channel ON transactions(channel);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_audit_created ON audit_log(created_at);
```

### 5.2 Connection Pool

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verifica conexões antes de usar
    pool_recycle=300,    # Recicla conexões a cada 5 min
    connect_args={
        "connect_timeout": 10,
        "sslmode": "require"
    }
)
```

---

## 6. Frontend Architecture

### 6.1 Estrutura de Componentes

```
src/
├── pages/                  # Páginas principais
│   ├── Dashboard.tsx       # KPIs e gráficos
│   ├── Transactions.tsx    # Lista de transações
│   ├── Calibration.tsx     # Ajustes de threshold
│   ├── Investigation.tsx   # Central de investigação
│   ├── ManualReview.tsx    # Fila HITL
│   ├── Monitoring.tsx      # Saúde do sistema
│   ├── Reports.tsx         # Geração de relatórios
│   ├── Metrics.tsx         # Contadores real-time
│   └── Alerts.tsx          # Central de alertas
│
├── components/
│   ├── ui/                 # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   ├── select.tsx
│   │   ├── slider.tsx
│   │   ├── switch.tsx
│   │   └── table.tsx
│   │
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Layout.tsx
│   │
│   └── charts/
│       ├── LineChart.tsx
│       ├── BarChart.tsx
│       └── PieChart.tsx
│
└── lib/
    ├── api.ts              # API client
    └── utils.ts            # Utilitários
```

### 6.2 API Client

```typescript
// lib/api.ts
const API_BASE = 'http://0.0.0.0:8445/api';

export async function fetchDashboardKPIs() {
  const response = await fetch(`${API_BASE}/dashboard/kpis`);
  return response.json();
}

export async function fetchTransactions(page = 1, limit = 50) {
  const response = await fetch(
    `${API_BASE}/transactions?page=${page}&limit=${limit}`
  );
  return response.json();
}

export async function predictFraud(transactions: Transaction[]) {
  const response = await fetch(`${API_BASE}/fraud/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transactions })
  });
  return response.json();
}
```

### 6.3 Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5000,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://0.0.0.0:8445',
        changeOrigin: true
      }
    }
  }
});
```

---

## 7. Segurança

### 7.1 Autenticação JWT

```python
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity

jwt = JWTManager(app)

@app.route('/api/protected')
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(user=current_user)
```

### 7.2 Rate Limiting

```python
# Limites por endpoint
@limiter.limit("1000/minute")
@app.route('/api/fraud/predict', methods=['POST'])
def predict():
    ...

@limiter.limit("100/minute")
@app.route('/api/fraud/batch', methods=['POST'])
def batch():
    ...
```

### 7.3 Input Validation

```python
def validate_transaction(data):
    required_fields = ['transaction_id', 'amount']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Campo obrigatório: {field}")
    
    if not isinstance(data['amount'], (int, float)):
        raise ValidationError("Amount deve ser numérico")
    
    if data['amount'] < 0:
        data['amount'] = abs(data['amount'])  # Normaliza
```

---

## 8. MLOps

### 8.1 Drift Detection

```python
class DriftDetector:
    """Detecta data drift usando Jensen-Shannon divergence"""
    
    def detect_drift(self, reference_data, current_data, threshold=0.1):
        for feature in self.features:
            ref_dist = self._get_distribution(reference_data[feature])
            cur_dist = self._get_distribution(current_data[feature])
            
            js_divergence = jensenshannon(ref_dist, cur_dist)
            
            if js_divergence > threshold:
                self.alerts.append({
                    'feature': feature,
                    'divergence': js_divergence,
                    'severity': self._classify_severity(js_divergence)
                })
        
        return self.alerts
```

### 8.2 A/B Testing

```python
class ABTestingManager:
    """Gerencia testes A/B entre modelos"""
    
    def __init__(self, variants, traffic_split):
        self.variants = variants  # {'control': model_v1, 'treatment': model_v2}
        self.traffic_split = traffic_split  # {'control': 0.5, 'treatment': 0.5}
    
    def route_request(self, transaction_id):
        # Hash-based routing para consistência
        hash_value = hash(transaction_id) % 100
        
        cumulative = 0
        for variant, split in self.traffic_split.items():
            cumulative += split * 100
            if hash_value < cumulative:
                return variant
        
        return 'control'
```

### 8.3 Canary Deployment

```python
class CanaryDeploymentManager:
    """Deploy gradual de novos modelos"""
    
    STAGES = [0.05, 0.10, 0.25, 0.50, 1.0]  # 5%, 10%, 25%, 50%, 100%
    
    def advance_stage(self):
        if self.health_check_passed():
            self.current_stage += 1
            self.traffic_to_canary = self.STAGES[self.current_stage]
            return True
        else:
            self.rollback()
            return False
```

---

## 9. Observabilidade

### 9.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "Fraud prediction completed",
    transaction_id=tx_id,
    score=score,
    is_fraud=is_fraud,
    latency_ms=latency,
    model_version=version
)
```

### 9.2 Métricas Coletadas

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `transactions_total` | Counter | Total de transações |
| `frauds_detected` | Counter | Fraudes detectadas |
| `prediction_latency` | Histogram | Latência de predição |
| `model_accuracy` | Gauge | Acurácia do modelo |
| `cache_hit_rate` | Gauge | Taxa de cache hit |
| `db_connection_pool` | Gauge | Conexões ativas |

### 9.3 Health Checks

```python
@app.route('/api/health')
def health():
    checks = {
        'database': check_database(),
        'ml_model': check_model_loaded(),
        'cache': check_cache(),
        'disk': check_disk_space()
    }
    
    status = 'healthy' if all(checks.values()) else 'degraded'
    
    return jsonify({
        'status': status,
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    })
```

---

## 10. Performance

### 10.1 Benchmarks Atuais

| Operação | Latência P50 | Latência P95 | Latência P99 |
|----------|--------------|--------------|--------------|
| Health Check | 0.15ms | 0.20ms | 0.25ms |
| Single Prediction | 22ms | 33ms | 45ms |
| Batch (50 txns) | 250ms | 400ms | 600ms |
| Dashboard KPIs | 0.4ms | 0.6ms | 1.0ms |
| Transactions List | 12ms | 15ms | 20ms |

### 10.2 Otimizações Implementadas

1. **Connection Pooling:** 10 conexões base, 20 overflow
2. **In-Memory Cache:** Fallback quando Redis indisponível
3. **Lazy Loading:** Modelos carregados sob demanda
4. **Batch Processing:** Vetorização de predições
5. **Index Optimization:** Índices em campos de busca frequente

---

## 11. Deployment

### 11.1 Variáveis de Ambiente

```bash
# Backend
FLASK_ENV=production
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
JWT_SECRET=<secret>
API_PORT=8445

# Frontend
VITE_API_URL=http://0.0.0.0:8445
```

### 11.2 Workflows Configurados

```yaml
# Backend API
name: Backend API
command: cd sankofa-enterprise-real/backend && python api/production_api.py

# Frontend
name: Sankofa Enterprise
command: cd sankofa-enterprise-real/frontend && npm run dev
```

---

*Documento técnico atualizado em 27 de Novembro de 2025*  
*Sankofa Enterprise Pro v11.0*
