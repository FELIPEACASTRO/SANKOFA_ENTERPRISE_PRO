# SANKOFA ENTERPRISE PRO - PLANO DE CORRECOES

## Resumo Executivo

- **Total de Correcoes**: 6
- **Esforco Estimado**: 2-4 horas
- **Risco de Regressao**: BAIXO (adicao de aliases, nao alteracao de logica)

## Plano de Correcoes Priorizado

### FASE 1: CRITICO (Imediato)

#### FIX-001: FraudGNN Graceful Degradation

**Arquivo**: `ml_engine/gnn/fraud_gnn.py`

**Problema**: Acesso a `torch.Tensor` quando torch e None

**Correcao**:

```python
# Linha ~317 - TimeEncoder class
# ANTES:
class TimeEncoder(_BaseModule):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frequencies = nn.Parameter(
            torch.randn(hidden_dim // 2) * 0.1,
            requires_grad=True
        )

# DEPOIS:
class TimeEncoder(_BaseModule):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        if HAS_TORCH:
            self.frequencies = nn.Parameter(
                torch.randn(hidden_dim // 2) * 0.1,
                requires_grad=True
            )
        else:
            self.frequencies = None
```

**Teste de Validacao**:

```python
def test_fraudgnn_without_pytorch():
    # Simular ausencia de PyTorch
    from ml_engine.gnn.fraud_gnn import FraudGNN, HAS_TORCH
    gnn = FraudGNN()
    assert hasattr(gnn, 'fallback_mode')
    # Nao deve lancar excecao
```

### FASE 2: ALTO (24h)

#### FIX-002: FeatureGenerator.generate() Alias

**Arquivo**: `ml_engine/feature_engineering.py`

**Correcao**: Adicionar alias `generate` para `generate_features`

```python
# Adicionar ao final da classe FeatureGenerator:
def generate(self, transaction: dict) -> dict:
    """Alias for generate_features for API consistency"""
    return self.generate_features(transaction)
```

**Teste de Validacao**:

```python
def test_feature_generator_generate_alias():
    from ml_engine.feature_engineering import FeatureGenerator
    fg = FeatureGenerator()
    tx = {"amount": 100.0, "hour": 14}
    result = fg.generate(tx)
    assert len(result) > len(tx)
```

### FASE 3: MEDIO (48h)

#### FIX-003: AutoencoderAnomalyDetector.detect() Alias

**Arquivo**: `ml_engine/autoencoder_anomaly_detector.py`

**Correcao**: Adicionar metodo `detect` como wrapper

```python
# Adicionar ao final da classe AutoencoderAnomalyDetector:
def detect(self, data: np.ndarray) -> dict:
    """Detect anomalies in data - wrapper for consistency"""
    is_anomaly, score = self.predict(data)
    return {
        "is_anomaly": bool(is_anomaly),
        "score": float(score),
        "threshold": self.threshold
    }
```

**Teste de Validacao**:

```python
def test_autoencoder_detect_method():
    import pickle
    with open("models/production/autoencoder_model.pkl", "rb") as f:
        ae = pickle.load(f)
    result = ae.detect(np.array([[100.0, 12, 0.1, 0, 0.5]]))
    assert "is_anomaly" in result
    assert "score" in result
```

#### FIX-004: BiLSTMSequenceAnalyzer Signature

**Arquivo**: `ml_engine/bilstm_sequence_analyzer.py`

**Correcao**: Tornar `current_transaction` opcional

```python
# ANTES:
def analyze_sequence(self, sequence: List[dict], current_transaction: dict) -> dict:

# DEPOIS:
def analyze_sequence(self, sequence: List[dict], current_transaction: dict = None) -> dict:
    if current_transaction is None and sequence:
        current_transaction = sequence[-1]
```

**Teste de Validacao**:

```python
def test_bilstm_analyze_without_current():
    import pickle
    with open("models/production/bilstm_model.pkl", "rb") as f:
        bilstm = pickle.load(f)
    sequence = [{"amount": 50.0}, {"amount": 100.0}]
    result = bilstm.analyze_sequence(sequence)  # Sem segundo argumento
    assert "risk_score" in str(result) or "suspicious" in str(result)
```

#### FIX-005: BahnsenFeatureEngineering.generate_features()

**Arquivo**: `ml_engine/bahnsen_feature_engineering.py`

**Correcao**: Adicionar metodo padrao

```python
# Adicionar a classe BahnsenFeatureEngineering:
def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Generate Bahnsen features for DataFrame"""
    return self.transform(df)

def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Transform DataFrame with Bahnsen features"""
    # Implementacao existente ou nova
    pass
```

### FASE 4: BAIXO (Opcional)

#### FIX-006: PIXFraudTaxonomy.classify() Alias

**Arquivo**: `ml_engine/pix_fraud_taxonomy.py`

**Correcao**: Adicionar alias

```python
def classify(self, transaction: dict) -> dict:
    """Alias for analyze_transaction"""
    return self.analyze_transaction(transaction)
```

## Cronograma de Implementacao

| Fase | Correcao | Responsavel | Prazo | Status |
|------|----------|-------------|-------|--------|
| 1 | FIX-001 | Dev Senior | Imediato | Pendente |
| 2 | FIX-002 | Dev | 24h | Pendente |
| 3 | FIX-003 | Dev | 48h | Pendente |
| 3 | FIX-004 | Dev | 48h | Pendente |
| 3 | FIX-005 | Dev | 48h | Pendente |
| 4 | FIX-006 | Dev Junior | Opcional | Pendente |

## Comandos para Executar Apos Correcoes

```bash
# Executar testes
cd backend
python scripts/forensic_test_final.py

# Executar pytest (quando disponivel)
pytest tests/ -v --tb=short

# Verificar cobertura
pytest tests/ --cov=ml_engine --cov-report=html
```

## Criterios de Aceitacao

1. Todos os 24 testes devem passar
2. Taxa de aprovacao >= 95%
3. Nenhum FAIL critico
4. Warnings <= 2

---

**Gerado em: 2025-12-15**
**Proxima Revisao: Apos implementacao das correcoes**
