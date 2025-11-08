# ⚠️ DEPRECATED FRAUD ENGINES ⚠️

## Status: **DEPRECATED - NÃO USAR**

Todos os fraud engines neste diretório (exceto `production_fraud_engine.py`) estão **DEPRECATED** e **NÃO DEVEM SER USADOS**.

### Engines Antigos (DEPRECATED):
- ❌ `ultra_fast_fraud_engine.py` 
- ❌ `final_balanced_fraud_engine.py`
- ❌ `hyper_optimized_fraud_engine_v3.py`
- ❌ `continuous_learning_system.py`
- ❌ `enhanced_fraud_engine_v4.py`
- ❌ `fast_balanced_fraud_engine.py`
- ❌ `fast_optimized_fraud_engine.py`
- ❌ `final_fraud_analyzer.py`
- ❌ `fraud_analyzer.py`
- ❌ `guaranteed_recall_fraud_engine.py`
- ❌ `optimized_fraud_analyzer.py`
- ❌ `optimized_fraud_engine.py`
- ❌ `real_model_training.py`
- ❌ `ultra_low_latency_fraud_engine.py`
- ❌ `ultra_precision_fraud_engine_v4.py`

**Total**: 15 arquivos, 6.483 linhas de código duplicado

### Engine Oficial (USE ESTE):
- ✅ **`production_fraud_engine.py`** - Engine consolidado production-grade

## Por que foram Deprecated?

1. **Código Duplicado Massivo**: 70%+ de código duplicado entre engines
2. **Manutenção Impossível**: 15 engines diferentes para manter
3. **Sem Engine "Oficial"**: Confusão sobre qual usar
4. **Performance Inconsistente**: Diferentes implementações = diferentes resultados
5. **Difícil de Testar**: 15 engines = 15x trabalho de testes

## Transformação Realizada

**Antes**:
- 15 engines diferentes
- 6.483 linhas de código
- Configurações hardcoded
- Logging não estruturado

**Depois**:
- 1 engine production-grade
- ~600 linhas de código (-90%)
- Configuração via environment vars
- Logging estruturado JSON
- Error handling enterprise

## Como Migrar

Se você está usando algum engine antigo:

### Antes:
```python
from backend.ml_engine.final_balanced_fraud_engine import FinalBalancedFraudEngine

engine = FinalBalancedFraudEngine()
engine.fit(X_train, y_train)
predictions = engine.predict(X_test)
```

### Depois:
```python
from backend.ml_engine.production_fraud_engine import get_fraud_engine

engine = get_fraud_engine()
engine.fit(X_train, y_train)
predictions = engine.predict(X_test)
```

## Uso via API

A forma recomendada é usar a **Production API**:

```bash
# Prediction endpoint
POST /api/fraud/predict
Body: {
  "transactions": [...]
}

# Batch prediction
POST /api/fraud/batch
Body: {
  "transactions": [...],
  "batch_size": 100
}
```

Ver: `backend/api/production_api.py`

## Quando Deletar?

Estes arquivos serão **DELETADOS** após:

1. ✅ Production API validada em testes
2. ✅ Todos os imports migrados
3. ✅ Testes de integração passando
4. ✅ Validação em staging

**Timeline Estimado**: 1-2 semanas

## Dúvidas?

Ver documentação completa:
- `docs/TRANSFORMATION_REPORT.md`
- `replit.md`
- `backend/ml_engine/production_fraud_engine.py` (docstrings completas)

---

**Status**: DEPRECATED desde 08/11/2025  
**Motivo**: Consolidação enterprise  
**Replacement**: `production_fraud_engine.py`
