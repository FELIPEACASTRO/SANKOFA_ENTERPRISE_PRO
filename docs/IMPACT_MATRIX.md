# 📊 Matriz de Impacto - Módulos vs Testes

**Quando você corrige um bug, execute os testes nesta matriz para validar.**

---

## 🎯 Legenda

| Coluna | Significado |
|--------|-------------|
| **Módulo** | Arquivo de código afetado |
| **Testes Diretos** | Testes que testam ESTE arquivo |
| **Dependentes** | Testes que usam este arquivo indiretamente |
| **Smoke Críticos** | Testes mínimos que DEVEM passar sempre |

---

## 📋 Matriz Completa

### **Backend - ML Engine**

| Módulo | Testes Diretos | Testes Dependentes | Smoke |
|--------|---|---|---|
| `ml_engine/production_fraud_engine.py` | 50 testes | 33 testes (test_e2e, test_ml, test_integration) | 3 |
| `ml_engine/feature_engineering.py` | 15 testes | 25 testes (ML engine, E2E) | 1 |
| `ml_engine/explainability_engine.py` | 8 testes | 10 testes (compliance, LGPD) | 1 |
| `ml_engine/device_fingerprint.py` | 12 testes | 8 testes (fraud prediction) | 1 |

**Total ML Engine: 85 testes diretos + 76 dependentes**

---

### **Backend - API**

| Módulo | Testes Diretos | Testes Dependentes | Smoke |
|--------|---|---|---|
| `api/production_api.py` | 40 testes | 31 testes (E2E, integration) | 2 |
| `api/services/*` | 20 testes | 15 testes (API endpoints) | 1 |
| `security/jwt_validator.py` | 15 testes | 25 testes (auth, RBAC, E2E) | 2 |
| `database/repository.py` | 30 testes | 20 testes (integration, persistence) | 1 |

**Total API: 105 testes diretos + 91 dependentes**

---

### **Frontend - React**

| Módulo | Testes Diretos | Testes Dependentes | Smoke |
|--------|---|---|---|
| `frontend/src/pages/Dashboard.tsx` | 20 testes | 5 testes (E2E, integration) | 1 |
| `frontend/src/components/*` | 30 testes | 10 testes (UI, E2E) | 1 |
| `frontend/src/hooks/*` | 15 testes | 8 testes (components) | 0 |

**Total Frontend: 65 testes diretos + 23 dependentes**

---

## 🚀 Como Usar Esta Matriz

### **Cenário 1: Corrigiu bug em `ml_engine/production_fraud_engine.py`**

```bash
# NÍVEL 1 - Re-executar o teste que falhou
pytest tests/unit/test_ml_engine_unit.py::test_que_falhou -v

# NÍVEL 2 - Suite inteira do módulo
pytest tests/unit/test_ml_engine_unit.py -v              # 50 testes

# NÍVEL 3 - Smoke tests críticos (3 testes)
pytest tests/functional/test_smoke.py::test_smoke_03 -v  # ML model loaded
pytest tests/functional/test_smoke.py::test_smoke_05 -v  # Prediction works
pytest tests/functional/test_smoke.py::test_smoke_01 -v  # API starts

# RESULTADO: Se tudo passou ✅, correção foi validada
```

---

### **Cenário 2: Corrigiu bug em `api/production_api.py`**

```bash
# NÍVEL 1
pytest tests/unit/test_api_layer_unit.py::test_que_falhou -v

# NÍVEL 2
pytest tests/unit/test_api_layer_unit.py -v              # 40 testes

# NÍVEL 3
pytest tests/functional/test_smoke.py::test_smoke_01 -v  # API starts
pytest tests/functional/test_smoke.py::test_smoke_04 -v  # Auth works

# DEPOIS: Executar E2E relacionados (testes 31)
pytest tests/test_e2e.py -v
```

---

### **Cenário 3: Corrigiu bug em `security/jwt_validator.py`**

```bash
# NÍVEL 1
pytest tests/unit/test_api_layer_unit.py::test_token_validation -v

# NÍVEL 2
pytest tests/unit/test_api_layer_unit.py -v              # 40 testes (inclui JWT)
pytest tests/security/ -v                                # 60 testes security

# NÍVEL 3
pytest tests/functional/test_smoke.py::test_smoke_04 -v  # Auth works

# DEPOIS: E2E completo
pytest tests/test_e2e.py -v
```

---

## 📌 Regra Ouro

**Se corrigir X, sempre executar:**
1. ✅ Teste original que falhou
2. ✅ Todos os testes do módulo X
3. ✅ Smoke tests (mínimo)
4. ✅ Se passou em 3: CORRIGIR validado ✅

---

## 📚 Veja também

- `docs/FIX_VALIDATION_CHECKLIST.md` - Checklist passo-a-passo de 3 níveis
- `docs/DEFECTS_LOG.md` - Log de todos os bugs
- `docs/DEFECT_TEMPLATE.md` - Como registrar um novo bug
