# 📊 Matriz de Impacto - Módulos vs Testes

**Quando você corrige um bug, execute os testes nesta matriz para validar.**

---

## 🎯 Inventário Real de Testes (185 testes em 6 arquivos)

| Arquivo | Testes | Descrição |
|---------|--------|-----------|
| `tests/test_domain.py` | 14 testes | Entidades, regras de negócio, feature engineering |
| `tests/test_e2e.py` | 31 testes | E2E completo (API, auth, prediction, persistence) |
| `tests/test_improvements.py` | ~40 testes | Explainability, calibração, métricas |
| `tests/test_qa_comprehensive.py` | ~50 testes | QA abrangente, security, performance |
| `tests/test_qa_expanded.py` | ~40 testes | QA expandido, edge cases |
| `tests/test_resilience.py` | 10 testes | Resiliência, error handling, cache |

**Total: 185 testes coletados**

---

## 📋 Matriz de Impacto por Módulo

### **Backend - ML Engine**

| Módulo | Testes Relacionados | Comando |
|--------|---------------------|---------|
| `ml_engine/production_fraud_engine.py` | test_domain, test_e2e, test_improvements | `pytest tests/test_domain.py tests/test_e2e.py tests/test_improvements.py -v` |
| `ml_engine/feature_engineering.py` | test_domain (TestFeatureEngineering) | `pytest tests/test_domain.py::TestFeatureEngineering -v` |
| `ml_engine/explainability_engine.py` | test_improvements (TestExplainabilityEngine) | `pytest tests/test_improvements.py::TestExplainabilityEngine -v` |
| `ml_engine/device_fingerprint.py` | test_qa_comprehensive | `pytest tests/test_qa_comprehensive.py -v` |

---

### **Backend - API**

| Módulo | Testes Relacionados | Comando |
|--------|---------------------|---------|
| `api/production_api.py` | test_e2e (todos) | `pytest tests/test_e2e.py -v` |
| `api/services/*` | test_e2e (TestE2EAPIEndpoints) | `pytest tests/test_e2e.py::TestE2EAPIEndpoints -v` |
| Security/Auth | test_e2e (TestE2EAuthentication) | `pytest tests/test_e2e.py::TestE2EAuthentication -v` |

---

### **Infraestrutura**

| Módulo | Testes Relacionados | Comando |
|--------|---------------------|---------|
| Database | test_e2e (TestE2EDataPersistence) | `pytest tests/test_e2e.py::TestE2EDataPersistence -v` |
| Cache | test_resilience (TestCacheResilience) | `pytest tests/test_resilience.py::TestCacheResilience -v` |
| Error Handling | test_resilience (TestErrorHandlingResilience) | `pytest tests/test_resilience.py::TestErrorHandlingResilience -v` |

---

## 🚀 Como Usar Esta Matriz

### **Cenário 1: Corrigiu bug em `ml_engine/production_fraud_engine.py`**

```bash
# NÍVEL 1 - Re-executar o teste que falhou
pytest tests/test_domain.py::TestNOME -v

# NÍVEL 2 - Suites relacionadas ao ML Engine
pytest tests/test_domain.py tests/test_improvements.py -v

# NÍVEL 3 - Smoke tests (E2E básico)
pytest tests/test_e2e.py::TestE2EInfrastructure -v
pytest tests/test_e2e.py::TestE2EMLPipeline -v

# RESULTADO: Se tudo passou ✅, correção foi validada
```

---

### **Cenário 2: Corrigiu bug em `api/production_api.py`**

```bash
# NÍVEL 1 - Teste específico que falhou
pytest tests/test_e2e.py::TestNOME -v

# NÍVEL 2 - Suite E2E completa
pytest tests/test_e2e.py -v

# NÍVEL 3 - Resiliência
pytest tests/test_resilience.py -v

# RESULTADO: Se tudo passou ✅, correção foi validada
```

---

### **Cenário 3: Corrigiu bug de Security/Auth**

```bash
# NÍVEL 1 - Teste específico
pytest tests/test_e2e.py::TestE2EAuthentication::test_NOME -v

# NÍVEL 2 - Suite de autenticação
pytest tests/test_e2e.py::TestE2EAuthentication -v

# NÍVEL 3 - QA comprehensive (inclui security)
pytest tests/test_qa_comprehensive.py -v

# RESULTADO: Se tudo passou ✅, correção foi validada
```

---

## 📌 Regra de Ouro

**Se corrigir X, sempre executar:**
1. ✅ Teste original que falhou
2. ✅ Suite relacionada ao módulo X
3. ✅ Smoke tests (E2E básico)
4. ✅ Se passou em 3: CORREÇÃO validada ✅

---

## 🔄 Comando Rápido: Executar TODOS os Testes

```bash
cd sankofa-enterprise-real/backend
python -m pytest tests/ -v

# Esperado: 185 passed
```

---

## 📚 Veja também

- `docs/FIX_VALIDATION_CHECKLIST.md` - Checklist passo-a-passo de 3 níveis
- `docs/DEFECTS_LOG.md` - Log de todos os bugs
- `docs/DEFECT_TEMPLATE.md` - Como registrar um novo bug
