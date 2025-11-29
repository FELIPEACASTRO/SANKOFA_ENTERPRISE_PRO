# ✅ Checklist de Validação de Correção (3 Níveis)

**Use este checklist TODA VEZ que corrigir um bug.**

---

## 📋 Informações da Correção

```
Defeito:       DEF-2025-XXX
Título:        [Descrição breve do bug]
Módulo:        [Caminho do arquivo]
Data Correção: YYYY-MM-DD
Desenvolvedor: [Seu nome]
```

---

## 🔍 NÍVEL 1: Re-executar Teste Falho (5 min)

**Objetivo:** Validar que o bug específico foi corrigido.

```bash
# Comando
pytest tests/unit/test_FILE.py::test_NOME -v

# Esperado
PASSED tests/unit/test_FILE.py::test_NOME [100%]

# Resultado
□ ✅ PASSOU → Continuar para Nível 2
□ ❌ FALHOU → PARAR! Voltar para correção
```

**Documentação:**
```
Teste Original:  test_XXX
Status:          [ ] PASSOU  [ ] FALHOU
Tempo:           ___ ms
```

---

## 🔄 NÍVEL 2: Suite do Módulo (10 min)

**Objetivo:** Garantir que correção não quebrou outros testes do mesmo módulo.

```bash
# Exemplo 1: ML Engine
pytest tests/unit/test_ml_engine_unit.py -v

# Exemplo 2: API
pytest tests/unit/test_api_layer_unit.py -v

# Exemplo 3: Banco de dados
pytest tests/unit/test_database_unit.py -v

# Esperado
Resultado: NNN passed in X.XXs

# Critério
□ ✅ TODOS PASSARAM → Continuar para Nível 3
□ ❌ ALGUM FALHOU → REGRESSÃO DETECTADA!
```

**Documentação:**
```
Suite Testada:   test_FILE.py
Total Testes:    ___
Passaram:        ___
Falharam:        ___
Regressões:      [ ] NÃO  [ ] SIM (documentar quais)
Tempo Total:     ___ min
```

---

## 🎯 NÍVEL 3: Smoke Tests Críticos (5 min)

**Objetivo:** Validar que funções críticas ainda funcionam (teste de fumaça).

```bash
# Executar TODOS os smoke tests
pytest tests/functional/test_smoke.py -v

# Esperado
10 passed in 0.35s

# Testes Críticos
□ test_smoke_01_backend_starts
□ test_smoke_02_database_accessible
□ test_smoke_03_ml_model_loaded
□ test_smoke_04_authentication_works
□ test_smoke_05_prediction_endpoint_responds
□ test_smoke_06_frontend_available
□ test_smoke_07_api_health_check
□ test_smoke_08_database_connection_pool
□ test_smoke_09_cache_operational
□ test_smoke_10_logs_working

# Resultado
□ ✅ 10/10 PASSARAM → CORREÇÃO VALIDADA ✅
□ ❌ ALGUM FALHOU → REGRESSÃO CRÍTICA!
```

**Documentação:**
```
Smoke Tests:     test_smoke.py
Status:          [ ] PASSOU 10/10  [ ] FALHOU
Falhas:          ________________________
Tempo:           ___ sec
```

---

## 📊 RESUMO RÁPIDO

| Nível | O quê | Tempo | Critério |
|-------|-------|-------|----------|
| **1** | Teste falho original | 5 min | ✅ 1/1 |
| **2** | Suite do módulo | 10 min | ✅ NN/NN |
| **3** | Smoke tests | 5 min | ✅ 10/10 |
| **TOTAL** | **Validação Completa** | **20 min** | **Sem regressões** |

---

## 🚨 O que fazer se falhar

### **Nível 1 Falhou**
```
→ Correção não funcionou
→ Voltar para editar o código
→ Re-executar Nível 1
```

### **Nível 2 Falhou**
```
→ Regressão em outro teste do mesmo módulo
→ Registrar como: DEF-2025-XXX-REGRESSÃO
→ Investigar causa
→ Voltar para editar o código
```

### **Nível 3 Falhou**
```
→ Regressão CRÍTICA
→ PARAR TUDO
→ Considerar reverter mudança
→ Chamar Tech Lead
```

---

## ✅ Aprovação Final

```
Desenvolvedor:  ________________  Data: ___/___/___
Validação:      ✅ PASSOU - Sem regressões
```

---

## 📚 Referência

- **DEFECTS_LOG.md** - Onde registrar o defeito
- **IMPACT_MATRIX.md** - Quais testes executar
- **DEFECT_TEMPLATE.md** - Template completo
