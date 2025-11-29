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
# Comando genérico
cd sankofa-enterprise-real/backend
python -m pytest tests/test_ARQUIVO.py::TestCLASSE::test_NOME -v

# Exemplos reais:
python -m pytest tests/test_domain.py::TestBusinessRules::test_high_amount_night_transaction_rule -v
python -m pytest tests/test_e2e.py::TestE2EAuthentication::test_login_success -v
python -m pytest tests/test_resilience.py::TestMLModelResilience::test_model_handles_invalid_data -v

# Esperado
PASSED tests/test_ARQUIVO.py::TestCLASSE::test_NOME [100%]

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
cd sankofa-enterprise-real/backend

# Para bugs de ML Engine:
python -m pytest tests/test_domain.py tests/test_improvements.py -v

# Para bugs de API:
python -m pytest tests/test_e2e.py -v

# Para bugs de Resiliência:
python -m pytest tests/test_resilience.py -v

# Para bugs de QA/Security:
python -m pytest tests/test_qa_comprehensive.py -v

# Esperado
Resultado: NN passed in X.XXs

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
cd sankofa-enterprise-real/backend

# Smoke tests: Infraestrutura E2E
python -m pytest tests/test_e2e.py::TestE2EInfrastructure -v

# Esperado: 4 tests
□ test_frontend_available
□ test_backend_health
□ test_database_connection
□ test_database_tables_exist

# Smoke tests: ML Pipeline
python -m pytest tests/test_e2e.py::TestE2EMLPipeline -v

# Esperado: 3 tests
□ test_model_loaded
□ test_prediction_consistency
□ test_feature_engineering_e2e

# Resultado
□ ✅ 7/7 PASSARAM → CORREÇÃO VALIDADA ✅
□ ❌ ALGUM FALHOU → REGRESSÃO CRÍTICA!
```

**Documentação:**
```
Smoke Tests:     TestE2EInfrastructure + TestE2EMLPipeline
Status:          [ ] PASSOU 7/7  [ ] FALHOU
Falhas:          ________________________
Tempo:           ___ sec
```

---

## 📊 RESUMO RÁPIDO

| Nível | O quê | Tempo | Critério |
|-------|-------|-------|----------|
| **1** | Teste falho original | 5 min | ✅ 1/1 |
| **2** | Suite relacionada | 10 min | ✅ NN/NN |
| **3** | Smoke tests (7 testes) | 5 min | ✅ 7/7 |
| **TOTAL** | **Validação Completa** | **20 min** | **Sem regressões** |

---

## 🔄 Comando Único: Suite Completa (185 testes)

```bash
cd sankofa-enterprise-real/backend
python -m pytest tests/ -v

# Esperado: 185 passed in ~30s
```

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
