# ⚖️ Guia Rápido de Governança - Sankofa Enterprise

**Referência rápida para o processo de defeitos, validação e compliance.**

---

## 🚀 Workflow de Defeito (De A até Z)

### 1️⃣ **Bug Encontrado**
```
Teste falha ou bug descoberto em homologação
         ↓
Abrir docs/DEFECTS_LOG.md
Criar: DEF-2025-NNN (próximo número)
Usar: docs/DEFECT_TEMPLATE.md (5 campos obrigatórios)
         ↓
```

### 2️⃣ **Corrigir Código**
```
Editar arquivo do bug
Adicionar comentário: # FIX DEF-2025-NNN: [descrição]
Testar localmente
         ↓
```

### 3️⃣ **Validar (3 Níveis)**
```
Usar: docs/FIX_VALIDATION_CHECKLIST.md

Nível 1 (5 min):  Re-executar teste falho
Nível 2 (10 min): Suite do módulo
Nível 3 (5 min):  Smoke tests

Se TUDO passou ✅ → Continuar
Se algo falhou ❌ → Voltar para correção
         ↓
```

### 4️⃣ **Documentar Resolução**
```
Atualizar: docs/DEFECTS_LOG.md
Status:    ✅ RESOLVIDO
           com datas, desenvolvedor, validação
         ↓
```

### 5️⃣ **Marcar como Pronto**
```
Commit com: "FIX DEF-2025-NNN: [descrição breve]"
Status:     ✅ PRONTO PARA PRODUÇÃO
```

---

## 📋 Matriz de Decisão

```
┌─ Bug encontrado?
│
├─ É CRÍTICA (security, dados perdidos)?
│  └─ SIM: Corrigir + validar imediatamente
│           Informar Tech Lead
│
├─ É ALTA (funcionalidade quebrada)?
│  └─ SIM: Corrigir + validar no mesmo dia
│
├─ É MÉDIA (funcionalidade parcial)?
│  └─ SIM: Agendar para próxima sprint
│           Documentar no DEFECTS_LOG.md
│
└─ É BAIXA (cosmético)?
   └─ Agendar para melhorias futuras
```

---

## ✅ Checklist Simples

```
□ Bug registrado em DEFECTS_LOG.md com ID DEF-XXXX?
□ Descrito em 5 campos: Título, Severidade, Módulo, Causa, Solução?
□ Código corrigido com comentário # FIX DEF-XXXX?
□ Nível 1 (teste falho): ✅ PASSOU?
□ Nível 2 (suite módulo): ✅ PASSOU?
□ Nível 3 (smoke tests): ✅ PASSOU?
□ Status atualizado para RESOLVIDO?
□ Commit feito?

RESULTADO: ✅ PRONTO OU ❌ VOLTAR?
```

---

## 🔗 Documentos Relacionados

| Documento | Usar quando |
|-----------|---|
| `DEFECT_TEMPLATE.md` | Registrar novo bug |
| `IMPACT_MATRIX.md` | Saber quais testes executar |
| `FIX_VALIDATION_CHECKLIST.md` | Validar correção (3 níveis) |
| `DEFECTS_LOG.md` | Ver histórico de bugs |

---

## 🎯 Exemplo Real (Passo-a-passo)

### Cenário: Feature extraction retorna NaN

```
1️⃣ DESCOBERTA
   Teste falha: test_domain.py::TestFeatureEngineering
   
2️⃣ REGISTRO
   Criar: DEF-2025-001
   Título: Feature extraction retorna NaN para valores > 1.000.000
   Severidade: ALTA
   Módulo: ml_engine/production_fraud_engine.py
   Causa: Divisão por zero em normalização
   
3️⃣ CORREÇÃO
   Editar: ml_engine/production_fraud_engine.py
   Adicionar: np.clip(value, -1e6, 1e6)
   
4️⃣ VALIDAÇÃO (usar FIX_VALIDATION_CHECKLIST.md)
   cd sankofa-enterprise-real/backend
   
   Nível 1: python -m pytest tests/test_domain.py::TestFeatureEngineering -v
            ✅ PASSED
   
   Nível 2: python -m pytest tests/test_domain.py tests/test_improvements.py -v
            ✅ ALL PASSED
   
   Nível 3: python -m pytest tests/test_e2e.py::TestE2EMLPipeline -v
            ✅ 3/3 PASSED
   
5️⃣ DOCUMENTAÇÃO
   DEFECTS_LOG.md:
   Status: ✅ RESOLVIDO
   Data: 2025-11-29
   Validação: 3 níveis completos
   
6️⃣ COMMIT
   git commit -m "FIX DEF-2025-001: Clipping em normalização de features"
```

---

## 🔄 Comandos Rápidos de Validação

```bash
cd sankofa-enterprise-real/backend

# Suite completa (185 testes)
python -m pytest tests/ -v

# Smoke tests rápidos (7 testes)
python -m pytest tests/test_e2e.py::TestE2EInfrastructure tests/test_e2e.py::TestE2EMLPipeline -v

# Por categoria:
python -m pytest tests/test_domain.py -v          # 14 testes - Domínio
python -m pytest tests/test_e2e.py -v             # 31 testes - E2E
python -m pytest tests/test_improvements.py -v    # ~40 testes - Improvements
python -m pytest tests/test_qa_comprehensive.py -v # ~50 testes - QA
python -m pytest tests/test_resilience.py -v      # 10 testes - Resiliência
```

---

## 🚨 Red Flags (Parar e chamar Tech Lead!)

```
❌ Nível 3 (smoke test) falhou após correção
   → Pode ser regressão crítica
   → Considerar reverter
   → Investigar causa raiz
   
❌ Múltiplos testes falhando ao corrigir 1 bug
   → Possível efeito colateral
   → Voltar e revisar solução
   
❌ Security test falhando
   → PARAR TUDO
   → Chamar Security Officer
   → Não fazer deploy
```

---

## 📞 Contatos Rápidos

| Situação | Quem chamar |
|----------|---|
| Bug de security | Security Officer |
| Bug de ML/Fraude | ML Lead |
| Bug de API/Backend | Backend Lead |
| Bug de Frontend | Frontend Lead |
| Regressão crítica | Tech Lead |

---

**Última atualização: 2025-11-29**
**Versão: 1.1 - Comandos Reais Validados**
