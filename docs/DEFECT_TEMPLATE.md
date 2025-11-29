# 🐛 Template de Registro de Defeito

Use este template para documentar CADA bug encontrado. **5 campos obrigatórios:**

---

## DEF-AAAA-NNN

```yaml
ID:          DEF-2025-001
TÍTULO:      Feature extraction retorna NaN para valores > 1.000.000
SEVERIDADE:  ALTA         # CRÍTICA | ALTA | MÉDIA | BAIXA
MÓDULO:      ml_engine/production_fraud_engine.py
STATUS:      ABERTO       # ABERTO | RESOLVIDO ✅ | CANCELADO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEMA:
├─ O quê?      Feature engineering retorna NaN
├─ Onde?       Normalização de amount (linha 145)
├─ Quando?     Valores de transação > R$ 1.000.000

CAUSA:
├─ Divisão por zero em normalização
├─ Falta de clipping de valores extremos

SOLUÇÃO:
├─ Adicionar: np.clip(value, -1e6, 1e6)
├─ Re-normalizar após clipping

TESTES FALHANDO:
├─ test_unit_ml_006
├─ test_integration_api_to_ml
├─ test_e2e_full_flow

VALIDAÇÃO:
✅ Teste falho passou?      SIM
✅ Suite do módulo passou?   SIM
✅ Smoke tests passaram?     SIM
```

---

## 📋 Exemplos de Campos

### SEVERIDADE
- **CRÍTICA**: Vulnerabilidade security, dados perdidos, sistema offline
- **ALTA**: Funcionalidade quebrada, impacta múltiplos fluxos
- **MÉDIA**: Funcionalidade parcial, workaround disponível
- **BAIXA**: Cosmético, melhorias, edge cases

### MÓDULO
Caminho do arquivo afetado:
- `ml_engine/production_fraud_engine.py`
- `api/production_api.py`
- `frontend/src/pages/Dashboard.tsx`
- `database/repository.py`

### STATUS
- **ABERTO**: Bug confirmado, aguardando correção
- **RESOLVIDO ✅**: Correção implementada e validada
- **CANCELADO**: Não é um bug, é behavior esperado

---

## 🔗 Documentação Completa

Veja mais detalhes em:
- `docs/DEFECTS_LOG.md` - Log de todos os defeitos
- `docs/IMPACT_MATRIX.md` - Quais testes executar após correção
- `docs/FIX_VALIDATION_CHECKLIST.md` - Checklist de 3 níveis
