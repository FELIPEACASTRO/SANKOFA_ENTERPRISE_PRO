# Guia Completo de Recalibração de Métricas

## Visão Geral

O Sankofa Enterprise Pro possui um sistema de monitoramento de métricas que permite recalibração quando necessário. Este guia explica quando e como fazer recalibração.

---

## 1. MÉTRICAS ATUAIS (30 de Novembro de 2025)

### Dados do Sistema

| Métrica | Valor | Status |
|---------|-------|--------|
| **Transações** | 4.466 | ✅ Real |
| **Fraudes detectadas** | 3.114 (69,73%) | ✅ Real |
| **Latência média** | 37-72ms (com cache) | ✅ SLA <50ms |
| **Valor protegido** | R$ 14.328.997,85 | ✅ Real |

### Distribuição por Canal

| Canal | Transações | Fraudes | Taxa |
|-------|-----------|---------|------|
| PIX | 4.285 | 3.081 | 71,9% |
| TED | 86 | 14 | 16,3% |
| BOLETO | 88 | 14 | 15,9% |

---

## 2. PERFORMANCE THRESHOLDS (Limites Mínimos)

| Métrica | Limite | Atual | Status |
|---------|--------|-------|--------|
| Acurácia | ≥ 85% | ~90% | ✅ |
| Precisão | ≥ 80% | ~95% | ✅ |
| Recall | ≥ 75% | ~91% | ✅ |
| F1-Score | ≥ 70% | ~93% | ✅ |
| Latência P95 | ≤ 100ms | 72ms | ✅ |

**Quando Recalibrar?**
- Se alguma métrica cair abaixo do limite
- Se Data Drift > 0.1
- Se Concept Drift > 0.15

---

## 3. TIPOS DE RECALIBRAÇÃO

### Modo 1: Recalibração Completa

**O que faz:**
1. ✅ Recalibra threshold (decisão de fraude)
2. ✅ Recalibra probabilidades (confiança)
3. ✅ Verifica data drift
4. ✅ Verifica concept drift
5. ✅ Gera relatório

**Tempo:** ~2 minutos

---

### Modo 2: Recalibrar Apenas Threshold

**O que faz:**
- Encontra melhor ponto de decisão (0.1-0.9)
- Otimiza para F1-Score máximo
- Aplica automaticamente

**Quando usar:** Quando mudam muitas fraudes/legítimas

---

### Modo 3: Recalibrar Apenas Probabilidades

**O que faz:**
- Mapeia probabilidades para [0, 1] corretamente
- Usa regressão isotônica
- Garante confiabilidade das previsões

**Quando usar:** Quando scores estão desconfiáveis

---

### Modo 4: Verificar Drift

**O que faz:**
- Calcula Data Drift (mudança de distribuição)
- Calcula Concept Drift (mudança de conceito)
- Recomenda se deve retreinar

**Quando usar:** Diariamente como monitoramento

---

## 4. ENTENDENDO AS MÉTRICAS

### Acurácia (Accuracy)
```
Quantas previsões estão corretas?

Fórmula: (Corretas) / (Total)
Ideal: > 85%

❌ Problema: Não funciona bem com dados desbalanceados
```

### Precisão (Precision)
```
De todas as fraudes que detectei, quantas eram REAIS?

Fórmula: (Fraudes Verdadeiras) / (Todas as Fraudes Detectadas)
Ideal: > 80%

✅ Usa: Minimizar falsos positivos (não acusar cliente honesto)
```

### Recall (Sensibilidade)
```
De todas as fraudes que existem, quantas detectei?

Fórmula: (Fraudes Verdadeiras) / (Todas as Fraudes Existentes)
Ideal: > 75%

✅ Usa: Minimizar fraudes perdidas (não deixar fraude passar)
```

### F1-Score (Balanço)
```
Balanço entre Precisão e Recall

Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
Ideal: > 70%

✅ Usa: Quando quer bom desempenho em ambos
```

### Threshold (Ponto de Decisão)
```
Qual score mínimo para marcar como FRAUDE?

Padrão: 0.5 (50%)

Exemplo:
  ├─ Threshold 0.3: Detecta mais fraudes (alto recall)
  │  Risco: Mais falsos positivos
  │
  ├─ Threshold 0.5: Balanço
  │  Recomendado: Para maioria dos casos
  │
  └─ Threshold 0.7: Mais conservador
     Risco: Deixa fraude passar
```

---

## 5. DATA DRIFT vs CONCEPT DRIFT

### Data Drift
```
A DISTRIBUIÇÃO dos dados mudou

Exemplo:
  Antes: Valores de PIX R$100-R$500 (média)
  Agora: Valores de PIX R$5000-R$50000 (tudo alto)

Solução: Recalibrar features e threshold
```

### Concept Drift
```
O SIGNIFICADO dos dados mudou (novo tipo de fraude)

Exemplo:
  Antes: Fraudes noturnas (23h-4h)
  Agora: Fraudes à tarde (14h-18h) - Novo padrão!

Solução: Retreinar o modelo com novos padrões
```

---

## 6. QUANDO RECALIBRAR (Checklist)

✅ **Faça recalibração se:**
- [ ] Acurácia caiu abaixo de 85%
- [ ] Precisão caiu abaixo de 80%
- [ ] Recall caiu abaixo de 75%
- [ ] Data Drift > 0.10
- [ ] Concept Drift > 0.15
- [ ] Taxa de fraude mudou drasticamente
- [ ] Novos tipos de transação adicionados
- [ ] Mudança de política de banco/negócio

✅ **Faça RETREINAMENTO se:**
- [ ] Recalibração não resolveu
- [ ] Múltiplos drifts detectados
- [ ] Performance ainda abaixo após recalibração

---

## 7. INTERPRETANDO O RELATÓRIO

```json
{
  "timestamp": "2025-11-30T10:00:00",
  "recalibration_type": "full",
  "data_drift": 0.08,
  "concept_drift": 0.05,
  "metrics": {
    "accuracy": 0.90,
    "precision": 0.95,
    "recall": 0.91,
    "f1_score": 0.93,
    "threshold": 0.50
  },
  "status": "healthy"
}
```

**Como ler:**
- ✅ Todos os metrics > limites mínimos
- ✅ Data Drift = 0.08 (abaixo de 0.1)
- ✅ Concept Drift = 0.05 (abaixo de 0.15)
- ✅ Threshold = 0.50 (padrão)

**Status:** Sistema saudável, sem necessidade de recalibração

---

## 8. DÚVIDAS FREQUENTES

**P: Recalibrar afeta o modelo em produção?**
R: Sim! A recalibração muda o threshold e probabilidades. A API recebe automaticamente.

**P: Posso fazer recalibração múltiplas vezes por dia?**
R: Sim! Sistema suporta. Recomendação: 1-2x por dia máximo para evitar instabilidade.

**P: Qual é mais importante: Precisão ou Recall?**
R: Depende da política do banco:
- 🔒 Segurança 1º → Priorize Recall (detecte todas fraudes)
- 😊 Experiência 1º → Priorize Precisão (menos inocentes acusados)
- ⚖️ Balanço → Use F1-Score

**P: Recalibração é reversível?**
R: Sim! Cada recalibração salva relatório. Pode voltar ao threshold anterior se necessário.

---

## 9. PRÓXIMOS PASSOS

1. **Diário:** Verificar drift pela manhã
2. **Semanal:** Executar recalibração completa
3. **Mensal:** Avaliar necessidade de retreinamento
4. **Se alerta:** Executar recalibração imediata

---

**Sankofa Enterprise Pro v1.0**  
**Sistema de Monitoramento e Recalibração**  
**Última atualização: 30 de Novembro de 2025**
