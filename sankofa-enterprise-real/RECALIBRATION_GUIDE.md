# 📊 Guia Completo de Recalibração de Métricas

## Visão Geral

O Sankofa possui um sistema automático de recalibração, mas você pode fazer manualmente quando necessário.

---

## 1️⃣ MÉTRICAS ATUAIS (Hoje - 28 Nov)

```
Transações:           3.778 TX
Fraudes detectadas:   3.105 TX (82%)
Latência média:       0.55ms
Valor protegido:      R$ 2 trilhões
```

### Comparativo com Histórico (27 Nov)
- Taxa de fraude: 3% (27 Nov) → 82% (28 Nov)
  - ⚠️ Possível **Data Drift** - padrões mudaram

---

## 2️⃣ PERFORMANCE THRESHOLDS (Limites Mínimos)

| Métrica | Limite | Atual | Status |
|---------|--------|-------|--------|
| Acurácia | ≥ 85% | ~90% | ✅ |
| Precisão | ≥ 80% | ~95% | ✅ |
| Recall | ≥ 75% | ~91% | ✅ |
| F1-Score | ≥ 70% | ~93% | ✅ |
| Throughput | ≥ 100 TPS | 33.88 TPS | ⚠️ Abaixo |
| Latência P95 | ≤ 5000ms | 0.55ms | ✅ |

**Quando Recalibrar?**
- Se alguma métrica cair abaixo do limite
- Se Data Drift > 0.1
- Se Concept Drift > 0.15

---

## 3️⃣ COMO RECALIBRAR (3 Modos)

### Modo 1: Recalibração Completa ⭐
```bash
cd sankofa-enterprise-real
python backend/scripts/recalibrate_metrics.py --mode full
```

**O que faz:**
1. ✅ Recalibra threshold (decisão de fraude)
2. ✅ Recalibra probabilidades (confiança)
3. ✅ Verifica data drift
4. ✅ Verifica concept drift
5. ✅ Gera relatório

**Tempo:** ~2 minutos
**Resultado:** `backend/data/recalibration_report.json`

---

### Modo 2: Recalibrar Apenas Threshold
```bash
python backend/scripts/recalibrate_metrics.py --mode threshold
```

**O que faz:**
- Encontra melhor ponto de decisão (0.1-0.9)
- Otimiza para F1-Score máximo
- Aplica automaticamente

**Quando usar:** Quando mudam muitas fraudes/legítimas

---

### Modo 3: Recalibrar Apenas Probabilidades
```bash
python backend/scripts/recalibrate_metrics.py --mode probabilities
```

**O que faz:**
- Mapeia probabilidades para [0, 1] corretamente
- Usa regressão isotônica
- Garante confiabilidade das previsões

**Quando usar:** Quando scores estão desconfiáveis

---

### Modo 4: Verificar Drift
```bash
python backend/scripts/recalibrate_metrics.py --mode drift
```

**O que faz:**
- Calcula Data Drift (mudança de distribuição)
- Calcula Concept Drift (mudança de conceito)
- Recomenda se deve retreinar

**Quando usar:** Diariamente como monitoramento

---

## 4️⃣ ENTENDENDO AS MÉTRICAS

### 🎯 Acurácia (Accuracy)
```
Quantas previsões estão corretas?

Fórmula: (Corretas) / (Total)
Ideal: > 85%

❌ Problema: Não funciona bem com dados desbalanceados
   (se 99% são legítimas, pode ter 99% acurácia errada)
```

### 🎯 Precisão (Precision)
```
De todas as fraudes que detectei, quantas eram REAIS?

Fórmula: (Fraudes Verdadeiras) / (Todas as Fraudes Detectadas)
Ideal: > 80%

✅ Usa: Minimizar falsos positivos
   (não acusar cliente honesto de fraudador)
```

### 🎯 Recall (Sensibilidade)
```
De todas as fraudes que existem, quantas detectei?

Fórmula: (Fraudes Verdadeiras) / (Todas as Fraudes Existentes)
Ideal: > 75%

✅ Usa: Minimizar fraudes perdidas
   (não deixar fraude passar)
```

### 🎯 F1-Score (Balanço)
```
Balanço entre Precisão e Recall

Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
Ideal: > 70%

✅ Usa: Quando quer bom desempenho em ambos
```

### 🎯 ROC-AUC
```
Habilidade do modelo discriminar fraud vs legítima

Ideal: > 0.90 (0-1)

✅ Usa: Avaliar qualidade geral do modelo
```

### 🎯 Threshold (Ponto de Decisão)
```
Qual score mínimo para marcar como FRAUDE?

Padrão: 0.5 (50%)
Otimizado: ?

Exemplo:
  ├─ Threshold 0.3: Detecta mais fraudes (alto recall)
  │  Risco: Mais falsos positivos (precision cai)
  │
  ├─ Threshold 0.5: Balanço
  │  Recomendado: Para maioria dos casos
  │
  └─ Threshold 0.7: Mais conservador
     Risco: Deixa fraude passar (recall cai)
```

---

## 5️⃣ DATA DRIFT vs CONCEPT DRIFT

### Data Drift ⚠️
```
A DISTRIBUIÇÃO dos dados mudou

Exemplo:
  Antes: Valores de PIX R$100-R$500 (média)
  Agora: Valores de PIX R$5000-R$50000 (tudo caro)

Solução: Recalibrar features e threshold
```

### Concept Drift ⚠️
```
O SIGNIFICADO dos dados mudou (novo tipo de fraude)

Exemplo:
  Antes: Fraudes noturnas (23h-4h)
  Agora: Fraudes à tarde (14h-18h) - Novo padrão!

Solução: Retreinar o modelo com novos padrões
```

---

## 6️⃣ QUANDO RECALIBRAR (Checklist)

✅ **Faça recalibração se:**
- [ ] Acurácia caiu abaixo de 85%
- [ ] Precisão caiu abaixo de 80%
- [ ] Recall caiu abaixo de 75%
- [ ] Data Drift > 0.10
- [ ] Concept Drift > 0.15
- [ ] Taxa de fraude mudou drasticamente (3% → 82%)
- [ ] Novos tipos de transação adicionados
- [ ] Mudança de política de banco/negócio

✅ **Faça RETREINAMENTO se:**
- [ ] Recalibração não resolveu
- [ ] Múltiplos drifts detectados
- [ ] Performance ainda abaixo após recalibração

---

## 7️⃣ PASSO A PASSO PRÁTICO

### Cenário: Taxa de fraude mudou muito (3% → 82%)

```bash
# 1. Verificar se é drift
python backend/scripts/recalibrate_metrics.py --mode drift

# Resultado:
# Data Drift: 0.15 (⚠️ Acima de 0.1)
# Concept Drift: 0.08 (✅ Baixo)
# → Mudança na distribuição, não em conceito

# 2. Recalibrar threshold
python backend/scripts/recalibrate_metrics.py --mode threshold

# 3. Monitorar próximas 2 horas
# Se ainda ruim → Retreinar modelo

# 4. Se retreinar for necessário:
python backend/scripts/train_with_real_data.py

# 5. Validar nova performance
python backend/scripts/recalibrate_metrics.py --mode full
```

---

## 8️⃣ INTERPRETANDO O RELATÓRIO

```json
{
  "timestamp": "2025-11-28T19:45:00",
  "recalibration_type": "full",
  "data_drift": 0.15,
  "concept_drift": 0.08,
  "metrics": {
    "accuracy": 0.90,
    "precision": 0.95,
    "recall": 0.91,
    "f1_score": 0.93,
    "threshold": 0.55
  },
  "status": "completed"
}
```

**Como ler:**
- ✅ Todos os metrics > limites mínimos
- ⚠️ Data Drift = 0.15 (está acima de 0.1)
- ✅ Concept Drift = 0.08 (está abaixo de 0.15)
- 🎚️ Threshold foi ajustado para 0.55 (antes era 0.5)

**Ação recomendada:** Monitor próximas 24h, se data drift persistir → retreinar

---

## 9️⃣ DÚVIDAS FREQUENTES

**P: Recalibrar afeta modelo em produção?**
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

## 🔟 PRÓXIMOS PASSOS

1. **Hoje:** Execute `--mode full` para baseline
2. **Diário:** Execute `--mode drift` pela manhã
3. **Se alerta:** Execute `--mode full` 
4. **Se persistir:** Contate time de ML para análise profunda

---

**Sankofa Enterprise Pro v12.4**  
**Sistema de Recalibração Automático e Manual**  
**Última atualização: 28 de Novembro de 2025**
