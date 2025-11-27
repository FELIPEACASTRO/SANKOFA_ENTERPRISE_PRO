# Sankofa Enterprise Pro - Documentacao v12.0

**Sistema de Deteccao de Fraudes Enterprise-Grade**  
**Versao:** 12.0 | **Ultima Atualizacao:** 27 de Novembro de 2025

---

## Status do Sistema

| Componente | Status | Versao |
|------------|--------|--------|
| Backend API (Flask) | ✅ Producao | 50+ endpoints |
| Frontend Dashboard (React) | ✅ Producao | 9 paginas |
| ML Stacking Ensemble | ✅ Producao | RF + GB + LR |
| Explicabilidade SHAP | ✅ Integrado na API | LGPD Compliant |
| Observabilidade | ✅ Producao | Prometheus/SLA |
| Infraestrutura Escala | ✅ Producao | 33.88 TPS testado |
| PostgreSQL | ✅ Integrado | Neon-backed |
| Testes Automatizados | ✅ 25 E2E passando | 100% |

---

## Indice de Documentacao

| Documento | Publico-Alvo | Descricao |
|-----------|--------------|-----------|
| [ARQUITETURA_TECNICA.md](./ARQUITETURA_TECNICA.md) | Desenvolvedores, Arquitetos, DevOps | Stack, componentes, APIs, banco de dados |
| [DOCUMENTACAO_FUNCIONAL.md](./DOCUMENTACAO_FUNCIONAL.md) | Analistas, Product Owners | Casos de uso, regras de negocio, compliance |
| [MANUAL_USUARIO.md](./MANUAL_USUARIO.md) | Analistas de Fraude, Gerentes | Guia pratico do dashboard |
| [DIAGRAMAS.md](./DIAGRAMAS.md) | Todos | Fluxogramas e diagramas ASCII/Mermaid |
| [USE_A_CABECA_FRAUDES.md](./USE_A_CABECA_FRAUDES.md) | Todos | **NOVO!** Guia didatico estilo Head First com casos reais, ilustracoes e exercicios |
| [RELATORIO_QA.md](./RELATORIO_QA.md) | QA, Desenvolvedores | Testes automatizados e metricas |
| [BLUEPRINT_MOTOR_FRAUDE_300M.md](./BLUEPRINT_MOTOR_FRAUDE_300M.md) | Arquitetos Enterprise | Blueprint para 300M req/dia |

---

## Novos Recursos v12.0

### 1. Explicabilidade LGPD (NOVO)

Cada predicao de fraude agora inclui explicacoes automaticas para compliance LGPD:

```json
{
  "predictions": [{
    "is_fraud": true,
    "risk_score": 87.5,
    "explanation_text": "Transacao de alto valor (R$ 15.000) em horario noturno (03:00) com velocidade acima do padrao",
    "top_risk_factors": [
      {"feature": "amount_normalized", "impact": 0.45},
      {"feature": "is_night", "impact": 0.32}
    ],
    "top_protective_factors": [
      {"feature": "device_risk_score", "impact": -0.15}
    ],
    "lgpd_compliant": true,
    "compliance_report": {
      "lgpd": "Explicacao fornecida conforme Art. 20 LGPD",
      "bacen": "Tempo de resposta dentro do SLA",
      "pci_dss": "Dados sensiveis mascarados"
    }
  }]
}
```

**Endpoint:** `POST /api/fraud/predict` com `include_explanation: true`

### 2. Observabilidade Prometheus (NOVO)

Sistema completo de metricas em tempo real:

| Endpoint | Descricao |
|----------|-----------|
| `/api/observability/metrics` | Metricas JSON (TPS, latencia, error rate) |
| `/api/observability/prometheus` | Formato Prometheus para Grafana |
| `/api/observability/sla` | Verificacao de compliance SLA |
| `/api/health/detailed` | Health check detalhado por componente |

**Metricas Disponiveis:**
- TPS (transacoes por segundo)
- Latencia p50, p95, p99
- Taxa de erro
- Taxa de fraude
- Alertas disparados

### 3. Infraestrutura de Escala (NOVO)

Processamento em batch otimizado para alta performance:

| Endpoint | Descricao | Throughput |
|----------|-----------|------------|
| `/api/infrastructure/batch/process` | Batch paralelo | 33.88 TPS |
| `/api/infrastructure/task/submit` | Fila assincrona | Prioridades |
| `/api/infrastructure/queue/metrics` | Metricas da fila | Circuit breaker |

**Componentes:**
- `AsyncTaskQueue`: Fila com prioridades e workers
- `BatchProcessor`: Processamento paralelo
- `CircuitBreaker`: Protecao contra falhas em cascata

---

## Navegacao Rapida

| Preciso de... | Documento |
|---------------|-----------|
| Entender a arquitetura do sistema | [ARQUITETURA_TECNICA.md](./ARQUITETURA_TECNICA.md) |
| Ver como funciona o fluxo de fraude | [DIAGRAMAS.md](./DIAGRAMAS.md) |
| Entender os casos de uso | [DOCUMENTACAO_FUNCIONAL.md](./DOCUMENTACAO_FUNCIONAL.md) |
| Aprender a usar o dashboard | [MANUAL_USUARIO.md](./MANUAL_USUARIO.md) |
| Ver regras de compliance | [DOCUMENTACAO_FUNCIONAL.md](./DOCUMENTACAO_FUNCIONAL.md#7-compliance-e-regulamentacao) |
| Entender o modelo ML | [ARQUITETURA_TECNICA.md](./ARQUITETURA_TECNICA.md#4-motor-de-machine-learning) |
| Configurar observabilidade | [ARQUITETURA_TECNICA.md](./ARQUITETURA_TECNICA.md#8-observabilidade) |
| Usar processamento em batch | [ARQUITETURA_TECNICA.md](./ARQUITETURA_TECNICA.md#9-infraestrutura-de-escala) |

---

## Metricas de Performance Validadas

| Metrica | Valor | Condicao |
|---------|-------|----------|
| Throughput Batch | 33.88 TPS | 50 transacoes paralelas |
| Latencia p50 | 28ms | Modelo aquecido |
| Latencia p95 | 300ms | Inclui cold start |
| Latencia p99 | 311ms | Inclui cold start |
| Testes E2E | 25/25 | 100% passando |
| Recall ML | 90.9% | Deteccao de fraude |
| Precisao ML | 100% | Sem falsos positivos |

---

## Versao da Documentacao

| Documento | Versao | Ultima Atualizacao |
|-----------|--------|-------------------|
| Arquitetura Tecnica | 12.0 | 27 Nov 2025 |
| Documentacao Funcional | 12.0 | 27 Nov 2025 |
| Manual do Usuario | 12.0 | 27 Nov 2025 |
| Relatorio QA | 12.0 | 27 Nov 2025 |
| Diagramas | 1.0.0 | Novembro 2025 |

---

## Contato

Para duvidas sobre a documentacao ou sugestoes de melhoria, entre em contato com a equipe de engenharia.

---

**Sankofa Enterprise Pro v12.0** - Protegendo instituicoes financeiras com inteligencia artificial.  
*Ultima atualizacao: 27 de Novembro de 2025*
