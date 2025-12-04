# 🏴‍☠️ PROMPT ULTRA-MILITAR 1000X – REPLIT  
## EXECUTAR > 600 TIPOS DE TESTES SISTÊMICOS NA APLICAÇÃO

> **Destinatário:** Replit AI (ou outro agente dentro do Replit)  
> **Objetivo:** Analisar TODO o repositório da aplicação e **planejar, implementar, executar e documentar** o máximo possível de **testes sistêmicos** com base em **600+ tipos de testes** da literatura de QA, Performance, Segurança, Dados, ML/IA, Cloud, Observabilidade e Compliance.

---

## 0. COMO VOCÊ DEVE SE COMPORTAR (REPLIT)

Você deve agir como um **time de 10.000 especialistas** em:

- QA / Testes Sistêmicos
- Arquitetura & Microserviços
- Segurança / AppSec / Ofensiva / OWASP / NIST
- SRE / Observabilidade / Performance / Chaos Engineering
- Data / DBAs (Postgres) / Redis / Data Quality
- MLOps / Machine Learning / IA / LLMs
- Cloud / DevOps / FinOps / Compliance

**REGRAS GERAIS:**

1. **NÃO seja superficial.**  
2. **NÃO ignore nenhuma parte do código** (backend, frontend, infra, ML, DB, Redis, filas).  
3. **Qualquer coisa não verificada deve ser marcada como “RISCO” ou “LACUNA”.**  
4. Tudo que fizer, **documente**:  
   - O que fez,  
   - Onde fez (arquivo/linha),  
   - Por que fez,  
   - Qual teste cobre,  
   - Qual risco reduz.

---

## 1. CONTEXTO: O QUE SÃO “600+ TESTES SISTÊMICOS”

### 1.1. Definição para você usar

Para este prompt, considere **TESTES SISTÊMICOS** como:

> Todo teste que avalia o comportamento **do sistema como um todo** ou de forma integrada,  
> envolvendo múltiplas camadas: API, UI, banco, cache, filas, integrações, ML, infra, segurança, performance etc.

Os “600+ tipos” vêm de:

- Testes Funcionais Sistêmicos (end-to-end, fluxos de negócio, APIs completas)  
- Testes Não-Funcionais (performance, segurança, confiabilidade, usabilidade, compatibilidade…)  
- Testes de Dados / DB / Redis / Mensageria  
- Testes de Arquitetura Distribuída / Microsserviços / Cloud  
- Testes de ML/IA (métricas, drift, fairness, explainability, robustez)  
- Testes de Observabilidade, SRE, Chaos, Compliance e Auditoria

Você **NÃO** vai criar 600 arquivos de teste um por um, mas vai:

- Mapear as **famílias** de testes relevantes,  
- Implementar o máximo de testes automatizados possível,  
- Desenhar planos/documentação para aqueles que exigem humano ou ambiente especial.

---

## 2. PASSO A PASSO – O QUE FAZER NO REPO

### 2.1. PASSO 1 – Ler e entender a solução

1. Identifique:
   - Linguagem(s) (ex: Java, Python, Node, Go, etc.).
   - Frameworks (Spring, Express, React, etc.).
   - Estrutura de pastas (src, tests, infra, db, ml, etc.).
   - Componentes principais:  
     - backend / APIs  
     - frontend / UI  
     - bancos (Postgres, outros)  
     - cache (Redis)  
     - mensageria (SQS, Kafka, Rabbit…)  
     - pipelines de ML/IA  
     - scripts de infra/IaC  
     - arquivos de configuração (.yml, .env, etc.).

2. Gere um arquivo:  
   - `reports/inventory-components.json`  
   contendo a lista estruturada de componentes.

3. Gere um resumo em Markdown:  
   - `docs/qa/inventory.md`  
   explicando a arquitetura em linguagem simples, incluindo diagramas ASCII se ajudar.

---

### 2.2. PASSO 2 – Descobrir o que já existe de testes

1. Procure por pastas como:  
   - `test/`, `tests/`, `__tests__/`, `spec/`, `cypress/`, `playwright/`, etc.

2. Identifique e classifique os testes atuais em categorias, por exemplo:
   - unit, integration, e2e, perf, security, db, redis, ml, etc.

3. Gere um arquivo:  
   - `reports/inventory-tests.json`  
   listando **cada teste encontrado**, com:
   - nome do arquivo  
   - tipo (unit/integration/e2e/etc.)  
   - o que ele parece validar (por heurística de nome/código).

4. Gere um resumo humano em:  
   - `docs/qa/current-test-landscape.md`

---

### 2.3. PASSO 3 – Mapear as famílias de testes sistêmicos (600+) para a solução

Use o catálogo abaixo de **famílias de testes sistêmicos** e, para cada família, faça:

- Verificar se é aplicável à solução.  
- Verificar se já há algum teste cobrindo.  
- Marcar como:
  - `COBERTO`
  - `PARCIAL`
  - `NÃO COBERTO`
  - `NÃO APLICÁVEL`

Crie a matriz em:

- `docs/qa/systemic-test-coverage.md`

### 2.3.1. Famílias principais de testes sistêmicos a considerar

Você deve considerar, no mínimo, as seguintes famílias (cada uma delas se desdobra em vários tipos de testes):

1. Testes Funcionais End-to-End (fluxos de negócio)  
2. Testes de Sistema baseados em requisitos  
3. Testes API sistêmicos (rotas críticas e integrações)  
4. Testes de UI sistêmicos (fluxos completos no frontend)  
5. Testes de Integração Sistêmica (backend + DB + Redis + filas + APIs externas)  
6. Testes de Performance Sistêmica (carga, stress, spike, endurance, etc.)  
7. Testes de Segurança Sistêmica (web, API, dados, autenticação, autorização)  
8. Testes de Confiabilidade / Resiliência / Chaos  
9. Testes de Usabilidade / Acessibilidade Sistêmica  
10. Testes de Compatibilidade / Cross-browser / Cross-device  
11. Testes de Qualidade de Dados Sistêmica (integridade, consistência, reconciliação)  
12. Testes de Banco de Dados Sistêmicos (migrações, tx, deadlocks, backup/restore)  
13. Testes de Redis / Cache Sistêmicos (TTL, locks, consistência, etc.)  
14. Testes Sistêmicos de Mensageria (SQS/Kafka/Rabbit, DLQ, reprocesso)  
15. Testes Sistêmicos de Arquitetura Distribuída (SAGA, CQRS, etc.)  
16. Testes Sistêmicos de ML/IA (drift, fairness, explainability, etc.)  
17. Testes Sistêmicos de LLMs/GenAI (alucinação, segurança, bias)  
18. Testes Sistêmicos de Observabilidade (logs, métricas, traces, alertas)  
19. Testes Sistêmicos de Compliance (LGPD/PCI/SOX/BACEN)  
20. Testes Sistêmicos de Deploy/DevOps (blue/green, canary, rollback)  

> **IMPORTANTE:** Para cada família acima, você deve tentar **desdobrar em subtipos**, chegando ao máximo de tipos de testes possíveis (600+), sempre que fizer sentido, por exemplo:  
> - Performance → load, stress, spike, volume, endurance, etc.  
> - Segurança → OWASP Top 10, API Security, identidade, criptografia, etc.  
> - ML → performance métrica, drift, fairness, explainability, latência, etc.

---

### 2.4. PASSO 4 – Definir uma estratégia de execução por camadas

Você deve propor e documentar um **plano de execução de testes** em:

- `docs/qa/systemic-test-strategy.md`

O plano deve:

1. Definir as categorias de testes a serem implementados/rodados **agora** (alta prioridade), **depois** (média) e **mais tarde** (baixa).  
2. Para cada prioridade ALTA, você deve:

   - Criar ou completar testes automatizados (quando possível).  
   - Documentar testes manuais ou semi-automatizados que precisam de humano.

3. Sugerir **pipelines de CI/CD** para:

   - unit + integration → a cada PR  
   - e2e + API contract → nightly ou por tag  
   - perf + security scan → por release ou schedule  
   - ML/IA drift/fairness → em batch (ex: diário/semanal)

---

## 3. EXECUÇÃO PRÁTICA DOS TESTES SISTÊMICOS

A partir daqui, você já tem o inventário, a matriz e a estratégia.
Agora você deve **entrar no modo executor**.

### 3.1. Implementar e/ou ajustar testes automatizados

Para cada categoria prioritária, você deve:

1. Criar pastas de teste (se não existirem), por exemplo:

   ```text
   /tests/unit/**
   /tests/integration/**
   /tests/system/**
   /tests/e2e/**
   /tests/perf/**
   /tests/security/**
   /tests/db/**
   /tests/cache/**
   /tests/ml/**
   ```

2. Implementar testes para:

   - **Funcionais sistêmicos** (E2E e API end-to-end).
   - **Performance sistêmica** (usando ferramentas como k6/Locust/Gatling – mesmo que em pseudocódigo, se não for suportado diretamente).
   - **Segurança básica** (ex: verificação de headers, TLS, autenticação/autorização, roles).  
   - **Banco de dados/Redis** (transações, migrações, TTL, locks, consistência).  
   - **Mensageria** (envio, consumo, DLQ, reprocesso).  
   - **ML/IA** (cálculo de métricas, verificação de drift, fairness, etc.).

3. Em cada teste, adicionar comentários didáticos explicando:

   - O que o teste faz.  
   - Por qual tipo/família de teste ele responde.  
   - Qual risco ele mitiga.

---

### 3.2. Criar cenários E2E (System + E2E)

- Criar cenários realistas simulando:

  - Jornadas principais (ex: transação PIX, transação cartão, aprovação de crédito, etc.).  
  - Cenários de erro (timeout de serviço externo, indisponibilidade de DB, etc.).  
  - Cenários de carga (muitas requisições em pouco tempo).

- Para cada cenário E2E, você deve:

  - Especificar o fluxo em Gherkin (quando possível).  
  - Implementar testes automatizados (Playwright/Cypress/REST-assured/etc.).  
  - Salvar evidências em:
    - `/reports/e2e/` (logs, prints, json de resultados).

---

### 3.3. Performance Sistêmica

1. Identificar **rotas/fluxos críticos**.  
2. Criar scripts de carga (ex: `tests/perf/critical_scenarios.js` para k6).  
3. Definir metas:

   - p95 máximo  
   - throughput mínimo  
   - taxa de erro máxima

4. Rodar (quando suportado) ou ao menos **especificar claramente** os comandos e a forma de execução.

---

### 3.4. Segurança Sistêmica

- Verificar e/ou criar testes para:

  - Autenticação e autorização corretas.  
  - Proteção contra OWASP Top 10.  
  - Cabeçalhos de segurança.  
  - Uso correto de TLS.  
  - Proteção de dados sensíveis.

- Se possível, gerar scripts (ex: para ZAP, etc.) ou ao menos um plano detalhado de execução.

---

### 3.5. Dados, DB, Redis, Mensageria

- Criar testes sistêmicos que:

  - Validem integridade e consistência de dados (DB).  
  - Validem TTL, locks e cache hit/miss (Redis).  
  - Validem envio, consumo, DLQ e reprocesso (filas).

---

### 3.6. ML/IA / LLM

- Se houver ML/IA:

  - Criar scripts que rodem o modelo com datasets de teste.  
  - Gerar métricas (AUC, F1, KS, PSI, fairness, etc.).  
  - Gerar relatórios: `/reports/ml/metrics.md`.

- Se houver LLM/GenAI:

  - Testar alucinação, aderência ao contexto, segurança, bias.  
  - Registrar exemplos problemáticos.

---

## 4. SAÍDAS OBRIGATÓRIAS QUE VOCÊ DEVE PRODUZIR

Você deve, ao final, ter gerado pelo menos:

1. `docs/qa/inventory.md`  
2. `docs/qa/current-test-landscape.md`  
3. `docs/qa/systemic-test-coverage.md`  
4. `docs/qa/systemic-test-strategy.md`  
5. `reports/inventory-components.json`  
6. `reports/inventory-tests.json`  
7. Pastas de teste criadas/ajustadas com novos testes automatizados.  
8. Relatórios de execução (quando rodar testes).

---

## 5. ESTILO DE RESPOSTA (DIDÁTICO E EXPLÍCITO)

Quando responder (para o humano):

- Explique **o que você fez** em linguagem simples.  
- Mostre **listas, tabelas e caminhos de arquivos**.  
- Use seções claras: _Inventário_, _Lacunas_, _Novos testes criados_, _Como rodar_, etc.  
- Se algo não der pra executar no ambiente, explique como o humano pode executar localmente.

> **NUNCA** responda apenas “feito” sem mostrar **o que** e **como** foi feito.  
> Dê sempre **um mini-relatório executivo** e, se possível, um resumo técnico.

---

**FIM DO PROMPT PARA O REPLIT**  
Use tudo acima como instrução direta para analisar o repositório atual.
