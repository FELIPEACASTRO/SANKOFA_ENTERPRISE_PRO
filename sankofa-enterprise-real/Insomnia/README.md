# Insomnia - Sankofa Fraud Engine API v12.0

## Colecao de Testes da API

Este diretorio contem a colecao completa de requisicoes para testar todos os endpoints da API do Motor de Fraude Sankofa.

## Estrutura do Diretorio

```
Insomnia/
├── collections/
│   └── sankofa_api_collection.json    # Colecao principal (70+ requests)
├── environments/
│   └── development.json                # Ambientes dev/prod
├── evidence/
│   └── INSOMNIA_TEST_EVIDENCE.md      # Documentacao de evidencias
└── screenshots/
    └── (capturas de tela dos testes)
```

## Como Importar no Insomnia

1. Abra o Insomnia Desktop
2. Clique em **Import/Export** > **Import Data** > **From File**
3. Selecione `collections/sankofa_api_collection.json`
4. Importe tambem `environments/development.json`

## Categorias de Endpoints

| # | Categoria | Endpoints | Descricao |
|---|-----------|-----------|-----------|
| 01 | Health & Status | 6 | Verificacao de saude do sistema |
| 02 | Autenticacao | 5 | Login, verificacao e refresh de token JWT |
| 03 | Deteccao de Fraude | 8 | Predicao de fraude (cenarios diversos) |
| 04 | Modelo ML | 3 | Metricas e treinamento do modelo |
| 05 | Explicabilidade LGPD | 2 | Explicacoes de decisoes para compliance |
| 06 | Dashboard | 9 | Dados do dashboard executivo |
| 07 | Observabilidade | 4 | Metricas Prometheus e SLA |
| 08 | Transacoes | 3 | Consulta e investigacao de transacoes |
| 09 | Revisao Manual | 3 | Fila de revisao e feedback |
| 10 | Configuracoes | 4 | Settings e calibracao |
| 11 | Listas | 6 | VIP, Hot List e Hard Rules |
| 12 | Alertas | 1 | Gerenciamento de alertas |
| 13 | Auditoria | 2 | Log de auditoria LGPD |
| 14 | Infraestrutura | 4 | Batch e processamento async |
| 15 | Relatorios | 4 | Geracao de relatorios |

**Total: 64 requisicoes organizadas em 15 categorias**

## Cenarios de Teste Incluidos

### Deteccao de Fraude
- Transacao de baixo risco (aprovacao automatica)
- Transacao de alto risco (bloqueio)
- Transacao de risco medio (revisao manual)
- Transacao internacional suspeita
- Transacao com alta velocidade (velocity)
- Batch processing de multiplas transacoes
- Payload invalido (validacao de erro)
- Valor negativo (tratamento de erro)

### Autenticacao
- Login como admin
- Login como analista
- Login com credenciais invalidas
- Verificacao de token
- Renovacao de token

### Performance
- Batch paralelo (5 transacoes)
- Task assincrona
- Metricas de fila

## Variaveis de Ambiente

| Variavel | Descricao | Valor Default |
|----------|-----------|---------------|
| `base_url` | URL base da API | http://localhost:8000 |
| `jwt_token` | Token JWT para autenticacao | (vazio, preencher apos login) |
| `admin_username` | Usuario admin | admin |
| `admin_password` | Senha admin | admin123 |

## Fluxo de Teste Recomendado

1. **Iniciar Backend**
   ```bash
   cd sankofa-enterprise-real/backend
   python api/production_api.py
   ```

2. **Testar Health**
   - Execute: `GET /api/health`
   - Esperado: `{"status": "healthy"}`

3. **Autenticar**
   - Execute: `POST /api/auth/login` com credenciais
   - Copie o `access_token` retornado
   - Cole em `jwt_token` nas variaveis de ambiente

4. **Testar Predicao**
   - Execute cada cenario de fraude
   - Verifique os campos `risk_score`, `is_fraud`, `explanation_text`

5. **Verificar Observabilidade**
   - Execute: `GET /api/observability/metrics`
   - Verifique latencias e TPS

## Documentacao de Referencia

- [ARQUITETURA_TECNICA.md](../docs/ARQUITETURA_TECNICA.md) - Detalhes tecnicos da API
- [DOCUMENTACAO_FUNCIONAL.md](../docs/DOCUMENTACAO_FUNCIONAL.md) - Casos de uso
- [RELATORIO_QA.md](../docs/RELATORIO_QA.md) - Resultados de testes automatizados

---

*Sankofa Enterprise Pro v12.0*  
*Colecao Insomnia criada em 27 de Novembro de 2025*
