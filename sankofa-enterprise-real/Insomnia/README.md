# Insomnia - Sankofa Fraud Engine API v13.0

## Colecao de Testes da API

Este diretorio contem a colecao completa de requisicoes para testar todos os endpoints da API do Motor de Fraude Sankofa.

## Estrutura do Diretorio

```
Insomnia/
├── collections/
│   └── sankofa_api_collection.json    # Colecao principal (60+ requests)
├── environments/
│   └── development.json                # Ambientes dev/replit/prod
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

## Categorias de Endpoints (14 categorias)

| # | Categoria | Endpoints | Descricao |
|---|-----------|-----------|-----------|
| 01 | Health & Status | 4 | Verificacao de saude do sistema |
| 02 | Autenticacao | 5 | Login, verificacao e refresh de token JWT |
| 03 | Deteccao de Fraude | 5 | Predicao de fraude (cenarios diversos) |
| 04 | Modelo ML | 3 | Metricas e treinamento do modelo |
| 05 | Explicabilidade LGPD | 2 | Explicacoes de decisoes para compliance |
| 06 | Dashboard | 9 | Dados do dashboard executivo |
| 07 | Transacoes | 6 | Acoes sobre transacoes (approve, reject, review, flag) |
| 08 | Revisao Manual | 5 | Human-in-the-Loop review |
| 09 | Alertas | 3 | Gerenciamento de alertas |
| 10 | Listas | 10 | VIP (whitelist), Hot (blacklist), Hard Rules |
| 11 | Calibracao | 5 | Ajuste de thresholds e parametros |
| 12 | Configuracoes | 3 | Settings gerais do sistema |
| 13 | Auditoria | 2 | Log de auditoria LGPD/BACEN |
| 14 | Metricas | 1 | Metricas consolidadas |

**Total: 63 requisicoes organizadas em 14 categorias**

## Cenarios de Teste Incluidos

### Deteccao de Fraude
- Transacao de baixo risco (aprovacao automatica)
- Transacao de alto risco (bloqueio)
- Transacao de risco medio (revisao manual)
- Batch processing de multiplas transacoes
- Payload invalido (validacao de erro)

### Autenticacao
- Login como admin
- Login como analista
- Login com credenciais invalidas
- Verificacao de token
- Renovacao de token

### Gestao de Listas
- VIP List (whitelist) - GET/POST/DELETE
- Hot List (blacklist) - GET/POST/DELETE
- Hard Rules - GET/POST/PUT/DELETE

### Acoes sobre Transacoes
- Aprovar transacao
- Rejeitar transacao
- Enviar para revisao
- Marcar flag
- Criar investigacao

## Variaveis de Ambiente

| Variavel | Descricao | Valor Default |
|----------|-----------|---------------|
| `base_url` | URL base da API | http://localhost:5000 |
| `jwt_token` | Token JWT para autenticacao | (vazio, preencher apos login) |
| `admin_username` | Usuario admin | admin |
| `admin_password` | Senha admin | admin123 |
| `analyst_username` | Usuario analista | analyst |
| `analyst_password` | Senha analista | analyst123 |

## Ambientes Disponiveis

1. **Development (Local)** - `http://localhost:5000`
2. **Replit Preview** - `https://sankofaenterprisepro--felipesp1983wor.replit.app`
3. **Production** - URL de producao (configurar)

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

5. **Testar Acoes**
   - Execute acoes de aprovar/rejeitar transacoes
   - Verifique listas VIP/Hot
   - Teste calibracao do modelo

## Documentacao de Referencia

- [ARQUITETURA_TECNICA.md](../docs/ARQUITETURA_TECNICA.md) - Detalhes tecnicos da API
- [DOCUMENTACAO_FUNCIONAL.md](../docs/DOCUMENTACAO_FUNCIONAL.md) - Casos de uso
- [RELATORIO_QA.md](../docs/RELATORIO_QA.md) - Resultados de testes automatizados

---

*Sankofa Enterprise Pro v13.0*  
*Colecao Insomnia atualizada em 01 de Dezembro de 2025*
