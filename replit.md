# Sankofa Enterprise Pro - Sistema de Detecção de Fraude Bancária

## Visão Geral do Projeto

O **Sankofa Enterprise Pro** é uma plataforma completa de detecção de fraude bancária em tempo real, desenvolvida para instituições financeiras de grande porte. O sistema combina Machine Learning avançado, MLOps automatizado e compliance bancário para oferecer proteção máxima contra fraudes financeiras.

## Estrutura do Projeto

```
sankofa-enterprise-real/
├── frontend/              # Interface React com Vite
│   ├── src/
│   │   ├── pages/        # Páginas do dashboard
│   │   ├── components/   # Componentes React reutilizáveis
│   │   └── lib/          # Utilitários
│   └── vite.config.js    # Configuração Vite (porta 5000)
│
├── backend/              # API Flask
│   ├── api/              # Endpoints da API
│   ├── ml_engine/        # Motor de ML para detecção de fraude
│   ├── compliance/       # Módulos de compliance (BACEN, LGPD, PCI-DSS)
│   ├── security/         # Sistema de segurança enterprise
│   ├── cache/            # Sistema de cache Redis
│   ├── performance/      # Otimizações de performance
│   ├── mlops/            # Pipeline MLOps automatizado
│   └── simple_api.py     # API simplificada para demonstração
│
├── docs/                 # Documentação técnica completa
├── models/               # Modelos ML treinados
├── tests/                # Suíte de testes QA
└── reports/              # Relatórios de validação
```

## Tecnologias Principais

### Frontend
- **React 19** com Vite 6
- **Tailwind CSS 4** para estilização
- **Radix UI** componentes acessíveis
- **Recharts** para visualizações
- **React Router** para navegação
- **Shadcn UI** componentes customizados

### Backend
- **Flask** framework web
- **Scikit-learn, XGBoost, LightGBM** para ML
- **Redis** para caching
- **PostgreSQL** para persistência
- **JWT** para autenticação
- **Gunicorn** para produção

## Estado Atual da Configuração

### ✅ Configurado e Funcionando
1. **Frontend**: Rodando na porta 5000 com Vite
2. **Backend**: API simplificada na porta 8445
3. **Proxy**: Vite configurado para redirecionar `/api` → `localhost:8445`
4. **Deployment**: Configurado para Autoscale com build otimizado

### 🔧 Configurações Específicas do Replit
- Frontend configurado com `host: 0.0.0.0` e `port: 5000`
- Backend configurado com `host: localhost` e `port: 8445`
- HMR (Hot Module Replacement) configurado para a porta 5000
- Workflow configurado para iniciar automaticamente o frontend

## Arquitetura e Componentes Principais

### Sistema de Detecção de Fraude
O projeto inclui um motor de ML sofisticado com:
- **47 técnicas de análise** (temporal, geográfica, comportamental)
- **Ensemble de modelos**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Latência ultra-baixa**: ~11ms P95
- **Throughput**: Testado com 118.720 TPS

### Compliance Bancário
- **BACEN**: Resolução Conjunta n° 6 implementada
- **LGPD**: Proteção de dados pessoais com mascaramento
- **PCI DSS**: Segurança de dados de cartão
- **SOX**: Controles internos e auditoria

### MLOps Pipeline
- CI/CD para modelos de ML
- Detecção de drift automática
- Testes adversariais
- Rollback automático
- Gestão de versões de modelos

## Como Executar Localmente

### Desenvolvimento
O workflow já está configurado para iniciar automaticamente. O frontend estará disponível na porta 5000.

Para iniciar manualmente:
```bash
# Frontend
cd sankofa-enterprise-real/frontend
npm run dev

# Backend (em outro terminal)
cd sankofa-enterprise-real/backend
python simple_api.py
```

### Production Build
```bash
cd sankofa-enterprise-real/frontend
npm run build
```

## Dependências Instaladas

### Backend Python
- Flask 2.3.3 (Framework web)
- Flask-CORS (CORS support)
- NumPy, Pandas (Data processing)
- Scikit-learn (ML)
- Redis (Caching)
- E várias outras bibliotecas para ML, segurança e compliance

### Frontend Node.js
- React 19.1.0
- Vite 6.3.5
- Tailwind CSS 4.1.7
- Radix UI components
- Recharts para gráficos
- React Router para navegação

## API Endpoints Disponíveis

### Health Check
- `GET /api/health` - Verifica status da API

### Dashboard
- `GET /api/dashboard/kpis` - Métricas principais
- `GET /api/dashboard/timeseries` - Dados de série temporal
- `GET /api/dashboard/channels` - Dados por canal
- `GET /api/dashboard/alerts` - Alertas do sistema
- `GET /api/dashboard/models` - Status dos modelos ML

### Transações
- `GET /api/transactions` - Lista de transações
- `GET /api/transactions/stats` - Estatísticas das transações

## Resultados de Testes QA

De acordo com a documentação original:
- **Throughput**: 118.720 TPS (1187x superior ao requisito)
- **Latência P95**: 11.08ms
- **Recall**: 90.9%
- **Precision**: 100%
- **F1-Score**: 95.2%
- **Disponibilidade**: 99.9%

## Próximos Passos Recomendados

1. **Integração com Redis**: Configurar Redis para caching de alta performance
2. **Banco de Dados**: Conectar PostgreSQL para persistência real
3. **Autenticação**: Implementar JWT completo para segurança
4. **Modelos ML**: Carregar e integrar os modelos treinados
5. **Monitoramento**: Integrar com DataDog para observabilidade

## Notas Importantes

- O projeto original foi projetado para Docker Compose com múltiplos serviços
- A versão atual usa uma API simplificada para facilitar a execução no Replit
- Todos os componentes complexos (Redis, PostgreSQL, Nginx, Prometheus, Grafana) estão disponíveis no código, mas não estão ativos por padrão
- Para produção bancária real, seria necessário configurar todos os serviços de infraestrutura

## Suporte e Documentação

Consulte a pasta `docs/` para documentação técnica detalhada:
- `DOCUMENTACAO_TECNICA_COMPLETA.md` - Documentação técnica completa
- `ANALISE_COMPLIANCE_BACEN.md` - Análise de compliance BACEN
- `ANALISE_COMPLIANCE_LGPD.md` - Análise de compliance LGPD
- `MANUAL_USUARIO_FINAL.md` - Manual do usuário
- `DEPLOYMENT_GUIDE.md` - Guia de deployment

---

**Status**: ✅ Projeto configurado e pronto para desenvolvimento
**Última atualização**: 08 de Novembro de 2025
