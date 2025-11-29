# Arquitetura do Frontend - Sankofa Enterprise Pro

**Versao:** 11.0  
**Data da Analise:** 29/11/2025  
**Analista:** Forca-Tarefa MODO ULTRA (35.000 Especialistas)

---

## 1. Visao Geral da Stack

### 1.1 Ferramentas de Build

| Ferramenta | Versao | Proposito |
|------------|--------|-----------|
| Vite | 6.3.5 | Bundler e dev server |
| React | 19.1.0 | Framework de UI |
| React Router | 7.6.1 | Roteamento SPA |
| TailwindCSS | 4.1.7 | Estilizacao utility-first |
| Radix UI | Various | Componentes acessiveis |

### 1.2 Dependencias Principais

- **UI Components**: Radix UI (15+ primitives), Lucide React (icons)
- **Charts**: Recharts 2.15.3
- **Forms**: React Hook Form + Zod validation
- **Animations**: Framer Motion 12.15.0
- **State Management**: React hooks (useState, useEffect)
- **Date Handling**: date-fns 3.6.0

---

## 2. Estrutura de Diretorios

```
sankofa-enterprise-real/frontend/
├── public/
│   ├── favicon.ico
│   ├── logo-booster.png
│   └── sankofa-logo.png
├── src/
│   ├── assets/
│   │   └── react.svg
│   ├── components/
│   │   ├── charts/
│   │   │   ├── KPICard.jsx          # Cards de metricas
│   │   │   └── SimpleChart.jsx       # Graficos (Line, Area, Bar, Pie)
│   │   ├── layout/
│   │   │   ├── AppBar.jsx           # Barra superior
│   │   │   ├── Layout.jsx           # Layout principal
│   │   │   └── Sidebar.jsx          # Navegacao lateral
│   │   └── ui/
│   │       ├── Badge.jsx            # Badges de status
│   │       ├── Button.jsx           # Botoes
│   │       ├── Card.jsx             # Cards
│   │       ├── Input.jsx            # Inputs
│   │       ├── Slider.jsx           # Sliders de calibracao
│   │       └── Switch.jsx           # Toggles
│   ├── hooks/
│   │   └── use-mobile.js            # Deteccao de mobile
│   ├── lib/
│   │   └── utils.js                 # Utilitarios (cn function)
│   ├── pages/                       # 16 paginas
│   │   ├── Alerts.jsx               # Central de alertas
│   │   ├── Audit.jsx                # Trilhas de auditoria
│   │   ├── Calibration.jsx          # Calibracao de algoritmos
│   │   ├── Dashboard.jsx            # Dashboard executivo
│   │   ├── Datasets.jsx             # Catalogo de datasets
│   │   ├── FeedbackAnalyst.jsx      # Feedback de analistas
│   │   ├── HardRules.jsx            # Regras rigidas
│   │   ├── HotList.jsx              # Lista negra
│   │   ├── Investigation.jsx        # Central de investigacao
│   │   ├── ManualReview.jsx         # Revisao manual
│   │   ├── Metrics.jsx              # Metricas em tempo real
│   │   ├── Monitoring.jsx           # Monitoramento do sistema
│   │   ├── Reports.jsx              # Central de relatorios
│   │   ├── Settings.jsx             # Configuracoes
│   │   ├── Transactions.jsx         # Lista de transacoes
│   │   └── VipList.jsx              # Lista branca
│   ├── providers/
│   │   └── ThemeProvider.jsx        # Provedor de tema
│   ├── styles/
│   │   └── tokens.css               # Design tokens
│   ├── App.jsx                      # Componente raiz
│   ├── App.css                      # Estilos globais
│   ├── index.css                    # Reset CSS
│   └── main.jsx                     # Entry point
├── index.html
├── vite.config.js
├── package.json
└── eslint.config.js
```

---

## 3. Mapa de Rotas

| Rota | Componente | Descricao |
|------|------------|-----------|
| `/` | Dashboard | Dashboard executivo com KPIs |
| `/transactions` | Transactions | Lista e busca de transacoes |
| `/calibration` | Calibration | Calibracao de algoritmos de ML |
| `/investigation` | Investigation | Central de investigacao de fraudes |
| `/manual-review` | ManualReview | Revisao manual Human-in-the-Loop |
| `/monitoring` | Monitoring | Monitoramento de sistema e modelos |
| `/reports` | Reports | Geracao de relatorios |
| `/metrics` | Metrics | Metricas em tempo real |
| `/alerts` | Alerts | Central de alertas |
| `/datasets` | Datasets | Catalogo de datasets |
| `/hard-rules` | HardRules | Regras rigidas de bloqueio |
| `/vip-list` | VipList | Lista branca (aprovacao direta) |
| `/hot-list` | HotList | Lista negra (bloqueio direto) |
| `/audit` | Audit | Trilhas de auditoria |
| `/settings` | Settings | Configuracoes do sistema |
| `*` | 404 | Pagina nao encontrada |

---

## 4. Integracao com Backend

### 4.1 Endpoints Consumidos

| Pagina | Endpoints | Metodo |
|--------|-----------|--------|
| Dashboard | `/api/dashboard/kpis`, `/api/dashboard/timeseries`, `/api/dashboard/channels`, `/api/dashboard/recent-alerts`, `/api/dashboard/model-status` | GET |
| Transactions | `/api/transactions`, `/api/transactions/{id}/approve`, `/api/transactions/{id}/reject`, `/api/transactions/{id}/flag`, `/api/investigations` | GET, POST |
| Calibration | `/api/calibration/config`, `/api/calibration/impact`, `/api/calibration/apply`, `/api/calibration/reset` | GET, POST |
| Investigation | `/api/investigations`, `/api/investigation/{id}` | GET, POST |
| Manual Review | `/api/manual-review` | GET, POST, PUT, DELETE |
| Monitoring | (dados mocados localmente) | - |
| Metrics | `/api/metrics/dashboard` | GET |
| Alerts | `/api/alerts`, `/api/alerts/{id}/status` | GET, PUT |
| Reports | `/api/reports`, `/api/reports/generate` | GET, POST |
| Datasets | `/api/datasets` | GET |
| Hard Rules | `/api/hard-rules` | GET, POST, PUT, DELETE |
| VIP List | `/api/vip-list` | GET, POST, DELETE |
| HOT List | `/api/hot-list` | GET, POST, DELETE |
| Audit | `/api/audit`, `/api/audit/export` | GET, POST |
| Settings | `/api/settings` | GET, PUT |

### 4.2 Proxy de API

```javascript
// vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    secure: false,
  },
}
```

**Nota:** Em producao, o Flask serve tanto a API quanto os arquivos estaticos do frontend na porta 5000.

---

## 5. Fluxo de Dependencias

```
main.jsx
└── App.jsx
    ├── ThemeProvider
    ├── Router
    └── Layout
        ├── AppBar
        ├── Sidebar
        └── Pages (16)
            ├── UI Components (Button, Card, Badge, Input, Slider, Switch)
            ├── Chart Components (KPICard, SimpleChart)
            └── fetch('/api/...')
```

---

## 6. Padroes Identificados

### 6.1 Padroes Positivos

1. **Separacao de Concerns**: Componentes de UI separados de logica de negocio
2. **Design System**: Uso consistente de design tokens via CSS variables
3. **Componentes Acessiveis**: Uso de Radix UI primitives
4. **Tratamento de Loading**: Estados de loading implementados em todas as paginas
5. **Auto-refresh**: Polling automatico para dados em tempo real

### 6.2 Problemas Identificados

1. **Console.logs em Producao**: 44+ console.logs ativos
2. **Dados Mocados**: Monitoring.jsx usa dados hardcoded localmente
3. **Falta de Aria Labels**: Apenas 2 aria-labels em todo o frontend
4. **Fetch Direto**: Algumas paginas usam fetch() diretamente sem camada de servico
5. **Estados Duplicados**: Logica de formatacao duplicada entre componentes

---

## 7. Arquivos de Configuracao

| Arquivo | Proposito |
|---------|-----------|
| `vite.config.js` | Configuracao Vite (proxy, aliases) |
| `package.json` | Dependencias e scripts |
| `eslint.config.js` | Regras de linting |
| `jsconfig.json` | Path aliases (@/) |
| `components.json` | Configuracao shadcn/ui |

---

## 8. Recomendacoes de Melhorias

### 8.1 Curto Prazo (Sprint Atual)

1. Criar camada de servicos (`src/services/api.js`)
2. Remover console.logs em build de producao
3. Adicionar aria-labels em componentes interativos
4. Substituir dados mocados em Monitoring.jsx por chamadas reais

### 8.2 Medio Prazo

1. Implementar testes unitarios com Vitest + Testing Library
2. Adicionar code splitting para lazy loading de rotas
3. Implementar cache de requisicoes com React Query

### 8.3 Longo Prazo

1. Testes E2E com Playwright
2. PWA capabilities
3. Offline support

---

*Documento gerado conforme diretrizes MODO ULTRA*
