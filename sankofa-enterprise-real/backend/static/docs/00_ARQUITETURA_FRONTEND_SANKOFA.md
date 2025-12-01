# 00 - Arquitetura do Frontend Sankofa Enterprise Pro

**Data da Analise:** 29/11/2025  
**Versao do Sistema:** v11.0  
**Metodologia:** MODO DOUBLE CHECK ULTRA (35.000 Especialistas)

---

## 1. Stack Tecnologica (Confirmada via package.json)

| Tecnologia | Versao | Proposito |
|------------|--------|-----------|
| React | 19.1.0 | Framework de UI |
| Vite | 6.3.5 | Bundler e Dev Server |
| React Router DOM | 7.6.1 | Roteamento SPA |
| TailwindCSS | 4.1.7 | Estilizacao utility-first |
| Radix UI | Various | Primitivos acessiveis |
| Lucide React | 0.513.0 | Iconografia |
| Recharts | 2.15.3 | Visualizacao de dados |
| Framer Motion | 12.15.0 | Animacoes |
| date-fns | 3.6.0 | Manipulacao de datas |
| clsx + tailwind-merge | - | Utilitarios de classes |

---

## 2. Estrutura de Diretorios (Mapa Real)

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
│   │   │   ├── KPICard.jsx
│   │   │   └── SimpleChart.jsx
│   │   ├── layout/
│   │   │   ├── AppBar.jsx
│   │   │   ├── Layout.jsx
│   │   │   └── Sidebar.jsx
│   │   └── ui/
│   │       ├── Badge.jsx
│   │       ├── Button.jsx
│   │       ├── Card.jsx
│   │       ├── Input.jsx
│   │       ├── Slider.jsx
│   │       └── Switch.jsx
│   ├── hooks/
│   │   └── use-mobile.js
│   ├── lib/
│   │   └── utils.js
│   ├── pages/
│   │   ├── Alerts.jsx
│   │   ├── Audit.jsx
│   │   ├── Calibration.jsx
│   │   ├── Dashboard.jsx
│   │   ├── Datasets.jsx
│   │   ├── FeedbackAnalyst.jsx
│   │   ├── HardRules.jsx
│   │   ├── HotList.jsx
│   │   ├── Investigation.jsx
│   │   ├── ManualReview.jsx
│   │   ├── Metrics.jsx
│   │   ├── Monitoring.jsx
│   │   ├── Reports.jsx
│   │   ├── Settings.jsx
│   │   ├── Transactions.jsx
│   │   └── VipList.jsx
│   ├── providers/
│   │   └── ThemeProvider.jsx
│   ├── styles/
│   │   └── tokens.css
│   ├── App.css
│   ├── App.jsx
│   ├── index.css
│   └── main.jsx
├── components.json
├── eslint.config.js
├── index.html
├── jsconfig.json
├── package.json
├── postcss.config.js
├── tailwind.config.js
└── vite.config.js
```

---

## 3. Configuracao do Build (vite.config.js)

```javascript
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@assets': path.resolve(__dirname, './src/assets'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
```

**Nota:** Em producao, o Flask serve os arquivos estaticos do frontend diretamente na porta 5000.

---

## 4. Entry Point (main.jsx)

```javascript
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import { ThemeProvider } from './providers/ThemeProvider.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
)
```

---

## 5. Sistema de Rotas (App.jsx)

| Rota | Componente | Responsabilidade |
|------|------------|------------------|
| `/` | Dashboard | KPIs e visao geral |
| `/transactions` | Transactions | Lista de transacoes |
| `/calibration` | Calibration | Calibracao de modelos |
| `/investigation` | Investigation | Investigacao de fraudes |
| `/manual-review` | ManualReview | Revisao Human-in-the-Loop |
| `/monitoring` | Monitoring | Monitoramento do sistema |
| `/reports` | Reports | Relatorios |
| `/metrics` | Metrics | Metricas em tempo real |
| `/alerts` | Alerts | Central de alertas |
| `/datasets` | Datasets | Catalogo de datasets |
| `/hard-rules` | HardRules | Regras rigidas |
| `/vip-list` | VipList | Lista branca |
| `/hot-list` | HotList | Lista negra |
| `/audit` | Audit | Trilhas de auditoria |
| `/settings` | Settings | Configuracoes |
| `/feedback-analyst` | FeedbackAnalyst | Feedback de analistas |
| `*` | - | 404 Not Found |

---

## 6. Design System

### 6.1 Design Tokens (tokens.css)

- Cores primarias da marca (brand)
- Cores semanticas (success, warning, error, info)
- Tipografia (font-family, sizes)
- Espacamento (spacing scale)
- Border radius
- Sombras

### 6.2 Componentes Base (Radix UI)

- Button (variants: primary, secondary, outline, ghost, destructive)
- Card (Card, CardHeader, CardTitle, CardContent, CardFooter)
- Badge (variants para status)
- Input (com labels e validacao)
- Slider (para calibracao)
- Switch (toggles)

---

## 7. Integracao com Backend

### 7.1 Proxy de Desenvolvimento

O Vite proxy redireciona `/api/*` para `localhost:8000` (Flask backend).

### 7.2 Producao

Em producao, o Flask serve:
- Arquivos estaticos do frontend (dist/)
- API REST na mesma porta 5000

### 7.3 Endpoints Principais Consumidos

- `/api/dashboard/*` - KPIs, timeseries, channels, alerts, model-status
- `/api/transactions` - CRUD de transacoes
- `/api/fraud/predict` - Predicao de fraude (CRITICO - latencia alta)
- `/api/calibration/*` - Configuracao de modelos
- `/api/alerts` - Central de alertas
- `/api/observability/*` - Metricas e SLA
- `/api/health/*` - Health checks

---

## 8. Bundle Analysis

| Arquivo | Tamanho | Otimizacao |
|---------|---------|------------|
| index-BktzNHsD.js | 878 KB | Minificado |
| index-BIM-x7Sm.css | ~50 KB | Minificado |
| recharts-*.js | ~475 KB | Tree-shaking parcial |
| framer-motion-*.js | ~138 KB | Lazy loading recomendado |
| react-dom-*.js | ~157 KB | Necessario |
| **TOTAL** | ~3.8 MB | - |

### Recomendacoes de Otimizacao

1. **Code Splitting**: Lazy load de rotas menos usadas
2. **Tree Shaking**: Importar apenas componentes necessarios do Recharts
3. **Compression**: Habilitar gzip/brotli no servidor

---

## 9. Scripts NPM

| Script | Comando | Descricao |
|--------|---------|-----------|
| dev | `vite` | Servidor de desenvolvimento |
| build | `vite build` | Build de producao |
| preview | `vite preview` | Preview do build |
| lint | `eslint .` | Linting |

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 0*
