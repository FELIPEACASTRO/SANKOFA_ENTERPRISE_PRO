# 01 - Inventario Completo de Arquivos do Frontend

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 1

---

## 1. Paginas (src/pages/)

| Arquivo | Proposito | Linhas |
|---------|-----------|--------|
| Dashboard.jsx | Dashboard executivo com KPIs, graficos e alertas recentes | ~300 |
| Transactions.jsx | Lista, busca, filtros e acoes em transacoes | ~600 |
| Calibration.jsx | Calibracao de 17+ algoritmos de ML (pesos, thresholds) | ~1400 |
| Investigation.jsx | Central de investigacao de casos de fraude | ~400 |
| ManualReview.jsx | Fila de revisao manual Human-in-the-Loop | ~300 |
| Monitoring.jsx | Monitoramento de sistema (CPU, memoria, latencia) | ~340 |
| Reports.jsx | Geracao e download de relatorios | ~300 |
| Metrics.jsx | Metricas e contadores em tempo real | ~160 |
| Alerts.jsx | Central de alertas com filtros e acoes | ~600 |
| Datasets.jsx | Catalogo de datasets para treinamento | ~250 |
| HardRules.jsx | CRUD de regras rigidas de bloqueio | ~350 |
| VipList.jsx | Gerenciamento de lista branca (aprovacao direta) | ~300 |
| HotList.jsx | Gerenciamento de lista negra (bloqueio direto) | ~300 |
| Audit.jsx | Trilhas de auditoria com exportacao | ~350 |
| Settings.jsx | Configuracoes do sistema | ~400 |
| FeedbackAnalyst.jsx | Formulario de feedback de analistas | ~250 |

---

## 2. Componentes de UI (src/components/ui/)

| Arquivo | Proposito | Exportacoes |
|---------|-----------|-------------|
| Button.jsx | Botao com variantes (primary, secondary, outline, ghost, destructive) | Button |
| Card.jsx | Container card com header, title, content, footer | Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter |
| Badge.jsx | Badges de status com cores semanticas | Badge, TransactionStatusBadge, RiskScoreBadge |
| Input.jsx | Campo de entrada com label e validacao | Input, FormField |
| Slider.jsx | Controle deslizante para calibracao | Slider, SliderControl |
| Switch.jsx | Toggle on/off | Switch, SwitchControl |

---

## 3. Componentes de Charts (src/components/charts/)

| Arquivo | Proposito | Exportacoes |
|---------|-----------|-------------|
| KPICard.jsx | Card de metrica com valor, variacao e icone | KPICard |
| SimpleChart.jsx | Wrapper para graficos Recharts | SimpleLineChart, SimpleAreaChart, SimpleBarChart, SimplePieChart |

---

## 4. Componentes de Layout (src/components/layout/)

| Arquivo | Proposito | Exportacoes |
|---------|-----------|-------------|
| Layout.jsx | Layout principal com sidebar e main content | Layout |
| Sidebar.jsx | Navegacao lateral colapsavel com 15 itens | Sidebar |
| AppBar.jsx | Barra superior com logo, search, theme toggle, notificacoes, user menu | AppBar |

---

## 5. Hooks (src/hooks/)

| Arquivo | Proposito | Retorno |
|---------|-----------|---------|
| use-mobile.js | Detecta se dispositivo e mobile (< 768px) | boolean |

---

## 6. Utilitarios (src/lib/)

| Arquivo | Proposito | Exportacoes |
|---------|-----------|-------------|
| utils.js | Funcao `cn()` para merge de classes Tailwind | cn |

---

## 7. Providers (src/providers/)

| Arquivo | Proposito | Exportacoes |
|---------|-----------|-------------|
| ThemeProvider.jsx | Gerencia tema claro/escuro | ThemeProvider, useTheme |

---

## 8. Estilos (src/styles/)

| Arquivo | Proposito |
|---------|-----------|
| tokens.css | Design tokens (cores, tipografia, espacamento) |

---

## 9. Arquivos Raiz (src/)

| Arquivo | Proposito |
|---------|-----------|
| main.jsx | Entry point (ReactDOM.render) |
| App.jsx | Componente raiz com rotas |
| App.css | Estilos globais do app |
| index.css | Reset CSS e imports Tailwind |

---

## 10. Arquivos de Configuracao

| Arquivo | Proposito |
|---------|-----------|
| vite.config.js | Configuracao Vite (plugins, proxy, aliases) |
| tailwind.config.js | Configuracao Tailwind (cores, fonts) |
| postcss.config.js | Plugins PostCSS |
| eslint.config.js | Regras ESLint |
| jsconfig.json | Path aliases para VS Code |
| components.json | Configuracao shadcn/ui |
| package.json | Dependencias e scripts |

---

## 11. Assets Publicos (public/)

| Arquivo | Proposito |
|---------|-----------|
| favicon.ico | Icone do browser |
| sankofa-logo.png | Logo principal |
| logo-booster.png | Logo secundario |

---

## 12. Estatisticas

| Metrica | Valor |
|---------|-------|
| Total de Paginas | 16 |
| Total de Componentes UI | 6 |
| Total de Componentes Charts | 2 |
| Total de Componentes Layout | 3 |
| Total de Hooks | 1 |
| Total de Providers | 1 |
| Total de Arquivos de Config | 7 |

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 1.1*
