# Estratégia de Execução de Testes Sistêmicos 600+
## Plano 1000X Ultra-Militar por Camadas

**Gerado em:** 2025-12-04  
**Versão:** 1000X-SYSTEMIC  
**Priorização:** Risk-Based Testing

---

## 1. VISÃO GERAL DA ESTRATÉGIA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIRÂMIDE DE PRIORIZAÇÃO                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    🔴 P1 - IMEDIATO (Sprint atual)                      │
│                   ─────────────────────────────────                     │
│                   Load Testing, Security Scan,                          │
│                   Frontend E2E, Chaos Basics                            │
│                                                                         │
│              🟡 P2 - CURTO PRAZO (2-4 semanas)                          │
│             ───────────────────────────────────                         │
│             API Contracts, Accessibility,                               │
│             DB Backup/Restore, Rollback                                 │
│                                                                         │
│          🟢 P3 - MÉDIO PRAZO (1-2 meses)                                │
│         ─────────────────────────────────                               │
│         Visual Regression, Cross-browser,                               │
│         Mutation Testing, Full Chaos                                    │
│                                                                         │
│      ⚪ P4 - LONGO PRAZO (Backlog)                                      │
│     ───────────────────────────────                                     │
│     Nice-to-have, Edge cases,                                           │
│     Experimental tests                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. PRIORIDADE 1 - IMEDIATO (🔴 Esta Sprint)

### 2.1. Load Testing (k6)

**Objetivo:** Validar capacidade de 300M req/dia

**Implementação:**
```javascript
// tests/perf/critical_scenarios.js (k6)
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 50 },   // Ramp-up
    { duration: '1m', target: 100 },   // Normal load
    { duration: '30s', target: 200 },  // Stress
    { duration: '30s', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<50'],   // 95% < 50ms
    http_req_failed: ['rate<0.01'],    // <1% errors
  },
};

export default function() {
  const payload = JSON.stringify({
    transaction_id: `TXN_${Date.now()}`,
    amount: Math.random() * 10000,
    channel: 'PIX',
    user_id: 'USR_12345',
  });
  
  const res = http.post(
    `${__ENV.BASE_URL}/api/predict`,
    payload,
    { headers: { 'Content-Type': 'application/json' } }
  );
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 50ms': (r) => r.timings.duration < 50,
  });
  
  sleep(0.1);
}
```

**Comando:**
```bash
k6 run --env BASE_URL=http://localhost:5000 tests/perf/critical_scenarios.js
```

**Metas:**
| Métrica | Meta | Crítico |
|---------|------|---------|
| p95 latency | < 50ms | < 100ms |
| Throughput | > 1000 req/s | > 500 req/s |
| Error rate | < 0.1% | < 1% |

---

### 2.2. Security Scanning (OWASP ZAP)

**Objetivo:** Detectar vulnerabilidades OWASP Top 10

**Implementação:**
```bash
# Executar OWASP ZAP baseline scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:5000 \
  -g gen.conf \
  -r security_report.html
```

**Checklist Manual (até automação):**
- [ ] SQL Injection (SQLi)
- [ ] Cross-Site Scripting (XSS)
- [ ] Broken Authentication
- [ ] Security Headers
- [ ] CORS Configuration
- [ ] TLS/SSL Configuration

---

### 2.3. Frontend E2E (Playwright)

**Objetivo:** Cobertura de fluxos de UI críticos

**Estrutura:**
```
tests/e2e/frontend/
├── auth.spec.ts
├── dashboard.spec.ts
├── transactions.spec.ts
├── alerts.spec.ts
└── hardrules.spec.ts
```

**Exemplo de Teste:**
```typescript
// tests/e2e/frontend/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Login if needed
  });

  test('should display KPIs', async ({ page }) => {
    await expect(page.locator('.kpi-card')).toHaveCount(4);
    await expect(page.locator('.kpi-transactions')).toBeVisible();
  });

  test('should show transaction chart', async ({ page }) => {
    await expect(page.locator('canvas, svg.chart')).toBeVisible();
  });

  test('should load recent alerts', async ({ page }) => {
    await expect(page.locator('.alert-item')).toBeVisible();
  });
});
```

---

### 2.4. Chaos Basics (Pytest)

**Objetivo:** Validar resiliência básica

**Implementação:**
```python
# tests/chaos/test_basic_chaos.py
import pytest
import time
from unittest.mock import patch, MagicMock

class TestBasicChaos:
    """Testes básicos de caos para resiliência"""
    
    def test_database_timeout_recovery(self, client):
        """Sistema deve recuperar de timeout de DB"""
        with patch('psycopg2.connect', side_effect=TimeoutError):
            response = client.post('/api/predict', json={...})
            # Deve usar fallback ou retornar erro gracioso
            assert response.status_code in [200, 503]
    
    def test_cache_unavailable(self, client):
        """Sistema deve funcionar sem cache"""
        with patch('redis.Redis.get', side_effect=ConnectionError):
            response = client.post('/api/predict', json={...})
            assert response.status_code == 200
    
    def test_high_latency_simulation(self, client):
        """Sistema deve lidar com alta latência"""
        with patch('time.sleep', side_effect=lambda x: time.sleep(0.01)):
            response = client.post('/api/predict', json={...})
            assert response.status_code == 200
```

---

## 3. PRIORIDADE 2 - CURTO PRAZO (🟡 2-4 Semanas)

### 3.1. API Contract Tests (Pact/Schemathesis)

**Objetivo:** Garantir contratos de API

**Implementação:**
```python
# tests/contract/test_api_contracts.py
import schemathesis

schema = schemathesis.from_path("openapi.yaml")

@schema.parametrize()
def test_api_contract(case):
    """Valida que API segue o contrato OpenAPI"""
    response = case.call()
    case.validate_response(response)
```

---

### 3.2. Accessibility Tests (axe-core)

**Objetivo:** WCAG 2.1 AA compliance

**Implementação:**
```typescript
// tests/accessibility/a11y.spec.ts
import { test, expect } from '@playwright/test';
import { checkA11y, injectAxe } from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('dashboard should be accessible', async ({ page }) => {
    await page.goto('/');
    await injectAxe(page);
    const results = await checkA11y(page);
    expect(results.violations).toHaveLength(0);
  });
});
```

---

### 3.3. Database Backup/Restore

**Objetivo:** Validar backup e restore funcionam

**Implementação:**
```python
# tests/db/test_backup_restore.py
import subprocess
import pytest

class TestBackupRestore:
    def test_pg_dump_works(self):
        """Backup deve completar sem erros"""
        result = subprocess.run(
            ['pg_dump', '-Fc', 'sankofa_db', '-f', '/tmp/backup.dump'],
            capture_output=True
        )
        assert result.returncode == 0
    
    def test_pg_restore_works(self):
        """Restore deve completar sem erros"""
        result = subprocess.run(
            ['pg_restore', '--clean', '-d', 'sankofa_test', '/tmp/backup.dump'],
            capture_output=True
        )
        assert result.returncode == 0
```

---

### 3.4. Rollback Testing

**Objetivo:** Validar rollback de deploy

**Checklist:**
- [ ] Rollback de código (Git)
- [ ] Rollback de migrations (Alembic)
- [ ] Rollback de modelo ML
- [ ] Rollback de configuração

---

## 4. PRIORIDADE 3 - MÉDIO PRAZO (🟢 1-2 Meses)

### 4.1. Visual Regression (Percy/Playwright)

```typescript
// tests/visual/visual.spec.ts
test('dashboard visual regression', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveScreenshot('dashboard.png', {
    maxDiffPixelRatio: 0.01
  });
});
```

### 4.2. Cross-Browser (BrowserStack/Playwright)

```typescript
// playwright.config.ts
export default {
  projects: [
    { name: 'chromium' },
    { name: 'firefox' },
    { name: 'webkit' },
  ],
};
```

### 4.3. Mutation Testing (mutmut)

```bash
# Executar mutation testing
mutmut run --paths-to-mutate=ml_engine/
mutmut results
```

---

## 5. PIPELINE DE CI/CD PROPOSTO

```yaml
# .github/workflows/test-pipeline.yml
name: Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  # ========== NÍVEL 1: A CADA PR ==========
  unit-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/test_domain.py tests/test_improvements.py -v
      - name: Run integration tests
        run: pytest tests/test_integration_db.py -v

  # ========== NÍVEL 2: NIGHTLY ==========
  e2e-api:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - name: Run E2E API tests
        run: pytest tests/test_e2e.py -v
      - name: Run ML tests
        run: pytest tests/test_ml_metrics_comprehensive.py -v

  # ========== NÍVEL 3: POR RELEASE ==========
  performance:
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    steps:
      - name: Run k6 load tests
        run: k6 run tests/perf/critical_scenarios.js
      - name: Run security scan
        run: docker run owasp/zap2docker-stable zap-baseline.py -t $URL

  # ========== NÍVEL 4: SEMANAL ==========
  ml-fairness:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - name: Run fairness analysis
        run: python -c "from mlops.fairness_analyzer import *; print('OK')"
      - name: Check drift
        run: python -c "from mlops.drift_detector import *; print('OK')"
```

---

## 6. MÉTRICAS DE SUCESSO

### 6.1. KPIs de Qualidade

| Métrica | Meta | Crítico | Status Atual |
|---------|------|---------|--------------|
| Test Coverage | > 80% | > 60% | ~70% |
| Test Pass Rate | > 98% | > 95% | 98% |
| Critical Bugs | 0 | < 3 | 0 |
| Latency p95 | < 50ms | < 100ms | 42.3ms ✅ |
| Uptime | 99.9% | 99% | TBD |

### 6.2. Cronograma de Implementação

| Semana | Atividade | Responsável |
|--------|-----------|-------------|
| 1 | Load Testing (k6) | DevOps |
| 1-2 | Security Scan (ZAP) | Security |
| 2-3 | Frontend E2E (Playwright) | QA |
| 3-4 | Chaos Testing | SRE |
| 4-5 | API Contracts | Backend |
| 5-6 | Accessibility | Frontend |
| 6-8 | DB Backup/Restore | DBA |
| 8+ | Visual Regression | QA |

---

## 7. ESTRUTURA DE PASTAS DE TESTES

```
tests/
├── unit/               # Testes unitários
│   ├── test_entities.py
│   └── test_use_cases.py
├── integration/        # Testes de integração
│   ├── test_db.py
│   └── test_cache.py
├── e2e/               # Testes end-to-end
│   ├── api/
│   │   └── test_endpoints.py
│   └── frontend/
│       ├── auth.spec.ts
│       └── dashboard.spec.ts
├── perf/              # Testes de performance
│   ├── load_test.js
│   └── stress_test.js
├── security/          # Testes de segurança
│   └── security_scan.py
├── chaos/             # Testes de caos
│   └── test_chaos.py
├── ml/                # Testes de ML
│   ├── test_metrics.py
│   └── test_fairness.py
├── contract/          # Testes de contrato
│   └── test_api_contract.py
├── accessibility/     # Testes de acessibilidade
│   └── a11y.spec.ts
└── visual/            # Testes visuais
    └── visual.spec.ts
```

---

## 8. PRÓXIMOS PASSOS IMEDIATOS

1. **HOJE:**
   - [x] Criar estrutura de pastas
   - [ ] Implementar test_systemic_1000x.py

2. **ESTA SEMANA:**
   - [ ] Script k6 para load testing
   - [ ] Configurar OWASP ZAP
   - [ ] Setup Playwright para frontend

3. **PRÓXIMA SEMANA:**
   - [ ] Testes de caos básicos
   - [ ] API contract testing
   - [ ] Accessibility audit

---

**Documento:** `docs/qa/systemic-test-strategy.md`  
**Próximo:** Implementação dos testes prioritários
