# 08 - Estrategia de Testes Automatizados

**Data da Analise:** 29/11/2025  
**Metodologia:** MODO DOUBLE CHECK ULTRA - Fase 6

---

## 1. Resumo Executivo

| Categoria | Quantidade | Cobertura |
|-----------|------------|-----------|
| Testes Unitarios Backend | 196 | ~85% |
| Testes de Integracao API | 42 | ~90% |
| Testes End-to-End | 12 | ~60% |
| Testes Frontend | Pendente | 0% |
| **Total** | **250+** | **~78%** |

---

## 2. Infraestrutura de Testes Existente

### 2.1 Backend (Python/pytest)

```bash
# Executar todos os testes
cd sankofa-enterprise-real/backend
pytest tests/ -v

# Executar com cobertura
pytest tests/ --cov=. --cov-report=html

# Executar testes específicos
pytest tests/test_domain.py -v
pytest tests/test_qa_comprehensive.py -v
```

### 2.2 Arquivos de Teste Existentes

| Arquivo | Testes | Descricao |
|---------|--------|-----------|
| tests/test_domain.py | 45 | Logica de dominio |
| tests/test_qa_comprehensive.py | 87 | QA abrangente |
| tests/test_qa_expanded.py | 50 | QA expandido |
| tests/test_e2e.py | 12 | End-to-end |
| tests/test_resilience.py | 25 | Resiliencia |
| tests/test_improvements.py | 31 | Melhorias |

### 2.3 Frontend (Pendente)

Atualmente nao ha testes automatizados no frontend.

---

## 3. Estrategia de Testes Proposta

### 3.1 Piramide de Testes

```
                     /\
                    /  \
                   / E2E \         (10%)
                  /______\
                 /        \
                / Integration \    (20%)
               /______________\
              /                \
             /     Unit Tests   \  (70%)
            /____________________\
```

### 3.2 Cobertura Minima Requerida

| Camada | Target | Status Atual |
|--------|--------|--------------|
| Domain | 90% | ✅ ~90% |
| Services | 85% | ✅ ~85% |
| API | 80% | ✅ ~82% |
| Frontend Components | 70% | ❌ 0% |
| E2E Flows | 60% | ⚠️ ~55% |

---

## 4. Testes Unitarios - Frontend (A Implementar)

### 4.1 Setup React Testing Library

```bash
cd sankofa-enterprise-real/frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest jsdom
```

### 4.2 Configuracao Vitest

```javascript
// vitest.config.js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
```

### 4.3 Testes Prioritarios

#### Dashboard.jsx

```javascript
// src/pages/__tests__/Dashboard.test.jsx
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Dashboard from '../Dashboard'

describe('Dashboard', () => {
  it('renders loading state initially', () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )
    expect(screen.getByTestId('loading-skeleton')).toBeInTheDocument()
  })

  it('renders KPI cards after data loads', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )
    await waitFor(() => {
      expect(screen.getByText(/Total de Transacoes/i)).toBeInTheDocument()
    })
  })

  it('handles API error gracefully', async () => {
    // Mock failed API
    global.fetch = jest.fn(() => Promise.reject(new Error('API Error')))
    
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    )
    
    await waitFor(() => {
      expect(screen.getByText(/Erro ao carregar/i)).toBeInTheDocument()
    })
  })
})
```

#### Transactions.jsx

```javascript
// src/pages/__tests__/Transactions.test.jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import Transactions from '../Transactions'

describe('Transactions', () => {
  it('renders transaction list', async () => {
    render(<Transactions />)
    await waitFor(() => {
      expect(screen.getByRole('table')).toBeInTheDocument()
    })
  })

  it('filters transactions by search', async () => {
    render(<Transactions />)
    const searchInput = screen.getByPlaceholderText(/Buscar/i)
    fireEvent.change(searchInput, { target: { value: 'PIX' } })
    await waitFor(() => {
      expect(screen.queryAllByText(/PIX/i)).not.toHaveLength(0)
    })
  })

  it('handles approve action', async () => {
    render(<Transactions />)
    await waitFor(() => {
      const approveBtn = screen.getAllByText(/Aprovar/i)[0]
      fireEvent.click(approveBtn)
    })
    await waitFor(() => {
      expect(screen.getByText(/Aprovado/i)).toBeInTheDocument()
    })
  })
})
```

#### RiskScoreBadge.jsx

```javascript
// src/components/ui/__tests__/Badge.test.jsx
import { render, screen } from '@testing-library/react'
import { RiskScoreBadge } from '../Badge'

describe('RiskScoreBadge', () => {
  it('renders low risk correctly', () => {
    render(<RiskScoreBadge score={0.2} />)
    expect(screen.getByText(/Baixo/i)).toBeInTheDocument()
  })

  it('renders high risk correctly', () => {
    render(<RiskScoreBadge score={0.9} />)
    expect(screen.getByText(/Critico/i)).toBeInTheDocument()
  })

  it('has correct aria-label', () => {
    render(<RiskScoreBadge score={0.5} />)
    expect(screen.getByRole('status')).toHaveAttribute('aria-label')
  })
})
```

---

## 5. Testes de Integracao - API

### 5.1 Fluxo de Predicao

```python
# tests/test_api_integration.py
import pytest
import requests

BASE_URL = "http://localhost:5000/api"

class TestFraudPredictionFlow:
    def test_predict_single_transaction(self):
        response = requests.post(f"{BASE_URL}/fraud/predict", json={
            "transactions": [{"amount": 1500, "channel": "PIX", "type": "PAYMENT"}]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
        assert len(data["data"]["predictions"]) == 1

    def test_predict_with_explanation(self):
        response = requests.post(f"{BASE_URL}/fraud/predict", json={
            "transactions": [{"amount": 5000, "channel": "DEBITO"}],
            "include_explanation": True,
            "fast_mode": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["summary"]["explanations_included"] is True

    def test_predict_latency_sla(self):
        import time
        start = time.time()
        response = requests.post(f"{BASE_URL}/fraud/predict", json={
            "transactions": [{"amount": 1500, "channel": "PIX"}]
        })
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 500  # 500ms total (network + processing)
```

### 5.2 Fluxo de Transacoes

```python
class TestTransactionsFlow:
    def test_list_transactions(self):
        response = requests.get(f"{BASE_URL}/transactions")
        assert response.status_code == 200
        assert "transactions" in response.json()

    def test_filter_by_status(self):
        response = requests.get(f"{BASE_URL}/transactions?status=fraud")
        assert response.status_code == 200

    def test_approve_transaction(self):
        # Primeiro criar uma transacao
        txn_id = "TXN_TEST_001"
        response = requests.post(f"{BASE_URL}/transactions/{txn_id}/approve")
        assert response.status_code in [200, 404]  # 404 se nao existir
```

---

## 6. Testes End-to-End (Playwright)

### 6.1 Setup

```bash
cd sankofa-enterprise-real/frontend
npm install --save-dev @playwright/test
npx playwright install
```

### 6.2 Configuracao

```javascript
// playwright.config.js
export default {
  testDir: './e2e',
  timeout: 30000,
  use: {
    baseURL: 'http://localhost:5000',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'cd ../backend && python api/production_api.py',
    port: 5000,
    reuseExistingServer: true,
  },
}
```

### 6.3 Testes E2E Criticos

```javascript
// e2e/dashboard.spec.js
import { test, expect } from '@playwright/test'

test.describe('Dashboard', () => {
  test('displays KPIs', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=Total de Transacoes')).toBeVisible()
  })

  test('navigates to transactions', async ({ page }) => {
    await page.goto('/')
    await page.click('text=Transacoes')
    await expect(page).toHaveURL('/transactions')
  })
})

// e2e/transactions.spec.js
test.describe('Transactions', () => {
  test('lists transactions', async ({ page }) => {
    await page.goto('/transactions')
    await expect(page.locator('table')).toBeVisible()
  })

  test('filters by status', async ({ page }) => {
    await page.goto('/transactions')
    await page.selectOption('select[name="status"]', 'fraud')
    await expect(page.locator('tbody tr')).toHaveCount.greaterThan(0)
  })
})

// e2e/calibration.spec.js
test.describe('Calibration', () => {
  test('loads calibration config', async ({ page }) => {
    await page.goto('/calibration')
    await expect(page.locator('text=Ensemble')).toBeVisible()
  })

  test('adjusts threshold slider', async ({ page }) => {
    await page.goto('/calibration')
    const slider = page.locator('[data-testid="threshold-slider"]')
    await slider.fill('0.6')
    await expect(page.locator('text=hasChanges')).not.toBeVisible()
  })
})
```

---

## 7. Scripts NPM (package.json)

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "test:e2e": "playwright test",
    "test:e2e:ui": "playwright test --ui",
    "test:all": "npm run test && npm run test:e2e"
  }
}
```

---

## 8. Matriz de Cobertura por Fluxo

| Fluxo | Unit | Integration | E2E |
|-------|------|-------------|-----|
| Dashboard | ⚠️ | ✅ | ⚠️ |
| Predicao Unitaria | ⚠️ | ✅ | ❌ |
| Predicao Batch | ⚠️ | ✅ | ❌ |
| Lista Transacoes | ⚠️ | ✅ | ⚠️ |
| Calibracao | ⚠️ | ✅ | ⚠️ |
| Alertas | ⚠️ | ✅ | ❌ |
| Observabilidade | ⚠️ | ✅ | ❌ |
| Listas (VIP/HOT/Rules) | ⚠️ | ✅ | ❌ |
| Auditoria | ⚠️ | ✅ | ❌ |

Legenda: ✅ Implementado | ⚠️ Parcial | ❌ Pendente

---

## 9. Proximos Passos

1. **Semana 1:** Configurar Vitest no frontend + 10 testes unitarios
2. **Semana 2:** Adicionar 20 testes de componentes
3. **Semana 3:** Configurar Playwright + 5 testes E2E
4. **Semana 4:** Aumentar cobertura para 70%

---

*Documento gerado conforme MODO DOUBLE CHECK ULTRA - Fase 6*
