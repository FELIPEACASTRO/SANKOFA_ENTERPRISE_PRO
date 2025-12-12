import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Investigation } from '@/pages/Investigation';
import { HardRules } from '@/pages/HardRules';
import { Calibration } from '@/pages/Calibration';

// Mock dados de Investigação
const mockInvestigationsData = {
  data: [
    {
      id: 'INV001',
      titulo: 'Fraude em série - PIX',
      status: 'investigando',
      prioridade: 'alta',
      valor_total: 150000,
      transacoes_vinculadas: 15,
      data_abertura: '2025-01-14T08:00:00'
    },
    {
      id: 'INV002',
      titulo: 'Padrão suspeito - Cartão',
      status: 'ativo',
      prioridade: 'media',
      valor_total: 45000,
      transacoes_vinculadas: 8,
      data_abertura: '2025-01-13T14:30:00'
    }
  ],
  summary: {
    active: 5,
    pending: 3,
    resolved: 12,
    total: 20
  }
};

// Mock dados de Hard Rules
const mockHardRulesData = {
  success: true,
  rules: [
    {
      id: 1,
      name: 'Bloqueio PIX > R$50k',
      description: 'Bloqueia automaticamente PIX acima de R$50.000',
      type: 'amount',
      enabled: true,
      condition: 'amount > 50000 AND type = PIX',
      action: 'BLOCK',
      hits: 234
    },
    {
      id: 2,
      name: 'País Bloqueado',
      description: 'Bloqueia transações de países de alto risco',
      type: 'location',
      enabled: true,
      condition: 'country IN (blocked_countries)',
      action: 'BLOCK',
      hits: 89
    }
  ]
};

const mockHardRulesMetadata = {
  types: ['amount', 'location', 'velocity', 'device', 'time'],
  actions: ['BLOCK', 'REVIEW', 'FLAG'],
  operators: ['>', '<', '=', '>=', '<=', 'IN', 'NOT IN']
};

// Mock dados de Calibração
const mockCalibrationConfig = {
  success: true,
  config: {
    threshold_fraud: 0.7,
    threshold_review: 0.4,
    sensitivity: 'medium',
    auto_block_enabled: true,
    realtime_enabled: true
  }
};

const mockCalibrationImpact = {
  success: true,
  impact: {
    estimated_blocks: 1250,
    estimated_reviews: 3400,
    false_positive_rate: 2.3,
    coverage: 98.5
  }
};

describe('Investigation Page', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockImplementation((url) => {
      if (url.includes('/api/investigations') && !url.includes('/transactions')) {
        return Promise.resolve({ json: () => Promise.resolve(mockInvestigationsData) });
      }
      if (url.includes('/transactions')) {
        return Promise.resolve({ json: () => Promise.resolve({ transactions: [] }) });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  it('renderiza título da página', async () => {
    render(<Investigation />);
    await waitFor(() => {
      expect(screen.getByText(/Central de Investigação/i)).toBeInTheDocument();
    });
  });

  it('carrega investigações da API', async () => {
    render(<Investigation />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/investigations');
    });
  });

  it('exibe filtros de status', async () => {
    render(<Investigation />);
    await waitFor(() => {
      expect(screen.getByText(/Atualizar/i)).toBeInTheDocument();
    });
  });

  it('mostra botão de nova investigação', async () => {
    render(<Investigation />);
    await waitFor(() => {
      expect(screen.getByText(/Nova Investigação/i)).toBeInTheDocument();
    });
  });
});

describe('HardRules Page', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockImplementation((url) => {
      if (url.includes('/api/hard-rules/metadata')) {
        return Promise.resolve({ json: () => Promise.resolve(mockHardRulesMetadata) });
      }
      if (url.includes('/api/hard-rules')) {
        return Promise.resolve({ json: () => Promise.resolve(mockHardRulesData) });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  it('renderiza título da página', async () => {
    render(<HardRules />);
    await waitFor(() => {
      expect(screen.getByText(/Regras Duras/i)).toBeInTheDocument();
    });
  });

  it('carrega regras da API', async () => {
    render(<HardRules />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/hard-rules');
    });
  });

  it('carrega metadata da API', async () => {
    render(<HardRules />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/hard-rules/metadata');
    });
  });
});

describe('Calibration Page', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockImplementation((url) => {
      if (url.includes('/api/calibration/config')) {
        return Promise.resolve({ json: () => Promise.resolve(mockCalibrationConfig) });
      }
      if (url.includes('/api/calibration/impact')) {
        return Promise.resolve({ json: () => Promise.resolve(mockCalibrationImpact) });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  it('renderiza título da página', async () => {
    render(<Calibration />);
    await waitFor(() => {
      expect(screen.getByText(/Calibração/i)).toBeInTheDocument();
    });
  });

  it('carrega configuração da API', async () => {
    render(<Calibration />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/calibration/config');
    });
  });

  it('carrega impacto da API', async () => {
    render(<Calibration />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/calibration/impact');
    });
  });
});
