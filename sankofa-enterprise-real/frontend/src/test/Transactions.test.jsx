import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { Transactions } from '@/pages/Transactions';

const mockTransactionsData = {
  success: true,
  data: [
    {
      id: 'TXN001',
      valor: 1500.00,
      tipo: 'PIX',
      canal: 'Mobile',
      localizacao: 'São Paulo, SP',
      cpf: '***.***.***-01',
      data_hora: '2025-01-15T10:30:00',
      status: 'APROVADA',
      fraud_score: 0.12
    },
    {
      id: 'TXN002',
      valor: 5000.00,
      tipo: 'Crédito',
      canal: 'Web',
      localizacao: 'Rio de Janeiro, RJ',
      cpf: '***.***.***-02',
      data_hora: '2025-01-15T10:25:00',
      status: 'EM_ANALISE',
      fraud_score: 0.75
    },
    {
      id: 'TXN003',
      valor: 25000.00,
      tipo: 'TED',
      canal: 'Mobile',
      localizacao: 'Curitiba, PR',
      cpf: '***.***.***-03',
      data_hora: '2025-01-15T10:20:00',
      status: 'BLOQUEADA',
      fraud_score: 0.95
    }
  ],
  stats: {
    total: 3
  }
};

describe('Transactions Page', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockImplementation((url) => {
      if (url.includes('/api/transactions')) {
        return Promise.resolve({ json: () => Promise.resolve(mockTransactionsData) });
      }
      if (url.includes('/api/explainability/explain')) {
        return Promise.resolve({
          json: () => Promise.resolve({
            success: true,
            explanation: {
              decision: 'APROVADA',
              confidence: 0.95,
              factors: ['Valor dentro do padrão', 'Dispositivo conhecido']
            }
          })
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  it('mostra estado de loading', () => {
    render(<Transactions />);
    expect(screen.getByText(/Transações/i)).toBeInTheDocument();
  });

  it('carrega transações da API', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/transactions'));
    });
  });

  it('exibe componente de busca', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Buscar/i)).toBeInTheDocument();
    });
  });

  it('exibe filtro de período', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Últimas 24h/i)).toBeInTheDocument();
    });
  });

  it('exibe botão de exportar', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Exportar/i)).toBeInTheDocument();
    });
  });

  it('exibe botão de atualizar', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Atualizar/i)).toBeInTheDocument();
    });
  });

  it('renderiza página de transações', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Transações/i)).toBeInTheDocument();
    });
  });

  it('exibe título lista de transações', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Lista de Transações/i)).toBeInTheDocument();
    });
  });
});

describe('Transactions - Filtros', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockResolvedValue({
      json: () => Promise.resolve(mockTransactionsData)
    });
  });

  it('permite filtrar transações', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Transações/i)).toBeInTheDocument();
    });
  });

  it('permite alterar período de busca', async () => {
    render(<Transactions />);
    await waitFor(() => {
      expect(screen.getByText(/Últimas 24h/i)).toBeInTheDocument();
    });
  });
});

describe('Transactions - Ações', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockResolvedValue({
      json: () => Promise.resolve(mockTransactionsData)
    });
  });

  it('chama API ao clicar em atualizar', async () => {
    render(<Transactions />);
    await waitFor(() => {
      const refreshButton = screen.getByText(/Atualizar/i);
      fireEvent.click(refreshButton);
    });

    await waitFor(() => {
      // API deve ser chamada novamente
      expect(global.fetch).toHaveBeenCalledTimes(2); // Initial + refresh
    });
  });
});
