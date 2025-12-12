import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Dashboard } from '@/pages/Dashboard';

// Mock dos dados da API
const mockKpisData = {
  data: {
    transacoes_hoje: 15420,
    transacoes_ontem: 14800,
    fraudes_detectadas: 23,
    fraudes_ontem: 19,
    taxa_aprovacao: 98.5,
    taxa_aprovacao_ontem: 98.2,
    latencia_media: 45,
    latencia_ontem: 52,
    valor_protegido_hoje: 2500000,
    valor_protegido_ano: 150000000,
    familias_protegidas: 45000
  }
};

const mockTimeseriesData = {
  data: {
    timeseries: [
      { time: '00:00', transactions: 420, latency: 42 },
      { time: '01:00', transactions: 380, latency: 38 },
      { time: '02:00', transactions: 290, latency: 35 },
    ]
  }
};

const mockChannelsData = {
  data: {
    channels: [
      { name: 'PIX', value: 45, frauds: 12 },
      { name: 'Crédito', value: 30, frauds: 8 },
      { name: 'Débito', value: 25, frauds: 3 }
    ]
  }
};

const mockAlertsData = {
  alerts: [
    { id: '1', message: 'Fraude detectada em PIX', severity: 'critico', time: '10:30' },
    { id: '2', message: 'Padrão suspeito identificado', severity: 'alto', time: '10:25' }
  ]
};

const mockModelsData = {
  models: [
    { name: 'XGBoost Ensemble', accuracy: 99.2, status: 'healthy' },
    { name: 'LightGBM', accuracy: 98.8, status: 'healthy' },
    { name: 'Neural Network', accuracy: 97.5, status: 'healthy' }
  ]
};

describe('Dashboard Page', () => {
  beforeEach(() => {
    vi.mocked(global.fetch).mockImplementation((url) => {
      if (url.includes('/api/dashboard/kpis')) {
        return Promise.resolve({ json: () => Promise.resolve(mockKpisData) });
      }
      if (url.includes('/api/dashboard/timeseries')) {
        return Promise.resolve({ json: () => Promise.resolve(mockTimeseriesData) });
      }
      if (url.includes('/api/dashboard/channels')) {
        return Promise.resolve({ json: () => Promise.resolve(mockChannelsData) });
      }
      if (url.includes('/api/dashboard/recent-alerts')) {
        return Promise.resolve({ json: () => Promise.resolve(mockAlertsData) });
      }
      if (url.includes('/api/dashboard/model-status')) {
        return Promise.resolve({ json: () => Promise.resolve(mockModelsData) });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  it('mostra estado de loading inicialmente', () => {
    render(<Dashboard />);
    expect(screen.getByText(/Carregando dados do dashboard/i)).toBeInTheDocument();
  });

  it('renderiza título do dashboard', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Dashboard Executivo')).toBeInTheDocument();
    });
  });

  it('exibe KPIs após carregar dados', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Transações Hoje')).toBeInTheDocument();
      expect(screen.getByText('Fraudes Detectadas')).toBeInTheDocument();
      expect(screen.getByText('Taxa de Aprovação')).toBeInTheDocument();
      expect(screen.getByText('Latência Média')).toBeInTheDocument();
    });
  });

  it('mostra status dos modelos de ML', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Status dos Modelos')).toBeInTheDocument();
    });
  });

  it('exibe alertas recentes', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Alertas Recentes')).toBeInTheDocument();
    });
  });

  it('mostra valor protegido', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Valor Protegido')).toBeInTheDocument();
    });
  });

  it('faz 5 chamadas de API ao carregar', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(5);
    });
  });

  it('chama endpoints corretos da API', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/dashboard/kpis');
      expect(global.fetch).toHaveBeenCalledWith('/api/dashboard/timeseries');
      expect(global.fetch).toHaveBeenCalledWith('/api/dashboard/channels');
      expect(global.fetch).toHaveBeenCalledWith('/api/dashboard/recent-alerts');
      expect(global.fetch).toHaveBeenCalledWith('/api/dashboard/model-status');
    });
  });

  it('mostra badge de sistema online', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Sistema Online')).toBeInTheDocument();
    });
  });

  it('mostra botão de atualizar', async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText('Atualizar')).toBeInTheDocument();
    });
  });
});
