import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/Button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';
import { 
  Activity, 
  RefreshCw, 
  Server, 
  Cpu, 
  HardDrive, 
  Clock, 
  Shield,
  AlertTriangle,
  CheckCircle,
  Wifi
} from 'lucide-react';

const Monitoring = () => {
  const [systemHealth, setSystemHealth] = useState({
    overall_status: 'healthy',
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    network_latency: 0,
    active_connections: 0,
    uptime: '0d 0h 0m',
    active_models: 0,
    transactions_per_second: 0,
    avg_response_time: 0,
    fraud_detection_rate: 0,
    false_positive_rate: 0,
    processed_today: 0
  });

  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  const loadMonitoringData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [healthRes, metricsRes, alertsRes, slaRes] = await Promise.all([
        fetch('/api/health/detailed').catch(() => null),
        fetch('/api/observability/metrics').catch(() => null),
        fetch('/api/observability/alerts').catch(() => null),
        fetch('/api/observability/sla').catch(() => null)
      ]);

      const safeJsonParse = async (response, defaultValue = {}) => {
        if (!response || !response.ok) return defaultValue;
        try {
          const text = await response.text();
          return text ? JSON.parse(text) : defaultValue;
        } catch {
          return defaultValue;
        }
      };

      const healthData = await safeJsonParse(healthRes, { status: 'unknown' });
      const metricsData = await safeJsonParse(metricsRes, {});
      const alertsData = await safeJsonParse(alertsRes, { alerts: [] });
      const slaData = await safeJsonParse(slaRes, { latency: {} });

      setSystemHealth(prev => ({
        ...prev,
        overall_status: healthData.status || healthData.data?.status || 'healthy',
        cpu_usage: metricsData.data?.system?.cpu_usage || metricsData.system?.cpu_usage || metricsData.cpu_usage || 0,
        memory_usage: metricsData.data?.system?.memory_usage || metricsData.system?.memory_usage || metricsData.memory_usage || 0,
        disk_usage: metricsData.data?.system?.disk_usage || metricsData.system?.disk_usage || metricsData.disk_usage || 0,
        network_latency: slaData.data?.latency?.p50 || slaData.latency?.p50 || metricsData.latency_p50 || 0,
        active_connections: metricsData.data?.active_connections || metricsData.active_connections || 0,
        uptime: healthData.data?.uptime || healthData.uptime || metricsData.uptime || '0d 0h 0m',
        active_models: healthData.data?.active_models || healthData.active_models || metricsData.models_active || 0,
        transactions_per_second: metricsData.data?.transactions_per_second || metricsData.transactions_per_second || metricsData.tps || 0,
        avg_response_time: slaData.data?.latency?.avg || slaData.latency?.avg || metricsData.avg_response_time || 0,
        fraud_detection_rate: metricsData.data?.fraud_detection_rate || metricsData.fraud_detection_rate || metricsData.recall || 0,
        false_positive_rate: metricsData.data?.false_positive_rate || metricsData.false_positive_rate || 0,
        processed_today: metricsData.data?.transactions_today || metricsData.transactions_today || metricsData.processed_today || 0
      }));

      if (alertsData.alerts) {
        setAlerts(alertsData.alerts.slice(0, 5));
      }

      setLastUpdate(new Date());
    } catch (err) {
      console.error('Erro ao carregar dados de monitoramento:', err);
      setError('Falha ao carregar dados. Tentando novamente...');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMonitoringData();

    if (autoRefresh) {
      const interval = setInterval(loadMonitoringData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (value, type = 'usage') => {
    if (type === 'usage') {
      if (value >= 85) return 'text-red-600 bg-red-100';
      if (value >= 70) return 'text-yellow-600 bg-yellow-100';
      return 'text-green-600 bg-green-100';
    } else if (type === 'health') {
      if (value >= 95) return 'text-green-600 bg-green-100';
      if (value >= 85) return 'text-yellow-600 bg-yellow-100';
      return 'text-red-600 bg-red-100';
    }
  };

  const getAlertColor = (type) => {
    switch (type) {
      case 'error': case 'critical': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'info': return 'border-blue-500 bg-blue-50';
      default: return 'border-gray-500 bg-gray-50';
    }
  };

  const getOverallStatusBadge = () => {
    const status = systemHealth.overall_status;
    if (status === 'healthy' || status === 'ok') {
      return <Badge variant="success">Saudável</Badge>;
    } else if (status === 'degraded' || status === 'warning') {
      return <Badge variant="warning">Degradado</Badge>;
    } else {
      return <Badge variant="error">Crítico</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-h1">Monitoramento do Sistema</h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Saúde dos modelos de IA e performance em tempo real
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdate && (
            <span className="text-sm text-[var(--color-text-secondary)]">
              Atualizado: {lastUpdate.toLocaleTimeString('pt-BR')}
            </span>
          )}
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant={autoRefresh ? "primary" : "secondary"}
            size="sm"
          >
            <Activity className="w-4 h-4 mr-2" />
            Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
          </Button>
          <Button onClick={loadMonitoringData} variant="secondary" size="sm" disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
        </div>
      </div>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-600" />
          <span className="text-yellow-800">{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Status Geral</p>
                <div className="mt-2">{getOverallStatusBadge()}</div>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Modelos Ativos</p>
                <p className="text-2xl font-bold text-blue-600">{systemHealth.active_models}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <Shield className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Transações/seg</p>
                <p className="text-2xl font-bold text-purple-600">{systemHealth.transactions_per_second}</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                <Activity className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Tempo Resposta</p>
                <p className="text-2xl font-bold text-orange-600">{systemHealth.avg_response_time}s</p>
              </div>
              <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                <Clock className="h-6 w-6 text-orange-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Taxa Detecção</p>
                <p className="text-2xl font-bold text-green-600">{systemHealth.fraud_detection_rate}%</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                <Shield className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Falsos Positivos</p>
                <p className="text-2xl font-bold text-yellow-600">{systemHealth.false_positive_rate}%</p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-yellow-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Processadas Hoje</p>
                <p className="text-2xl font-bold text-blue-600">{systemHealth.processed_today.toLocaleString()}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <Server className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--color-text-secondary)]">Uptime</p>
                <p className="text-2xl font-bold text-green-600">{systemHealth.uptime}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                <Clock className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recursos do Sistema</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-3">
                <Cpu className="h-8 w-8 text-blue-600" />
              </div>
              <p className="text-sm font-medium text-[var(--color-text-secondary)]">CPU</p>
              <p className="text-2xl font-bold">{systemHealth.cpu_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.cpu_usage)}`}>
                {systemHealth.cpu_usage >= 85 ? 'Alto' : systemHealth.cpu_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-green-100 rounded-full flex items-center justify-center mb-3">
                <Server className="h-8 w-8 text-green-600" />
              </div>
              <p className="text-sm font-medium text-[var(--color-text-secondary)]">Memória</p>
              <p className="text-2xl font-bold">{systemHealth.memory_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.memory_usage)}`}>
                {systemHealth.memory_usage >= 85 ? 'Alto' : systemHealth.memory_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-yellow-100 rounded-full flex items-center justify-center mb-3">
                <HardDrive className="h-8 w-8 text-yellow-600" />
              </div>
              <p className="text-sm font-medium text-[var(--color-text-secondary)]">Disco</p>
              <p className="text-2xl font-bold">{systemHealth.disk_usage.toFixed(1)}%</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(systemHealth.disk_usage)}`}>
                {systemHealth.disk_usage >= 85 ? 'Alto' : systemHealth.disk_usage >= 70 ? 'Médio' : 'Normal'}
              </span>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto bg-purple-100 rounded-full flex items-center justify-center mb-3">
                <Wifi className="h-8 w-8 text-purple-600" />
              </div>
              <p className="text-sm font-medium text-[var(--color-text-secondary)]">Latência</p>
              <p className="text-2xl font-bold">{systemHealth.network_latency.toFixed(1)}ms</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(100 - systemHealth.network_latency, 'health')}`}>
                {systemHealth.network_latency <= 20 ? 'Excelente' : systemHealth.network_latency <= 50 ? 'Bom' : 'Lento'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Alertas Recentes
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <div className="text-center py-8">
                <CheckCircle className="h-12 w-12 mx-auto text-green-500" />
                <p className="text-[var(--color-text-secondary)] mt-2">Nenhum alerta ativo</p>
              </div>
            ) : (
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div key={alert.id} className={`p-4 rounded-lg border-l-4 ${getAlertColor(alert.type || alert.severity)}`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium">{alert.message || alert.titulo}</p>
                        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                          {new Date(alert.timestamp || alert.created_at).toLocaleString('pt-BR')}
                        </p>
                      </div>
                      <Badge variant={
                        (alert.severity === 'high' || alert.severity === 'critico') ? 'error' :
                        (alert.severity === 'medium' || alert.severity === 'medio') ? 'warning' : 'info'
                      }>
                        {alert.severity === 'high' || alert.severity === 'critico' ? 'Alta' : 
                         alert.severity === 'medium' || alert.severity === 'medio' ? 'Média' : 'Baixa'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Informações do Sistema
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Conexões Ativas</span>
                <span className="font-medium">{systemHealth.active_connections}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Uptime do Sistema</span>
                <span className="font-medium">{systemHealth.uptime}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Última Atualização</span>
                <span className="font-medium">{lastUpdate ? lastUpdate.toLocaleTimeString('pt-BR') : '-'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Status dos Modelos</span>
                <Badge variant="success">Todos Online</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Modo de Operação</span>
                <Badge variant="info">Produção</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[var(--color-text-secondary)]">Auto-refresh</span>
                <Badge variant={autoRefresh ? 'success' : 'default'}>
                  {autoRefresh ? 'Ativo' : 'Inativo'}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Monitoring;
