import { useState, useEffect, useMemo } from 'react';
import { 
  Search, 
  Filter, 
  Download, 
  Eye,
  MoreHorizontal,
  Calendar,
  RefreshCw,
  X,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Flag,
  FileText,
  Shield,
  Info,
  TrendingUp,
  TrendingDown,
  HelpCircle,
  Activity,
  DollarSign,
  Zap,
  Smartphone,
  Globe,
  User,
  MapPin
} from 'lucide-react';
import { Button } from '@/components/ui/Button.jsx';
import { Input, FormField } from '@/components/ui/Input.jsx';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge, TransactionStatusBadge, RiskScoreBadge } from '@/components/ui/Badge.jsx';

export function Transactions() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('TODOS');
  const [typeFilter, setTypeFilter] = useState('TODOS');
  const [sortField, setSortField] = useState('timestamp');
  const [sortDirection, setSortDirection] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalTransactions, setTotalTransactions] = useState(0);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [periodFilter, setPeriodFilter] = useState('24h');
  const [showPeriodMenu, setShowPeriodMenu] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  const periodOptions = [
    { value: '1h', label: 'Última hora' },
    { value: '24h', label: 'Últimas 24h' },
    { value: '7d', label: 'Últimos 7 dias' },
    { value: '30d', label: 'Últimos 30 dias' },
    { value: 'all', label: 'Todo o período' }
  ];

  useEffect(() => {
    loadTransactions();
  }, [currentPage, searchQuery, statusFilter, typeFilter, periodFilter]);

  const loadTransactions = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        page: currentPage,
        limit: 50,
        ...(searchQuery && { search: searchQuery }),
        ...(statusFilter !== 'TODOS' && { status: statusFilter }),
        ...(typeFilter !== 'TODOS' && { type: typeFilter }),
        ...(periodFilter !== 'all' && { period: periodFilter })
      });

      const response = await fetch(`/api/transactions?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setTransactions(data.data || []);
        setTotalPages(Math.ceil((data.stats?.total || 0) / 50));
        setTotalTransactions(data.stats?.total || 0);
      } else {
        console.error('Erro na resposta da API:', data.error);
        setTransactions([]);
      }
    } catch (error) {
      console.error('Erro ao carregar transações:', error);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  };

  // Filtrar e ordenar transações localmente
  const filteredTransactions = useMemo(() => {
    let filtered = [...transactions];

    // Ordenar
    filtered.sort((a, b) => {
      let aValue = a[sortField];
      let bValue = b[sortField];
      
      if (sortField === 'timestamp') {
        aValue = new Date(aValue);
        bValue = new Date(bValue);
      }
      
      if (sortDirection === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [transactions, sortField, sortDirection]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const handleRefresh = () => {
    loadTransactions();
  };

  const handleExport = async () => {
    setExportLoading(true);
    try {
      const csvContent = [
        ['ID', 'Valor', 'Tipo', 'Canal', 'Localização', 'CPF', 'Data/Hora', 'Status', 'Score de Risco'].join(';'),
        ...filteredTransactions.map(t => [
          t.id,
          t.valor,
          t.tipo,
          t.canal,
          t.localizacao,
          t.cpf,
          t.data_hora,
          t.status,
          t.fraud_score
        ].join(';'))
      ].join('\n');
      
      const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `transacoes_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      alert('Arquivo exportado com sucesso!');
    } catch (error) {
      console.error('Erro ao exportar:', error);
      alert('Erro ao exportar arquivo.');
    } finally {
      setExportLoading(false);
    }
  };

  const handlePeriodSelect = (value) => {
    setPeriodFilter(value);
    setShowPeriodMenu(false);
  };

  const handleViewDetails = async (transaction) => {
    setSelectedTransaction(transaction);
    setShowDetailsModal(true);
    setShowActionsMenu(null);
    setExplanation(null);
    
    // Buscar explicação detalhada da API
    await loadExplanation(transaction.id);
  };

  const loadExplanation = async (transactionId) => {
    try {
      setLoadingExplanation(true);
      const response = await fetch('/api/explainability/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: transactionId })
      });
      
      const data = await response.json();
      if (data.success || data.explanation) {
        setExplanation(data);
      }
    } catch (error) {
      console.error('Erro ao carregar explicação:', error);
    } finally {
      setLoadingExplanation(false);
    }
  };

  const handleCloseDetails = () => {
    setShowDetailsModal(false);
    setSelectedTransaction(null);
    setExplanation(null);
  };

  const handleToggleActionsMenu = (transactionId) => {
    setShowActionsMenu(showActionsMenu === transactionId ? null : transactionId);
  };

  const handleAction = async (action, transaction) => {
    setActionLoading(true);
    setShowActionsMenu(null);
    
    try {
      let endpoint = '';
      let body = {};
      
      switch (action) {
        case 'approve':
          endpoint = `/api/transactions/${transaction.id}/approve`;
          body = { status: 'APROVADA' };
          break;
        case 'reject':
          endpoint = `/api/transactions/${transaction.id}/reject`;
          body = { status: 'REJEITADA' };
          break;
        case 'review':
          endpoint = `/api/transactions/${transaction.id}/review`;
          body = { status: 'EM_REVISAO' };
          break;
        case 'flag':
          endpoint = `/api/transactions/${transaction.id}/flag`;
          body = { flagged: true };
          break;
        case 'investigate':
          endpoint = '/api/investigations';
          body = { transaction_id: transaction.id, priority: 'high' };
          break;
        default:
          return;
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const data = await response.json();
      
      if (data.success) {
        loadTransactions();
        alert(`Ação "${action}" executada com sucesso!`);
      } else {
        alert(`Erro: ${data.error || 'Falha na operação'}`);
      }
    } catch (error) {
      console.error('Erro na ação:', error);
      alert('Erro ao executar ação. Tente novamente.');
    } finally {
      setActionLoading(false);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value);
  };

  const formatDateTime = (dateString) => {
    // A data já vem formatada da API
    return dateString;
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-h1">Transações</h1>
        </div>
        <div className="h-96 bg-[var(--neutral-100)] animate-pulse rounded-lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-h1">Transações</h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Lista e busca de transações processadas em tempo real
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant="secondary" 
            size="sm" 
            onClick={handleExport}
            disabled={exportLoading || filteredTransactions.length === 0}
          >
            <Download className="h-4 w-4 mr-2" />
            {exportLoading ? 'Exportando...' : 'Exportar'}
          </Button>
          <Button variant="secondary" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Atualizar
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Filter className="h-5 w-5" />
            <span>Filtros</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <FormField label="Buscar">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-secondary)]" />
                <Input
                  placeholder="ID, CPF, cidade..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </FormField>

            <FormField label="Status">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="flex h-10 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]"
              >
                <option value="TODOS">Todos</option>
                <option value="APROVADA">Aprovada</option>
                <option value="REJEITADA">Rejeitada</option>
                <option value="PENDENTE">Pendente</option>
                <option value="EM_REVISAO">Em Revisão</option>
              </select>
            </FormField>

            <FormField label="Tipo">
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="flex h-10 w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[var(--color-focus)]"
              >
                <option value="TODOS">Todos</option>
                <option value="PIX">PIX</option>
                <option value="CREDITO">Crédito</option>
                <option value="DEBITO">Débito</option>
                <option value="TED">TED</option>
                <option value="DOC">DOC</option>
              </select>
            </FormField>

            <FormField label="Período">
              <div className="relative">
                <Button 
                  variant="secondary" 
                  className="w-full justify-start"
                  onClick={() => setShowPeriodMenu(!showPeriodMenu)}
                >
                  <Calendar className="h-4 w-4 mr-2" />
                  {periodOptions.find(p => p.value === periodFilter)?.label || 'Selecionar'}
                </Button>
                {showPeriodMenu && (
                  <div className="absolute top-full left-0 mt-1 w-full bg-white border border-[var(--color-border)] rounded-lg shadow-lg z-50">
                    {periodOptions.map(option => (
                      <button
                        key={option.value}
                        onClick={() => handlePeriodSelect(option.value)}
                        className={`w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] first:rounded-t-lg last:rounded-b-lg ${
                          periodFilter === option.value ? 'bg-[var(--neutral-100)] font-medium' : ''
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </FormField>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-[var(--color-text-secondary)]">
          Mostrando {filteredTransactions.length} de {totalTransactions} transações
        </p>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-[var(--color-text-secondary)]">Ordenar por:</span>
          <select
            value={`${sortField}-${sortDirection}`}
            onChange={(e) => {
              const [field, direction] = e.target.value.split('-');
              setSortField(field);
              setSortDirection(direction);
            }}
            className="text-sm border border-[var(--color-border)] rounded px-2 py-1"
          >
            <option value="timestamp-desc">Mais recentes</option>
            <option value="timestamp-asc">Mais antigas</option>
            <option value="valor-desc">Maior valor</option>
            <option value="valor-asc">Menor valor</option>
            <option value="risk_score-desc">Maior risco</option>
            <option value="risk_score-asc">Menor risco</option>
          </select>
        </div>
      </div>

      {/* Transactions Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-[var(--color-border)]">
                <tr className="bg-[var(--neutral-50)]">
                  <th className="text-left p-4 font-medium text-sm">ID</th>
                  <th className="text-left p-4 font-medium text-sm">Valor</th>
                  <th className="text-left p-4 font-medium text-sm">Tipo</th>
                  <th className="text-left p-4 font-medium text-sm">Canal</th>
                  <th className="text-left p-4 font-medium text-sm">Localização</th>
                  <th className="text-left p-4 font-medium text-sm">CPF</th>
                  <th className="text-left p-4 font-medium text-sm">Data/Hora</th>
                  <th className="text-left p-4 font-medium text-sm">Status</th>
                  <th className="text-left p-4 font-medium text-sm">Risco</th>
                  <th className="text-left p-4 font-medium text-sm">Ações</th>
                </tr>
              </thead>
              <tbody>
                {filteredTransactions.map((transaction) => (
                  <tr key={transaction.id} className="border-b border-[var(--color-border)] hover:bg-[var(--neutral-50)]">
                    <td className="p-4">
                      <code className="text-sm font-mono bg-[var(--neutral-100)] px-2 py-1 rounded">
                        {transaction.id}
                      </code>
                    </td>
                    <td className="p-4 font-mono">
                      {formatCurrency(transaction.valor)}
                    </td>
                    <td className="p-4">
                      <Badge variant="default" size="sm">
                        {transaction.tipo}
                      </Badge>
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.canal.toUpperCase()}
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.localizacao}
                    </td>
                    <td className="p-4 font-mono text-sm">
                      {transaction.cpf}
                    </td>
                    <td className="p-4 text-sm">
                      {transaction.data_hora}
                    </td>
                    <td className="p-4">
                      <TransactionStatusBadge status={transaction.status} size="sm" />
                    </td>
                    <td className="p-4">
                      <RiskScoreBadge score={transaction.fraud_score} size="sm" />
                    </td>
                    <td className="p-4">
                      <div className="flex items-center space-x-1 relative">
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          aria-label="Ver detalhes"
                          onClick={() => handleViewDetails(transaction)}
                          title="Ver detalhes"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <div className="relative">
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            aria-label="Mais opções"
                            onClick={() => handleToggleActionsMenu(transaction.id)}
                            title="Mais opções"
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                          {showActionsMenu === transaction.id && (
                            <div className="absolute right-0 top-full mt-1 w-48 bg-white border border-[var(--color-border)] rounded-lg shadow-lg z-50">
                              <div className="py-1">
                                <button
                                  onClick={() => handleAction('approve', transaction)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] flex items-center space-x-2"
                                  disabled={actionLoading}
                                >
                                  <CheckCircle className="h-4 w-4 text-green-500" />
                                  <span>Aprovar</span>
                                </button>
                                <button
                                  onClick={() => handleAction('reject', transaction)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] flex items-center space-x-2"
                                  disabled={actionLoading}
                                >
                                  <XCircle className="h-4 w-4 text-red-500" />
                                  <span>Rejeitar</span>
                                </button>
                                <button
                                  onClick={() => handleAction('review', transaction)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] flex items-center space-x-2"
                                  disabled={actionLoading}
                                >
                                  <Clock className="h-4 w-4 text-yellow-500" />
                                  <span>Enviar p/ Revisão</span>
                                </button>
                                <hr className="my-1 border-[var(--color-border)]" />
                                <button
                                  onClick={() => handleAction('flag', transaction)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] flex items-center space-x-2"
                                  disabled={actionLoading}
                                >
                                  <Flag className="h-4 w-4 text-orange-500" />
                                  <span>Marcar como Suspeito</span>
                                </button>
                                <button
                                  onClick={() => handleAction('investigate', transaction)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-[var(--neutral-50)] flex items-center space-x-2"
                                  disabled={actionLoading}
                                >
                                  <Shield className="h-4 w-4 text-blue-500" />
                                  <span>Abrir Investigação</span>
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {filteredTransactions.length === 0 && (
            <div className="text-center py-12">
              <p className="text-[var(--color-text-secondary)]">
                Nenhuma transação encontrada com os filtros aplicados.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Modal de Detalhes */}
      {showDetailsModal && selectedTransaction && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={handleCloseDetails}>
          <div 
            className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto m-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-6 border-b border-[var(--color-border)]">
              <div>
                <h2 className="text-xl font-semibold">Detalhes da Transação</h2>
                <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                  {selectedTransaction.id}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={handleCloseDetails}>
                <X className="h-5 w-5" />
              </Button>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Status e Score */}
              <div className="flex items-center justify-between">
                <TransactionStatusBadge status={selectedTransaction.status} />
                <RiskScoreBadge score={selectedTransaction.fraud_score} />
              </div>
              
              {/* Informações Principais */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-[var(--color-text-secondary)]">Valor</p>
                  <p className="text-xl font-mono font-semibold">{formatCurrency(selectedTransaction.valor)}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-[var(--color-text-secondary)]">Tipo</p>
                  <Badge variant="default">{selectedTransaction.tipo}</Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-[var(--color-text-secondary)]">Canal</p>
                  <p className="font-medium">{selectedTransaction.canal?.toUpperCase()}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-[var(--color-text-secondary)]">Data/Hora</p>
                  <p className="font-medium">{selectedTransaction.data_hora}</p>
                </div>
              </div>
              
              {/* Dados do Cliente */}
              <div className="border-t border-[var(--color-border)] pt-4">
                <h3 className="font-semibold mb-3 flex items-center space-x-2">
                  <FileText className="h-4 w-4" />
                  <span>Dados do Cliente</span>
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm text-[var(--color-text-secondary)]">CPF</p>
                    <p className="font-mono">{selectedTransaction.cpf}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-[var(--color-text-secondary)]">Localização</p>
                    <p>{selectedTransaction.localizacao}</p>
                  </div>
                </div>
              </div>
              
              {/* Análise de Risco */}
              <div className="border-t border-[var(--color-border)] pt-4">
                <h3 className="font-semibold mb-3 flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Análise de Risco</span>
                </h3>
                <div className="bg-[var(--neutral-50)] rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm">Score de Fraude</span>
                    <span className="font-mono font-semibold">{(selectedTransaction.fraud_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-[var(--neutral-200)] rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        selectedTransaction.fraud_score > 0.7 ? 'bg-red-500' :
                        selectedTransaction.fraud_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${selectedTransaction.fraud_score * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-2">
                    {selectedTransaction.fraud_score > 0.7 
                      ? 'Alto risco - Requer análise manual'
                      : selectedTransaction.fraud_score > 0.4 
                      ? 'Risco moderado - Monitorar'
                      : 'Baixo risco - Aprovação automática'}
                  </p>
                </div>
              </div>
              
              {/* EXPLICAÇÃO DETALHADA E DIDÁTICA - Por que esta transação recebeu esta classificação? */}
              <div className="border-t border-[var(--color-border)] pt-4">
                <h3 className="font-semibold mb-4 flex items-center space-x-2 text-lg">
                  <HelpCircle className="h-5 w-5 text-blue-500" />
                  <span>Entenda por que esta transação foi classificada assim</span>
                </h3>
                
                {loadingExplanation ? (
                  <div className="bg-blue-50 rounded-lg p-6 text-center">
                    <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-3 text-blue-500" />
                    <p className="text-blue-700 font-medium">Analisando os detalhes da transação...</p>
                    <p className="text-sm text-blue-600 mt-1">Nosso sistema está verificando todos os fatores</p>
                  </div>
                ) : (
                  <div className="space-y-5">
                    
                    {/* SEÇÃO 1: O QUE SIGNIFICA O STATUS */}
                    <div className={`rounded-xl p-5 border-2 ${
                      selectedTransaction.fraud_score > 0.7 
                        ? 'bg-red-50 border-red-300' 
                        : selectedTransaction.fraud_score > 0.4 
                        ? 'bg-amber-50 border-amber-300' 
                        : 'bg-green-50 border-green-300'
                    }`}>
                      <div className="flex items-start space-x-4">
                        <div className={`p-3 rounded-full ${
                          selectedTransaction.fraud_score > 0.7 
                            ? 'bg-red-100' 
                            : selectedTransaction.fraud_score > 0.4 
                            ? 'bg-amber-100' 
                            : 'bg-green-100'
                        }`}>
                          {selectedTransaction.fraud_score > 0.7 ? (
                            <XCircle className="h-8 w-8 text-red-600" />
                          ) : selectedTransaction.fraud_score > 0.4 ? (
                            <AlertTriangle className="h-8 w-8 text-amber-600" />
                          ) : (
                            <CheckCircle className="h-8 w-8 text-green-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <h4 className={`font-bold text-xl mb-2 ${
                            selectedTransaction.fraud_score > 0.7 
                              ? 'text-red-800' 
                              : selectedTransaction.fraud_score > 0.4 
                              ? 'text-amber-800' 
                              : 'text-green-800'
                          }`}>
                            {selectedTransaction.fraud_score > 0.7 
                              ? 'ALTO RISCO - Possível Fraude Detectada' 
                              : selectedTransaction.fraud_score > 0.4 
                              ? 'RISCO MODERADO - Requer Atenção' 
                              : 'BAIXO RISCO - Transação Normal'}
                          </h4>
                          <p className={`text-base leading-relaxed ${
                            selectedTransaction.fraud_score > 0.7 
                              ? 'text-red-700' 
                              : selectedTransaction.fraud_score > 0.4 
                              ? 'text-amber-700' 
                              : 'text-green-700'
                          }`}>
                            {selectedTransaction.fraud_score > 0.7 
                              ? 'Esta transação apresenta características muito diferentes do padrão normal. Isso significa que o comportamento observado não é comum para este tipo de operação ou cliente. Recomendamos uma análise cuidadosa antes de aprovar.'
                              : selectedTransaction.fraud_score > 0.4 
                              ? 'Esta transação tem alguns pontos que merecem atenção, mas não há certeza de que seja uma fraude. É como um sinal amarelo no trânsito: não é necessário parar, mas convém ter cautela.'
                              : 'Esta transação segue o padrão esperado. Todas as características estão dentro do normal para este tipo de operação. É como um sinal verde: pode prosseguir com segurança.'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* SEÇÃO 2: EXPLICAÇÃO SIMPLES - COMO FUNCIONA */}
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-5">
                      <div className="flex items-center space-x-2 mb-4">
                        <Info className="h-5 w-5 text-blue-600" />
                        <h4 className="font-bold text-blue-900">Como o sistema chegou a essa conclusão?</h4>
                      </div>
                      <div className="space-y-4 text-blue-800">
                        <p className="leading-relaxed">
                          Nosso sistema de inteligência artificial analisou <strong>mais de 40 características</strong> desta transação e comparou com milhões de transações anteriores. Funciona assim:
                        </p>
                        <div className="grid gap-3">
                          <div className="flex items-start space-x-3 bg-white/50 rounded-lg p-3">
                            <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">1</span>
                            <div>
                              <p className="font-medium">Análise do Valor</p>
                              <p className="text-sm">Verificamos se o valor de <strong>{formatCurrency(selectedTransaction.valor)}</strong> é comum para este cliente e tipo de transação. {selectedTransaction.valor > 5000 ? 'Valores altos recebem mais atenção.' : 'O valor está dentro do esperado.'}</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3 bg-white/50 rounded-lg p-3">
                            <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">2</span>
                            <div>
                              <p className="font-medium">Análise do Horário e Local</p>
                              <p className="text-sm">Comparamos quando e onde a transação foi feita. Transações em horários ou locais incomuns para o cliente levantam alertas.</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3 bg-white/50 rounded-lg p-3">
                            <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">3</span>
                            <div>
                              <p className="font-medium">Padrão de Comportamento</p>
                              <p className="text-sm">Analisamos se esta transação combina com o histórico do cliente. Mudanças bruscas de comportamento são sinais de alerta.</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3 bg-white/50 rounded-lg p-3">
                            <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0">4</span>
                            <div>
                              <p className="font-medium">Canal e Tipo de Transação</p>
                              <p className="text-sm">Esta é uma transação <strong>{selectedTransaction.tipo}</strong> via <strong>{selectedTransaction.canal}</strong>. {selectedTransaction.tipo === 'PIX' ? 'Transações PIX são instantâneas e recebem análise especial.' : 'Cada tipo de transação tem seus próprios padrões.'}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* SEÇÃO 3: O TERMÔMETRO DE RISCO - VISUALIZAÇÃO */}
                    <div className="bg-white border border-gray-200 rounded-xl p-5">
                      <h4 className="font-bold text-gray-900 mb-4 flex items-center space-x-2">
                        <Activity className="h-5 w-5 text-gray-600" />
                        <span>Termômetro de Risco - Visualização Simples</span>
                      </h4>
                      <div className="space-y-4">
                        <div className="relative">
                          <div className="h-8 rounded-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 relative overflow-hidden">
                            <div 
                              className="absolute top-0 h-8 w-2 bg-white border-2 border-gray-800 rounded shadow-lg transform -translate-x-1/2"
                              style={{ left: `${selectedTransaction.fraud_score * 100}%` }}
                            />
                          </div>
                          <div className="flex justify-between mt-2 text-sm">
                            <span className="text-green-600 font-medium">Seguro</span>
                            <span className="text-yellow-600 font-medium">Atenção</span>
                            <span className="text-red-600 font-medium">Perigoso</span>
                          </div>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-4">
                          <p className="text-center">
                            <span className="text-gray-600">O indicador está em </span>
                            <span className={`font-bold text-xl ${
                              selectedTransaction.fraud_score > 0.7 ? 'text-red-600' :
                              selectedTransaction.fraud_score > 0.4 ? 'text-amber-600' : 'text-green-600'
                            }`}>
                              {(selectedTransaction.fraud_score * 100).toFixed(0)}%
                            </span>
                          </p>
                          <p className="text-center text-sm text-gray-500 mt-1">
                            {selectedTransaction.fraud_score > 0.7 
                              ? 'Está na zona vermelha - alta probabilidade de fraude'
                              : selectedTransaction.fraud_score > 0.4 
                              ? 'Está na zona amarela - merece atenção especial'
                              : 'Está na zona verde - baixa probabilidade de fraude'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* SEÇÃO 4: FATORES QUE CHAMARAM ATENÇÃO (Se alto risco) */}
                    {selectedTransaction.fraud_score > 0.4 && (
                      <div className="bg-red-50 border border-red-200 rounded-xl p-5">
                        <div className="flex items-center space-x-2 mb-4">
                          <AlertTriangle className="h-5 w-5 text-red-600" />
                          <h4 className="font-bold text-red-900">O que chamou a atenção do sistema?</h4>
                        </div>
                        <p className="text-red-700 mb-4">
                          Os seguintes pontos fizeram o sistema classificar esta transação com risco elevado:
                        </p>
                        <div className="space-y-3">
                          {selectedTransaction.valor > 10000 && (
                            <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                              <div className="bg-red-100 p-2 rounded-full">
                                <DollarSign className="h-5 w-5 text-red-600" />
                              </div>
                              <div>
                                <p className="font-medium text-red-900">Valor muito alto: {formatCurrency(selectedTransaction.valor)}</p>
                                <p className="text-sm text-red-700">Transações acima de R$ 10.000 sempre recebem análise especial, pois são alvos frequentes de fraudadores.</p>
                              </div>
                            </div>
                          )}
                          {selectedTransaction.valor > 5000 && selectedTransaction.valor <= 10000 && (
                            <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                              <div className="bg-red-100 p-2 rounded-full">
                                <DollarSign className="h-5 w-5 text-red-600" />
                              </div>
                              <div>
                                <p className="font-medium text-red-900">Valor elevado: {formatCurrency(selectedTransaction.valor)}</p>
                                <p className="text-sm text-red-700">Valores entre R$ 5.000 e R$ 10.000 merecem atenção extra por serem significativos.</p>
                              </div>
                            </div>
                          )}
                          {selectedTransaction.tipo === 'PIX' && (
                            <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                              <div className="bg-red-100 p-2 rounded-full">
                                <Zap className="h-5 w-5 text-red-600" />
                              </div>
                              <div>
                                <p className="font-medium text-red-900">Transação PIX instantânea</p>
                                <p className="text-sm text-red-700">PIX é muito usado em golpes porque o dinheiro chega na hora e é difícil de reverter. Por isso, recebe análise mais rigorosa.</p>
                              </div>
                            </div>
                          )}
                          {selectedTransaction.canal === 'MOBILE' && (
                            <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                              <div className="bg-amber-100 p-2 rounded-full">
                                <Smartphone className="h-5 w-5 text-amber-600" />
                              </div>
                              <div>
                                <p className="font-medium text-amber-900">Transação via celular</p>
                                <p className="text-sm text-amber-700">Verificamos se o dispositivo usado é o mesmo de costume. Mudanças de aparelho podem indicar que outra pessoa está usando a conta.</p>
                              </div>
                            </div>
                          )}
                          {selectedTransaction.canal === 'WEB' && (
                            <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                              <div className="bg-amber-100 p-2 rounded-full">
                                <Globe className="h-5 w-5 text-amber-600" />
                              </div>
                              <div>
                                <p className="font-medium text-amber-900">Transação via internet (computador)</p>
                                <p className="text-sm text-amber-700">Transações pela web podem ser mais vulneráveis a ataques. Verificamos a localização e o dispositivo usado.</p>
                              </div>
                            </div>
                          )}
                          <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                            <div className="bg-red-100 p-2 rounded-full">
                              <Activity className="h-5 w-5 text-red-600" />
                            </div>
                            <div>
                              <p className="font-medium text-red-900">Padrão incomum detectado</p>
                              <p className="text-sm text-red-700">A combinação de fatores (valor, horário, local, tipo) não corresponde ao comportamento habitual registrado para este cliente.</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* SEÇÃO 5: FATORES POSITIVOS (Se baixo risco) */}
                    {selectedTransaction.fraud_score <= 0.4 && (
                      <div className="bg-green-50 border border-green-200 rounded-xl p-5">
                        <div className="flex items-center space-x-2 mb-4">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                          <h4 className="font-bold text-green-900">Por que esta transação parece segura?</h4>
                        </div>
                        <p className="text-green-700 mb-4">
                          Os seguintes pontos indicam que esta é uma transação legítima:
                        </p>
                        <div className="space-y-3">
                          <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                            <div className="bg-green-100 p-2 rounded-full">
                              <User className="h-5 w-5 text-green-600" />
                            </div>
                            <div>
                              <p className="font-medium text-green-900">Comportamento consistente</p>
                              <p className="text-sm text-green-700">Esta transação está de acordo com o histórico do cliente. O valor, horário e tipo são semelhantes às operações anteriores.</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                            <div className="bg-green-100 p-2 rounded-full">
                              <MapPin className="h-5 w-5 text-green-600" />
                            </div>
                            <div>
                              <p className="font-medium text-green-900">Local conhecido</p>
                              <p className="text-sm text-green-700">A transação foi realizada em {selectedTransaction.cidade || 'uma localidade'} que faz parte do padrão habitual deste cliente.</p>
                            </div>
                          </div>
                          <div className="flex items-start space-x-3 bg-white/60 rounded-lg p-4">
                            <div className="bg-green-100 p-2 rounded-full">
                              <Clock className="h-5 w-5 text-green-600" />
                            </div>
                            <div>
                              <p className="font-medium text-green-900">Horário adequado</p>
                              <p className="text-sm text-green-700">A transação foi feita em um horário normal de operação, sem características suspeitas como madrugada ou feriados.</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* SEÇÃO 6: O QUE O ANALISTA DEVE FAZER */}
                    <div className={`rounded-xl p-5 border-2 ${
                      selectedTransaction.fraud_score > 0.7 
                        ? 'bg-purple-50 border-purple-200' 
                        : 'bg-indigo-50 border-indigo-200'
                    }`}>
                      <div className="flex items-center space-x-2 mb-4">
                        <User className="h-5 w-5 text-purple-600" />
                        <h4 className="font-bold text-purple-900">Recomendação para o Analista</h4>
                      </div>
                      <div className={`p-4 rounded-lg ${
                        selectedTransaction.fraud_score > 0.7 ? 'bg-purple-100' : 'bg-indigo-100'
                      }`}>
                        {selectedTransaction.fraud_score > 0.7 ? (
                          <div className="space-y-3">
                            <p className="text-purple-900 font-medium">Esta transação PRECISA de análise manual. Sugerimos:</p>
                            <ul className="space-y-2 text-purple-800">
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                <span>Verificar se o cliente reconhece esta transação (ligar ou enviar notificação)</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                <span>Confirmar os dados do destinatário</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                <span>Verificar se houve outras tentativas suspeitas recentes</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                                <span>Se não confirmar, REJEITAR a transação imediatamente</span>
                              </li>
                            </ul>
                          </div>
                        ) : selectedTransaction.fraud_score > 0.4 ? (
                          <div className="space-y-3">
                            <p className="text-indigo-900 font-medium">Esta transação merece atenção moderada. Sugerimos:</p>
                            <ul className="space-y-2 text-indigo-800">
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-indigo-500 rounded-full"></span>
                                <span>Verificar rapidamente o histórico recente do cliente</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-indigo-500 rounded-full"></span>
                                <span>Se tudo parecer normal, pode aprovar com monitoramento</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-indigo-500 rounded-full"></span>
                                <span>Registrar observações para análise futura se necessário</span>
                              </li>
                            </ul>
                          </div>
                        ) : (
                          <div className="space-y-3">
                            <p className="text-indigo-900 font-medium">Esta transação pode ser aprovada com segurança:</p>
                            <ul className="space-y-2 text-indigo-800">
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                <span>Todos os indicadores estão dentro do normal</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                <span>Pode aprovar automaticamente ou com revisão rápida</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                <span>Não há necessidade de contato com o cliente</span>
                              </li>
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* SEÇÃO 7: COMPLIANCE LGPD */}
                    <div className="bg-gray-100 border border-gray-300 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <Shield className="h-5 w-5 text-gray-600 mt-0.5" />
                        <div>
                          <p className="font-medium text-gray-800">Transparência e seus direitos (LGPD)</p>
                          <p className="text-sm text-gray-600 mt-1">
                            Conforme a Lei Geral de Proteção de Dados (Art. 20), você tem direito a entender como decisões automatizadas afetam você. 
                            Esta explicação foi gerada automaticamente pelo sistema Sankofa e pode ser contestada ou revista por um humano a qualquer momento.
                          </p>
                        </div>
                      </div>
                    </div>
                    
                  </div>
                )}
              </div>
              
              {/* Ações */}
              <div className="border-t border-[var(--color-border)] pt-4 flex flex-wrap gap-2">
                <Button 
                  variant="primary" 
                  size="sm"
                  onClick={() => { handleAction('approve', selectedTransaction); handleCloseDetails(); }}
                  disabled={actionLoading}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Aprovar
                </Button>
                <Button 
                  variant="secondary" 
                  size="sm"
                  onClick={() => { handleAction('reject', selectedTransaction); handleCloseDetails(); }}
                  disabled={actionLoading}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Rejeitar
                </Button>
                <Button 
                  variant="secondary" 
                  size="sm"
                  onClick={() => { handleAction('review', selectedTransaction); handleCloseDetails(); }}
                  disabled={actionLoading}
                >
                  <Clock className="h-4 w-4 mr-2" />
                  Enviar p/ Revisão
                </Button>
                <Button 
                  variant="secondary" 
                  size="sm"
                  onClick={() => { handleAction('investigate', selectedTransaction); handleCloseDetails(); }}
                  disabled={actionLoading}
                >
                  <Shield className="h-4 w-4 mr-2" />
                  Investigar
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-[var(--color-text-secondary)]">
            Página {currentPage} de {totalPages}
          </p>
          <div className="flex items-center space-x-2">
            <Button 
              variant="secondary" 
              size="sm" 
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(currentPage - 1)}
            >
              Anterior
            </Button>
            <Button 
              variant="secondary" 
              size="sm" 
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(currentPage + 1)}
            >
              Próxima
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

