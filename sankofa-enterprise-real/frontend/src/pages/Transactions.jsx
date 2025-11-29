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
  Shield
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

  const handleViewDetails = (transaction) => {
    setSelectedTransaction(transaction);
    setShowDetailsModal(true);
    setShowActionsMenu(null);
  };

  const handleCloseDetails = () => {
    setShowDetailsModal(false);
    setSelectedTransaction(null);
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

