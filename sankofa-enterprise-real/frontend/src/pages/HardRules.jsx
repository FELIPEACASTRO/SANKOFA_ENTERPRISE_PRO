import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/Button.jsx';
import { 
  Plus, 
  Edit, 
  Trash2, 
  Play, 
  Pause, 
  Calendar,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Filter,
  Download,
  Upload,
  ChevronDown,
  ChevronUp,
  Layers,
  Zap,
  Shield,
  Bell,
  Target,
  Info,
  Lightbulb,
  BookOpen
} from 'lucide-react';

const HardRules = () => {
  const [rules, setRules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showDialog, setShowDialog] = useState(false);
  const [editingRule, setEditingRule] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [ruleExplanation, setRuleExplanation] = useState(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    conditions: [{ field: '', operator: '', value: '' }],
    logic_operator: 'AND',
    action: 'block',
    action_config: {},
    rule_type: 'blocking',
    priority: 1,
    enabled: true
  });

  useEffect(() => {
    loadRules();
    loadMetadata();
  }, []);

  const fetchExplanation = useCallback(async () => {
    const validConditions = formData.conditions.filter(c => c.field && c.operator && c.value);
    if (validConditions.length === 0) {
      setRuleExplanation(null);
      return;
    }
    
    setLoadingExplanation(true);
    try {
      const response = await fetch('/api/hard-rules/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conditions_json: validConditions,
          logic_operator: formData.logic_operator,
          action: formData.action,
          rule_type: formData.rule_type,
          name: formData.name
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setRuleExplanation(data.data);
      }
    } catch (error) {
      console.error('Erro ao buscar explicação:', error);
    } finally {
      setLoadingExplanation(false);
    }
  }, [formData.conditions, formData.logic_operator, formData.action, formData.rule_type, formData.name]);

  useEffect(() => {
    if (showDialog) {
      const timeoutId = setTimeout(() => {
        fetchExplanation();
      }, 500);
      return () => clearTimeout(timeoutId);
    }
  }, [formData.conditions, formData.logic_operator, formData.action, showDialog, fetchExplanation]);

  const loadMetadata = async () => {
    try {
      const response = await fetch('/api/hard-rules/metadata');
      if (response.ok) {
        const data = await response.json();
        setMetadata(data.data);
      }
    } catch (error) {
      console.error('Erro ao carregar metadados:', error);
    }
  };

  const loadRules = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/hard-rules');
      if (response.ok) {
        const data = await response.json();
        const rulesList = data.data?.rules || data.rules || [];
        setRules(rulesList);
      }
    } catch (error) {
      console.error('Erro ao carregar regras:', error);
    } finally {
      setLoading(false);
    }
  };

  const getActionLabel = (action) => {
    const actionMap = {
      'block': 'Bloquear',
      'review': 'Revisar',
      'alert': 'Alertar',
      'approve': 'Aprovar',
      'step_up': 'Step-Up',
      'score_adjust': 'Ajustar Score'
    };
    return actionMap[action] || action;
  };

  const getRuleTypeIcon = (ruleType) => {
    const icons = {
      'blocking': <Shield className="w-4 h-4 text-red-500" />,
      'scoring': <Target className="w-4 h-4 text-blue-500" />,
      'routing': <Layers className="w-4 h-4 text-purple-500" />,
      'alerting': <Bell className="w-4 h-4 text-orange-500" />
    };
    return icons[ruleType] || <Zap className="w-4 h-4 text-gray-500" />;
  };

  const getRuleTypeLabel = (ruleType) => {
    const labels = {
      'blocking': 'Bloqueio',
      'scoring': 'Pontuação',
      'routing': 'Roteamento',
      'alerting': 'Alerta'
    };
    return labels[ruleType] || ruleType;
  };

  const addCondition = () => {
    setFormData({
      ...formData,
      conditions: [...formData.conditions, { field: '', operator: '', value: '' }]
    });
  };

  const removeCondition = (index) => {
    if (formData.conditions.length > 1) {
      const newConditions = formData.conditions.filter((_, i) => i !== index);
      setFormData({ ...formData, conditions: newConditions });
    }
  };

  const updateCondition = (index, key, value) => {
    const newConditions = [...formData.conditions];
    newConditions[index] = { ...newConditions[index], [key]: value };
    setFormData({ ...formData, conditions: newConditions });
  };

  const getFieldInfo = (fieldValue) => {
    if (!metadata) return null;
    return metadata.fields.find(f => f.value === fieldValue);
  };

  const getOperatorsForField = (fieldValue) => {
    if (!metadata) return [];
    const fieldInfo = getFieldInfo(fieldValue);
    if (!fieldInfo) return metadata.operators;
    return metadata.operators.filter(op => op.types.includes(fieldInfo.type));
  };

  const buildConditionString = () => {
    const parts = formData.conditions
      .filter(c => c.field && c.operator && c.value)
      .map(c => `${c.field} ${c.operator} ${c.value}`);
    return parts.join(` ${formData.logic_operator} `);
  };

  const handleSave = async () => {
    try {
      const url = editingRule ? `/api/hard-rules/${editingRule.id}` : '/api/hard-rules';
      const method = editingRule ? 'PUT' : 'POST';
      
      const conditionString = buildConditionString();
      const payload = {
        name: formData.name,
        description: formData.description,
        condition: conditionString,
        conditions_json: formData.conditions.filter(c => c.field && c.operator && c.value),
        logic_operator: formData.logic_operator,
        action: formData.action,
        action_config: formData.action_config,
        rule_type: formData.rule_type,
        priority: formData.priority,
        enabled: formData.enabled
      };
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        await loadRules();
        setShowDialog(false);
        resetForm();
      }
    } catch (error) {
      console.error('Erro ao salvar regra:', error);
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Tem certeza que deseja excluir esta regra?')) {
      try {
        const response = await fetch(`/api/hard-rules/${id}`, { method: 'DELETE' });
        if (response.ok) {
          await loadRules();
        }
      } catch (error) {
        console.error('Erro ao excluir regra:', error);
      }
    }
  };

  const handleToggleActive = async (rule) => {
    try {
      const response = await fetch(`/api/hard-rules/${rule.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !rule.enabled }),
      });
      if (response.ok) {
        await loadRules();
      }
    } catch (error) {
      console.error('Erro ao alterar status da regra:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      conditions: [{ field: '', operator: '', value: '' }],
      logic_operator: 'AND',
      action: 'block',
      action_config: {},
      rule_type: 'blocking',
      priority: 1,
      enabled: true
    });
    setEditingRule(null);
    setRuleExplanation(null);
  };

  const openDialog = (rule = null) => {
    console.log('openDialog chamado com:', rule);
    if (rule) {
      let conditions = [{ field: '', operator: '', value: '' }];
      
      if (rule.conditions_json) {
        if (Array.isArray(rule.conditions_json) && rule.conditions_json.length > 0) {
          conditions = rule.conditions_json.map(c => ({
            field: c.field || '',
            operator: c.operator || '',
            value: String(c.value || '')
          }));
        } else if (typeof rule.conditions_json === 'string') {
          try {
            const parsed = JSON.parse(rule.conditions_json);
            if (Array.isArray(parsed) && parsed.length > 0) {
              conditions = parsed.map(c => ({
                field: c.field || '',
                operator: c.operator || '',
                value: String(c.value || '')
              }));
            }
          } catch (e) {
            console.error('Erro ao parsear conditions_json:', e);
          }
        }
      }
      
      console.log('Condições processadas:', conditions);
      
      const newFormData = {
        name: rule.name || '',
        description: rule.description || '',
        conditions: conditions,
        logic_operator: rule.logic_operator || 'AND',
        action: rule.action || 'block',
        action_config: rule.action_config || {},
        rule_type: rule.rule_type || 'blocking',
        priority: rule.priority || 1,
        enabled: rule.enabled !== false
      };
      
      console.log('FormData a ser definido:', newFormData);
      setFormData(newFormData);
      setEditingRule(rule);
    } else {
      resetForm();
    }
    setShowDialog(true);
  };

  const filteredFields = metadata?.fields?.filter(
    f => selectedCategory === 'all' || f.category === selectedCategory
  ) || [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Regras Rígidas Avançadas
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Construtor visual de regras com condições múltiplas AND/OR
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => openDialog()}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Plus className="w-4 h-4 mr-2" />
            Nova Regra
          </Button>
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Exportar
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{rules.length}</p>
            </div>
            <Filter className="w-6 h-6 text-blue-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Ativas</p>
              <p className="text-2xl font-bold text-green-600">{rules.filter(r => r.enabled).length}</p>
            </div>
            <CheckCircle className="w-6 h-6 text-green-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Bloqueio</p>
              <p className="text-2xl font-bold text-red-600">{rules.filter(r => r.rule_type === 'blocking').length}</p>
            </div>
            <Shield className="w-6 h-6 text-red-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Pontuação</p>
              <p className="text-2xl font-bold text-blue-600">{rules.filter(r => r.rule_type === 'scoring').length}</p>
            </div>
            <Target className="w-6 h-6 text-blue-500" />
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Alerta</p>
              <p className="text-2xl font-bold text-orange-600">{rules.filter(r => r.rule_type === 'alerting').length}</p>
            </div>
            <Bell className="w-6 h-6 text-orange-500" />
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg border">
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Lista de Regras</h2>
          <div className="text-sm text-gray-500">Ordenado por prioridade</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Prioridade</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Nome</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Tipo</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Condições</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Ação</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Ações</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {loading ? (
                <tr>
                  <td colSpan="7" className="px-4 py-8 text-center text-gray-500">Carregando...</td>
                </tr>
              ) : rules.length === 0 ? (
                <tr>
                  <td colSpan="7" className="px-4 py-8 text-center text-gray-500">
                    Nenhuma regra encontrada. Clique em "Nova Regra" para criar.
                  </td>
                </tr>
              ) : (
                rules.map((rule) => (
                  <tr key={rule.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-4">
                      <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium">
                        {rule.priority || 1}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm font-medium text-gray-900 dark:text-white">{rule.name}</div>
                      {rule.description && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">{rule.description}</div>
                      )}
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-2">
                        {getRuleTypeIcon(rule.rule_type)}
                        <span className="text-sm">{getRuleTypeLabel(rule.rule_type)}</span>
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <div className="max-w-xs">
                        {rule.conditions_json && rule.conditions_json.length > 0 ? (
                          <div className="space-y-1">
                            {rule.conditions_json.map((cond, idx) => (
                              <div key={idx} className="text-xs bg-gray-100 dark:bg-gray-700 rounded px-2 py-1 inline-block mr-1">
                                {cond.field} {cond.operator} {cond.value}
                              </div>
                            ))}
                            {rule.conditions_json.length > 1 && (
                              <span className="text-xs text-blue-600 font-medium ml-1">
                                ({rule.logic_operator})
                              </span>
                            )}
                          </div>
                        ) : (
                          <span className="text-xs text-gray-500">{rule.condition}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        rule.action === 'block' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        rule.action === 'review' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        rule.action === 'alert' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200' :
                        'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      }`}>
                        {getActionLabel(rule.action)}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <button
                        onClick={() => handleToggleActive(rule)}
                        className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${
                          rule.enabled 
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}
                      >
                        {rule.enabled ? <CheckCircle className="w-3 h-3 mr-1" /> : <XCircle className="w-3 h-3 mr-1" />}
                        {rule.enabled ? 'Ativa' : 'Inativa'}
                      </button>
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => openDialog(rule)}
                          className="text-blue-600 hover:text-blue-900 dark:text-blue-400"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(rule.id)}
                          className="text-red-600 hover:text-red-900 dark:text-red-400"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {showDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              {editingRule ? 'Editar Regra' : 'Nova Regra Avançada'}
            </h2>
            
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Nome da Regra *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
                    placeholder="Ex: Bloqueio PIX Alto Valor Noturno"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Prioridade (1 = mais alta)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={formData.priority}
                    onChange={(e) => setFormData({ ...formData, priority: parseInt(e.target.value) || 1 })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Descrição
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
                  rows="2"
                  placeholder="Descreva o objetivo desta regra..."
                />
              </div>

              <div className="border dark:border-gray-600 rounded-lg p-4 bg-gray-50 dark:bg-gray-700/50">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Construtor de Condições
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Combinar com:</span>
                    <select
                      value={formData.logic_operator}
                      onChange={(e) => setFormData({ ...formData, logic_operator: e.target.value })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                    >
                      <option value="AND">E (todas)</option>
                      <option value="OR">OU (qualquer)</option>
                    </select>
                  </div>
                </div>

                <div className="mb-3">
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Filtrar campos por categoria:</label>
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                  >
                    <option value="all">Todas as Categorias</option>
                    {metadata?.field_categories?.map(cat => (
                      <option key={cat.value} value={cat.value}>{cat.label}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-3">
                  {formData.conditions.map((condition, index) => (
                    <div key={index} className="flex items-center gap-2 bg-white dark:bg-gray-800 p-3 rounded-lg border dark:border-gray-600">
                      {index > 0 && (
                        <span className="text-xs font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/50 px-2 py-1 rounded">
                          {formData.logic_operator}
                        </span>
                      )}
                      <select
                        value={condition.field}
                        onChange={(e) => updateCondition(index, 'field', e.target.value)}
                        className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                      >
                        <option value="">Selecione o campo...</option>
                        {(selectedCategory === 'all' ? metadata?.fields : filteredFields)?.map(field => (
                          <option key={field.value} value={field.value}>
                            {field.label} ({field.category})
                          </option>
                        ))}
                      </select>
                      <select
                        value={condition.operator}
                        onChange={(e) => updateCondition(index, 'operator', e.target.value)}
                        className="w-40 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                      >
                        <option value="">Operador...</option>
                        {getOperatorsForField(condition.field).map(op => (
                          <option key={op.value} value={op.value}>{op.label}</option>
                        ))}
                      </select>
                      {getFieldInfo(condition.field)?.options ? (
                        <select
                          value={condition.value}
                          onChange={(e) => updateCondition(index, 'value', e.target.value)}
                          className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                        >
                          <option value="">Selecione...</option>
                          {getFieldInfo(condition.field).options.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type={getFieldInfo(condition.field)?.type === 'number' ? 'number' : 'text'}
                          value={condition.value}
                          onChange={(e) => updateCondition(index, 'value', e.target.value)}
                          className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white text-sm"
                          placeholder="Valor..."
                        />
                      )}
                      {formData.conditions.length > 1 && (
                        <button
                          onClick={() => removeCondition(index)}
                          className="p-2 text-red-500 hover:text-red-700"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>

                <button
                  onClick={addCondition}
                  className="mt-3 flex items-center gap-2 text-blue-600 hover:text-blue-800 dark:text-blue-400 text-sm font-medium"
                >
                  <Plus className="w-4 h-4" />
                  Adicionar Condição
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Tipo de Regra
                  </label>
                  <select
                    value={formData.rule_type}
                    onChange={(e) => setFormData({ ...formData, rule_type: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
                  >
                    {metadata?.rule_types?.map(type => (
                      <option key={type.value} value={type.value}>{type.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Ação
                  </label>
                  <select
                    value={formData.action}
                    onChange={(e) => setFormData({ ...formData, action: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md dark:bg-gray-700 dark:text-white"
                  >
                    {metadata?.actions?.map(action => (
                      <option key={action.value} value={action.value}>{action.label}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center pt-6">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formData.enabled}
                      onChange={(e) => setFormData({ ...formData, enabled: e.target.checked })}
                      className="mr-2 w-4 h-4"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">Regra Ativa</span>
                  </label>
                </div>
              </div>

              {formData.conditions.filter(c => c.field && c.operator && c.value).length > 0 && (
                <div className="space-y-4">
                  <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">Preview da Regra:</h4>
                    <code className="text-sm text-blue-700 dark:text-blue-300 font-mono">
                      SE ({buildConditionString()}) ENTÃO {getActionLabel(formData.action).toUpperCase()}
                    </code>
                  </div>

                  {loadingExplanation ? (
                    <div className="bg-gray-50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 rounded-lg p-4">
                      <div className="flex items-center gap-2 text-gray-500">
                        <div className="animate-spin w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full"></div>
                        <span className="text-sm">Analisando regra...</span>
                      </div>
                    </div>
                  ) : ruleExplanation && (
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 border border-green-200 dark:border-green-800 rounded-lg p-4">
                      <div className="flex items-start gap-3 mb-3">
                        <Lightbulb className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                        <div>
                          <h4 className="text-sm font-semibold text-green-800 dark:text-green-200 mb-1">
                            Explicação da Regra
                          </h4>
                          <p className="text-sm text-green-700 dark:text-green-300">
                            {ruleExplanation.explanation}
                          </p>
                        </div>
                      </div>
                      
                      {ruleExplanation.risk_analysis && ruleExplanation.risk_analysis.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-green-200 dark:border-green-700">
                          <div className="flex items-start gap-2">
                            <Info className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
                            <div>
                              <h5 className="text-xs font-medium text-amber-700 dark:text-amber-300 mb-1">
                                Análise de Risco (Baseada em Dados Reais)
                              </h5>
                              <ul className="text-xs text-amber-600 dark:text-amber-400 space-y-1">
                                {ruleExplanation.risk_analysis.map((analysis, idx) => (
                                  <li key={idx} className="flex items-start gap-1">
                                    <span className="text-amber-500">•</span>
                                    {analysis}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      )}

                      {ruleExplanation.data_insights && (
                        <div className="mt-3 pt-3 border-t border-green-200 dark:border-green-700">
                          <div className="flex items-start gap-2">
                            <BookOpen className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                            <div>
                              <h5 className="text-xs font-medium text-blue-700 dark:text-blue-300 mb-2">
                                Insights dos Dados Históricos
                              </h5>
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="bg-white dark:bg-gray-800 rounded p-2">
                                  <span className="text-gray-500">PIX:</span>
                                  <span className="ml-1 font-medium text-red-600">{ruleExplanation.data_insights.pix_fraud_rate}</span>
                                  <span className="text-gray-400"> fraude</span>
                                </div>
                                <div className="bg-white dark:bg-gray-800 rounded p-2">
                                  <span className="text-gray-500">Mobile:</span>
                                  <span className="ml-1 font-medium text-red-600">{ruleExplanation.data_insights.mobile_fraud_rate}</span>
                                  <span className="text-gray-400"> fraude</span>
                                </div>
                                <div className="bg-white dark:bg-gray-800 rounded p-2">
                                  <span className="text-gray-500">Noturno:</span>
                                  <span className="ml-1 font-medium text-orange-600">{ruleExplanation.data_insights.night_fraud_rate}</span>
                                  <span className="text-gray-400"> fraude</span>
                                </div>
                                <div className="bg-white dark:bg-gray-800 rounded p-2">
                                  <span className="text-gray-500">Alto Valor:</span>
                                  <span className="ml-1 font-medium text-red-600">{ruleExplanation.data_insights.high_value_fraud_rate}</span>
                                  <span className="text-gray-400"> fraude</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="mt-3 pt-3 border-t border-green-200 dark:border-green-700">
                        <p className="text-xs text-green-600 dark:text-green-400 italic">
                          {ruleExplanation.recommendation}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="flex justify-end gap-3 pt-4 border-t dark:border-gray-600">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowDialog(false);
                    resetForm();
                  }}
                >
                  Cancelar
                </Button>
                <Button
                  onClick={handleSave}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                  disabled={!formData.name || formData.conditions.every(c => !c.field || !c.operator || !c.value)}
                >
                  {editingRule ? 'Salvar Alterações' : 'Criar Regra'}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HardRules;
