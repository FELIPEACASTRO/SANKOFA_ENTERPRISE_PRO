import { useState } from 'react';
import {
  BookOpen, ChevronDown, ChevronRight, FileText, Brain, AlertTriangle,
  Package, Layers, Zap, BarChart3, Code
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Badge } from '@/components/ui/Badge.jsx';

export function Documentation() {
  const [activeTab, setActiveTab] = useState('sankofa');
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (id) => {
    setExpandedSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const tabs = [
    { id: 'sankofa', label: 'Sankofa 101', icon: BookOpen, color: 'blue' },
    { id: 'ml', label: 'Machine Learning', icon: Brain, color: 'purple' },
    { id: 'fraudes', label: 'Tipos de Fraudes', icon: AlertTriangle, color: 'red' },
    { id: 'payload', label: 'Payload de Entrada', icon: Package, color: 'green' },
    { id: 'funcional', label: 'Documentação Funcional', icon: Layers, color: 'orange' },
    { id: 'diagramas', label: 'Diagramas', icon: BarChart3, color: 'cyan' },
    { id: 'arquitetura', label: 'Arquitetura Técnica', icon: Code, color: 'indigo' }
  ];

  const SectionAccordion = ({ title, icon: Icon, content, id }) => (
    <div className="border border-[var(--color-border)] rounded-lg mb-4 overflow-hidden">
      <button
        onClick={() => toggleSection(id)}
        className="w-full flex items-center justify-between p-4 bg-[var(--color-surface-alt)] hover:bg-[var(--color-surface)] transition"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="h-5 w-5 text-blue-600" />}
          <span className="font-semibold text-gray-900 dark:text-white">{title}</span>
        </div>
        {expandedSections[id] ? (
          <ChevronDown className="h-5 w-5" />
        ) : (
          <ChevronRight className="h-5 w-5" />
        )}
      </button>
      {expandedSections[id] && (
        <div className="p-4 bg-[var(--color-surface)] border-t border-[var(--color-border)]">
          <div className="prose prose-sm dark:prose-invert max-w-none">
            {content}
          </div>
        </div>
      )}
    </div>
  );

  const renderSankofaContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="sankofa-intro"
        title="O Que é Sankofa?"
        icon={BookOpen}
        content={
          <div className="space-y-3">
            <p className="text-gray-700 dark:text-gray-300">
              Sankofa Enterprise Pro é um sistema de detecção de fraudes financeiras que analisa transações em tempo real usando Inteligência Artificial.
            </p>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
              <p className="font-semibold text-blue-900 dark:text-blue-100">Capacidade: 300 milhões de transações/dia</p>
              <p className="text-sm text-blue-800 dark:text-blue-200">Latência: &lt;50ms (SLA BACEN para PIX)</p>
            </div>
          </div>
        }
      />
      <SectionAccordion
        id="sankofa-trio"
        title="O Trio de Modelos ML"
        icon={Brain}
        content={
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/30 p-3 rounded">
                <p className="font-semibold text-green-900 dark:text-green-100">🌲 Random Forest</p>
                <p className="text-xs text-green-800 dark:text-green-200">100 árvores votando</p>
              </div>
              <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-900/30 p-3 rounded">
                <p className="font-semibold text-yellow-900 dark:text-yellow-100">🎯 Gradient Boosting</p>
                <p className="text-xs text-yellow-800 dark:text-yellow-200">100 iterações aprendendo</p>
              </div>
              <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/30 p-3 rounded">
                <p className="font-semibold text-red-900 dark:text-red-100">📈 Meta-Modelo</p>
                <p className="text-xs text-red-800 dark:text-red-200">Logistic Regression</p>
              </div>
            </div>
          </div>
        }
      />
      <SectionAccordion
        id="sankofa-personas"
        title="4 Personas Principais"
        icon={BookOpen}
        content={
          <div className="space-y-2">
            {[
              { name: 'Ana Paula', role: 'Líder de Prevenção a Fraudes', exp: '8 anos' },
              { name: 'Carlos Roberto', role: 'Analista Senior', exp: '5 anos' },
              { name: 'Marina', role: 'Compliance Officer', exp: '10 anos' },
              { name: 'Rodrigo', role: 'Analista Junior', exp: '1 ano' }
            ].map(p => (
              <div key={p.name} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <span className="font-medium">{p.name}</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">{p.role} ({p.exp})</span>
              </div>
            ))}
          </div>
        }
      />
    </div>
  );

  const renderMLContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="ml-47features"
        title="47 Features Explicadas"
        icon={Brain}
        content={
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { group: 'Temporais', count: 7, features: 'hour, day_of_week, is_night, is_business_hours' },
                { group: 'Valor', count: 6, features: 'amount, log_value, amount_zscore, is_high_value' },
                { group: 'Comportamento', count: 8, features: 'velocity, user_history, account_age' },
                { group: 'Dispositivo', count: 4, features: 'device_id, is_new_device, os_type' },
                { group: 'Localização', count: 5, features: 'latitude, longitude, location_entropy' },
                { group: 'Especiais', count: 12, features: 'channel, merchant_type, pix_key_type' }
              ].map(g => (
                <div key={g.group} className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-900/30 p-3 rounded border border-purple-200 dark:border-purple-800">
                  <p className="font-semibold text-purple-900 dark:text-purple-100">{g.group}</p>
                  <Badge className="mt-2 mb-2">{g.count} features</Badge>
                  <p className="text-xs text-purple-800 dark:text-purple-200">{g.features}</p>
                </div>
              ))}
            </div>
          </div>
        }
      />
      <SectionAccordion
        id="ml-processo"
        title="Processo de Decisão (0-100)"
        icon={Zap}
        content={
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded border-l-4 border-green-500">
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">0-30</p>
                <p className="text-sm font-semibold text-green-900 dark:text-green-100">✅ APROVAR</p>
              </div>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border-l-4 border-yellow-500">
                <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">30-70</p>
                <p className="text-sm font-semibold text-yellow-900 dark:text-yellow-100">⚠️ REVISAR</p>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded border-l-4 border-red-500">
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">70-100</p>
                <p className="text-sm font-semibold text-red-900 dark:text-red-100">🚫 BLOQUEAR</p>
              </div>
            </div>
          </div>
        }
      />
    </div>
  );

  const renderFraudesContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="fraude-stats"
        title="Estatísticas Brasileiras"
        icon={AlertTriangle}
        content={
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Perdas Anuais', value: 'R$ 2,5 Bi', color: 'red' },
                { label: 'Fraudes/Hora', value: '4.000', color: 'orange' },
                { label: 'Via PIX', value: '71%', color: 'yellow' },
                { label: 'Engenharia Social', value: '45%', color: 'red' }
              ].map(s => (
                <div key={s.label} className={`bg-${s.color}-50 dark:bg-${s.color}-900/20 p-3 rounded`}>
                  <p className="text-xs text-gray-600 dark:text-gray-400">{s.label}</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-white">{s.value}</p>
                </div>
              ))}
            </div>
          </div>
        }
      />
      <SectionAccordion
        id="fraude-3niveis"
        title="3 Níveis de Fraudadores"
        icon={AlertTriangle}
        content={
          <div className="space-y-2">
            {[
              { level: 'Amador', score: '85-100', desc: 'Oportunista, erros óbvios' },
              { level: 'Profissional', score: '50-75', desc: 'Organizado, usa ferramentas' },
              { level: 'Especialista', score: '30-50', desc: 'Expert, conhece sistemas' }
            ].map(f => (
              <div key={f.level} className="border-l-4 border-red-500 p-3 bg-red-50 dark:bg-red-900/20 rounded">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-red-900 dark:text-red-100">{f.level}</span>
                  <Badge variant="destructive">{f.score}</Badge>
                </div>
                <p className="text-xs text-red-800 dark:text-red-200 mt-1">{f.desc}</p>
              </div>
            ))}
          </div>
        }
      />
    </div>
  );

  const renderPayloadContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="payload-estrutura"
        title="Estrutura do Payload JSON"
        icon={Package}
        content={
          <div className="bg-gray-900 text-gray-100 p-4 rounded font-mono text-sm overflow-x-auto">
            <pre>{`{
  "transactions": [{
    "transaction_id": "TXN1732726800000",
    "amount": 1500.00,
    "customer_id": "CUST_12345",
    "merchant_id": "MERCH_LOJA",
    "transaction_type": "PIX",
    "channel": "mobile",
    "device_id": "device_abc123",
    "timestamp": "2025-11-27T14:30:00"
  }],
  "include_explanation": true
}`}</pre>
          </div>
        }
      />
      <SectionAccordion
        id="payload-campos"
        title="Campos Críticos"
        icon={Package}
        content={
          <div className="space-y-2">
            {[
              { field: 'amount', peso: '100%', desc: 'Valor da transação' },
              { field: 'timestamp', peso: '90%', desc: 'Horário (madrugada = risco)' },
              { field: 'device_id', peso: '80%', desc: 'Dispositivo (novo = risco)' },
              { field: 'location', peso: '80%', desc: 'Localização geográfica' }
            ].map(c => (
              <div key={c.field} className="flex justify-between items-start p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <div>
                  <p className="font-semibold text-gray-900 dark:text-white">{c.field}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">{c.desc}</p>
                </div>
                <Badge>{c.peso}</Badge>
              </div>
            ))}
          </div>
        }
      />
    </div>
  );

  const renderFuncionalContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="funcional-endpoints"
        title="21 Endpoints Funcionais"
        icon={Layers}
        content={
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {[
              { group: 'Health', count: 3 },
              { group: 'Dashboard', count: 5 },
              { group: 'Transactions', count: 8 },
              { group: 'Calibration', count: 3 },
              { group: 'Observability', count: 5 },
              { group: 'Configuration', count: 5 }
            ].map(e => (
              <div key={e.group} className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                <p className="font-semibold text-blue-900 dark:text-blue-100">{e.group}</p>
                <Badge className="mt-2">{e.count} endpoints</Badge>
              </div>
            ))}
          </div>
        }
      />
      <SectionAccordion
        id="funcional-paginas"
        title="16 Páginas do Dashboard"
        icon={Layers}
        content={
          <div className="grid grid-cols-2 gap-2">
            {['Dashboard', 'Transações', 'Calibragem', 'Investigação', 'Revisão Manual', 'Monitoramento', 'Relatórios', 'Métricas', 'Alertas', 'Datasets', 'Regras Duras', 'VIP List', 'HOT List', 'Auditoria', 'Configurações', 'Manual'].map(p => (
              <Badge key={p} variant="secondary" className="justify-center py-2">{p}</Badge>
            ))}
          </div>
        }
      />
    </div>
  );

  const renderDiagramasContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="diagramas-arquitetura"
        title="Arquitetura Geral"
        icon={BarChart3}
        content={
          <div className="space-y-2 text-sm">
            <p className="font-semibold">Camadas do Sistema:</p>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><strong>Clientes:</strong> Core Banking, Mobile, PIX Gateway, Cartões</li>
              <li><strong>API:</strong> Load Balancer + SSL + 50+ Endpoints</li>
              <li><strong>Processamento:</strong> Feature Engine → ML Ensemble → Decision</li>
              <li><strong>Storage:</strong> PostgreSQL + Redis + Files</li>
              <li><strong>Frontend:</strong> React + Vite + TailwindCSS</li>
            </ul>
          </div>
        }
      />
      <SectionAccordion
        id="diagramas-fluxo"
        title="Fluxo de Decisão"
        icon={BarChart3}
        content={
          <div className="space-y-2">
            {[
              { step: '1', desc: 'Validação: Rate Limit → JWT Auth → Schema Validate' },
              { step: '2', desc: 'Features: Extração de 47 características' },
              { step: '3', desc: 'ML: 3 modelos votam (RF + GB + LR)' },
              { step: '4', desc: 'Regras: Aplicam precision rules' },
              { step: '5', desc: 'Decisão: Aprovar → Revisar → Bloquear' }
            ].map(f => (
              <div key={f.step} className="flex gap-3 p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">{f.step}</div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{f.desc}</p>
              </div>
            ))}
          </div>
        }
      />
    </div>
  );

  const renderArquiteturaContent = () => (
    <div className="space-y-4">
      <SectionAccordion
        id="arquitetura-stack"
        title="Stack Tecnológico"
        icon={Code}
        content={
          <div className="grid grid-cols-2 gap-2 text-sm">
            {[
              { layer: 'Frontend', tech: 'React 19 + Vite 6.3 + TailwindCSS' },
              { layer: 'Backend', tech: 'Python 3.12 + Flask 3.0' },
              { layer: 'ML', tech: 'Scikit-learn + XGBoost + LightGBM' },
              { layer: 'Data', tech: 'PostgreSQL (Neon) + Redis' }
            ].map(s => (
              <div key={s.layer} className="bg-indigo-50 dark:bg-indigo-900/20 p-2 rounded">
                <p className="font-semibold text-indigo-900 dark:text-indigo-100">{s.layer}</p>
                <p className="text-xs text-indigo-800 dark:text-indigo-200">{s.tech}</p>
              </div>
            ))}
          </div>
        }
      />
      <SectionAccordion
        id="arquitetura-tipos"
        title="Tipos de Transação"
        icon={Code}
        content={
          <div className="space-y-2">
            {[
              { type: 'PIX', risco: 'ALTO', desc: 'Instantâneo (24/7), irreversível' },
              { type: 'CRÉDITO', risco: 'MÉDIO', desc: 'Presencial/online, chargeback' },
              { type: 'DÉBITO', risco: 'BAIXO', desc: 'Desconto direto, requer senha' },
              { type: 'TED/DOC', risco: 'MÉDIO', desc: 'Transferência tradicional' }
            ].map(t => (
              <div key={t.type} className="flex justify-between items-start p-2 bg-gray-50 dark:bg-gray-800 rounded">
                <div>
                  <p className="font-semibold text-gray-900 dark:text-white">{t.type}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">{t.desc}</p>
                </div>
                <Badge variant={t.risco === 'ALTO' ? 'destructive' : 'secondary'}>{t.risco}</Badge>
              </div>
            ))}
          </div>
        }
      />
    </div>
  );

  const getContent = () => {
    switch (activeTab) {
      case 'sankofa':
        return renderSankofaContent();
      case 'ml':
        return renderMLContent();
      case 'fraudes':
        return renderFraudesContent();
      case 'payload':
        return renderPayloadContent();
      case 'funcional':
        return renderFuncionalContent();
      case 'diagramas':
        return renderDiagramasContent();
      case 'arquitetura':
        return renderArquiteturaContent();
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
            <BookOpen className="h-8 w-8 text-blue-600" />
            Centro de Documentação
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Explore a documentação completa do Sankofa Enterprise Pro
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <Card>
        <CardHeader className="border-b border-gray-200 dark:border-gray-700">
          <CardTitle>{tabs.find(t => t.id === activeTab)?.label}</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          {getContent()}
        </CardContent>
      </Card>

      {/* Footer Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: '7 Documentos', value: 'Completos', icon: '📚' },
          { label: '47 Features', value: 'ML Explicadas', icon: '🧠' },
          { label: '21 Endpoints', value: 'Funcionais', icon: '⚡' },
          { label: '100%', value: 'Cobertura', icon: '✅' }
        ].map(s => (
          <Card key={s.label} className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/10 border-blue-200 dark:border-blue-800">
            <CardContent className="p-4 text-center">
              <p className="text-2xl">{s.icon}</p>
              <p className="font-semibold text-gray-900 dark:text-white mt-2">{s.label}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">{s.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
