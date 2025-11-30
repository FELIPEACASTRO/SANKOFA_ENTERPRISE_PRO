import { AlertTriangle, CheckCircle, XCircle, Clock, User, Shield, TrendingUp, Zap, Eye, Brain, Target, ArrowRight, ArrowDown, Phone, Mail, Building, CreditCard, Smartphone } from 'lucide-react';

// Persona Card - Mostra analistas com foto, cargo e contexto
export function PersonaCard({ name, role, avatar, department, experience, quote, color = 'blue' }) {
  const colors = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    orange: 'from-orange-500 to-orange-600'
  };
  
  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      <div className={`bg-gradient-to-r ${colors[color]} p-4 text-white`}>
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-white/20 flex items-center justify-center text-2xl font-bold">
            {avatar}
          </div>
          <div>
            <h3 className="text-xl font-bold">{name}</h3>
            <p className="text-white/90">{role}</p>
            <p className="text-sm text-white/80">{department}</p>
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
          <Clock className="h-4 w-4" />
          <span>{experience}</span>
        </div>
        <blockquote className="italic text-gray-700 border-l-4 border-blue-300 pl-3">
          "{quote}"
        </blockquote>
      </div>
    </div>
  );
}

// Scenario Timeline - Mostra uma história passo a passo
export function ScenarioTimeline({ title, icon: Icon, steps, outcome, outcomeType = 'success' }) {
  const outcomeColors = {
    success: 'bg-green-100 border-green-500 text-green-800',
    warning: 'bg-yellow-100 border-yellow-500 text-yellow-800',
    danger: 'bg-red-100 border-red-500 text-red-800'
  };
  
  return (
    <div className="bg-gradient-to-br from-gray-50 to-white rounded-xl p-6 border border-gray-200">
      <div className="flex items-center gap-3 mb-6">
        {Icon && <Icon className="h-6 w-6 text-blue-600" />}
        <h3 className="text-lg font-bold text-gray-900">{title}</h3>
      </div>
      
      <div className="relative">
        {/* Linha vertical conectora */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-blue-200" />
        
        <div className="space-y-4">
          {steps.map((step, index) => (
            <div key={index} className="relative flex gap-4">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center z-10 text-white text-sm font-bold ${
                step.type === 'action' ? 'bg-blue-500' :
                step.type === 'alert' ? 'bg-red-500' :
                step.type === 'success' ? 'bg-green-500' :
                'bg-gray-400'
              }`}>
                {index + 1}
              </div>
              <div className="flex-1 bg-white rounded-lg p-4 shadow-sm border border-gray-100">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-gray-500">{step.time}</span>
                  {step.badge && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      step.badge === 'PIX' ? 'bg-green-100 text-green-700' :
                      step.badge === 'ALERTA' ? 'bg-red-100 text-red-700' :
                      step.badge === 'BLOQUEIO' ? 'bg-red-100 text-red-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {step.badge}
                    </span>
                  )}
                </div>
                <p className="text-gray-800 font-medium">{step.title}</p>
                <p className="text-sm text-gray-600">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {outcome && (
        <div className={`mt-6 p-4 rounded-lg border-l-4 ${outcomeColors[outcomeType]}`}>
          <p className="font-semibold flex items-center gap-2">
            {outcomeType === 'success' && <CheckCircle className="h-5 w-5" />}
            {outcomeType === 'warning' && <AlertTriangle className="h-5 w-5" />}
            {outcomeType === 'danger' && <XCircle className="h-5 w-5" />}
            Resultado:
          </p>
          <p>{outcome}</p>
        </div>
      )}
    </div>
  );
}

// Flow Diagram - Diagrama de fluxo visual
export function FlowDiagram({ title, nodes }) {
  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
      {title && <h4 className="text-center font-bold text-gray-800 mb-6">{title}</h4>}
      
      <div className="flex flex-col items-center gap-2">
        {nodes.map((node, index) => (
          <div key={index} className="flex flex-col items-center">
            <div className={`
              px-6 py-3 rounded-lg text-center font-medium shadow-sm
              ${node.type === 'start' ? 'bg-blue-500 text-white rounded-full' : ''}
              ${node.type === 'process' ? 'bg-white border-2 border-blue-300 text-gray-800' : ''}
              ${node.type === 'decision' ? 'bg-yellow-100 border-2 border-yellow-400 text-gray-800 transform rotate-0' : ''}
              ${node.type === 'success' ? 'bg-green-500 text-white' : ''}
              ${node.type === 'danger' ? 'bg-red-500 text-white' : ''}
              ${node.type === 'end' ? 'bg-gray-700 text-white rounded-full' : ''}
            `}>
              {node.icon && <node.icon className="inline h-4 w-4 mr-2" />}
              {node.label}
            </div>
            
            {index < nodes.length - 1 && (
              <ArrowDown className="h-6 w-6 text-blue-400 my-1" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Risk Thermometer - Termômetro de risco visual
export function RiskThermometer({ score, label }) {
  const getColor = (s) => {
    if (s <= 30) return 'from-green-400 to-green-500';
    if (s <= 50) return 'from-yellow-400 to-yellow-500';
    if (s <= 70) return 'from-orange-400 to-orange-500';
    return 'from-red-500 to-red-600';
  };
  
  const getLabel = (s) => {
    if (s <= 30) return 'Baixo Risco';
    if (s <= 50) return 'Risco Moderado';
    if (s <= 70) return 'Alto Risco';
    return 'Risco Crítico';
  };
  
  return (
    <div className="bg-white rounded-lg p-4 border border-gray-200">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">{label || 'Score de Risco'}</span>
        <span className="text-2xl font-bold text-gray-900">{score}</span>
      </div>
      <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${getColor(score)} transition-all duration-500`}
          style={{ width: `${score}%` }}
        />
      </div>
      <p className="text-xs text-center mt-2 text-gray-500">{getLabel(score)}</p>
    </div>
  );
}

// Transaction Card - Card de transação visual
export function TransactionCard({ id, cpf, amount, channel, time, status, score, location }) {
  const statusConfig = {
    approved: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50', label: 'Aprovada' },
    blocked: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50', label: 'Bloqueada' },
    pending: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-50', label: 'Em Análise' }
  };
  
  const channelConfig = {
    PIX: { icon: Zap, color: 'bg-green-500' },
    TED: { icon: Building, color: 'bg-blue-500' },
    CARTAO: { icon: CreditCard, color: 'bg-purple-500' }
  };
  
  const config = statusConfig[status] || statusConfig.pending;
  const channelCfg = channelConfig[channel] || channelConfig.PIX;
  const StatusIcon = config.icon;
  const ChannelIcon = channelCfg.icon;
  
  return (
    <div className={`rounded-xl border-2 ${config.bg} border-gray-200 p-4 shadow-sm`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`${channelCfg.color} text-white p-2 rounded-lg`}>
            <ChannelIcon className="h-5 w-5" />
          </div>
          <div>
            <p className="font-bold text-gray-900">{channel}</p>
            <p className="text-xs text-gray-500">{time}</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <StatusIcon className={`h-5 w-5 ${config.color}`} />
          <span className={`text-sm font-medium ${config.color}`}>{config.label}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-gray-500">ID</p>
          <p className="font-mono text-gray-800">{id}</p>
        </div>
        <div>
          <p className="text-gray-500">CPF</p>
          <p className="font-mono text-gray-800">{cpf}</p>
        </div>
        <div>
          <p className="text-gray-500">Valor</p>
          <p className="font-bold text-gray-900 text-lg">{amount}</p>
        </div>
        <div>
          <p className="text-gray-500">Score</p>
          <RiskThermometer score={score} />
        </div>
      </div>
      
      {location && (
        <div className="mt-3 pt-3 border-t border-gray-200 text-sm text-gray-600">
          📍 {location}
        </div>
      )}
    </div>
  );
}

// KPI Card - Card de indicador
export function KPICard({ title, value, change, changeType, icon: Icon, color = 'blue' }) {
  const colors = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    red: 'from-red-500 to-red-600',
    orange: 'from-orange-500 to-orange-600',
    purple: 'from-purple-500 to-purple-600'
  };
  
  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      <div className={`bg-gradient-to-r ${colors[color]} p-4`}>
        <div className="flex items-center justify-between">
          <div className="text-white">
            <p className="text-sm opacity-90">{title}</p>
            <p className="text-3xl font-bold">{value}</p>
          </div>
          {Icon && (
            <div className="bg-white/20 p-3 rounded-lg">
              <Icon className="h-8 w-8 text-white" />
            </div>
          )}
        </div>
      </div>
      {change && (
        <div className="p-3 flex items-center gap-2">
          <TrendingUp className={`h-4 w-4 ${changeType === 'up' ? 'text-green-500' : 'text-red-500'}`} />
          <span className={`text-sm ${changeType === 'up' ? 'text-green-600' : 'text-red-600'}`}>
            {change}
          </span>
          <span className="text-gray-500 text-sm">vs. ontem</span>
        </div>
      )}
    </div>
  );
}

// Alert Box - Caixa de alerta
export function AlertBox({ type, title, children }) {
  const configs = {
    info: { bg: 'bg-blue-50', border: 'border-blue-500', icon: Eye, iconColor: 'text-blue-500' },
    success: { bg: 'bg-green-50', border: 'border-green-500', icon: CheckCircle, iconColor: 'text-green-500' },
    warning: { bg: 'bg-yellow-50', border: 'border-yellow-500', icon: AlertTriangle, iconColor: 'text-yellow-500' },
    danger: { bg: 'bg-red-50', border: 'border-red-500', icon: XCircle, iconColor: 'text-red-500' },
    tip: { bg: 'bg-purple-50', border: 'border-purple-500', icon: Brain, iconColor: 'text-purple-500' }
  };
  
  const config = configs[type] || configs.info;
  const Icon = config.icon;
  
  return (
    <div className={`${config.bg} border-l-4 ${config.border} rounded-r-lg p-4`}>
      <div className="flex items-start gap-3">
        <Icon className={`h-6 w-6 ${config.iconColor} flex-shrink-0 mt-0.5`} />
        <div>
          {title && <p className="font-bold text-gray-900 mb-1">{title}</p>}
          <div className="text-gray-700">{children}</div>
        </div>
      </div>
    </div>
  );
}

// Checklist - Lista de verificação
export function Checklist({ items, title }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      {title && <h4 className="font-bold text-gray-900 mb-3">{title}</h4>}
      <ul className="space-y-2">
        {items.map((item, index) => (
          <li key={index} className="flex items-start gap-3">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
              item.done ? 'bg-green-500' : 'bg-gray-200'
            }`}>
              {item.done ? (
                <CheckCircle className="h-4 w-4 text-white" />
              ) : (
                <span className="text-xs text-gray-500">{index + 1}</span>
              )}
            </div>
            <span className={item.done ? 'text-gray-500 line-through' : 'text-gray-800'}>
              {item.text}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Screen Preview - Prévia de tela
export function ScreenPreview({ title, description, elements }) {
  return (
    <div className="bg-gray-900 rounded-xl overflow-hidden shadow-xl">
      {/* Browser header */}
      <div className="bg-gray-800 px-4 py-2 flex items-center gap-2">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
        </div>
        <div className="flex-1 ml-4">
          <div className="bg-gray-700 rounded-lg px-3 py-1 text-xs text-gray-300 max-w-xs">
            sankofa.app/{title.toLowerCase().replace(/\s/g, '-')}
          </div>
        </div>
      </div>
      
      {/* Screen content */}
      <div className="bg-gray-100 p-4">
        <div className="bg-white rounded-lg shadow-sm p-4">
          <h3 className="text-lg font-bold text-gray-900 mb-2">{title}</h3>
          <p className="text-sm text-gray-600 mb-4">{description}</p>
          
          <div className="grid grid-cols-2 gap-3">
            {elements.map((el, i) => (
              <div key={i} className={`h-16 rounded-lg flex items-center justify-center text-sm font-medium ${
                el.type === 'chart' ? 'bg-blue-100 text-blue-700' :
                el.type === 'table' ? 'bg-green-100 text-green-700' :
                el.type === 'kpi' ? 'bg-purple-100 text-purple-700' :
                'bg-gray-100 text-gray-700'
              }`}>
                {el.label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default {
  PersonaCard,
  ScenarioTimeline,
  FlowDiagram,
  RiskThermometer,
  TransactionCard,
  KPICard,
  AlertBox,
  Checklist,
  ScreenPreview
};
