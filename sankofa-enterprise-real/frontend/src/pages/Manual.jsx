import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Users, Target, Shield, AlertTriangle, Clock, Zap, Eye, Brain, Settings, FileText, BarChart3, Database, Bell, Lock, Star, CheckCircle, XCircle, TrendingUp, Phone, Building, HelpCircle, Search, Filter, Download, Upload, RefreshCw, Play, Pause, Edit, Trash2, Plus, ArrowRight, ArrowLeft, Info, MessageSquare, ThumbsUp, ThumbsDown, Activity, Cpu, Server, Globe, Calendar, DollarSign, Percent, Hash, List, Grid, PieChart, LineChart, Table, Map, Flag, Award, Bookmark, ExternalLink, Copy, Share, Mail, Send } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { 
  PersonaCard, 
  ScenarioTimeline, 
  FlowDiagram, 
  RiskThermometer, 
  TransactionCard, 
  KPICard, 
  AlertBox, 
  Checklist,
  ScreenPreview 
} from '@/components/manual/ManualComponents.jsx';

const personas = {
  anaPaula: {
    name: 'Ana Paula Oliveira',
    role: 'Líder de Prevenção a Fraudes',
    avatar: 'AP',
    department: 'Banco Digital Nexus - Matriz SP',
    experience: '8 anos em prevenção a fraudes',
    quote: 'Cada fraude bloqueada é um cliente que continua confiando em nós.',
    color: 'blue'
  },
  carlosRoberto: {
    name: 'Carlos Roberto Silva',
    role: 'Analista de Fraudes Sênior',
    avatar: 'CR',
    department: 'Operações de Risco - Turno Diurno',
    experience: '5 anos analisando transações',
    quote: 'O segredo é entender o padrão normal do cliente antes de julgar.',
    color: 'green'
  },
  marinaFernandes: {
    name: 'Marina Fernandes',
    role: 'Compliance Officer',
    avatar: 'MF',
    department: 'Jurídico e Compliance',
    experience: '10 anos em regulação bancária',
    quote: 'LGPD e BACEN não são obstáculos, são nossos aliados.',
    color: 'purple'
  },
  rodrigoMendes: {
    name: 'Rodrigo Mendes',
    role: 'Analista de Fraudes Júnior',
    avatar: 'RM',
    department: 'Operações de Risco - Turno Noturno',
    experience: '1 ano no setor bancário',
    quote: 'Estou aprendendo que cada transação conta uma história.',
    color: 'orange'
  },
  patriciaLima: {
    name: 'Patrícia Lima',
    role: 'Gerente de Operações',
    avatar: 'PL',
    department: 'Diretoria de Riscos',
    experience: '15 anos em bancos',
    quote: 'Métricas não mentem. Precisamos de dados para tomar decisões.',
    color: 'blue'
  }
};

const allScenarios = {
  golpePix: {
    title: '🚨 Golpe do PIX Falso - Fraude Confirmada',
    icon: AlertTriangle,
    steps: [
      { time: '14:32:01', badge: 'PIX', type: 'action', title: 'Transação Recebida', description: 'PIX de R$ 4.850,00 para conta nova (Banco 341 - Ag 0001). CPF destino: ***.***. 789-01. Chave: celular.' },
      { time: '14:32:02', badge: 'ML', type: 'action', title: 'Análise de 40 Features', description: 'Modelo analisa: horário (14h = normal), valor (3x média), destinatário (novo), device (mesmo), geolocalização (SP).' },
      { time: '14:32:03', badge: 'SCORE', type: 'alert', title: 'Score 87/100 - ALTO RISCO', description: 'Fatores: valor atípico (+35 pontos), destinatário novo (+25), padrão de golpe conhecido (+15), horário OK (-5).' },
      { time: '14:32:04', badge: 'BLOQUEIO', type: 'alert', title: 'Transação Bloqueada Automaticamente', description: 'Regra: Score > 70 = bloqueio. Cliente notificado por SMS e push. Alerta criado no sistema.' },
      { time: '14:35:00', type: 'action', title: 'Analista Carlos Assume', description: 'Carlos vê alerta na fila. Abre investigação. Verifica histórico de 90 dias do cliente.' },
      { time: '14:38:00', type: 'action', title: 'Análise de Histórico', description: 'Cliente faz PIX médio de R$ 1.200. Nunca enviou para este destinatário. Último PIX grande foi há 6 meses.' },
      { time: '14:40:00', type: 'action', title: 'Contato com Cliente', description: 'Carlos liga para cliente. "Senhor João, confirma um PIX de R$ 4.850?" Cliente: "Não fiz nada!"' },
      { time: '14:42:00', type: 'success', title: 'Fraude Confirmada', description: 'Cliente confirma golpe de WhatsApp. Alguém se passou por filho pedindo dinheiro urgente.' }
    ],
    outcome: 'R$ 4.850,00 SALVOS! Cliente protegido. Conta laranja reportada ao BACEN via STR. HOT List atualizada.',
    outcomeType: 'success'
  },
  falsoPositivo: {
    title: '✅ Falso Positivo - Viagem de Negócios',
    icon: CheckCircle,
    steps: [
      { time: '23:15:01', badge: 'PIX', type: 'action', title: 'PIX Noturno Alto Valor', description: 'R$ 12.000,00 de SP para Salvador. Pagamento para Hotel Fasano. CPF: ***.***. 456-78.' },
      { time: '23:15:02', badge: 'SCORE', type: 'alert', title: 'Score 72 - Análise Manual', description: 'Horário noturno (+20), valor alto (+15), destino incomum (+20), merchant conhecido (-10), device usual (0).' },
      { time: '23:15:03', badge: 'HOLD', type: 'alert', title: 'Bloqueio Preventivo', description: 'Score entre 60-80 = retenção para análise. Cliente recebe SMS: "Transação em análise de segurança."' },
      { time: '23:18:00', type: 'action', title: 'Cliente Liga Irritado', description: '"Estou na recepção do hotel, preciso pagar a reserva! Por que bloquearam?"' },
      { time: '23:20:00', type: 'action', title: 'Ana Paula Analisa', description: 'Verifica: cliente é empresário, viaja frequentemente, tem histórico de hotéis de luxo.' },
      { time: '23:22:00', type: 'action', title: 'Validação Cruzada', description: 'Consulta sistema de RH da empresa: viagem autorizada para Salvador, dias 30/11 a 02/12.' },
      { time: '23:25:00', type: 'success', title: 'Liberação Manual', description: 'Ana Paula aprova manualmente. Adiciona feedback: "Viagem corporativa confirmada". Modelo aprende.' }
    ],
    outcome: 'Cliente satisfeito! PIX liberado em 10 minutos. Feedback treinou modelo: viagens de negócios são padrão normal.',
    outcomeType: 'success'
  },
  golpeMotoboy: {
    title: '🏍️ Golpe do Motoboy - Cartão Clonado',
    icon: AlertTriangle,
    steps: [
      { time: '10:05:00', badge: 'CARTÃO', type: 'action', title: 'Compra Online Suspeita', description: 'Cartão crédito usado em e-commerce de eletrônicos. Valor: R$ 8.900,00. iPhone 15 Pro Max.' },
      { time: '10:05:01', badge: 'DEVICE', type: 'alert', title: 'Device Novo Detectado', description: 'Compra feita de Android (cliente usa iPhone há 3 anos). IP: VPN comercial.' },
      { time: '10:05:02', badge: 'SCORE', type: 'alert', title: 'Score 92 - CRÍTICO', description: 'Device novo (+40), categoria alto risco (+20), valor alto (+15), VPN (+15), merchant novo (+10).' },
      { time: '10:05:03', badge: 'BLOQUEIO', type: 'alert', title: 'Transação Negada', description: 'Compra rejeitada na origem. Cliente recebe alerta: "Compra negada por segurança."' },
      { time: '10:30:00', type: 'action', title: 'Investigação Proativa', description: 'Rodrigo abre caso mesmo sem reclamação. Padrão indica clonagem de cartão.' },
      { time: '10:45:00', type: 'action', title: 'Contato Preventivo', description: 'Rodrigo liga para cliente: "Senhor Pedro, tentou comprar um iPhone?" Cliente: "Não! Meu cartão está aqui."' },
      { time: '11:00:00', type: 'success', title: 'Cartão Bloqueado', description: 'Cartão cancelado preventivamente. Novo cartão enviado. Cliente orientado sobre golpes.' }
    ],
    outcome: 'R$ 8.900,00 PROTEGIDOS! Fraude impedida antes de acontecer. Cliente elogiou atendimento proativo.',
    outcomeType: 'success'
  },
  tedFraudulenta: {
    title: '🏦 TED Fraudulenta - Conta Corporativa',
    icon: AlertTriangle,
    steps: [
      { time: '16:45:00', badge: 'TED', type: 'action', title: 'TED Alto Valor PJ', description: 'TED de R$ 89.500,00 da conta empresa ABC Ltda para PF desconhecida. Motivo: "Serviços".' },
      { time: '16:45:01', badge: 'REGRA', type: 'alert', title: 'Hard Rule Disparada', description: 'Regra: "TED PJ > R$ 50k para PF nova = BLOQUEAR". Transação retida.' },
      { time: '16:45:02', badge: 'ALERTA', type: 'alert', title: 'Alerta Prioritário', description: 'Caso marcado como CRÍTICO. Valor alto + primeira vez = risco máximo.' },
      { time: '16:50:00', type: 'action', title: 'Dupla Verificação', description: 'Ana Paula e Carlos analisam juntos. Empresa tem 2 sócios: Maria e João.' },
      { time: '17:00:00', type: 'action', title: 'Contato com Empresa', description: 'Ligam para telefone cadastrado. Maria atende: "Não autorizei nenhuma TED!"' },
      { time: '17:05:00', type: 'alert', title: 'Acesso Comprometido', description: 'Investigação revela: João (sócio) teve celular hackeado. Fraudador acessou app.' },
      { time: '17:30:00', type: 'success', title: 'Fraude Bloqueada', description: 'TED cancelada. Conta empresarial bloqueada temporariamente. Novos tokens gerados.' }
    ],
    outcome: 'R$ 89.500,00 SALVOS! Empresa protegida de perda catastrófica. Caso reportado à polícia.',
    outcomeType: 'success'
  },
  boletoVerdadeiro: {
    title: '📄 Boleto Legítimo - Compra de Veículo',
    icon: CheckCircle,
    steps: [
      { time: '09:00:00', badge: 'BOLETO', type: 'action', title: 'Boleto Alto Valor', description: 'Pagamento de boleto R$ 75.000,00 para Concessionária Toyota. Vencimento hoje.' },
      { time: '09:00:01', badge: 'SCORE', type: 'alert', title: 'Score 55 - Zona Cinza', description: 'Valor alto (+25), beneficiário conhecido (-15), cliente antigo (-10), horário comercial (-5).' },
      { time: '09:00:02', badge: 'HOLD', type: 'action', title: 'Retenção para Análise', description: 'Score entre 50-60 = verificação adicional. Tempo máximo: 15 minutos.' },
      { time: '09:05:00', type: 'action', title: 'Análise de Contexto', description: 'Carlos verifica: cliente recebeu R$ 50k de FGTS há 2 dias. Padrão de compra grande.' },
      { time: '09:08:00', type: 'action', title: 'Verificação de Beneficiário', description: 'Concessionária Toyota é cadastrada no CNPJ. Boleto registrado na CIP. Tudo regular.' },
      { time: '09:10:00', type: 'success', title: 'Liberação Aprovada', description: 'Carlos aprova. Feedback: "Compra de veículo confirmada por contexto financeiro."' }
    ],
    outcome: 'Boleto pago em 10 minutos. Cliente conseguiu retirar carro no mesmo dia!',
    outcomeType: 'success'
  },
  contaLaranja: {
    title: '🍊 Detecção de Conta Laranja',
    icon: AlertTriangle,
    steps: [
      { time: '11:00:00', badge: 'PADRÃO', type: 'action', title: 'Análise de Rede', description: 'Sistema detecta conta recebendo PIX de 15 CPFs diferentes em 24h. Total: R$ 45.000.' },
      { time: '11:00:01', badge: 'ML', type: 'alert', title: 'Modelo de Rede Neural', description: 'Padrão de "conta laranja" detectado: múltiplos remetentes, saque rápido, sem histórico.' },
      { time: '11:05:00', type: 'action', title: 'Investigação em Massa', description: 'Marina abre investigação. Verifica os 15 CPFs remetentes.' },
      { time: '11:30:00', type: 'action', title: 'Padrão de Fraude', description: 'Todos os 15 remetentes são vítimas de golpe do PIX. Mesma história: "filho pedindo ajuda".' },
      { time: '12:00:00', badge: 'BACEN', type: 'alert', title: 'Report ao BACEN', description: 'Marina gera relatório STR automático. Conta marcada como fraudulenta.' },
      { time: '12:30:00', type: 'success', title: 'Bloqueio e Ressarcimento', description: 'Conta bloqueada. R$ 32.000 ainda disponíveis são devolvidos às vítimas via MED.' }
    ],
    outcome: 'Conta laranja identificada e bloqueada. 8 vítimas ressarcidas. Fraudador reportado.',
    outcomeType: 'success'
  }
};

function ManualSection({ id, title, icon: Icon, children, defaultOpen = false, priority = 'normal' }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  const priorityStyles = {
    critical: 'border-l-4 border-l-red-500',
    high: 'border-l-4 border-l-orange-500',
    normal: '',
    info: 'border-l-4 border-l-blue-500'
  };
  
  return (
    <Card className={`overflow-hidden ${priorityStyles[priority]}`}>
      <button onClick={() => setIsOpen(!isOpen)} className="w-full text-left">
        <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3">
              {Icon && <Icon className="h-6 w-6 text-blue-600" />}
              {title}
            </CardTitle>
            {isOpen ? <ChevronUp className="h-5 w-5 text-gray-400" /> : <ChevronDown className="h-5 w-5 text-gray-400" />}
          </div>
        </CardHeader>
      </button>
      {isOpen && <CardContent className="pt-0">{children}</CardContent>}
    </Card>
  );
}

function StepByStep({ title, steps }) {
  return (
    <div className="bg-gray-50 rounded-xl p-6 my-4">
      <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
        <List className="h-5 w-5 text-blue-600" />
        {title}
      </h4>
      <div className="space-y-3">
        {steps.map((step, i) => (
          <div key={i} className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold flex-shrink-0">
              {i + 1}
            </div>
            <div className="flex-1">
              <p className="font-semibold text-gray-900">{step.action}</p>
              <p className="text-sm text-gray-600">{step.details}</p>
              {step.tip && (
                <p className="text-sm text-blue-600 mt-1 flex items-center gap-1">
                  <Info className="h-4 w-4" /> {step.tip}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FAQ({ questions }) {
  const [openIndex, setOpenIndex] = useState(null);
  
  return (
    <div className="space-y-2">
      {questions.map((q, i) => (
        <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
          <button 
            onClick={() => setOpenIndex(openIndex === i ? null : i)}
            className="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50"
          >
            <span className="font-medium text-gray-900 flex items-center gap-2">
              <HelpCircle className="h-5 w-5 text-blue-500" />
              {q.question}
            </span>
            {openIndex === i ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
          </button>
          {openIndex === i && (
            <div className="p-4 bg-blue-50 border-t border-gray-200">
              <p className="text-gray-700">{q.answer}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function KeyboardShortcut({ keys, description }) {
  return (
    <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
      <span className="text-sm text-gray-600">{description}</span>
      <div className="flex gap-1">
        {keys.map((key, i) => (
          <span key={i}>
            <kbd className="px-2 py-1 bg-gray-200 rounded text-xs font-mono">{key}</kbd>
            {i < keys.length - 1 && <span className="mx-1 text-gray-400">+</span>}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Manual() {
  const [activeTab, setActiveTab] = useState('inicio');
  
  const tabs = [
    { id: 'inicio', label: 'Início', icon: BookOpen },
    { id: 'dia-a-dia', label: 'Dia a Dia do Analista', icon: Calendar },
    { id: 'telas', label: 'Todas as Telas', icon: Grid },
    { id: 'cenarios', label: 'Cenários Reais', icon: Target },
    { id: 'troubleshooting', label: 'Problemas', icon: AlertTriangle },
    { id: 'glossario', label: 'Glossário', icon: FileText }
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-800 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-center gap-4 mb-4">
          <div className="bg-white/20 p-3 rounded-xl">
            <BookOpen className="h-10 w-10" />
          </div>
          <div>
            <h1 className="text-4xl font-bold">Manual Completo do Sankofa v1.0</h1>
            <p className="text-lg text-blue-100">Sistema de Detecção de Fraudes Bancárias - Guia Definitivo</p>
          </div>
        </div>
        <p className="text-blue-100 max-w-3xl">
          Este é o manual mais completo para analistas de fraude. Contém todas as telas, 
          todos os cenários possíveis, troubleshooting detalhado e glossário completo. 
          Use este guia como sua referência diária de trabalho.
        </p>
        <div className="flex items-center gap-6 mt-6 text-sm flex-wrap">
          <span className="flex items-center gap-2"><Clock className="h-4 w-4" /> Atualizado: 30/11/2025</span>
          <span className="flex items-center gap-2"><Users className="h-4 w-4" /> Para: Analistas de Fraude</span>
          <span className="flex items-center gap-2"><Shield className="h-4 w-4" /> Compliance: LGPD/BACEN/PCI DSS</span>
          <span className="flex items-center gap-2"><FileText className="h-4 w-4" /> Versão: 1.0.0</span>
        </div>
        
        {/* Navigation Tabs */}
        <div className="flex gap-2 mt-6 flex-wrap">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                activeTab === tab.id 
                  ? 'bg-white text-blue-700 font-semibold' 
                  : 'bg-white/20 hover:bg-white/30'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'inicio' && (
        <>
          {/* Personas */}
          <ManualSection id="personas" title="👥 Conheça Nossa Equipe de Especialistas" icon={Users} defaultOpen={true}>
            <p className="text-gray-600 mb-6">
              Acompanhe as histórias de 5 profissionais ao longo deste manual. 
              Cada um tem experiência e perspectiva diferente - você vai aprender com todos eles.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <PersonaCard {...personas.anaPaula} />
              <PersonaCard {...personas.carlosRoberto} />
              <PersonaCard {...personas.marinaFernandes} />
              <PersonaCard {...personas.rodrigoMendes} />
              <PersonaCard {...personas.patriciaLima} />
            </div>
            
            <div className="mt-6 bg-blue-50 rounded-xl p-6">
              <h4 className="font-bold text-gray-900 mb-4">Organização da Equipe</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-blue-700">Turno Diurno (06h-14h)</h5>
                  <ul className="text-sm text-gray-600 mt-2 space-y-1">
                    <li>• Carlos Roberto (Sênior)</li>
                    <li>• 2 Analistas Plenos</li>
                    <li>• Foco: Alta volumetria PIX</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-green-700">Turno Vespertino (14h-22h)</h5>
                  <ul className="text-sm text-gray-600 mt-2 space-y-1">
                    <li>• Ana Paula (Líder)</li>
                    <li>• 3 Analistas Plenos</li>
                    <li>• Foco: TED e Cartões</li>
                  </ul>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-purple-700">Turno Noturno (22h-06h)</h5>
                  <ul className="text-sm text-gray-600 mt-2 space-y-1">
                    <li>• Rodrigo (Júnior + backup)</li>
                    <li>• 1 Analista Pleno</li>
                    <li>• Foco: Monitoramento 24/7</li>
                  </ul>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Como Funciona */}
          <ManualSection id="como-funciona" title="🧠 Como o Sankofa Funciona - Explicação Completa" icon={Brain}>
            <div className="space-y-6">
              <AlertBox type="info" title="O Nome 'Sankofa'">
                "Sankofa" é um símbolo africano que significa "voltar e buscar". Representa a 
                ideia de aprender com o passado para construir o futuro. Nosso sistema faz 
                exatamente isso: analisa padrões históricos para prever fraudes futuras.
              </AlertBox>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Arquitetura do Sistema</h3>
                  <FlowDiagram 
                    title="Fluxo de Análise em Tempo Real"
                    nodes={[
                      { label: '📥 Transação Recebida', type: 'start' },
                      { label: '🔍 Extração de 40+ Features', type: 'process' },
                      { label: '🤖 3 Modelos de ML Analisam', type: 'process' },
                      { label: '📊 Score Final (0-100)', type: 'decision' },
                      { label: '✅ Score < 30: APROVAR', type: 'success' },
                      { label: '⚠️ Score 30-70: REVISAR', type: 'process' },
                      { label: '❌ Score > 70: BLOQUEAR', type: 'danger' }
                    ]}
                  />
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-gray-900">Os 3 Modelos de IA</h3>
                  
                  <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4 border border-green-200">
                    <div className="flex items-center gap-2 mb-2">
                      <Cpu className="h-5 w-5 text-green-600" />
                      <span className="font-bold text-green-800">Modelo 1: Random Forest</span>
                    </div>
                    <p className="text-sm text-gray-700">
                      Especialista em padrões de comportamento. Analisa 500 "árvores de decisão" 
                      para encontrar combinações suspeitas. Melhor para detectar fraudes de cartão.
                    </p>
                    <div className="mt-2 text-xs text-green-600">Acurácia: 94.2% | Tempo: 5ms</div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
                    <div className="flex items-center gap-2 mb-2">
                      <Brain className="h-5 w-5 text-blue-600" />
                      <span className="font-bold text-blue-800">Modelo 2: Gradient Boosting</span>
                    </div>
                    <p className="text-sm text-gray-700">
                      Especialista em valores e horários. Aprende iterativamente com erros 
                      anteriores. Melhor para detectar fraudes de PIX de alto valor.
                    </p>
                    <div className="mt-2 text-xs text-blue-600">Acurácia: 95.8% | Tempo: 8ms</div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4 border border-purple-200">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="h-5 w-5 text-purple-600" />
                      <span className="font-bold text-purple-800">Modelo 3: CatBoost</span>
                    </div>
                    <p className="text-sm text-gray-700">
                      Especialista em dados categóricos. Entende contexto como "tipo de merchant", 
                      "cidade", "canal". Melhor para detectar padrões regionais de fraude.
                    </p>
                    <div className="mt-2 text-xs text-purple-600">Acurácia: 96.1% | Tempo: 6ms</div>
                  </div>
                  
                  <AlertBox type="tip" title="Como os Modelos Votam">
                    Os 3 modelos "votam" e o score final é a média ponderada. 
                    Se 2 de 3 dizem "fraude", provavelmente é fraude!
                  </AlertBox>
                </div>
              </div>
              
              <h3 className="text-xl font-bold text-gray-900 mt-8">As 40+ Features Analisadas</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { icon: Clock, title: 'Horário', items: ['Hora do dia', 'Dia da semana', 'Feriado?', 'Madrugada?'] },
                  { icon: DollarSign, title: 'Valor', items: ['Valor absoluto', 'vs. Média cliente', 'vs. Desvio padrão', 'Arredondado?'] },
                  { icon: Target, title: 'Destino', items: ['CPF/CNPJ novo?', 'Já recebeu antes?', 'Está na HOT List?', 'Banco destino'] },
                  { icon: Globe, title: 'Localização', items: ['Cidade origem', 'Estado destino', 'Distância km', 'Velocidade impossível?'] },
                  { icon: Cpu, title: 'Device', items: ['Celular novo?', 'Modelo device', 'Jailbreak?', 'Emulador?'] },
                  { icon: Server, title: 'Rede', items: ['IP conhecido?', 'VPN/Proxy?', 'Tor Network?', 'Geolocalização IP'] },
                  { icon: Users, title: 'Cliente', items: ['Tempo de conta', 'Transações 30d', 'Reclamações?', 'Score interno'] },
                  { icon: Activity, title: 'Padrão', items: ['Horário usual?', 'Valor usual?', 'Canal usual?', 'Comportamento'] }
                ].map((cat, i) => (
                  <div key={i} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <cat.icon className="h-5 w-5 text-blue-600" />
                      <span className="font-bold text-gray-900">{cat.title}</span>
                    </div>
                    <ul className="text-xs text-gray-600 space-y-1">
                      {cat.items.map((item, j) => (
                        <li key={j}>• {item}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
              
              <h3 className="text-xl font-bold text-gray-900 mt-8">Latência e Performance</h3>
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-3xl font-bold text-green-400">37ms</div>
                    <div className="text-sm text-gray-400">Latência Média</div>
                    <div className="text-xs text-green-500">✓ Abaixo do SLA 50ms</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-blue-400">300M+</div>
                    <div className="text-sm text-gray-400">Requests/Dia</div>
                    <div className="text-xs text-blue-500">Capacidade máxima</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-purple-400">99.9%</div>
                    <div className="text-sm text-gray-400">Disponibilidade</div>
                    <div className="text-xs text-purple-500">Uptime garantido</div>
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-yellow-400">4.467</div>
                    <div className="text-sm text-gray-400">Transações Hoje</div>
                    <div className="text-xs text-yellow-500">Dados reais do banco</div>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Primeiro Dia de Trabalho */}
          <ManualSection id="primeiro-dia" title="🎓 Seu Primeiro Dia - Guia de Iniciante" icon={Star} priority="high">
            <AlertBox type="info" title="Bem-vindo à Equipe!">
              Se você está começando hoje, este é o seu guia de sobrevivência. 
              Siga estes passos e você estará operando em 1 hora.
            </AlertBox>
            
            <StepByStep 
              title="Primeiros 60 Minutos"
              steps={[
                { action: 'Faça Login no Sistema', details: 'Use seu CPF como usuário e a senha temporária recebida por e-mail. Troque a senha no primeiro acesso.', tip: 'Senha deve ter 12+ caracteres, maiúscula, número e símbolo.' },
                { action: 'Conheça o Dashboard', details: 'Fique 10 minutos observando os KPIs. Veja o volume de transações, quantas fraudes detectadas, latência.', tip: 'O dashboard atualiza a cada 30 segundos automaticamente.' },
                { action: 'Abra uma Transação Exemplo', details: 'Vá em Transações, clique em qualquer uma. Observe todos os campos: valor, canal, score, motivos.', tip: 'Use o botão "Ver Detalhes" para ver a explicação da IA.' },
                { action: 'Encontre a Fila de Revisão', details: 'Vá em "Revisão Manual". Aqui ficam as transações esperando sua análise humana.', tip: 'Transações na fila têm SLA de 15 minutos para análise.' },
                { action: 'Faça Sua Primeira Análise', details: 'Escolha uma transação simples (score 40-50). Analise o histórico do cliente. Decida: legítima ou fraude?', tip: 'Na dúvida, peça ajuda ao analista sênior. Nunca chute!' },
                { action: 'Registre Seu Feedback', details: 'Depois de decidir, clique em "Legítima" ou "Fraude" e escreva o motivo. Isso treina o modelo.', tip: 'Feedbacks detalhados melhoram a IA em 30 dias.' }
              ]}
            />
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <div className="bg-green-50 rounded-xl p-6 border border-green-200">
                <h4 className="font-bold text-green-800 flex items-center gap-2">
                  <ThumbsUp className="h-5 w-5" /> O Que FAZER
                </h4>
                <ul className="mt-4 space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Pergunte sempre que tiver dúvida</li>
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Documente suas decisões</li>
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Verifique o histórico do cliente (90 dias)</li>
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Use a explicabilidade da IA</li>
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Confirme valores altos com 2 fontes</li>
                  <li className="flex items-start gap-2"><CheckCircle className="h-4 w-4 text-green-500 mt-0.5" /> Reporte padrões novos que você notar</li>
                </ul>
              </div>
              
              <div className="bg-red-50 rounded-xl p-6 border border-red-200">
                <h4 className="font-bold text-red-800 flex items-center gap-2">
                  <ThumbsDown className="h-5 w-5" /> O Que NÃO FAZER
                </h4>
                <ul className="mt-4 space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Aprovar sem analisar ("só pra sair da fila")</li>
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Bloquear baseado só no valor alto</li>
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Ignorar alertas críticos (vermelhos)</li>
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Compartilhar dados de clientes</li>
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Alterar HOT List sem aprovação</li>
                  <li className="flex items-start gap-2"><XCircle className="h-4 w-4 text-red-500 mt-0.5" /> Deixar transação na fila além do SLA</li>
                </ul>
              </div>
            </div>
            
            <div className="mt-6">
              <h4 className="font-bold text-gray-900 mb-4">Atalhos de Teclado Essenciais</h4>
              <div className="bg-gray-50 rounded-lg p-4">
                <KeyboardShortcut keys={['Ctrl', 'K']} description="Busca rápida de transação" />
                <KeyboardShortcut keys={['Ctrl', 'Enter']} description="Aprovar transação atual" />
                <KeyboardShortcut keys={['Ctrl', 'Shift', 'Enter']} description="Bloquear transação atual" />
                <KeyboardShortcut keys={['Ctrl', 'H']} description="Ver histórico do cliente" />
                <KeyboardShortcut keys={['Ctrl', 'E']} description="Ver explicabilidade da IA" />
                <KeyboardShortcut keys={['F5']} description="Atualizar dados" />
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {activeTab === 'dia-a-dia' && (
        <>
          {/* Introdução ao Dia a Dia */}
          <ManualSection id="intro-dia" title="👨‍💼 Um Dia na Vida de Carlos Roberto - Analista Sênior" icon={Users} defaultOpen={true}>
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl p-6">
                <div className="flex items-center gap-4">
                  <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center text-3xl font-bold">
                    CR
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold">Carlos Roberto Silva</h3>
                    <p className="text-green-100">Analista de Fraudes Sênior | Turno Diurno (06:00 - 14:00)</p>
                    <p className="text-green-200 text-sm mt-1">5 anos de experiência | Especialista em PIX e Cartões</p>
                  </div>
                </div>
              </div>
              
              <AlertBox type="info" title="Sobre Este Guia">
                Acompanhe um dia completo de trabalho do Carlos Roberto. Você verá TODAS as situações 
                onde um Analista Sênior precisa atuar, desde o momento que chega até o fim do turno.
              </AlertBox>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-blue-600">~150</div>
                  <div className="text-sm text-gray-600">Transações analisadas/dia</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-green-600">~25</div>
                  <div className="text-sm text-gray-600">Alertas críticos/dia</div>
                </div>
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-orange-600">~5</div>
                  <div className="text-sm text-gray-600">Fraudes confirmadas/dia</div>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-purple-600">~10</div>
                  <div className="text-sm text-gray-600">Ligações para clientes/dia</div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* 06:00 - Início do Turno */}
          <ManualSection id="turno-0600" title="⏰ 06:00 - Início do Turno" icon={Clock}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-yellow-500 text-black px-3 py-1 rounded-full text-sm font-bold">06:00</div>
                  <span className="text-lg">Carlos chega e assume o turno</span>
                </div>
                <p className="text-gray-300">
                  O turno noturno (Rodrigo) está finalizando. Carlos precisa fazer a passagem de turno 
                  e entender o que aconteceu durante a madrugada.
                </p>
              </div>
              
              <StepByStep 
                title="Rotina de Início de Turno"
                steps={[
                  { action: 'Fazer Login no Sistema', details: 'Abrir o Sankofa, inserir CPF e senha. Sistema registra horário de entrada.', tip: 'Sempre use autenticação de 2 fatores (SMS ou app).' },
                  { action: 'Ler o Log de Passagem de Turno', details: 'Rodrigo deixou anotações no sistema: "2 fraudes confirmadas às 03:00, conta laranja identificada."', tip: 'Tela: Auditoria > Log de Turno' },
                  { action: 'Verificar Dashboard Imediatamente', details: 'Olhar KPIs do turno noturno. Algum spike? Algum modelo offline? Alertas pendentes?', tip: 'Foco nos números vermelhos primeiro.' },
                  { action: 'Checar Fila de Revisão Manual', details: 'Quantas transações estão esperando análise? SLA está sendo cumprido?', tip: 'Idealmente a fila deve ter < 20 itens no início do turno.' },
                  { action: 'Confirmar Status dos Modelos de IA', details: 'Os 3 modelos (RF, GB, CB) devem estar "Online" e "Healthy".', tip: 'Se algum estiver offline, verificar com TI imediatamente.' }
                ]}
              />
              
              <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                <h4 className="font-bold text-yellow-800 mb-3">📝 Exemplo de Log de Passagem de Turno</h4>
                <div className="bg-white rounded-lg p-4 font-mono text-sm">
                  <p><span className="text-gray-500">30/11/2025 05:55</span> - Rodrigo Mendes</p>
                  <p className="mt-2">Resumo do turno noturno (22:00 - 06:00):</p>
                  <ul className="mt-2 space-y-1 text-gray-700">
                    <li>• Total transações: 1.245</li>
                    <li>• Bloqueios automáticos: 89</li>
                    <li>• Análises manuais: 23</li>
                    <li>• <span className="text-red-600 font-bold">Fraudes confirmadas: 2</span> (PIX R$ 3.200 e R$ 8.900)</li>
                    <li>• <span className="text-orange-600">Conta laranja identificada:</span> CPF ***.***. 555-66</li>
                    <li>• Modelo CatBoost reiniciado às 04:30 (instabilidade)</li>
                    <li>• Pendências para turno diurno: 5 transações na fila</li>
                  </ul>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* 06:30 - Primeira Análise */}
          <ManualSection id="turno-0630" title="⏰ 06:30 - Limpando a Fila de Revisão" icon={Eye}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-blue-500 px-3 py-1 rounded-full text-sm font-bold">06:30</div>
                  <span className="text-lg">Carlos começa a analisar transações pendentes</span>
                </div>
                <p className="text-gray-300">
                  Existem 5 transações do turno noturno esperando análise. Carlos precisa resolver 
                  antes que o volume do dia aumente.
                </p>
              </div>
              
              <h4 className="font-bold text-gray-900">Transações na Fila</h4>
              <div className="space-y-4">
                <div className="bg-orange-50 border-2 border-orange-300 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-gray-900">Transação #1 - PIX Madrugada</span>
                    <span className="bg-orange-500 text-white px-2 py-1 rounded text-sm">Score: 58</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div><span className="text-gray-500">Valor:</span> R$ 2.340,00</div>
                    <div><span className="text-gray-500">Horário:</span> 03:45</div>
                    <div><span className="text-gray-500">Canal:</span> PIX</div>
                    <div><span className="text-gray-500">Destino:</span> Conhecido (irmão)</div>
                  </div>
                  <div className="mt-3 p-3 bg-white rounded-lg">
                    <p className="text-sm text-gray-600"><strong>Motivo do score:</strong> Horário atípico (+30), mas destinatário é familiar conhecido (-15), valor dentro do padrão (-5).</p>
                    <p className="text-sm text-green-700 font-semibold mt-2">✓ Decisão de Carlos: LEGÍTIMA - Cliente costuma trabalhar à noite</p>
                  </div>
                </div>
                
                <div className="bg-red-50 border-2 border-red-300 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-gray-900">Transação #2 - TED Alto Valor</span>
                    <span className="bg-red-500 text-white px-2 py-1 rounded text-sm">Score: 74</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div><span className="text-gray-500">Valor:</span> R$ 45.000,00</div>
                    <div><span className="text-gray-500">Horário:</span> 02:15</div>
                    <div><span className="text-gray-500">Canal:</span> TED</div>
                    <div><span className="text-gray-500">Destino:</span> CPF novo</div>
                  </div>
                  <div className="mt-3 p-3 bg-white rounded-lg">
                    <p className="text-sm text-gray-600"><strong>Motivo do score:</strong> Valor muito alto (+25), madrugada (+20), destinatário novo (+20), device habitual (-10).</p>
                    <p className="text-sm text-orange-700 font-semibold mt-2">⏳ Decisão de Carlos: PRECISA CONTATO - Ligar para cliente às 08:00</p>
                  </div>
                </div>
              </div>
              
              <AlertBox type="tip" title="Dica do Carlos">
                "Eu sempre resolvo os casos mais simples primeiro (score 40-55). Isso limpa a fila 
                rapidamente e me deixa tempo para investigar os casos complexos com calma."
              </AlertBox>
            </div>
          </ManualSection>

          {/* 08:00 - Pico de Transações */}
          <ManualSection id="turno-0800" title="⏰ 08:00 - Pico Matinal de Transações" icon={TrendingUp}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-red-500 px-3 py-1 rounded-full text-sm font-bold">08:00</div>
                  <span className="text-lg">Volume de transações aumenta 300%!</span>
                </div>
                <p className="text-gray-300">
                  O Brasil acorda e começa a fazer PIX. O volume salta de 50 tx/min para 200 tx/min. 
                  Mais transações = mais alertas.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-4">O Que Carlos Monitora Agora</h4>
                  <Checklist 
                    items={[
                      { text: 'Dashboard: Volume subindo normalmente?', done: true },
                      { text: 'Latência: Ainda abaixo de 50ms?', done: true },
                      { text: 'Fila: Crescendo muito rápido?', done: false },
                      { text: 'Alertas críticos: Algum novo vermelho?', done: false },
                      { text: 'Modelos: Todos online e respondendo?', done: true }
                    ]}
                  />
                </div>
                
                <div className="bg-blue-50 rounded-xl p-6">
                  <h4 className="font-bold text-gray-900 mb-3">📞 Ligação para Cliente (TED de R$ 45k)</h4>
                  <div className="bg-white rounded-lg p-4 text-sm">
                    <p className="text-gray-500 mb-2">08:05 - Carlos liga para o cliente</p>
                    <div className="space-y-2 italic text-gray-700">
                      <p><strong>Carlos:</strong> "Bom dia, Sr. Marcelo. Aqui é Carlos do Banco Nexus, setor de segurança. Tudo bem?"</p>
                      <p><strong>Cliente:</strong> "Bom dia! Sim, tudo bem."</p>
                      <p><strong>Carlos:</strong> "Detectamos uma TED de R$ 45.000 às 2h da manhã. O senhor confirma essa operação?"</p>
                      <p><strong>Cliente:</strong> "Sim! Comprei um carro usado ontem. O vendedor pediu transferência urgente."</p>
                      <p><strong>Carlos:</strong> "Perfeito! Pode me confirmar o nome do destinatário?"</p>
                      <p><strong>Cliente:</strong> "José Carlos... da Revenda Carros Sul."</p>
                      <p><strong>Carlos:</strong> "Confirmado! Vou liberar a transação agora. Desculpe qualquer inconveniente."</p>
                    </div>
                    <div className="mt-4 p-2 bg-green-100 rounded text-green-800 text-sm">
                      ✓ Transação liberada | Feedback registrado: "Compra de veículo confirmada por telefone"
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* 09:30 - Alerta Crítico */}
          <ManualSection id="turno-0930" title="🚨 09:30 - ALERTA CRÍTICO: Spike de Fraudes!" icon={AlertTriangle} priority="critical">
            <div className="space-y-6">
              <div className="bg-red-600 text-white rounded-xl p-6 animate-pulse">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-white text-red-600 px-3 py-1 rounded-full text-sm font-bold">09:30</div>
                  <span className="text-lg font-bold">⚠️ ALERTA: +450% de bloqueios nos últimos 15 minutos!</span>
                </div>
                <p>
                  O sistema detectou um aumento anormal de transações suspeitas. Pode ser um ataque coordenado!
                </p>
              </div>
              
              <StepByStep 
                title="Protocolo de Resposta a Incidente"
                steps={[
                  { action: 'PAUSAR outras atividades', details: 'Carlos para tudo que está fazendo. Incidente crítico tem prioridade máxima.', tip: 'Avise colegas que você está em incidente.' },
                  { action: 'Abrir Dashboard de Métricas', details: 'Verificar: de onde vêm as transações? Mesmo IP? Mesmo device? Mesmo padrão?', tip: 'Tela: Métricas > Tempo Real' },
                  { action: 'Identificar Padrão', details: 'Carlos vê: 45 transações com mesmo device_id, todos para contas diferentes, valor ~R$ 1.000.', tip: 'Isso é um ataque automatizado (bot)!' },
                  { action: 'Criar Hard Rule de Emergência', details: 'Bloquear temporariamente o device_id suspeito.', tip: 'Precisa de aprovação da Ana Paula (líder).' },
                  { action: 'Notificar Gerência', details: 'Ligar para Patrícia Lima (Gerente) e reportar o incidente.', tip: 'Use o canal de emergência no Teams.' },
                  { action: 'Documentar Tudo', details: 'Registrar no sistema: horário, IPs, devices, ações tomadas.', tip: 'Tela: Auditoria > Novo Incidente' }
                ]}
              />
              
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <h4 className="font-bold mb-4">📊 Análise do Carlos no Dashboard</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div className="bg-red-900 rounded-lg p-3">
                    <div className="text-2xl font-bold text-red-400">45</div>
                    <div className="text-xs text-gray-400">Transações suspeitas</div>
                  </div>
                  <div className="bg-red-900 rounded-lg p-3">
                    <div className="text-2xl font-bold text-red-400">1</div>
                    <div className="text-xs text-gray-400">Device ID comum</div>
                  </div>
                  <div className="bg-red-900 rounded-lg p-3">
                    <div className="text-2xl font-bold text-red-400">38</div>
                    <div className="text-xs text-gray-400">Contas destino únicas</div>
                  </div>
                  <div className="bg-red-900 rounded-lg p-3">
                    <div className="text-2xl font-bold text-red-400">R$ 42k</div>
                    <div className="text-xs text-gray-400">Valor total bloqueado</div>
                  </div>
                </div>
              </div>
              
              <AlertBox type="success" title="Resultado do Incidente">
                Carlos identificou e bloqueou um ataque de bot em 8 minutos. 
                R$ 42.000 foram salvos. O device foi adicionado à HOT List permanentemente.
              </AlertBox>
            </div>
          </ManualSection>

          {/* 10:30 - Investigação Profunda */}
          <ManualSection id="turno-1030" title="🔍 10:30 - Investigação Profunda de Fraude" icon={Search}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-purple-500 px-3 py-1 rounded-full text-sm font-bold">10:30</div>
                  <span className="text-lg">Carlos investiga uma rede de contas laranjas</span>
                </div>
                <p className="text-gray-300">
                  Durante a análise do ataque, Carlos notou que várias contas destino tinham padrão suspeito. 
                  Hora de investigar a fundo.
                </p>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-4">Telas Utilizadas na Investigação</h4>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3 p-3 bg-purple-50 rounded-lg">
                      <Target className="h-6 w-6 text-purple-600" />
                      <div>
                        <p className="font-semibold">1. Investigação</p>
                        <p className="text-sm text-gray-600">Ver timeline completa da transação, todas as features, explicabilidade da IA.</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                      <Users className="h-6 w-6 text-blue-600" />
                      <div>
                        <p className="font-semibold">2. Transações</p>
                        <p className="text-sm text-gray-600">Buscar todas as transações relacionadas ao CPF/conta suspeita.</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-orange-50 rounded-lg">
                      <Activity className="h-6 w-6 text-orange-600" />
                      <div>
                        <p className="font-semibold">3. Métricas</p>
                        <p className="text-sm text-gray-600">Ver padrões de rede: quais contas se conectam?</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg">
                      <XCircle className="h-6 w-6 text-red-600" />
                      <div>
                        <p className="font-semibold">4. HOT List</p>
                        <p className="text-sm text-gray-600">Adicionar contas confirmadas como fraudulentas.</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-6">
                  <h4 className="font-bold text-gray-900 mb-4">🕵️ Descobertas da Investigação</h4>
                  <div className="space-y-3">
                    <div className="bg-white rounded-lg p-3 border-l-4 border-red-500">
                      <p className="font-semibold text-gray-900">5 contas laranja identificadas</p>
                      <p className="text-sm text-gray-600">Todas abertas nos últimos 30 dias, sem histórico.</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 border-l-4 border-orange-500">
                      <p className="font-semibold text-gray-900">Padrão: recebem e sacam no mesmo dia</p>
                      <p className="text-sm text-gray-600">PIX entra → saque em 2h → conta volta a ficar zerada.</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 border-l-4 border-purple-500">
                      <p className="font-semibold text-gray-900">Mesmo endereço de IP em 3 contas</p>
                      <p className="text-sm text-gray-600">Provável: mesma pessoa controlando múltiplas contas.</p>
                    </div>
                    <div className="bg-white rounded-lg p-3 border-l-4 border-blue-500">
                      <p className="font-semibold text-gray-900">Conexão com fraude de ontem</p>
                      <p className="text-sm text-gray-600">Uma das contas recebeu PIX de vítima identificada pelo Rodrigo.</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <AlertBox type="danger" title="Ação Tomada">
                Carlos adiciona as 5 contas à HOT List e gera relatório STR para o BACEN. 
                Marina (Compliance) é notificada para validar o relatório antes do envio.
              </AlertBox>
            </div>
          </ManualSection>

          {/* 12:00 - Almoço e Métricas */}
          <ManualSection id="turno-1200" title="🍽️ 12:00 - Intervalo e Revisão de Métricas" icon={BarChart3}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-green-500 px-3 py-1 rounded-full text-sm font-bold">12:00</div>
                  <span className="text-lg">Carlos faz pausa para almoço</span>
                </div>
                <p className="text-gray-300">
                  Antes de sair, Carlos verifica se há pendências críticas e delega para os colegas.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-4">Checklist Pré-Almoço</h4>
                  <Checklist 
                    items={[
                      { text: 'Fila de revisão: menos de 10 itens?', done: true },
                      { text: 'Alertas críticos: todos resolvidos?', done: true },
                      { text: 'Colega assumiu monitoramento?', done: true },
                      { text: 'Log de atividades atualizado?', done: true },
                      { text: 'Celular de emergência ligado?', done: true }
                    ]}
                  />
                </div>
                
                <div className="bg-blue-50 rounded-xl p-6">
                  <h4 className="font-bold text-gray-900 mb-4">📊 Métricas do Turno Até Agora</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-blue-600">2.847</div>
                      <div className="text-xs text-gray-600">Transações processadas</div>
                    </div>
                    <div className="bg-white rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-green-600">67</div>
                      <div className="text-xs text-gray-600">Análises manuais</div>
                    </div>
                    <div className="bg-white rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-red-600">3</div>
                      <div className="text-xs text-gray-600">Fraudes confirmadas</div>
                    </div>
                    <div className="bg-white rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold text-purple-600">R$ 89k</div>
                      <div className="text-xs text-gray-600">Valor protegido</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* 13:00 - Feedback e Treinamento do Modelo */}
          <ManualSection id="turno-1300" title="🧠 13:00 - Feedback para Treinar a IA" icon={Brain}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-purple-500 px-3 py-1 rounded-full text-sm font-bold">13:00</div>
                  <span className="text-lg">Carlos registra feedbacks das análises do dia</span>
                </div>
                <p className="text-gray-300">
                  Todo dia, Carlos dedica 30 minutos para registrar feedbacks detalhados. 
                  Isso treina os modelos de IA para serem mais precisos.
                </p>
              </div>
              
              <AlertBox type="info" title="Por Que Feedback é Importante?">
                Cada feedback que você registra ensina a IA a tomar melhores decisões. 
                Em 30 dias, o modelo "aprende" seus padrões e fica mais preciso.
                Quanto mais detalhado o feedback, melhor o aprendizado!
              </AlertBox>
              
              <h4 className="font-bold text-gray-900">Exemplos de Feedback do Carlos</h4>
              <div className="space-y-4">
                <div className="bg-green-50 rounded-xl p-4 border border-green-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-green-800">Transação #PIX-2024-11-30-081523</span>
                    <span className="bg-green-500 text-white px-2 py-1 rounded text-sm">LEGÍTIMA</span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2"><strong>Decisão original da IA:</strong> Score 58 - Bloqueio</p>
                  <p className="text-sm text-gray-700 mb-2"><strong>Decisão do Carlos:</strong> Liberar</p>
                  <div className="bg-white rounded-lg p-3 mt-2">
                    <p className="text-sm text-gray-600"><strong>Feedback detalhado:</strong></p>
                    <p className="text-sm text-gray-700 mt-1">
                      "Cliente trabalha como motorista de aplicativo. PIX de madrugada é padrão normal 
                      para ele (recebe corridas). Destinatário é a mãe dele, transferência recorrente. 
                      Modelo deve aprender: profissionais noturnos têm padrão diferente."
                    </p>
                  </div>
                </div>
                
                <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-red-800">Transação #PIX-2024-11-30-093012</span>
                    <span className="bg-red-500 text-white px-2 py-1 rounded text-sm">FRAUDE</span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2"><strong>Decisão original da IA:</strong> Score 72 - Bloqueio</p>
                  <p className="text-sm text-gray-700 mb-2"><strong>Decisão do Carlos:</strong> Confirmar fraude</p>
                  <div className="bg-white rounded-lg p-3 mt-2">
                    <p className="text-sm text-gray-600"><strong>Feedback detalhado:</strong></p>
                    <p className="text-sm text-gray-700 mt-1">
                      "Golpe do WhatsApp clássico. Fraudador se passou por filho pedindo R$ 4.850 urgente. 
                      Padrão: valor quebrado (não redondo), urgência na mensagem, conta destino nova. 
                      Cliente confirmou por telefone que não fez a transação. Conta destino é laranja."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* 13:45 - Passagem de Turno */}
          <ManualSection id="turno-1345" title="🔄 13:45 - Preparação para Passagem de Turno" icon={RefreshCw}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-orange-500 px-3 py-1 rounded-full text-sm font-bold">13:45</div>
                  <span className="text-lg">Carlos prepara o log de passagem para Ana Paula</span>
                </div>
                <p className="text-gray-300">
                  O turno de Carlos termina às 14:00. Ele precisa documentar tudo que aconteceu 
                  para que Ana Paula (turno vespertino) continue o trabalho.
                </p>
              </div>
              
              <StepByStep 
                title="Rotina de Fim de Turno"
                steps={[
                  { action: 'Finalizar Análises em Andamento', details: 'Não deixar transações "em análise" - decidir ou delegar.', tip: 'Transações abandonadas afetam SLA.' },
                  { action: 'Escrever Log de Passagem', details: 'Documentar: incidentes, fraudes confirmadas, pendências, observações.', tip: 'Seja detalhado - você não estará lá para explicar.' },
                  { action: 'Atualizar Relatório Diário', details: 'Preencher métricas do turno: transações, análises, fraudes, tempo médio.', tip: 'Tela: Relatórios > Diário > Meu Turno' },
                  { action: 'Verificar Alertas Pendentes', details: 'Há algo crítico que precisa de atenção imediata da Ana Paula?', tip: 'Marcar como "Prioritário" se sim.' },
                  { action: 'Fazer Handoff Verbal', details: 'Conversar 5 minutos com Ana Paula sobre pontos importantes.', tip: 'Destaque o incidente do bot às 09:30!' }
                ]}
              />
              
              <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200">
                <h4 className="font-bold text-yellow-800 mb-3">📝 Log de Passagem do Carlos</h4>
                <div className="bg-white rounded-lg p-4 font-mono text-sm">
                  <p><span className="text-gray-500">30/11/2025 13:50</span> - Carlos Roberto Silva</p>
                  <p className="mt-2 font-bold">Resumo do turno diurno (06:00 - 14:00):</p>
                  <ul className="mt-2 space-y-1 text-gray-700">
                    <li>• Total transações: 4.523</li>
                    <li>• Bloqueios automáticos: 312</li>
                    <li>• Análises manuais: 89</li>
                    <li>• <span className="text-red-600 font-bold">Fraudes confirmadas: 4</span></li>
                    <li className="ml-4">- PIX R$ 4.850 (golpe WhatsApp)</li>
                    <li className="ml-4">- Ataque de bot: 45 transações bloqueadas (R$ 42k)</li>
                    <li className="ml-4">- Rede de 5 contas laranja identificada</li>
                    <li>• <span className="text-green-600">Valor total protegido: R$ 138.500</span></li>
                    <li>• STR enviado ao BACEN (aguardando validação Marina)</li>
                    <li>• Hard Rule criada: device_id XPTO bloqueado</li>
                    <li>• <span className="text-orange-600 font-bold">ATENÇÃO:</span> Monitorar se bot tenta novo device</li>
                  </ul>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Resumo do Dia */}
          <ManualSection id="resumo-dia" title="📊 Resumo: Todos os Momentos de Atuação" icon={List} priority="high">
            <div className="space-y-6">
              <AlertBox type="info" title="Quando o Analista Sênior PRECISA Atuar">
                Este é o resumo de TODAS as situações onde Carlos precisou intervir durante o dia.
              </AlertBox>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="p-3 text-left">Horário</th>
                      <th className="p-3 text-left">Situação</th>
                      <th className="p-3 text-left">Tela Usada</th>
                      <th className="p-3 text-left">Ação Tomada</th>
                      <th className="p-3 text-left">Prioridade</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    <tr className="bg-white">
                      <td className="p-3 font-mono">06:00</td>
                      <td className="p-3">Início de turno - passagem</td>
                      <td className="p-3">Dashboard, Auditoria</td>
                      <td className="p-3">Ler log do turno anterior</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Rotina</span></td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-mono">06:30</td>
                      <td className="p-3">Transações pendentes na fila</td>
                      <td className="p-3">Revisão Manual</td>
                      <td className="p-3">Analisar e decidir (5 itens)</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs">Média</span></td>
                    </tr>
                    <tr className="bg-white">
                      <td className="p-3 font-mono">08:00</td>
                      <td className="p-3">Pico de transações matinal</td>
                      <td className="p-3">Dashboard, Métricas</td>
                      <td className="p-3">Monitorar volume e latência</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Rotina</span></td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-mono">08:05</td>
                      <td className="p-3">Cliente questiona bloqueio</td>
                      <td className="p-3">Transações, Investigação</td>
                      <td className="p-3">Ligar para cliente, liberar TED</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs">Média</span></td>
                    </tr>
                    <tr className="bg-red-50">
                      <td className="p-3 font-mono">09:30</td>
                      <td className="p-3 font-bold text-red-700">INCIDENTE: Ataque de bot</td>
                      <td className="p-3">Dashboard, Hard Rules, HOT List</td>
                      <td className="p-3">Bloquear device, escalar gerência</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">CRÍTICA</span></td>
                    </tr>
                    <tr className="bg-white">
                      <td className="p-3 font-mono">10:30</td>
                      <td className="p-3">Investigação de rede de fraude</td>
                      <td className="p-3">Investigação, HOT List, Relatórios</td>
                      <td className="p-3">Mapear contas laranjas, STR BACEN</td>
                      <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">Alta</span></td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-mono">11:00</td>
                      <td className="p-3">Análise de transação score 65</td>
                      <td className="p-3">Revisão Manual</td>
                      <td className="p-3">Verificar histórico, aprovar</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs">Média</span></td>
                    </tr>
                    <tr className="bg-white">
                      <td className="p-3 font-mono">12:00</td>
                      <td className="p-3">Intervalo - delegação</td>
                      <td className="p-3">Dashboard</td>
                      <td className="p-3">Verificar fila, delegar para colega</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Rotina</span></td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-mono">13:00</td>
                      <td className="p-3">Registro de feedbacks</td>
                      <td className="p-3">Feedback Analista</td>
                      <td className="p-3">Treinar modelo com decisões</td>
                      <td className="p-3"><span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs">Importante</span></td>
                    </tr>
                    <tr className="bg-white">
                      <td className="p-3 font-mono">13:45</td>
                      <td className="p-3">Passagem de turno</td>
                      <td className="p-3">Auditoria, Relatórios</td>
                      <td className="p-3">Documentar dia, handoff verbal</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Rotina</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="bg-blue-50 rounded-xl p-6 text-center">
                  <div className="text-4xl font-bold text-blue-600">16</div>
                  <div className="text-gray-600 mt-2">Telas do sistema utilizadas</div>
                </div>
                <div className="bg-green-50 rounded-xl p-6 text-center">
                  <div className="text-4xl font-bold text-green-600">89</div>
                  <div className="text-gray-600 mt-2">Decisões tomadas</div>
                </div>
                <div className="bg-purple-50 rounded-xl p-6 text-center">
                  <div className="text-4xl font-bold text-purple-600">R$ 138k</div>
                  <div className="text-gray-600 mt-2">Valor protegido</div>
                </div>
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {activeTab === 'telas' && (
        <>
          {/* Dashboard Detalhado */}
          <ManualSection id="dashboard" title="📊 TELA: Dashboard Executivo" icon={BarChart3} defaultOpen={true}>
            <div className="space-y-6">
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <h3 className="text-xl font-bold text-gray-900 mb-2">Objetivo desta Tela</h3>
                <p className="text-gray-700">
                  O Dashboard é sua visão geral do sistema. Aqui você monitora em tempo real 
                  a saúde do sistema, volume de transações, fraudes detectadas e performance dos modelos.
                </p>
              </div>
              
              <h4 className="font-bold text-gray-900">KPIs Principais</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <KPICard title="Transações Hoje" value="4.467" change="+12%" changeType="up" icon={Zap} color="blue" />
                <KPICard title="Fraudes Detectadas" value="3.115" change="+8%" changeType="up" icon={Shield} color="red" />
                <KPICard title="Taxa de Aprovação" value="30.3%" change="-2%" changeType="down" icon={Percent} color="green" />
                <KPICard title="Latência Média" value="37ms" change="-5ms" changeType="up" icon={Clock} color="purple" />
              </div>
              
              <StepByStep 
                title="Como Usar o Dashboard"
                steps={[
                  { action: 'Verifique o Status do Sistema', details: 'No canto superior direito, veja "Sistema Online" em verde. Se estiver vermelho, alerte o suporte.', tip: 'Status amarelo = degradação parcial.' },
                  { action: 'Analise os KPIs', details: 'Compare os valores de hoje com ontem. Variações de ±10% são normais. Acima disso, investigue.', tip: 'Clique no KPI para ver gráfico detalhado.' },
                  { action: 'Observe o Gráfico de Transações/Hora', details: 'Picos são normais às 10h, 14h e 18h. Picos fora desse horário podem indicar ataque.', tip: 'Use zoom para analisar períodos específicos.' },
                  { action: 'Verifique Alertas Recentes', details: 'A lista mostra os últimos 10 alertas. Vermelhos são críticos e precisam de ação imediata.', tip: 'Clique no alerta para ir direto à transação.' },
                  { action: 'Confira Status dos Modelos', details: 'Os 3 modelos devem estar "Online". Se algum estiver offline, o backup assume.', tip: 'Modelo offline por mais de 5 min = alerte TI.' }
                ]}
              />
              
              <FAQ questions={[
                { question: 'Por que a taxa de fraude está tão alta (69%)?', answer: 'Isso é normal para um sistema de detecção! A taxa mostra quantas transações SUSPEITAS foram bloqueadas. Muitas são falsos positivos que serão liberados após análise.' },
                { question: 'O que significa latência?', answer: 'É o tempo que o sistema leva para analisar uma transação. 37ms significa que em 0.037 segundos já temos uma decisão! O SLA do BACEN para PIX é 10 segundos, estamos muito abaixo.' },
                { question: 'Por que os números não atualizam?', answer: 'O cache do sistema atualiza a cada 30 segundos para economizar recursos. Clique em "Atualizar" para forçar uma atualização imediata.' }
              ]} />
            </div>
          </ManualSection>

          {/* Transações */}
          <ManualSection id="transacoes" title="💳 TELA: Transações" icon={FileText}>
            <div className="space-y-6">
              <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                <h3 className="text-xl font-bold text-gray-900 mb-2">Objetivo desta Tela</h3>
                <p className="text-gray-700">
                  Aqui você busca, filtra e analisa qualquer transação do sistema. 
                  Use quando precisar investigar um caso específico ou quando um cliente ligar reclamando.
                </p>
              </div>
              
              <h4 className="font-bold text-gray-900">Filtros Disponíveis</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <Search className="h-6 w-6 text-blue-600 mb-2" />
                  <p className="font-semibold">Busca por CPF</p>
                  <p className="text-sm text-gray-600">Digite os 3 últimos dígitos ou CPF mascarado</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <Calendar className="h-6 w-6 text-green-600 mb-2" />
                  <p className="font-semibold">Período</p>
                  <p className="text-sm text-gray-600">Últimas 24h, 7d, 30d ou personalizado</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <Zap className="h-6 w-6 text-yellow-600 mb-2" />
                  <p className="font-semibold">Canal</p>
                  <p className="text-sm text-gray-600">PIX, TED, Cartão, Boleto</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <Filter className="h-6 w-6 text-purple-600 mb-2" />
                  <p className="font-semibold">Status</p>
                  <p className="text-sm text-gray-600">Aprovadas, Bloqueadas, Em Análise</p>
                </div>
              </div>
              
              <StepByStep 
                title="Cenário: Cliente Liga Reclamando de Bloqueio"
                steps={[
                  { action: 'Anote os Dados do Cliente', details: 'CPF (últimos 3 dígitos), data/hora aproximada, valor, canal (PIX/TED).', tip: 'Peça o ID da transação se o cliente tiver.' },
                  { action: 'Abra a Tela de Transações', details: 'Menu lateral → Transações', tip: 'Atalho: Ctrl + T' },
                  { action: 'Configure os Filtros', details: 'CPF: ***.***. XXX-YY | Período: Hoje | Canal: PIX', tip: 'Quanto mais filtros, mais rápida a busca.' },
                  { action: 'Clique em Buscar', details: 'A lista mostrará todas as transações correspondentes.', tip: 'Se não encontrar, amplie o período.' },
                  { action: 'Identifique a Transação', details: 'Compare valor e horário com o que o cliente informou.', tip: 'Transações bloqueadas têm ícone vermelho.' },
                  { action: 'Clique em "Ver Detalhes"', details: 'Abre a página de investigação completa.', tip: 'Você verá o score, motivos e histórico.' }
                ]}
              />
              
              <div className="bg-yellow-50 rounded-xl p-6 border border-yellow-200 mt-6">
                <h4 className="font-bold text-yellow-800 mb-4">Exemplo Prático: Transação Encontrada</h4>
                <TransactionCard 
                  id="PIX-2024-11-30-143201"
                  cpf="***.***. 789-01"
                  amount="R$ 4.850,00"
                  channel="PIX"
                  time="30/11/2025 14:32"
                  status="blocked"
                  score={87}
                  location="São Paulo, SP → Recife, PE"
                />
                <div className="mt-4 bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-gray-900">Motivos do Bloqueio (Explicabilidade IA):</h5>
                  <ul className="mt-2 space-y-1 text-sm text-gray-600">
                    <li>• <span className="text-red-600">+35 pontos</span>: Valor 3x acima da média do cliente</li>
                    <li>• <span className="text-red-600">+25 pontos</span>: Destinatário nunca visto antes</li>
                    <li>• <span className="text-red-600">+15 pontos</span>: Padrão similar a golpes conhecidos</li>
                    <li>• <span className="text-red-600">+12 pontos</span>: Distância origem/destino: 2.660km</li>
                    <li>• <span className="text-green-600">-5 pontos</span>: Horário comercial (14h)</li>
                  </ul>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Revisão Manual */}
          <ManualSection id="revisao" title="👁️ TELA: Revisão Manual (Human-in-the-Loop)" icon={Eye}>
            <div className="space-y-6">
              <AlertBox type="warning" title="IMPORTANTE">
                Esta é a tela mais crítica do sistema! Aqui você toma decisões que afetam 
                diretamente clientes e a segurança do banco. Cada decisão tem impacto real.
              </AlertBox>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-4">Por Que Existe Revisão Manual?</h4>
                  <p className="text-gray-600 mb-4">
                    A IA não é perfeita. Quando ela tem "dúvida" (score entre 40-60), 
                    a transação vem para você decidir. Sua experiência humana complementa 
                    a inteligência artificial.
                  </p>
                  
                  <div className="space-y-3">
                    <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                      <div className="bg-green-500 text-white p-2 rounded-full">
                        <CheckCircle className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-semibold">Legítima</p>
                        <p className="text-sm text-gray-600">Transação é real e segura. Liberar.</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg">
                      <div className="bg-red-500 text-white p-2 rounded-full">
                        <XCircle className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-semibold">Fraude</p>
                        <p className="text-sm text-gray-600">Transação é fraudulenta. Manter bloqueio.</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg">
                      <div className="bg-yellow-500 text-white p-2 rounded-full">
                        <Clock className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-semibold">Preciso de Mais Informação</p>
                        <p className="text-sm text-gray-600">Ligar para cliente ou aguardar documentos.</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-bold text-gray-900 mb-4">Checklist de Análise</h4>
                  <Checklist 
                    items={[
                      { text: 'Li o motivo do bloqueio (explicabilidade)', done: false },
                      { text: 'Verifiquei histórico de 90 dias', done: false },
                      { text: 'Comparei valor com média do cliente', done: false },
                      { text: 'Verifiquei se destinatário é conhecido', done: false },
                      { text: 'Analisei horário e localização', done: false },
                      { text: 'Consultei se CPF está em HOT List', done: false },
                      { text: 'Registrei minha decisão com justificativa', done: false }
                    ]}
                  />
                  
                  <AlertBox type="danger" title="Nunca Esqueça!" className="mt-4">
                    SEMPRE registre a justificativa da sua decisão. 
                    Isso é exigido pela LGPD e pode ser auditado pelo BACEN.
                  </AlertBox>
                </div>
              </div>
              
              <h4 className="font-bold text-gray-900 mt-6">SLAs de Tempo de Resposta</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-red-50 border-2 border-red-500 rounded-xl p-4 text-center">
                  <div className="text-3xl font-bold text-red-600">5 min</div>
                  <div className="text-sm text-red-700 font-medium">Crítico (Score 80+)</div>
                  <div className="text-xs text-red-600">Pode ser fraude em andamento!</div>
                </div>
                <div className="bg-orange-50 border-2 border-orange-500 rounded-xl p-4 text-center">
                  <div className="text-3xl font-bold text-orange-600">15 min</div>
                  <div className="text-sm text-orange-700 font-medium">Alto (Score 60-80)</div>
                  <div className="text-xs text-orange-600">Cliente pode estar esperando</div>
                </div>
                <div className="bg-yellow-50 border-2 border-yellow-500 rounded-xl p-4 text-center">
                  <div className="text-3xl font-bold text-yellow-600">30 min</div>
                  <div className="text-sm text-yellow-700 font-medium">Médio (Score 40-60)</div>
                  <div className="text-xs text-yellow-600">Análise padrão</div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Calibragem */}
          <ManualSection id="calibragem" title="⚙️ TELA: Calibragem" icon={Settings}>
            <div className="space-y-6">
              <AlertBox type="danger" title="ACESSO RESTRITO">
                Esta tela é apenas para LÍDERES e gestores. 
                Mudanças aqui afetam TODO o sistema imediatamente.
              </AlertBox>
              
              <div className="bg-gradient-to-r from-gray-100 to-gray-200 rounded-xl p-6">
                <h4 className="font-bold text-gray-900 mb-4">O Que é Threshold (Limiar)?</h4>
                <p className="text-gray-700 mb-4">
                  O threshold define a partir de qual score uma transação é bloqueada automaticamente.
                  Se o threshold é 45, qualquer transação com score 46 ou mais será bloqueada.
                </p>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                  <div>
                    <h5 className="font-semibold text-gray-900 mb-3">Termômetro de Sensibilidade</h5>
                    <div className="bg-gradient-to-b from-red-200 via-yellow-200 to-green-200 rounded-xl p-4">
                      <div className="space-y-3">
                        <div className="flex items-center gap-4">
                          <span className="w-12 text-center font-bold text-red-700">100</span>
                          <div className="flex-1 h-8 bg-red-500 rounded flex items-center px-3">
                            <span className="text-white text-sm">Ultra Rigoroso - Bloqueia quase tudo</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="w-12 text-center font-bold text-orange-700">70</span>
                          <div className="flex-1 h-8 bg-orange-500 rounded flex items-center px-3">
                            <span className="text-white text-sm">Muito Rigoroso</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="w-12 text-center font-bold text-yellow-700">45</span>
                          <div className="flex-1 h-10 bg-yellow-500 rounded flex items-center px-3 border-4 border-yellow-700">
                            <span className="text-gray-900 text-sm font-bold">⭐ ATUAL - Balanço ideal</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="w-12 text-center font-bold text-lime-700">30</span>
                          <div className="flex-1 h-8 bg-lime-500 rounded flex items-center px-3">
                            <span className="text-gray-900 text-sm">Permissivo</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="w-12 text-center font-bold text-green-700">0</span>
                          <div className="flex-1 h-8 bg-green-500 rounded flex items-center px-3">
                            <span className="text-white text-sm">Ultra Permissivo - Aprova quase tudo</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h5 className="font-semibold text-gray-900 mb-3">Quando Ajustar?</h5>
                    <div className="space-y-4">
                      <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                        <p className="font-bold text-red-800">😠 Muitas reclamações de clientes?</p>
                        <p className="text-sm text-red-700">"Meu PIX foi bloqueado injustamente!"</p>
                        <p className="text-sm font-semibold text-red-900 mt-2">→ DIMINUA o threshold (ex: 45 → 35)</p>
                        <p className="text-xs text-red-600">Menos falsos positivos, mais aprovações.</p>
                      </div>
                      
                      <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                        <p className="font-bold text-orange-800">😱 Fraudes passando?</p>
                        <p className="text-sm text-orange-700">"3 fraudes confirmadas ontem!"</p>
                        <p className="text-sm font-semibold text-orange-900 mt-2">→ AUMENTE o threshold (ex: 45 → 55)</p>
                        <p className="text-xs text-orange-600">Mais bloqueios, mais segurança.</p>
                      </div>
                      
                      <AlertBox type="warning" title="Regra de Ouro">
                        Nunca mude mais de 10 pontos por vez. 
                        Mude, espere 24h, analise, depois ajuste novamente.
                      </AlertBox>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {/* Hard Rules */}
          <ManualSection id="hard-rules" title="🔒 TELA: Hard Rules (Regras Duras)" icon={Lock}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <h4 className="text-xl font-bold mb-4">O Que São Hard Rules?</h4>
                <p className="text-gray-300 mb-4">
                  São regras que SEMPRE disparam, independente do score da IA. 
                  São usadas para casos extremos e conhecidos.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm">
                    <p className="text-green-400">// Regra 1: Valor Absurdo</p>
                    <p className="text-white">SE valor &gt; R$ 100.000</p>
                    <p className="text-white">E horário = 00:00-05:00</p>
                    <p className="text-red-400">ENTÃO = BLOQUEAR + ALERTA</p>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4 font-mono text-sm">
                    <p className="text-green-400">// Regra 2: Velocidade Impossível</p>
                    <p className="text-white">SE distância &gt; 500km</p>
                    <p className="text-white">E tempo &lt; 30 minutos</p>
                    <p className="text-red-400">ENTÃO = BLOQUEAR + INVESTIGAR</p>
                  </div>
                </div>
              </div>
              
              <StepByStep 
                title="Como Criar uma Nova Hard Rule"
                steps={[
                  { action: 'Documente o Padrão', details: 'Antes de criar, documente: qual padrão de fraude você detectou? Em quantos casos?', tip: 'Mínimo 5 casos confirmados para criar regra.' },
                  { action: 'Solicite Aprovação', details: 'Preencha formulário de "Nova Regra" e envie para Ana Paula ou gestor.', tip: 'Regras precisam de aprovação de compliance.' },
                  { action: 'Defina as Condições', details: 'Use linguagem clara: SE campo = valor E campo2 = valor2 ENTÃO ação', tip: 'Quanto mais específica, melhor.' },
                  { action: 'Configure a Ação', details: 'Escolha: BLOQUEAR, ALERTAR, RETER PARA ANÁLISE', tip: 'BLOQUEAR é irreversível até revisão humana.' },
                  { action: 'Teste em Sandbox', details: 'Antes de ativar, teste com dados históricos de 30 dias.', tip: 'Veja quantas transações seriam afetadas.' },
                  { action: 'Ative com Monitoramento', details: 'Ative a regra e monitore por 48h os resultados.', tip: 'Desative imediatamente se muitos falsos positivos.' }
                ]}
              />
            </div>
          </ManualSection>

          {/* VIP e HOT List */}
          <ManualSection id="listas" title="📋 TELAS: VIP List e HOT List" icon={List}>
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-xl p-6">
                  <Star className="h-10 w-10 mb-4" />
                  <h4 className="text-2xl font-bold mb-2">VIP List ✨</h4>
                  <p className="text-green-100 mb-4">
                    Clientes de total confiança. Transações são aprovadas com menos rigor.
                  </p>
                  <div className="bg-green-700 rounded-lg p-4">
                    <p className="font-semibold mb-2">Quem pode estar aqui:</p>
                    <ul className="text-sm space-y-1">
                      <li>• Diretores e C-Level da empresa</li>
                      <li>• Clientes com histórico perfeito (5+ anos)</li>
                      <li>• Contas institucionais verificadas</li>
                    </ul>
                  </div>
                  <AlertBox type="warning" title="Atenção" className="mt-4">
                    VIPs ainda são monitorados! Apenas têm threshold mais alto (80+).
                  </AlertBox>
                </div>
                
                <div className="bg-gradient-to-br from-red-500 to-red-600 text-white rounded-xl p-6">
                  <XCircle className="h-10 w-10 mb-4" />
                  <h4 className="text-2xl font-bold mb-2">HOT List 🔥</h4>
                  <p className="text-red-100 mb-4">
                    Contas com fraude confirmada. TODAS as transações são bloqueadas.
                  </p>
                  <div className="bg-red-700 rounded-lg p-4">
                    <p className="font-semibold mb-2">Quem entra aqui:</p>
                    <ul className="text-sm space-y-1">
                      <li>• CPFs com fraude confirmada</li>
                      <li>• Contas laranja identificadas</li>
                      <li>• Devices usados em ataques</li>
                    </ul>
                  </div>
                  <AlertBox type="danger" title="Permanente!" className="mt-4">
                    Uma vez na HOT List, a conta é bloqueada para sempre. Sem exceções.
                  </AlertBox>
                </div>
              </div>
              
              <FAQ questions={[
                { question: 'Quem pode adicionar alguém na VIP List?', answer: 'Apenas líderes (Ana Paula) ou gestores podem adicionar à VIP List. É necessária justificativa documentada e aprovação de compliance.' },
                { question: 'E se eu adicionar um cliente errado na HOT List?', answer: 'Isso é um incidente grave! A remoção precisa de aprovação do jurídico e compliance. Por isso, SEMPRE confirme com 2 fontes antes de adicionar.' },
                { question: 'A HOT List é compartilhada entre bancos?', answer: 'Sim! Via sistema BACEN de compartilhamento de fraudes. Se adicionamos alguém aqui, outros bancos são notificados.' }
              ]} />
            </div>
          </ManualSection>

          {/* Mais Telas */}
          <ManualSection id="outras-telas" title="📱 Outras Telas do Sistema" icon={Grid}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[
                { icon: Target, title: 'Investigação', desc: 'Análise profunda de casos. Timeline completa, explicabilidade LGPD, documentos anexos.', color: 'blue' },
                { icon: BarChart3, title: 'Métricas', desc: 'Dashboards operacionais: TPS, latência P50/P95/P99, taxa de erro, throughput.', color: 'green' },
                { icon: Activity, title: 'Monitoramento', desc: 'Saúde dos 3 modelos de IA. Drift detection, retraining status, accuracy history.', color: 'purple' },
                { icon: FileText, title: 'Relatórios', desc: 'Gerador de PDFs para gerência e BACEN. Customizáveis por período e tipo.', color: 'orange' },
                { icon: Database, title: 'Datasets', desc: 'Catálogo de dados para análises. Exportação CSV/Excel para investigações.', color: 'blue' },
                { icon: MessageSquare, title: 'Feedback', desc: 'Tela para treinar o modelo. Cada feedback melhora a precisão em 30 dias.', color: 'green' },
                { icon: Bell, title: 'Alertas', desc: 'Central de notificações. Críticos, avisos e informativos organizados.', color: 'red' },
                { icon: Settings, title: 'Configurações', desc: 'Preferências do usuário, tema, notificações, idioma.', color: 'gray' },
                { icon: FileText, title: 'Auditoria', desc: 'Log de todas as ações do sistema. Exigido por LGPD e BACEN.', color: 'purple' }
              ].map((item, i) => (
                <div key={i} className={`bg-${item.color}-50 rounded-xl p-4 border border-${item.color}-200`}>
                  <item.icon className={`h-8 w-8 text-${item.color}-600 mb-3`} />
                  <h4 className="font-bold text-gray-900">{item.title}</h4>
                  <p className="text-sm text-gray-600 mt-1">{item.desc}</p>
                </div>
              ))}
            </div>
          </ManualSection>
        </>
      )}

      {activeTab === 'cenarios' && (
        <>
          <ManualSection id="cenario1" title="🚨 Cenário 1: Golpe do PIX - Fraude Confirmada" icon={AlertTriangle} defaultOpen={true} priority="critical">
            <ScenarioTimeline {...allScenarios.golpePix} />
            <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TransactionCard 
                id="PIX-2024-11-30-143201"
                cpf="***.***. 789-01"
                amount="R$ 4.850,00"
                channel="PIX"
                time="30/11/2025 14:32"
                status="blocked"
                score={87}
                location="São Paulo, SP → Recife, PE"
              />
              <div>
                <h4 className="font-bold text-gray-900 mb-3">Lições Aprendidas</h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex gap-2"><CheckCircle className="h-5 w-5 text-green-500" /> Score alto (87) disparou bloqueio automático</li>
                  <li className="flex gap-2"><CheckCircle className="h-5 w-5 text-green-500" /> Analista confirmou em 8 minutos</li>
                  <li className="flex gap-2"><CheckCircle className="h-5 w-5 text-green-500" /> Cliente foi contatado proativamente</li>
                  <li className="flex gap-2"><CheckCircle className="h-5 w-5 text-green-500" /> Conta laranja adicionada à HOT List</li>
                </ul>
              </div>
            </div>
          </ManualSection>

          <ManualSection id="cenario2" title="✅ Cenário 2: Falso Positivo - Viagem de Negócios" icon={CheckCircle} priority="info">
            <ScenarioTimeline {...allScenarios.falsoPositivo} />
          </ManualSection>

          <ManualSection id="cenario3" title="🏍️ Cenário 3: Golpe do Motoboy - Cartão Clonado" icon={AlertTriangle} priority="high">
            <ScenarioTimeline {...allScenarios.golpeMotoboy} />
          </ManualSection>

          <ManualSection id="cenario4" title="🏦 Cenário 4: TED Fraudulenta - Conta PJ" icon={AlertTriangle} priority="critical">
            <ScenarioTimeline {...allScenarios.tedFraudulenta} />
          </ManualSection>

          <ManualSection id="cenario5" title="📄 Cenário 5: Boleto Legítimo - Compra de Veículo" icon={CheckCircle} priority="info">
            <ScenarioTimeline {...allScenarios.boletoVerdadeiro} />
          </ManualSection>

          <ManualSection id="cenario6" title="🍊 Cenário 6: Detecção de Conta Laranja" icon={AlertTriangle} priority="critical">
            <ScenarioTimeline {...allScenarios.contaLaranja} />
          </ManualSection>
        </>
      )}

      {activeTab === 'troubleshooting' && (
        <>
          <ManualSection id="erros-comuns" title="🔧 Erros Comuns e Soluções" icon={AlertTriangle} defaultOpen={true}>
            <div className="space-y-4">
              {[
                { problema: 'Sistema mostra "Carregando..." infinitamente', causa: 'Cache desatualizado ou problema de rede', solucao: 'Pressione Ctrl+Shift+R para hard refresh. Se persistir, contate TI.' },
                { problema: 'Não consigo ver transações do cliente', causa: 'CPF digitado incorretamente ou sem permissão', solucao: 'Verifique se está usando formato ***.***. XXX-YY. Confirme seu nível de acesso.' },
                { problema: 'Botão "Aprovar" está desabilitado', causa: 'Transação já foi decidida ou você não tem permissão', solucao: 'Verifique status da transação. Se "Concluída", já foi decidida por outro analista.' },
                { problema: 'Score não aparece na transação', causa: 'Modelo de IA temporariamente offline', solucao: 'Verifique status dos modelos no Dashboard. Se offline, use análise manual.' },
                { problema: 'Alerta crítico não aparece na minha fila', causa: 'Alerta foi assumido por outro analista', solucao: 'Alertas são exclusivos. Verifique se colega já está trabalhando nele.' },
                { problema: 'Não consigo gerar relatório', causa: 'Período muito longo ou muitos dados', solucao: 'Reduza o período para máximo 30 dias. Para mais, peça à TI.' },
                { problema: 'Gráficos não carregam', causa: 'Problema de JavaScript ou bloqueador de anúncios', solucao: 'Desative extensões do navegador. Use Chrome atualizado.' }
              ].map((item, i) => (
                <div key={i} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-6 w-6 text-orange-500 flex-shrink-0 mt-1" />
                    <div>
                      <p className="font-bold text-gray-900">{item.problema}</p>
                      <p className="text-sm text-gray-500">Causa: {item.causa}</p>
                      <p className="text-sm text-green-700 font-medium mt-1">✓ Solução: {item.solucao}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ManualSection>

          <ManualSection id="escalonamento" title="📞 Quando Escalonar" icon={Phone}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-yellow-50 rounded-xl p-6 border-2 border-yellow-400">
                <h4 className="font-bold text-yellow-800">Nível 1: Líder de Turno</h4>
                <ul className="mt-4 text-sm space-y-2">
                  <li>• Dúvida em análise</li>
                  <li>• Score na "zona cinza"</li>
                  <li>• Cliente VIP insatisfeito</li>
                </ul>
                <p className="mt-4 font-semibold text-yellow-900">Ana Paula / Carlos</p>
              </div>
              
              <div className="bg-orange-50 rounded-xl p-6 border-2 border-orange-400">
                <h4 className="font-bold text-orange-800">Nível 2: Gerente</h4>
                <ul className="mt-4 text-sm space-y-2">
                  <li>• Fraude confirmada &gt; R$ 50k</li>
                  <li>• Ataque coordenado</li>
                  <li>• Reclamação formal</li>
                </ul>
                <p className="mt-4 font-semibold text-orange-900">Patrícia Lima</p>
              </div>
              
              <div className="bg-red-50 rounded-xl p-6 border-2 border-red-400">
                <h4 className="font-bold text-red-800">Nível 3: Diretoria</h4>
                <ul className="mt-4 text-sm space-y-2">
                  <li>• Fraude &gt; R$ 500k</li>
                  <li>• Vazamento de dados</li>
                  <li>• Notificação BACEN</li>
                </ul>
                <p className="mt-4 font-semibold text-red-900">Diretor de Riscos</p>
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {activeTab === 'glossario' && (
        <>
          <ManualSection id="glossario-termos" title="📖 Glossário Completo" icon={FileText} defaultOpen={true}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { termo: 'Score de Risco', definicao: 'Pontuação de 0 a 100 que indica a probabilidade de uma transação ser fraudulenta. 0 = seguro, 100 = fraude certa.' },
                { termo: 'Threshold (Limiar)', definicao: 'Valor de corte do score. Transações com score acima do threshold são bloqueadas automaticamente.' },
                { termo: 'Falso Positivo', definicao: 'Quando o sistema bloqueia uma transação legítima por engano. Afeta experiência do cliente.' },
                { termo: 'Falso Negativo', definicao: 'Quando o sistema aprova uma transação fraudulenta. O pior cenário possível.' },
                { termo: 'Human-in-the-Loop', definicao: 'Processo onde humanos revisam decisões da IA nos casos de dúvida (score 40-60).' },
                { termo: 'Hard Rule', definicao: 'Regra fixa que dispara independente do score. Usada para padrões conhecidos de fraude.' },
                { termo: 'VIP List', definicao: 'Lista de clientes de alta confiança que têm threshold mais alto (menos bloqueios).' },
                { termo: 'HOT List', definicao: 'Lista negra de CPFs/contas com fraude confirmada. Bloqueio permanente.' },
                { termo: 'Feature', definicao: 'Característica analisada pela IA (ex: valor, horário, device). O sistema analisa 40+ features.' },
                { termo: 'Data Drift', definicao: 'Quando padrões de fraude mudam e o modelo fica desatualizado. Requer retraining.' },
                { termo: 'Explicabilidade', definicao: 'Capacidade de explicar POR QUE a IA tomou uma decisão. Exigida pela LGPD.' },
                { termo: 'STR', definicao: 'Sistema de Transmissão de Reservas. Relatório obrigatório ao BACEN para transações suspeitas.' },
                { termo: 'MED', definicao: 'Mecanismo Especial de Devolução. Permite devolver PIX fraudulento em até 90 dias.' },
                { termo: 'Conta Laranja', definicao: 'Conta bancária usada para receber e movimentar dinheiro de fraudes. Crime federal.' },
                { termo: 'TPS', definicao: 'Transações Por Segundo. Métrica de performance do sistema.' },
                { termo: 'Latência P99', definicao: '99% das transações são processadas abaixo deste tempo. SLA crítico.' }
              ].map((item, i) => (
                <div key={i} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <p className="font-bold text-blue-700">{item.termo}</p>
                  <p className="text-sm text-gray-600 mt-1">{item.definicao}</p>
                </div>
              ))}
            </div>
          </ManualSection>
        </>
      )}

      {/* Footer */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <div className="flex items-center justify-center gap-2">
              <Shield className="h-8 w-8 text-blue-600" />
              <span className="text-2xl font-bold text-gray-900">Sankofa Enterprise Pro v1.0</span>
            </div>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Este manual completo foi criado para capacitar analistas de fraude a proteger 
              milhões de reais todos os dias. Use-o como referência diária em seu trabalho.
            </p>
            <div className="flex items-center justify-center gap-6 text-sm text-gray-500">
              <span className="flex items-center gap-1"><Shield className="h-4 w-4" /> LGPD Compliant</span>
              <span className="flex items-center gap-1"><Building className="h-4 w-4" /> BACEN Approved</span>
              <span className="flex items-center gap-1"><Lock className="h-4 w-4" /> PCI DSS Ready</span>
            </div>
            <p className="text-sm text-gray-400">
              Última atualização: 30 de Novembro de 2025 | Versão do Manual: 1.0.0
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
