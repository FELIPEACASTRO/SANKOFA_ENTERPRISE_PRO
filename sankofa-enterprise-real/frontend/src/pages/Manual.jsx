import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Users, Target, Shield, AlertTriangle, Clock, Zap, Eye, Brain, Settings, FileText, BarChart3, Database, Bell, Lock, Star, CheckCircle, XCircle, TrendingUp, Phone, Building } from 'lucide-react';
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

// Dados das Personas Brasileiras
const personas = {
  anaPaula: {
    name: 'Ana Paula Oliveira',
    role: 'Líder de Prevenção a Fraudes',
    avatar: 'AP',
    department: 'Banco Digital Nexus',
    experience: '8 anos em prevenção a fraudes',
    quote: 'Cada fraude bloqueada é um cliente que continua confiando em nós.',
    color: 'blue'
  },
  carlosRoberto: {
    name: 'Carlos Roberto Silva',
    role: 'Analista de Fraudes Sênior',
    avatar: 'CR',
    department: 'Operações de Risco',
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
  }
};

// Cenários Reais Brasileiros
const scenarios = {
  golpePix: {
    title: '🚨 Golpe do PIX Falso - Caso Real',
    icon: AlertTriangle,
    steps: [
      { time: '14:32', badge: 'PIX', type: 'action', title: 'Transação Recebida', description: 'PIX de R$ 4.850,00 para conta nova. CPF: ***.***. 789-01' },
      { time: '14:32', badge: 'ALERTA', type: 'alert', title: 'Sankofa Detecta Anomalia', description: 'Score 87/100. Motivos: horário atípico, destinatário novo, valor 3x acima da média' },
      { time: '14:33', badge: 'BLOQUEIO', type: 'alert', title: 'Transação Bloqueada', description: 'Sistema bloqueia automaticamente. Cliente notificado por SMS.' },
      { time: '14:35', type: 'action', title: 'Analista Investiga', description: 'Carlos Roberto abre investigação. Verifica histórico do cliente.' },
      { time: '14:40', type: 'success', title: 'Fraude Confirmada', description: 'Cliente confirma: "Não fiz essa transferência!" Conta destino era de laranja.' }
    ],
    outcome: 'R$ 4.850,00 salvos. Cliente protegido. Conta laranja reportada ao BACEN.',
    outcomeType: 'success'
  },
  falsoPositivo: {
    title: '✅ Falso Positivo Resolvido - Viagem de Negócios',
    icon: CheckCircle,
    steps: [
      { time: '23:15', badge: 'PIX', type: 'action', title: 'PIX Noturno Detectado', description: 'R$ 12.000,00 de São Paulo para Salvador. Score: 72/100' },
      { time: '23:15', badge: 'ALERTA', type: 'alert', title: 'Bloqueio Preventivo', description: 'Horário noturno + valor alto + destino incomum' },
      { time: '23:20', type: 'action', title: 'Cliente Liga Reclamando', description: '"Estou em Salvador a trabalho, preciso pagar o hotel!"' },
      { time: '23:22', type: 'action', title: 'Analista Valida', description: 'Ana Paula confirma viagem no sistema da empresa. Cliente real.' },
      { time: '23:25', type: 'success', title: 'Transação Liberada', description: 'PIX aprovado manualmente. Feedback inserido no sistema.' }
    ],
    outcome: 'Cliente satisfeito. Modelo aprende: viagens a trabalho são legítimas.',
    outcomeType: 'success'
  }
};

function ManualSection({ id, title, icon: Icon, children, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <Card className="overflow-hidden">
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

export function Manual() {
  return (
    <div className="space-y-6 pb-12">
      {/* Header Principal */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-800 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-center gap-4 mb-4">
          <div className="bg-white/20 p-3 rounded-xl">
            <BookOpen className="h-10 w-10" />
          </div>
          <div>
            <h1 className="text-4xl font-bold">Manual do Sankofa v1.0</h1>
            <p className="text-lg text-blue-100">Sistema de Detecção de Fraudes Bancárias</p>
          </div>
        </div>
        <p className="text-blue-100 max-w-2xl">
          Guia completo e visual para analistas de fraude. Aprenda a proteger milhões 
          de reais com exemplos reais, cenários práticos e dicas de especialistas.
        </p>
        <div className="flex items-center gap-6 mt-6 text-sm">
          <span className="flex items-center gap-2"><Clock className="h-4 w-4" /> Atualizado: 30/11/2025</span>
          <span className="flex items-center gap-2"><Users className="h-4 w-4" /> Para: Analistas de Fraude</span>
          <span className="flex items-center gap-2"><Shield className="h-4 w-4" /> Compliance: LGPD/BACEN</span>
        </div>
      </div>

      {/* Conheça Nossa Equipe */}
      <ManualSection id="personas" title="👥 Conheça Nossa Equipe de Especialistas" icon={Users} defaultOpen={true}>
        <p className="text-gray-600 mb-6">
          Acompanhe as histórias de Ana Paula, Carlos e Marina ao longo deste manual. 
          Eles vão mostrar como usar cada funcionalidade na prática do dia a dia.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <PersonaCard {...personas.anaPaula} />
          <PersonaCard {...personas.carlosRoberto} />
          <PersonaCard {...personas.marinaFernandes} />
        </div>
      </ManualSection>

      {/* Como o Sankofa Funciona */}
      <ManualSection id="como-funciona" title="🧠 Como o Sankofa Detecta Fraudes" icon={Brain}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-bold text-gray-900 mb-4">Fluxo de Análise em Tempo Real</h3>
            <FlowDiagram 
              nodes={[
                { label: 'Transação Recebida', type: 'start', icon: Zap },
                { label: 'Análise de 40+ Características', type: 'process' },
                { label: 'Score de Risco (0-100)', type: 'decision' },
                { label: 'Score < 30: Aprovar', type: 'success' },
                { label: 'Score > 70: Bloquear', type: 'danger' },
                { label: 'Score 30-70: Revisão Manual', type: 'end' }
              ]}
            />
          </div>
          
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-gray-900 mb-4">O Que Analisamos?</h3>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-blue-50 rounded-lg p-4">
                <Clock className="h-6 w-6 text-blue-600 mb-2" />
                <p className="font-semibold text-gray-900">Horário</p>
                <p className="text-sm text-gray-600">PIX às 3h da manhã? Suspeito!</p>
              </div>
              <div className="bg-green-50 rounded-lg p-4">
                <TrendingUp className="h-6 w-6 text-green-600 mb-2" />
                <p className="font-semibold text-gray-900">Valor</p>
                <p className="text-sm text-gray-600">Acima da média do cliente?</p>
              </div>
              <div className="bg-purple-50 rounded-lg p-4">
                <Target className="h-6 w-6 text-purple-600 mb-2" />
                <p className="font-semibold text-gray-900">Destinatário</p>
                <p className="text-sm text-gray-600">Conta nova nunca vista?</p>
              </div>
              <div className="bg-orange-50 rounded-lg p-4">
                <Building className="h-6 w-6 text-orange-600 mb-2" />
                <p className="font-semibold text-gray-900">Localização</p>
                <p className="text-sm text-gray-600">SP para AC em 5 min?</p>
              </div>
            </div>
            
            <AlertBox type="tip" title="Dica da Ana Paula">
              "O Sankofa não olha apenas um fator isolado. Ele combina TODOS os fatores 
              para calcular o score. Um PIX noturno pode ser normal se o cliente sempre 
              faz isso. O contexto é tudo!"
            </AlertBox>
          </div>
        </div>
      </ManualSection>

      {/* Cenário 1: Golpe do PIX */}
      <ManualSection id="cenario-golpe" title="🚨 Caso Real: Golpe do PIX Detectado" icon={AlertTriangle}>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ScenarioTimeline {...scenarios.golpePix} />
          </div>
          
          <div className="space-y-4">
            <h4 className="font-bold text-gray-900">Transação Bloqueada</h4>
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
            
            <AlertBox type="success" title="Resultado">
              Cliente protegido! O dinheiro permaneceu na conta. 
              Conta laranja foi reportada ao BACEN e incluída na HOT List.
            </AlertBox>
          </div>
        </div>
      </ManualSection>

      {/* Cenário 2: Falso Positivo */}
      <ManualSection id="cenario-fp" title="✅ Caso Real: Falso Positivo Resolvido" icon={CheckCircle}>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ScenarioTimeline {...scenarios.falsoPositivo} />
          </div>
          
          <div className="space-y-4">
            <h4 className="font-bold text-gray-900">Transação Liberada</h4>
            <TransactionCard 
              id="PIX-2024-11-30-231501"
              cpf="***.***. 456-78"
              amount="R$ 12.000,00"
              channel="PIX"
              time="30/11/2025 23:15"
              status="approved"
              score={72}
              location="São Paulo, SP → Salvador, BA"
            />
            
            <AlertBox type="info" title="Aprendizado">
              O feedback da Ana Paula treinou o modelo. Agora o Sankofa 
              sabe que viagens de negócios geram PIX noturnos legítimos.
            </AlertBox>
          </div>
        </div>
      </ManualSection>

      {/* Dashboard */}
      <ManualSection id="dashboard" title="📊 Dashboard - Sua Central de Comando" icon={BarChart3}>
        <div className="space-y-6">
          <p className="text-gray-600">
            O Dashboard é como a cabine de um piloto: você vê TUDO em uma tela. 
            KPIs, gráficos, alertas e status dos modelos de IA.
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KPICard title="Transações Hoje" value="4.467" change="+12%" changeType="up" icon={Zap} color="blue" />
            <KPICard title="Fraudes Bloqueadas" value="3.115" change="+8%" changeType="up" icon={Shield} color="red" />
            <KPICard title="Valor Protegido" value="R$ 14.3M" change="+15%" changeType="up" icon={TrendingUp} color="green" />
            <KPICard title="Latência Média" value="37ms" change="-5ms" changeType="up" icon={Clock} color="purple" />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ScreenPreview 
              title="Dashboard"
              description="Visão geral do sistema em tempo real"
              elements={[
                { label: 'Gráfico de Fraudes/Hora', type: 'chart' },
                { label: 'KPIs Principais', type: 'kpi' },
                { label: 'Alertas Recentes', type: 'table' },
                { label: 'Status dos Modelos', type: 'kpi' }
              ]}
            />
            
            <div className="space-y-4">
              <h4 className="font-bold text-gray-900">Rotina do Carlos Roberto</h4>
              <Checklist 
                title="Início do Turno (08:00)"
                items={[
                  { text: 'Verificar alertas críticos (vermelhos)', done: true },
                  { text: 'Checar KPIs vs. dia anterior', done: true },
                  { text: 'Validar status dos 3 modelos de IA', done: true },
                  { text: 'Revisar fila de análise manual', done: false }
                ]}
              />
              
              <AlertBox type="tip" title="Dica do Carlos">
                "Deixo o Dashboard aberto em um monitor o dia todo. 
                A cada 30 segundos ele atualiza automaticamente. 
                Se algo estranho acontece, eu vejo na hora!"
              </AlertBox>
            </div>
          </div>
        </div>
      </ManualSection>

      {/* Transações */}
      <ManualSection id="transacoes" title="💳 Transações - Busque Qualquer Operação" icon={FileText}>
        <div className="space-y-6">
          <div className="bg-gray-50 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 mb-4">Situação Hipotética: Cliente Ligou Reclamando</h4>
            <div className="bg-white rounded-lg p-4 border border-gray-200 mb-4">
              <div className="flex items-start gap-3">
                <Phone className="h-6 w-6 text-blue-500 mt-1" />
                <div>
                  <p className="font-semibold">Chamada recebida às 15:45</p>
                  <p className="text-gray-600 italic">
                    "Alô, meu nome é José Roberto, CPF final 789-01. Fiz um PIX de R$ 2.500 
                    às 14h para minha filha e foi bloqueado! Preciso resolver URGENTE!"
                  </p>
                </div>
              </div>
            </div>
            
            <h4 className="font-bold text-gray-900 mb-3">Como Resolver:</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">Passo 1</span>
                <p className="font-semibold mt-2">Abra Transações</p>
                <p className="text-sm text-gray-600">Menu → Transações</p>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">Passo 2</span>
                <p className="font-semibold mt-2">Filtre por CPF</p>
                <p className="text-sm text-gray-600">Digite "***.***. 789-01" + Data de hoje</p>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">Passo 3</span>
                <p className="font-semibold mt-2">Clique em "Ver Detalhes"</p>
                <p className="text-sm text-gray-600">Abre a investigação completa</p>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 mb-3">Filtros Disponíveis</h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <span className="text-2xl">🔍</span>
                  <div>
                    <p className="font-semibold">CPF</p>
                    <p className="text-sm text-gray-600">Busca por cliente específico (mascarado)</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <span className="text-2xl">📅</span>
                  <div>
                    <p className="font-semibold">Período</p>
                    <p className="text-sm text-gray-600">Últimas 24h, 7 dias, 30 dias ou personalizado</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <span className="text-2xl">⚡</span>
                  <div>
                    <p className="font-semibold">Canal</p>
                    <p className="text-sm text-gray-600">PIX, TED, Cartão, Boleto</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <span className="text-2xl">🚦</span>
                  <div>
                    <p className="font-semibold">Status</p>
                    <p className="text-sm text-gray-600">Aprovadas ✅, Bloqueadas ❌, Em Análise ⏳</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 mb-3">Exemplo de Resultado</h4>
              <TransactionCard 
                id="PIX-2024-11-30-140023"
                cpf="***.***. 789-01"
                amount="R$ 2.500,00"
                channel="PIX"
                time="30/11/2025 14:00"
                status="blocked"
                score={65}
                location="São Paulo, SP → Campinas, SP"
              />
              
              <AlertBox type="warning" title="Por que foi bloqueado?" className="mt-4">
                Score 65: Destinatário novo (primeira vez) + Valor acima da média do cliente.
                Clique em "Ver Detalhes" para ver todos os motivos.
              </AlertBox>
            </div>
          </div>
        </div>
      </ManualSection>

      {/* Revisão Manual */}
      <ManualSection id="revisao" title="👁️ Revisão Manual - Sua Decisão Importa" icon={Eye}>
        <div className="space-y-6">
          <AlertBox type="info" title="O que é Human-in-the-Loop?">
            Quando o Sankofa tem dúvida (score entre 40-60), a transação vem para VOCÊ decidir. 
            Sua experiência humana complementa a inteligência artificial!
          </AlertBox>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl p-6 border border-yellow-200">
              <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Clock className="h-5 w-5 text-yellow-600" />
                Fila de Revisão Atual
              </h4>
              
              <div className="space-y-3">
                <div className="bg-white rounded-lg p-4 border border-yellow-300">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-500">PIX-2024-11-30-154532</span>
                    <RiskThermometer score={52} label="Score" />
                  </div>
                  <p className="font-semibold">R$ 8.900,00 para conta nova</p>
                  <p className="text-sm text-gray-600">Motivo: Valor 4x acima da média</p>
                  <div className="flex gap-2 mt-3">
                    <button className="flex-1 bg-green-500 text-white py-2 rounded-lg text-sm font-semibold hover:bg-green-600">
                      ✅ Legítima
                    </button>
                    <button className="flex-1 bg-red-500 text-white py-2 rounded-lg text-sm font-semibold hover:bg-red-600">
                      ❌ Fraude
                    </button>
                  </div>
                </div>
                
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-500">TED-2024-11-30-161245</span>
                    <RiskThermometer score={48} label="Score" />
                  </div>
                  <p className="font-semibold">R$ 15.000,00 horário comercial</p>
                  <p className="text-sm text-gray-600">Motivo: Novo destinatário</p>
                  <div className="flex gap-2 mt-3">
                    <button className="flex-1 bg-green-500 text-white py-2 rounded-lg text-sm font-semibold hover:bg-green-600">
                      ✅ Legítima
                    </button>
                    <button className="flex-1 bg-red-500 text-white py-2 rounded-lg text-sm font-semibold hover:bg-red-600">
                      ❌ Fraude
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Por Que Sua Decisão Importa?</h4>
              
              <div className="space-y-4">
                <div className="flex gap-3">
                  <div className="bg-blue-100 p-2 rounded-lg h-fit">
                    <Brain className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-semibold">Treina o Modelo</p>
                    <p className="text-sm text-gray-600">
                      Cada decisão sua ensina o Sankofa. Em 30 dias, 
                      ele aprende seus padrões e fica mais preciso.
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="bg-green-100 p-2 rounded-lg h-fit">
                    <Users className="h-6 w-6 text-green-600" />
                  </div>
                  <div>
                    <p className="font-semibold">Protege Clientes</p>
                    <p className="text-sm text-gray-600">
                      Você impede que clientes legítimos sejam bloqueados 
                      injustamente. Experiência do cliente melhora!
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="bg-purple-100 p-2 rounded-lg h-fit">
                    <Shield className="h-6 w-6 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-semibold">Captura Fraudes Novas</p>
                    <p className="text-sm text-gray-600">
                      Fraudadores inventam golpes novos. Você detecta 
                      padrões que o modelo ainda não conhece.
                    </p>
                  </div>
                </div>
              </div>
              
              <AlertBox type="tip" title="Meta da Ana Paula" className="mt-4">
                "Minha equipe revisa 100 transações por dia. Em 6 meses, 
                reduzimos falsos positivos de 15% para 3%. O modelo aprendeu conosco!"
              </AlertBox>
            </div>
          </div>
        </div>
      </ManualSection>

      {/* Calibragem */}
      <ManualSection id="calibragem" title="⚙️ Calibragem - Ajuste a Sensibilidade" icon={Settings}>
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Termômetro de Sensibilidade</h4>
              <div className="bg-gradient-to-b from-red-100 via-yellow-100 to-green-100 rounded-xl p-6 border">
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="w-16 text-center">
                      <span className="text-2xl font-bold text-red-600">100</span>
                      <p className="text-xs text-red-600">Máximo</p>
                    </div>
                    <div className="flex-1 h-8 bg-red-400 rounded-lg flex items-center px-3">
                      <span className="text-white text-sm">Bloqueia quase TUDO</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="w-16 text-center">
                      <span className="text-2xl font-bold text-orange-600">70</span>
                    </div>
                    <div className="flex-1 h-8 bg-orange-400 rounded-lg flex items-center px-3">
                      <span className="text-white text-sm">Muito rigoroso</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="w-16 text-center">
                      <span className="text-2xl font-bold text-yellow-600">45</span>
                      <p className="text-xs text-yellow-600">⭐ Atual</p>
                    </div>
                    <div className="flex-1 h-10 bg-yellow-400 rounded-lg flex items-center px-3 border-4 border-yellow-600">
                      <span className="text-gray-900 text-sm font-bold">RECOMENDADO (Balanço ideal)</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="w-16 text-center">
                      <span className="text-2xl font-bold text-lime-600">30</span>
                    </div>
                    <div className="flex-1 h-8 bg-lime-400 rounded-lg flex items-center px-3">
                      <span className="text-gray-900 text-sm">Permissivo</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="w-16 text-center">
                      <span className="text-2xl font-bold text-green-600">0</span>
                      <p className="text-xs text-green-600">Mínimo</p>
                    </div>
                    <div className="flex-1 h-8 bg-green-400 rounded-lg flex items-center px-3">
                      <span className="text-white text-sm">Aprova quase TUDO</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Quando Ajustar?</h4>
              
              <div className="space-y-4">
                <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                  <p className="font-bold text-red-800 flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Clientes Reclamando Muito?
                  </p>
                  <p className="text-sm text-red-700 mt-1">
                    "Meu PIX foi bloqueado injustamente!"
                  </p>
                  <p className="text-sm font-semibold mt-2">
                    → DIMINUA o threshold (45 → 35)
                  </p>
                </div>
                
                <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                  <p className="font-bold text-orange-800 flex items-center gap-2">
                    <XCircle className="h-5 w-5" />
                    Fraudes Passando?
                  </p>
                  <p className="text-sm text-orange-700 mt-1">
                    "Tivemos 3 fraudes confirmadas ontem que passaram!"
                  </p>
                  <p className="text-sm font-semibold mt-2">
                    → AUMENTE o threshold (45 → 55)
                  </p>
                </div>
                
                <AlertBox type="warning" title="Regra de Ouro da Marina">
                  "Nunca mude mais de 10 pontos por vez. Mude, espere 24 horas, 
                  analise os resultados, depois ajuste novamente se necessário."
                </AlertBox>
              </div>
            </div>
          </div>
        </div>
      </ManualSection>

      {/* Regras Duras e Listas */}
      <ManualSection id="regras" title="🔒 Regras Duras, VIP e HOT Lists" icon={Lock}>
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-900 text-white rounded-xl p-6">
              <Lock className="h-8 w-8 mb-3" />
              <h4 className="text-xl font-bold mb-2">Hard Rules</h4>
              <p className="text-gray-300 text-sm mb-4">
                Regras automáticas que SEMPRE disparam, independente do score.
              </p>
              <div className="bg-gray-800 rounded-lg p-3 text-sm font-mono">
                SE valor &gt; R$ 100.000<br/>
                E horário = 23h-05h<br/>
                ENTÃO = BLOQUEAR
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-xl p-6">
              <Star className="h-8 w-8 mb-3" />
              <h4 className="text-xl font-bold mb-2">VIP List ✨</h4>
              <p className="text-green-100 text-sm mb-4">
                Clientes de TOTAL confiança. Transações aprovadas automaticamente.
              </p>
              <div className="bg-green-700 rounded-lg p-3 text-sm">
                <p className="font-semibold">Exemplo:</p>
                <p>Diretor Geral da empresa</p>
                <p>CPF: ***.***. 111-00</p>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-red-500 to-red-600 text-white rounded-xl p-6">
              <XCircle className="h-8 w-8 mb-3" />
              <h4 className="text-xl font-bold mb-2">HOT List 🔥</h4>
              <p className="text-red-100 text-sm mb-4">
                Contas BLOQUEADAS permanentemente. Fraude confirmada.
              </p>
              <div className="bg-red-700 rounded-lg p-3 text-sm">
                <p className="font-semibold">Exemplo:</p>
                <p>Conta laranja detectada</p>
                <p>CPF: ***.***. 999-99</p>
              </div>
            </div>
          </div>
          
          <AlertBox type="danger" title="⚠️ Cuidado!">
            Adicionar alguém na VIP ou HOT List é uma decisão PERMANENTE. 
            Sempre documente o motivo e tenha aprovação do gestor.
          </AlertBox>
        </div>
      </ManualSection>

      {/* Alertas */}
      <ManualSection id="alertas" title="🚨 Alertas - Ações Urgentes" icon={Bell}>
        <div className="space-y-4">
          <p className="text-gray-600">
            Alertas são notificações que exigem sua atenção IMEDIATA. 
            Aparecem quando algo fora do normal acontece.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-red-50 border-2 border-red-500 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-4 h-4 rounded-full bg-red-500 animate-pulse" />
                <span className="font-bold text-red-700">CRÍTICO</span>
              </div>
              <p className="font-semibold text-gray-900">Spike de Fraudes</p>
              <p className="text-sm text-gray-600">+250% de fraudes vs. média horária</p>
              <p className="text-xs text-red-600 mt-2">Ação: URGENTE - Investigar agora!</p>
            </div>
            
            <div className="bg-orange-50 border-2 border-orange-500 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-4 h-4 rounded-full bg-orange-500" />
                <span className="font-bold text-orange-700">AVISO</span>
              </div>
              <p className="font-semibold text-gray-900">Modelo Offline</p>
              <p className="text-sm text-gray-600">CatBoost não respondendo há 5 min</p>
              <p className="text-xs text-orange-600 mt-2">Ação: Verificar em 15 minutos</p>
            </div>
            
            <div className="bg-yellow-50 border-2 border-yellow-500 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-4 h-4 rounded-full bg-yellow-500" />
                <span className="font-bold text-yellow-700">INFO</span>
              </div>
              <p className="font-semibold text-gray-900">Recalibração Sugerida</p>
              <p className="text-sm text-gray-600">Taxa de falso positivo subiu para 8%</p>
              <p className="text-xs text-yellow-600 mt-2">Ação: Analisar quando possível</p>
            </div>
          </div>
        </div>
      </ManualSection>

      {/* Outras Telas */}
      <ManualSection id="outras" title="📚 Outras Funcionalidades" icon={Database}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { icon: BarChart3, title: 'Métricas', desc: 'Contadores em tempo real: TPS, latência, taxas' },
            { icon: Eye, title: 'Monitoramento', desc: 'Saúde dos 3 modelos de IA e data drift' },
            { icon: FileText, title: 'Relatórios', desc: 'Gere PDFs para gerência e compliance' },
            { icon: Database, title: 'Datasets', desc: 'Catálogo de dados para análises' },
            { icon: Target, title: 'Investigação', desc: 'Análise profunda com explicabilidade LGPD' },
            { icon: Users, title: 'Feedback', desc: 'Treine o modelo com suas decisões' }
          ].map((item, i) => (
            <div key={i} className="bg-gray-50 rounded-xl p-4 flex items-start gap-3">
              <div className="bg-blue-100 p-2 rounded-lg">
                <item.icon className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="font-bold text-gray-900">{item.title}</p>
                <p className="text-sm text-gray-600">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </ManualSection>

      {/* Footer */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200">
        <CardContent className="pt-6">
          <div className="text-center space-y-4">
            <div className="flex items-center justify-center gap-2">
              <Shield className="h-6 w-6 text-blue-600" />
              <span className="text-lg font-bold text-gray-900">Sankofa Enterprise Pro v1.0</span>
            </div>
            <p className="text-sm text-gray-600 max-w-lg mx-auto">
              Este manual foi criado para ajudar analistas de fraude a proteger 
              milhões de reais todos os dias. Dúvidas? Fale com seu gestor.
            </p>
            <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
              <span>📜 LGPD Compliant</span>
              <span>🏦 BACEN Approved</span>
              <span>🔐 PCI DSS Ready</span>
            </div>
            <p className="text-xs text-gray-400">
              Última atualização: 30 de Novembro de 2025
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
