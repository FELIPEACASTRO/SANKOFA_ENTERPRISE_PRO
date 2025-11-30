import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Users, Target, Shield, AlertTriangle, Clock, Zap, Eye, Brain, Settings, FileText, BarChart3, Database, Bell, Lock, Star, CheckCircle, XCircle, TrendingUp, Phone, Building, HelpCircle, Search, Filter, Download, Upload, RefreshCw, Play, Pause, Edit, Trash2, Plus, ArrowRight, ArrowLeft, Info, MessageSquare, ThumbsUp, ThumbsDown, Activity, Cpu, Server, Globe, Calendar, DollarSign, Percent, Hash, List, Grid, PieChart, LineChart, Table, Map, Flag, Award, Bookmark, ExternalLink, Copy, Share, Mail, Send, Layers, GitBranch, Box, Terminal, Code, Workflow, Boxes, Network, Gauge, Timer, Sparkles, GraduationCap, Lightbulb, BookMarked, CircuitBoard } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';

const personas = {
  anaPaula: {
    name: 'Ana Paula Oliveira',
    role: 'Líder de Prevenção a Fraudes',
    avatar: 'AP',
    department: 'Banco Digital Nexus - Matriz SP',
    experience: '8 anos em prevenção a fraudes',
    quote: 'Cada fraude bloqueada é um cliente que continua confiando em nós.',
    color: 'blue',
    responsibilities: ['Supervisionar equipe de 12 analistas', 'Aprovar Hard Rules novas', 'Escalar incidentes críticos', 'Relatórios semanais para diretoria'],
    kpis: ['Taxa de detecção > 95%', 'Falsos positivos < 3%', 'SLA < 50ms', 'Satisfação cliente > 4.5']
  },
  carlosRoberto: {
    name: 'Carlos Roberto Silva',
    role: 'Analista de Fraudes Sênior',
    avatar: 'CR',
    department: 'Operações de Risco - Turno Diurno',
    experience: '5 anos analisando transações',
    quote: 'O segredo é entender o padrão normal do cliente antes de julgar.',
    color: 'green',
    responsibilities: ['Analisar 150+ transações/dia', 'Treinar analistas júnior', 'Investigar fraudes complexas', 'Dar feedback para IA'],
    kpis: ['Tempo médio análise < 3min', 'Precisão > 98%', 'Feedbacks/dia > 20', 'Escalonamentos < 5%']
  },
  marinaFernandes: {
    name: 'Marina Fernandes',
    role: 'Compliance Officer',
    avatar: 'MF',
    department: 'Jurídico e Compliance',
    experience: '10 anos em regulação bancária',
    quote: 'LGPD e BACEN não são obstáculos, são nossos aliados.',
    color: 'purple',
    responsibilities: ['Validar relatórios STR', 'Auditar mascaramento LGPD', 'Revisar políticas de dados', 'Interface com BACEN'],
    kpis: ['100% STRs válidos', 'Zero violações LGPD', 'Auditorias sem ressalvas', 'Tempo resposta BACEN < 24h']
  },
  rodrigoMendes: {
    name: 'Rodrigo Mendes',
    role: 'Analista de Fraudes Júnior',
    avatar: 'RM',
    department: 'Operações de Risco - Turno Noturno',
    experience: '1 ano no setor bancário',
    quote: 'Estou aprendendo que cada transação conta uma história.',
    color: 'orange',
    responsibilities: ['Monitorar alertas noturnos', 'Escalar casos complexos', 'Aprender com seniors', 'Documentar procedimentos'],
    kpis: ['Tempo resposta < 5min', 'Escalonamentos corretos > 90%', 'Erros < 2%', 'Treinamentos completos']
  },
  patriciaLima: {
    name: 'Patrícia Lima',
    role: 'Gerente de Operações',
    avatar: 'PL',
    department: 'Diretoria de Riscos',
    experience: '15 anos em bancos',
    quote: 'Métricas não mentem. Precisamos de dados para tomar decisões.',
    color: 'blue',
    responsibilities: ['Definir metas da área', 'Aprovar orçamento de TI', 'Reportar ao C-Level', 'Estratégia anti-fraude'],
    kpis: ['Perdas < 0.01% do volume', 'ROI sistema > 500%', 'NPS interno > 80', 'Turnover < 10%']
  }
};

const allScenarios = {
  golpePix: {
    title: '🚨 Golpe do PIX Falso - Fraude Confirmada',
    icon: AlertTriangle,
    difficulty: 'Média',
    timeToResolve: '10 minutos',
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
    difficulty: 'Baixa',
    timeToResolve: '10 minutos',
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
  }
};

const allFeatures = {
  transactionBasic: {
    category: 'Dados da Transação',
    icon: DollarSign,
    color: 'blue',
    features: [
      { name: 'amount', type: 'float', description: 'Valor da transação em reais (R$)', example: '4850.00', impact: 'ALTO', riskDirection: 'up', explanation: 'Valores muito acima da média do cliente aumentam o risco. Se você costuma fazer PIX de R$ 500, um de R$ 5.000 é suspeito.' },
      { name: 'channel', type: 'string', description: 'Canal: PIX, TED, CARTAO, BOLETO', example: 'PIX', impact: 'MÉDIO', riskDirection: 'varies', explanation: 'PIX é o canal mais arriscado por ser instantâneo e irreversível. TED pode ser revertida. Boleto tem prazo.' },
      { name: 'transaction_hour', type: 'int', description: 'Hora da transação (0-23)', example: '3', impact: 'ALTO', riskDirection: 'up', explanation: 'Transações entre 00h e 06h são 4x mais propensas a fraude. Criminosos agem quando vítimas dormem.' },
      { name: 'day_of_week', type: 'int', description: 'Dia da semana (0=Seg, 6=Dom)', example: '5', impact: 'BAIXO', riskDirection: 'neutral', explanation: 'Fins de semana têm padrões diferentes. Menos transações legítimas = proporção maior de fraudes.' },
      { name: 'is_weekend', type: 'bool', description: 'Se é sábado ou domingo', example: 'true', impact: 'BAIXO', riskDirection: 'up', explanation: 'Fins de semana têm equipes menores nos bancos, facilitando ação de fraudadores.' }
    ]
  },
  velocity: {
    category: 'Velocidade e Frequência',
    icon: Zap,
    color: 'orange',
    features: [
      { name: 'velocity_1h', type: 'int', description: 'Quantidade de transações na última hora', example: '5', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Se você fez 0 PIX na última hora e de repente faz 5, algo está errado. Fraudadores agem rápido.' },
      { name: 'velocity_24h', type: 'int', description: 'Quantidade de transações nas últimas 24h', example: '12', impact: 'ALTO', riskDirection: 'up', explanation: 'Compara com seu padrão normal. Se você faz 3/dia e hoje já fez 12, alerta vermelho.' },
      { name: 'amount_velocity_1h', type: 'float', description: 'Soma dos valores na última hora', example: '15000.00', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Total movimentado recentemente. R$ 15k em 1 hora é muito diferente de R$ 500.' },
      { name: 'avg_time_between_tx', type: 'float', description: 'Tempo médio entre transações (minutos)', example: '2.5', impact: 'ALTO', riskDirection: 'down', explanation: 'Transações muito próximas (< 1 min) indicam automação ou pressa de fraudador.' },
      { name: 'velocity_ratio', type: 'float', description: 'Razão velocidade atual / velocidade média', example: '8.5', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Se hoje você está fazendo 8.5x mais transações que o normal, algo mudou drasticamente.' }
    ]
  },
  behavioral: {
    category: 'Comportamento do Cliente',
    icon: Users,
    color: 'green',
    features: [
      { name: 'avg_transaction_amount', type: 'float', description: 'Média histórica de valores do cliente', example: '1200.00', impact: 'ALTO', riskDirection: 'varies', explanation: 'Seu "normal". Transação muito acima ou abaixo da média é suspeita.' },
      { name: 'std_transaction_amount', type: 'float', description: 'Desvio padrão dos valores', example: '500.00', impact: 'MÉDIO', riskDirection: 'varies', explanation: 'Se você sempre gasta valores parecidos, um valor muito diferente é estranho.' },
      { name: 'amount_deviation', type: 'float', description: 'Quantos desvios padrão do normal', example: '3.5', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Valor > 3 desvios = muito anormal. É como uma nota de prova muito diferente das outras.' },
      { name: 'days_since_last_tx', type: 'int', description: 'Dias desde a última transação', example: '45', impact: 'MÉDIO', riskDirection: 'up', explanation: 'Conta inativa há 45 dias que de repente movimenta milhares é suspeita.' },
      { name: 'account_age_days', type: 'int', description: 'Idade da conta em dias', example: '30', impact: 'ALTO', riskDirection: 'down', explanation: 'Contas muito novas (< 30 dias) são frequentemente usadas para fraude (contas laranja).' }
    ]
  },
  recipient: {
    category: 'Destinatário',
    icon: Target,
    color: 'red',
    features: [
      { name: 'recipient_is_new', type: 'bool', description: 'Se é a primeira transação para este destinatário', example: 'true', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Você nunca enviou dinheiro para esta pessoa. Em golpes, o destino é sempre novo.' },
      { name: 'recipient_risk_score', type: 'float', description: 'Score de risco do recebedor (0-100)', example: '85', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Se a conta de destino já recebeu dinheiro de outras fraudes, ela é de alto risco.' },
      { name: 'recipient_account_age', type: 'int', description: 'Idade da conta do destinatário (dias)', example: '7', impact: 'ALTO', riskDirection: 'down', explanation: 'Conta laranja típica tem menos de 30 dias. Abrem, usam para fraude, e abandonam.' },
      { name: 'recipient_tx_count', type: 'int', description: 'Quantas transações o destinatário já recebeu', example: '3', impact: 'ALTO', riskDirection: 'down', explanation: 'Conta que só recebe (nunca envia) e tem poucas transações é suspeita.' },
      { name: 'is_known_merchant', type: 'bool', description: 'Se o destinatário é um comerciante conhecido', example: 'false', impact: 'MÉDIO', riskDirection: 'down', explanation: 'Pagamentos para Uber, iFood, Netflix são seguros. Para CPFs desconhecidos, menos.' }
    ]
  },
  device: {
    category: 'Dispositivo e Localização',
    icon: Globe,
    color: 'purple',
    features: [
      { name: 'device_is_new', type: 'bool', description: 'Se o dispositivo nunca foi usado antes', example: 'true', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Você sempre usa o mesmo celular. Se de repente usa outro, pode ser fraudador com sua senha.' },
      { name: 'device_fingerprint_match', type: 'float', description: 'Similaridade com devices anteriores (0-1)', example: '0.2', impact: 'ALTO', riskDirection: 'down', explanation: 'Compara características do aparelho. Mesmo que troque de celular, há padrões.' },
      { name: 'ip_is_vpn', type: 'bool', description: 'Se o IP é de uma VPN conhecida', example: 'true', impact: 'ALTO', riskDirection: 'up', explanation: 'VPNs escondem localização real. Fraudadores usam para parecer que estão em outro lugar.' },
      { name: 'geolocation_distance_km', type: 'float', description: 'Distância da localização habitual (km)', example: '2500', impact: 'ALTO', riskDirection: 'up', explanation: 'Se você está em SP e a transação vem do Nordeste, como você viajou 2500km em 1 hora?' },
      { name: 'location_risk_score', type: 'float', description: 'Score de risco da região', example: '75', impact: 'MÉDIO', riskDirection: 'up', explanation: 'Algumas regiões têm mais fraudes que outras. Dado estatístico, não preconceito.' }
    ]
  },
  temporal: {
    category: 'Padrões Temporais',
    icon: Clock,
    color: 'indigo',
    features: [
      { name: 'is_night_transaction', type: 'bool', description: 'Se é entre 00h e 06h', example: 'true', impact: 'ALTO', riskDirection: 'up', explanation: 'Madrugada é horário preferido de fraudadores. Menos supervisão, vítima dormindo.' },
      { name: 'is_rush_hour', type: 'bool', description: 'Se é horário de pico (7-9h, 17-19h)', example: 'false', impact: 'BAIXO', riskDirection: 'down', explanation: 'Transações em horário comercial são mais normais.' },
      { name: 'time_since_account_login', type: 'int', description: 'Minutos desde o último login', example: '2', impact: 'MÉDIO', riskDirection: 'down', explanation: 'Transação logo após login longo pode ser sessão hackeada.' },
      { name: 'usual_hour_deviation', type: 'float', description: 'Desvio do horário habitual', example: '8.5', impact: 'ALTO', riskDirection: 'up', explanation: 'Se você sempre faz transações às 14h e agora faz às 3h, mudança suspeita.' },
      { name: 'days_since_password_change', type: 'int', description: 'Dias desde a última troca de senha', example: '1', impact: 'ALTO', riskDirection: 'down', explanation: 'Transação grande logo após troca de senha pode indicar que fraudador trocou.' }
    ]
  },
  network: {
    category: 'Análise de Rede',
    icon: Network,
    color: 'teal',
    features: [
      { name: 'shared_device_count', type: 'int', description: 'Quantas contas usam o mesmo device', example: '5', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Se 5 CPFs diferentes usam o mesmo celular, provavelmente é uma quadrilha.' },
      { name: 'shared_ip_count', type: 'int', description: 'Quantas contas usam o mesmo IP', example: '8', impact: 'ALTO', riskDirection: 'up', explanation: 'Múltiplas contas no mesmo IP = possível operação de fraude.' },
      { name: 'network_fraud_rate', type: 'float', description: 'Taxa de fraude na rede do cliente', example: '0.15', impact: 'ALTO', riskDirection: 'up', explanation: 'Se pessoas conectadas a você cometeram fraude, você é mais suspeito.' },
      { name: 'degree_centrality', type: 'float', description: 'Centralidade na rede de transações', example: '0.85', impact: 'MÉDIO', riskDirection: 'varies', explanation: 'Contas muito conectadas podem ser laranjas distribuindo dinheiro.' },
      { name: 'community_risk', type: 'float', description: 'Risco médio da comunidade de transações', example: '0.4', impact: 'ALTO', riskDirection: 'up', explanation: 'Se você transaciona com pessoas de alto risco, seu risco aumenta.' }
    ]
  },
  mlDerived: {
    category: 'Features Derivadas por IA',
    icon: Brain,
    color: 'pink',
    features: [
      { name: 'anomaly_score', type: 'float', description: 'Score de anomalia (0-1)', example: '0.92', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Quão diferente esta transação é de todas as outras. 0.92 = muito anormal.' },
      { name: 'cluster_distance', type: 'float', description: 'Distância do centróide do cluster', example: '3.5', impact: 'ALTO', riskDirection: 'up', explanation: 'Transações são agrupadas por similaridade. Estar longe do grupo = diferente.' },
      { name: 'fraud_probability', type: 'float', description: 'Probabilidade de fraude (0-1)', example: '0.87', impact: 'CRÍTICO', riskDirection: 'up', explanation: 'Resultado final do modelo. 0.87 = 87% de chance de ser fraude.' },
      { name: 'ensemble_agreement', type: 'float', description: 'Concordância entre os 3 modelos', example: '0.95', impact: 'ALTO', riskDirection: 'varies', explanation: 'Se os 3 modelos concordam, confiamos mais. Discordância = incerteza.' },
      { name: 'confidence_score', type: 'float', description: 'Confiança da predição', example: '0.88', impact: 'MÉDIO', riskDirection: 'varies', explanation: 'O quão certo o modelo está. Baixa confiança = revisar manualmente.' }
    ]
  }
};

const datasets = {
  kaggle: {
    name: 'Credit Card Fraud Detection (Kaggle)',
    source: 'Kaggle - Machine Learning Group ULB',
    records: '284.807 transações',
    fraudRate: '0.172% (492 fraudes)',
    icon: Database,
    color: 'blue',
    description: 'Dataset público mais famoso para detecção de fraudes. Contém transações de cartão de crédito de setembro de 2013 por titulares europeus.',
    features: [
      { name: 'V1-V28', description: 'Features transformadas por PCA (Principal Component Analysis) para proteger privacidade' },
      { name: 'Time', description: 'Segundos decorridos entre esta e a primeira transação do dataset' },
      { name: 'Amount', description: 'Valor da transação' },
      { name: 'Class', description: 'Label: 0 = legítima, 1 = fraude' }
    ],
    preprocessing: [
      'Normalização do Amount usando StandardScaler',
      'SMOTE para balanceamento (oversampling da classe minoritária)',
      'Split 70/15/15 (treino/validação/teste)',
      'Remoção de outliers extremos (> 5 desvios padrão)'
    ],
    limitations: [
      'Features V1-V28 são anônimas (não sabemos o significado)',
      'Dados de 2013 (padrões de fraude evoluíram)',
      'Apenas cartão de crédito (sem PIX, TED)',
      'Contexto europeu (diferente do Brasil)'
    ],
    usage: 'Pré-treinamento inicial dos modelos. Fornece base estatística robusta.'
  },
  production: {
    name: 'Transações de Produção (PostgreSQL)',
    source: 'Sistema Sankofa - Dados Reais',
    records: '4.467 transações',
    fraudRate: '69.73% (3.115 fraudes)',
    icon: Server,
    color: 'green',
    description: 'Dados reais do sistema em produção. Inclui PIX, TED, cartões e boletos com contexto brasileiro.',
    features: [
      { name: 'transaction_id', description: 'ID único da transação (UUID)' },
      { name: 'amount', description: 'Valor em reais (R$)' },
      { name: 'channel', description: 'PIX, TED, CREDIT_CARD, DEBIT_CARD, BOLETO' },
      { name: 'risk_score', description: 'Score calculado pelo modelo (0-100)' },
      { name: 'is_fraud', description: 'Se foi confirmada como fraude' },
      { name: 'created_at', description: 'Timestamp da transação' },
      { name: 'customer_id', description: 'ID do cliente (mascarado por LGPD)' }
    ],
    distribution: [
      { channel: 'PIX', count: 4285, frauds: 3081, fraudRate: '71.9%' },
      { channel: 'TED', count: 86, frauds: 14, fraudRate: '16.3%' },
      { channel: 'BOLETO', count: 88, frauds: 14, fraudRate: '15.9%' },
      { channel: 'CARTÃO', count: 8, frauds: 6, fraudRate: '75.0%' }
    ],
    preprocessing: [
      'Mascaramento de CPF (LGPD)',
      'Hash de device_id',
      'Anonimização de geolocalização',
      'Retenção de 90 dias (compliance)'
    ],
    usage: 'Fine-tuning dos modelos para contexto brasileiro. Validação de performance real.'
  },
  feedback: {
    name: 'Feedback dos Analistas',
    source: 'Human-in-the-Loop',
    records: '~50 feedbacks/dia',
    icon: MessageSquare,
    color: 'purple',
    description: 'Correções e confirmações feitas por analistas humanos. Fundamental para aprendizado contínuo.',
    fields: [
      { name: 'transaction_id', description: 'ID da transação avaliada' },
      { name: 'analyst_decision', description: 'FRAUD, LEGITIMATE, NEEDS_REVIEW' },
      { name: 'confidence', description: 'Confiança do analista (1-5)' },
      { name: 'reasoning', description: 'Texto explicando a decisão' },
      { name: 'analyst_id', description: 'ID do analista' },
      { name: 'timestamp', description: 'Quando o feedback foi dado' }
    ],
    workflow: [
      'Modelo faz predição inicial',
      'Analista revisa e confirma/corrige',
      'Feedback é armazenado no banco',
      'Diariamente: batch de retraining',
      'Modelo é recalibrado com novos dados'
    ],
    importance: [
      'Captura novos padrões de fraude',
      'Corrige vieses do modelo',
      'Adapta a mudanças no comportamento',
      'Mantém modelo atualizado'
    ],
    usage: 'Continuous Learning. Modelo evolui com feedback humano.'
  }
};

const transferLearning = {
  phases: [
    {
      phase: 1,
      name: 'Pré-Treinamento',
      icon: Database,
      color: 'blue',
      duration: '~2 horas',
      description: 'Modelo aprende padrões básicos de fraude com dataset Kaggle',
      details: [
        'Carrega 284.807 transações do Kaggle',
        'Treina Random Forest com 100 árvores',
        'Treina Gradient Boosting com 100 estimadores',
        'Treina CatBoost com 500 iterações',
        'Valida com 15% dos dados reservados',
        'AUC-ROC > 0.95 nesta fase'
      ],
      metrics: { accuracy: '99.2%', auc: '0.97', precision: '85%', recall: '82%' }
    },
    {
      phase: 2,
      name: 'Adaptação de Domínio',
      icon: RefreshCw,
      color: 'green',
      duration: '~30 minutos',
      description: 'Modelo se adapta ao contexto brasileiro e tipos de transação locais',
      details: [
        'Fine-tune com dados de produção (4.467 transações)',
        'Adiciona features específicas de PIX',
        'Aprende padrões de TED brasileira',
        'Ajusta pesos para horários locais (fuso)',
        'Calibra thresholds para falsos positivos aceitáveis',
        'Valida com transações recentes'
      ],
      metrics: { accuracy: '97.8%', auc: '0.94', precision: '92%', recall: '89%' }
    },
    {
      phase: 3,
      name: 'Ensemble Voting',
      icon: Layers,
      color: 'purple',
      duration: 'Tempo real',
      description: 'Os 3 modelos votam juntos para decisão final mais robusta',
      details: [
        'Random Forest dá probabilidade P1',
        'Gradient Boosting dá probabilidade P2',
        'CatBoost dá probabilidade P3',
        'Score final = média ponderada (RF: 0.3, GB: 0.35, CB: 0.35)',
        'Pesos definidos por performance no validation set',
        'Discordância alta = encaminha para análise manual'
      ],
      metrics: { accuracy: '98.5%', auc: '0.96', precision: '94%', recall: '91%' }
    },
    {
      phase: 4,
      name: 'Aprendizado Contínuo',
      icon: Brain,
      color: 'pink',
      duration: 'Contínuo',
      description: 'Modelo evolui diariamente com feedback dos analistas',
      details: [
        'Coleta ~50 feedbacks/dia dos analistas',
        'Batch retraining às 04:00 (baixo volume)',
        'Incremental learning: não descarta conhecimento anterior',
        'Detecta concept drift (mudança de padrões)',
        'Alerta se performance degradar',
        'Rollback automático se nova versão piorar'
      ],
      metrics: { accuracy: '99.1%', auc: '0.97', precision: '95%', recall: '93%' }
    }
  ],
  models: {
    randomForest: {
      name: 'Random Forest',
      icon: GitBranch,
      weight: 0.30,
      strengths: ['Robusto a outliers', 'Interpreta importância de features', 'Paralelo = rápido'],
      weaknesses: ['Menos preciso em dados novos', 'Pode overfit em datasets pequenos'],
      hyperparameters: {
        n_estimators: 100,
        max_depth: 15,
        min_samples_split: 5,
        min_samples_leaf: 2,
        class_weight: 'balanced'
      }
    },
    gradientBoosting: {
      name: 'Gradient Boosting',
      icon: TrendingUp,
      weight: 0.35,
      strengths: ['Excelente em dados tabulares', 'Captura relações complexas', 'Bom com classes desbalanceadas'],
      weaknesses: ['Mais lento que RF', 'Sensível a outliers'],
      hyperparameters: {
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 6,
        subsample: 0.8,
        colsample_bytree: 0.8
      }
    },
    catboost: {
      name: 'CatBoost',
      icon: Cpu,
      weight: 0.35,
      strengths: ['Lida bem com categóricas', 'Pouco overfitting', 'State-of-the-art em fraude'],
      weaknesses: ['Treinamento mais demorado', 'Mais complexo de tunar'],
      hyperparameters: {
        iterations: 500,
        learning_rate: 0.05,
        depth: 8,
        l2_leaf_reg: 3,
        auto_class_weights: 'Balanced'
      }
    }
  }
};

const compliance = {
  lgpd: {
    name: 'LGPD - Lei Geral de Proteção de Dados',
    icon: Shield,
    color: 'blue',
    law: 'Lei 13.709/2018',
    articles: [
      { number: 'Art. 6º', title: 'Princípios', description: 'Finalidade, adequação, necessidade, livre acesso, qualidade, transparência, segurança, prevenção.' },
      { number: 'Art. 7º', title: 'Bases Legais', description: 'Consentimento, execução de contrato, legítimo interesse, proteção ao crédito.' },
      { number: 'Art. 18', title: 'Direitos do Titular', description: 'Confirmação, acesso, correção, portabilidade, eliminação, revogação.' },
      { number: 'Art. 46', title: 'Segurança', description: 'Medidas técnicas e administrativas para proteger dados de acessos não autorizados.' }
    ],
    implementation: [
      { feature: 'Mascaramento de CPF', status: '✅ Implementado', details: 'CPF exibido como ***.***. XXX-XX' },
      { feature: 'Hash de Device ID', status: '✅ Implementado', details: 'SHA-256 irreversível' },
      { feature: 'Anonimização de IP', status: '✅ Implementado', details: 'Últimos octetos removidos' },
      { feature: 'Retenção de 90 dias', status: '✅ Implementado', details: 'Auto-purge de dados antigos' },
      { feature: 'Audit Trail', status: '✅ Implementado', details: 'Todas as ações são logadas' },
      { feature: 'Direito ao Esquecimento', status: '✅ Implementado', details: 'Endpoint para exclusão de dados' }
    ]
  },
  bacen: {
    name: 'BACEN - Banco Central do Brasil',
    icon: Building,
    color: 'green',
    regulations: [
      { number: 'Res. 4.893/2021', title: 'Política de Segurança Cibernética', description: 'Requisitos mínimos para segurança em instituições financeiras.' },
      { number: 'Circ. 3.978/2020', title: 'Prevenção à Lavagem de Dinheiro', description: 'Procedimentos para detecção e comunicação de operações suspeitas.' },
      { number: 'Res. BCB 1/2020', title: 'PIX - Regulamento', description: 'SLA de 10 segundos para liquidação, fraude zero tolerance.' }
    ],
    requirements: [
      { requirement: 'SLA PIX < 10s', status: '✅ Cumprido', details: 'Latência atual: 37-50ms' },
      { requirement: 'STR em 24h', status: '✅ Cumprido', details: 'Sistema gera STR automático' },
      { requirement: 'Audit Trail', status: '✅ Cumprido', details: '38 registros ativos' },
      { requirement: 'Treinamento Equipe', status: '✅ Cumprido', details: 'Certificação anual' },
      { requirement: 'Incidentes < 24h', status: '✅ Cumprido', details: 'Notificação automática' }
    ]
  },
  pciDss: {
    name: 'PCI DSS - Payment Card Industry Data Security Standard',
    icon: Lock,
    color: 'purple',
    version: 'v4.0',
    requirements: [
      { number: 'Req. 3', title: 'Proteger Dados do Titular', status: '✅', details: 'Dados de cartão nunca são armazenados em claro.' },
      { number: 'Req. 4', title: 'Criptografar Transmissão', status: '✅', details: 'TLS 1.3 em todas as comunicações.' },
      { number: 'Req. 6', title: 'Desenvolver com Segurança', status: '✅', details: 'SAST/DAST em CI/CD.' },
      { number: 'Req. 7', title: 'Acesso Need-to-Know', status: '✅', details: 'RBAC com 5 roles.' },
      { number: 'Req. 8', title: 'Autenticação Forte', status: '✅', details: 'JWT + 2FA.' },
      { number: 'Req. 10', title: 'Monitorar Acessos', status: '✅', details: 'Logs de todas as ações.' }
    ]
  }
};

function PersonaCard({ name, role, avatar, department, experience, quote, color, responsibilities, kpis }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    orange: 'bg-orange-500'
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
      <div className={`${colorClasses[color]} text-white p-4`}>
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center text-2xl font-bold">
            {avatar}
          </div>
          <div>
            <h3 className="text-xl font-bold">{name}</h3>
            <p className="text-white/80">{role}</p>
          </div>
        </div>
      </div>
      <div className="p-4 space-y-3">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Building className="h-4 w-4" />
          {department}
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Clock className="h-4 w-4" />
          {experience}
        </div>
        <blockquote className="italic text-gray-500 border-l-4 border-gray-200 pl-3 py-1">
          "{quote}"
        </blockquote>
        {responsibilities && (
          <div className="mt-4">
            <h4 className="font-semibold text-gray-700 text-sm mb-2">Responsabilidades:</h4>
            <ul className="text-xs text-gray-600 space-y-1">
              {responsibilities.slice(0, 3).map((r, i) => (
                <li key={i} className="flex items-center gap-1">
                  <CheckCircle className="h-3 w-3 text-green-500" /> {r}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const styles = {
    info: { bg: 'bg-blue-50', border: 'border-blue-200', icon: Info, iconColor: 'text-blue-600', titleColor: 'text-blue-800' },
    success: { bg: 'bg-green-50', border: 'border-green-200', icon: CheckCircle, iconColor: 'text-green-600', titleColor: 'text-green-800' },
    warning: { bg: 'bg-yellow-50', border: 'border-yellow-200', icon: AlertTriangle, iconColor: 'text-yellow-600', titleColor: 'text-yellow-800' },
    danger: { bg: 'bg-red-50', border: 'border-red-200', icon: XCircle, iconColor: 'text-red-600', titleColor: 'text-red-800' },
    tip: { bg: 'bg-purple-50', border: 'border-purple-200', icon: Lightbulb, iconColor: 'text-purple-600', titleColor: 'text-purple-800' }
  };
  const { bg, border, icon: Icon, iconColor, titleColor } = styles[type];

  return (
    <div className={`${bg} ${border} border rounded-xl p-4`}>
      <div className="flex items-start gap-3">
        <Icon className={`h-5 w-5 ${iconColor} mt-0.5`} />
        <div>
          <h4 className={`font-semibold ${titleColor}`}>{title}</h4>
          <div className="mt-1 text-sm text-gray-700">{children}</div>
        </div>
      </div>
    </div>
  );
}

function Checklist({ items }) {
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50">
          {item.done ? (
            <CheckCircle className="h-5 w-5 text-green-500" />
          ) : (
            <div className="h-5 w-5 border-2 border-gray-300 rounded-full" />
          )}
          <span className={item.done ? 'text-gray-700' : 'text-gray-500'}>{item.text}</span>
        </div>
      ))}
    </div>
  );
}

function KPICard({ title, value, change, changeType, icon: Icon, color }) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    red: 'bg-red-50 text-red-600',
    purple: 'bg-purple-50 text-purple-600',
    orange: 'bg-orange-50 text-orange-600'
  };

  return (
    <div className="bg-white rounded-xl shadow p-4 border">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-500">{title}</span>
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <div className="mt-2">
        <span className="text-2xl font-bold text-gray-900">{value}</span>
        {change && (
          <span className={`ml-2 text-sm ${changeType === 'up' ? 'text-green-600' : 'text-red-600'}`}>
            {change}
          </span>
        )}
      </div>
    </div>
  );
}

function StepByStep({ title, steps }) {
  return (
    <div className="bg-gray-50 rounded-xl p-6">
      <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
        <List className="h-5 w-5 text-blue-600" />
        {title}
      </h4>
      <div className="space-y-4">
        {steps.map((step, i) => (
          <div key={i} className="flex gap-4">
            <div className="flex flex-col items-center">
              <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-sm">
                {i + 1}
              </div>
              {i < steps.length - 1 && <div className="w-0.5 h-full bg-blue-200 mt-2" />}
            </div>
            <div className="flex-1 pb-6">
              <h5 className="font-semibold text-gray-900">{step.action}</h5>
              <p className="text-sm text-gray-600 mt-1">{step.details}</p>
              {step.tip && (
                <p className="text-xs text-purple-600 mt-2 flex items-center gap-1">
                  <Lightbulb className="h-3 w-3" /> Dica: {step.tip}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureCard({ feature, category }) {
  const riskColors = {
    up: 'text-red-600 bg-red-50',
    down: 'text-green-600 bg-green-50',
    varies: 'text-yellow-600 bg-yellow-50',
    neutral: 'text-gray-600 bg-gray-50'
  };

  const impactColors = {
    'CRÍTICO': 'bg-red-500 text-white',
    'ALTO': 'bg-orange-500 text-white',
    'MÉDIO': 'bg-yellow-500 text-white',
    'BAIXO': 'bg-gray-400 text-white'
  };

  return (
    <div className="bg-white rounded-lg border p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div>
          <code className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">{feature.name}</code>
          <span className={`ml-2 text-xs px-2 py-1 rounded ${impactColors[feature.impact]}`}>{feature.impact}</span>
        </div>
        <div className={`px-2 py-1 rounded text-xs ${riskColors[feature.riskDirection]}`}>
          {feature.riskDirection === 'up' ? '↑ Aumenta risco' : feature.riskDirection === 'down' ? '↓ Diminui risco' : '↔ Varia'}
        </div>
      </div>
      <p className="text-sm text-gray-600 mb-2">{feature.description}</p>
      <p className="text-xs text-gray-500">Exemplo: <code className="bg-gray-100 px-1 rounded">{feature.example}</code></p>
      <div className="mt-3 p-3 bg-blue-50 rounded-lg">
        <p className="text-xs text-blue-800"><strong>Explicação:</strong> {feature.explanation}</p>
      </div>
    </div>
  );
}

function DatasetCard({ dataset }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500'
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className={`${colorClasses[dataset.color]} text-white p-4`}>
        <div className="flex items-center gap-3">
          <dataset.icon className="h-8 w-8" />
          <div>
            <h3 className="font-bold text-lg">{dataset.name}</h3>
            <p className="text-sm text-white/80">{dataset.source}</p>
          </div>
        </div>
      </div>
      <div className="p-4 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xl font-bold text-gray-900">{dataset.records}</div>
            <div className="text-xs text-gray-500">Registros</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <div className="text-xl font-bold text-red-600">{dataset.fraudRate}</div>
            <div className="text-xs text-gray-500">Taxa de Fraude</div>
          </div>
        </div>
        <p className="text-sm text-gray-600">{dataset.description}</p>
      </div>
    </div>
  );
}

function TransferLearningPhase({ phase }) {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
    pink: 'bg-pink-500'
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className={`${colorClasses[phase.color]} text-white p-4`}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center font-bold">
            {phase.phase}
          </div>
          <div>
            <h3 className="font-bold">{phase.name}</h3>
            <p className="text-sm text-white/80">Duração: {phase.duration}</p>
          </div>
        </div>
      </div>
      <div className="p-4 space-y-4">
        <p className="text-sm text-gray-700">{phase.description}</p>
        <ul className="text-xs text-gray-600 space-y-1">
          {phase.details.map((d, i) => (
            <li key={i} className="flex items-start gap-2">
              <ArrowRight className="h-3 w-3 text-gray-400 mt-1" />
              {d}
            </li>
          ))}
        </ul>
        <div className="grid grid-cols-4 gap-2 mt-4">
          <div className="bg-gray-50 rounded p-2 text-center">
            <div className="text-sm font-bold text-gray-900">{phase.metrics.accuracy}</div>
            <div className="text-[10px] text-gray-500">Accuracy</div>
          </div>
          <div className="bg-gray-50 rounded p-2 text-center">
            <div className="text-sm font-bold text-gray-900">{phase.metrics.auc}</div>
            <div className="text-[10px] text-gray-500">AUC-ROC</div>
          </div>
          <div className="bg-gray-50 rounded p-2 text-center">
            <div className="text-sm font-bold text-gray-900">{phase.metrics.precision}</div>
            <div className="text-[10px] text-gray-500">Precision</div>
          </div>
          <div className="bg-gray-50 rounded p-2 text-center">
            <div className="text-sm font-bold text-gray-900">{phase.metrics.recall}</div>
            <div className="text-[10px] text-gray-500">Recall</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ManualSection({ id, title, icon: Icon, children, defaultOpen = false, priority }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  const priorityStyles = {
    critical: 'border-l-4 border-l-red-500',
    high: 'border-l-4 border-l-orange-500',
    default: ''
  };

  return (
    <div id={id} className={`bg-white rounded-xl shadow-lg overflow-hidden ${priorityStyles[priority] || ''}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="h-6 w-6 text-blue-600" />}
          <h2 className="text-lg font-bold text-gray-900">{title}</h2>
        </div>
        {isOpen ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
      </button>
      {isOpen && <div className="p-6 border-t">{children}</div>}
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
    { id: 'dia-a-dia', label: 'Dia a Dia', icon: Calendar },
    { id: 'features', label: 'Features de IA', icon: Brain },
    { id: 'datasets', label: 'DataSets', icon: Database },
    { id: 'transfer-learning', label: 'Transfer Learning', icon: Layers },
    { id: 'telas', label: 'Todas as Telas', icon: Grid },
    { id: 'compliance', label: 'Compliance', icon: Shield },
    { id: 'cenarios', label: 'Cenários Reais', icon: Target },
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
            <p className="text-lg text-blue-100">Sistema de Detecção de Fraudes Bancárias - Guia Definitivo Ultra-Detalhado</p>
          </div>
        </div>
        <p className="text-blue-100 max-w-4xl">
          Este é o manual mais completo e didático para analistas de fraude. Contém TODAS as features de IA explicadas,
          TODOS os datasets utilizados no treinamento, explicação completa de Transfer Learning, todas as 16 telas,
          cenários reais, compliance LGPD/BACEN/PCI DSS, e muito mais. Use como sua referência diária.
        </p>
        <div className="flex items-center gap-6 mt-6 text-sm flex-wrap">
          <span className="flex items-center gap-2"><Clock className="h-4 w-4" /> Atualizado: 30/11/2025</span>
          <span className="flex items-center gap-2"><Users className="h-4 w-4" /> Para: Analistas de Fraude</span>
          <span className="flex items-center gap-2"><Shield className="h-4 w-4" /> Compliance: LGPD/BACEN/PCI DSS</span>
          <span className="flex items-center gap-2"><Brain className="h-4 w-4" /> 40+ Features de IA</span>
          <span className="flex items-center gap-2"><Database className="h-4 w-4" /> 3 DataSets Explicados</span>
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

      {/* ========== ABA: INÍCIO ========== */}
      {activeTab === 'inicio' && (
        <>
          <ManualSection id="bem-vindo" title="🎓 Bem-Vindo ao Sankofa - Seu Guia Completo" icon={GraduationCap} defaultOpen={true}>
            <div className="space-y-6">
              <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl p-6">
                <h3 className="text-2xl font-bold mb-2">O que você vai aprender neste manual?</h3>
                <p className="text-green-100">
                  Este manual foi criado para ser extremamente didático e completo. Independente do seu nível de experiência,
                  você encontrará explicações claras para TUDO que precisa saber sobre detecção de fraudes bancárias.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-xl p-6 text-center">
                  <Brain className="h-12 w-12 text-blue-600 mx-auto mb-3" />
                  <h4 className="font-bold text-gray-900">40+ Features de IA</h4>
                  <p className="text-sm text-gray-600 mt-2">
                    Todas as características que a Inteligência Artificial analisa em cada transação, explicadas de forma simples.
                  </p>
                </div>
                <div className="bg-green-50 rounded-xl p-6 text-center">
                  <Database className="h-12 w-12 text-green-600 mx-auto mb-3" />
                  <h4 className="font-bold text-gray-900">3 DataSets Completos</h4>
                  <p className="text-sm text-gray-600 mt-2">
                    Kaggle (284k transações), Produção (4.467), e Feedback dos analistas. Saiba como cada um treina o modelo.
                  </p>
                </div>
                <div className="bg-purple-50 rounded-xl p-6 text-center">
                  <Layers className="h-12 w-12 text-purple-600 mx-auto mb-3" />
                  <h4 className="font-bold text-gray-900">Transfer Learning</h4>
                  <p className="text-sm text-gray-600 mt-2">
                    Como o modelo aprende com dados públicos e depois se adapta ao Brasil. Explicação passo a passo.
                  </p>
                </div>
              </div>
              
              <AlertBox type="info" title="Como usar este manual">
                <p>
                  Use as abas acima para navegar entre as seções. Cada seção é independente - você pode começar por onde
                  preferir. Recomendamos começar pelo "Dia a Dia" se você é iniciante, ou ir direto para "Features de IA" 
                  se quer entender o motor de Machine Learning.
                </p>
              </AlertBox>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <KPICard title="Transações Processadas" value="4.467" icon={Zap} color="blue" />
                <KPICard title="Fraudes Detectadas" value="3.115" icon={Shield} color="red" />
                <KPICard title="Latência Média" value="37ms" icon={Clock} color="green" />
                <KPICard title="Modelos de IA" value="3" icon={Brain} color="purple" />
              </div>
            </div>
          </ManualSection>

          <ManualSection id="personas" title="👥 Conheça Nossa Equipe de Especialistas" icon={Users}>
            <p className="text-gray-600 mb-6">
              Acompanhe as histórias de 5 profissionais reais ao longo deste manual. 
              Cada um tem experiência e perspectiva diferente - você vai aprender com todos eles.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <PersonaCard {...personas.anaPaula} />
              <PersonaCard {...personas.carlosRoberto} />
              <PersonaCard {...personas.marinaFernandes} />
              <PersonaCard {...personas.rodrigoMendes} />
              <PersonaCard {...personas.patriciaLima} />
            </div>
          </ManualSection>

          <ManualSection id="como-funciona" title="🧠 Como o Sankofa Funciona - Visão Geral" icon={Cpu}>
            <div className="space-y-6">
              <AlertBox type="tip" title="Resumo Executivo">
                O Sankofa analisa CADA transação bancária em menos de 50 milissegundos (0.05 segundos!) usando 3 modelos 
                de Inteligência Artificial que trabalham juntos. Ele examina mais de 40 características de cada transação 
                para decidir se é legítima ou suspeita.
              </AlertBox>
              
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <h4 className="font-bold mb-4 flex items-center gap-2">
                  <Workflow className="h-5 w-5" /> Fluxo de uma Transação (Passo a Passo)
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-center">
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">📱</div>
                    <div className="font-semibold">1. Cliente</div>
                    <div className="text-xs text-gray-400">Faz PIX no app</div>
                    <div className="text-[10px] text-green-400 mt-1">0ms</div>
                  </div>
                  <div className="flex items-center justify-center">
                    <ArrowRight className="h-6 w-6 text-gray-500" />
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">🔍</div>
                    <div className="font-semibold">2. Features</div>
                    <div className="text-xs text-gray-400">40+ características</div>
                    <div className="text-[10px] text-yellow-400 mt-1">5ms</div>
                  </div>
                  <div className="flex items-center justify-center">
                    <ArrowRight className="h-6 w-6 text-gray-500" />
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">🤖</div>
                    <div className="font-semibold">3. IA Analisa</div>
                    <div className="text-xs text-gray-400">3 modelos votam</div>
                    <div className="text-[10px] text-orange-400 mt-1">25ms</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-center mt-4">
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">📊</div>
                    <div className="font-semibold">4. Score</div>
                    <div className="text-xs text-gray-400">0-100 pontos</div>
                    <div className="text-[10px] text-purple-400 mt-1">30ms</div>
                  </div>
                  <div className="flex items-center justify-center">
                    <ArrowRight className="h-6 w-6 text-gray-500" />
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">⚖️</div>
                    <div className="font-semibold">5. Decisão</div>
                    <div className="text-xs text-gray-400">Aprovar/Bloquear</div>
                    <div className="text-[10px] text-cyan-400 mt-1">35ms</div>
                  </div>
                  <div className="flex items-center justify-center">
                    <ArrowRight className="h-6 w-6 text-gray-500" />
                  </div>
                  <div className="bg-green-800 rounded-lg p-4">
                    <div className="text-3xl mb-2">✅</div>
                    <div className="font-semibold">6. Resultado</div>
                    <div className="text-xs text-gray-300">PIX aprovado!</div>
                    <div className="text-[10px] text-green-400 mt-1">37ms total</div>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-blue-50 rounded-xl p-5">
                  <h4 className="font-bold text-blue-800 flex items-center gap-2">
                    <GitBranch className="h-5 w-5" /> Random Forest
                  </h4>
                  <p className="text-sm text-gray-600 mt-2">
                    Imagine 100 especialistas votando. Cada um analisa a transação de um ângulo diferente. 
                    A maioria decide se é fraude ou não. É robusto porque um erro individual não afeta o resultado.
                  </p>
                  <div className="mt-3 text-xs text-blue-600">Peso na decisão: 30%</div>
                </div>
                <div className="bg-green-50 rounded-xl p-5">
                  <h4 className="font-bold text-green-800 flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" /> Gradient Boosting
                  </h4>
                  <p className="text-sm text-gray-600 mt-2">
                    Funciona como um time que aprende com erros. O primeiro modelo faz uma previsão, 
                    o segundo corrige os erros do primeiro, o terceiro corrige os erros do segundo, e assim por diante.
                  </p>
                  <div className="mt-3 text-xs text-green-600">Peso na decisão: 35%</div>
                </div>
                <div className="bg-purple-50 rounded-xl p-5">
                  <h4 className="font-bold text-purple-800 flex items-center gap-2">
                    <Cpu className="h-5 w-5" /> CatBoost
                  </h4>
                  <p className="text-sm text-gray-600 mt-2">
                    O mais moderno dos três. Desenvolvido pelo Yandex (Google russo), é especialmente bom 
                    com dados categóricos como "tipo de transação" ou "dia da semana". State-of-the-art em fraude.
                  </p>
                  <div className="mt-3 text-xs text-purple-600">Peso na decisão: 35%</div>
                </div>
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: DIA A DIA ========== */}
      {activeTab === 'dia-a-dia' && (
        <>
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

          <ManualSection id="turno-0600" title="⏰ 06:00 - Início do Turno e Passagem" icon={Clock}>
            <div className="space-y-6">
              <div className="bg-gray-900 text-white rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="bg-yellow-500 text-black px-3 py-1 rounded-full text-sm font-bold">06:00</div>
                  <span className="text-lg">Carlos chega e assume o turno</span>
                </div>
                <p className="text-gray-300">
                  O turno noturno (Rodrigo) está finalizando. Carlos precisa fazer a passagem de turno 
                  e entender o que aconteceu durante a madrugada antes de começar a trabalhar.
                </p>
              </div>
              
              <StepByStep 
                title="Rotina de Início de Turno"
                steps={[
                  { action: 'Fazer Login no Sistema', details: 'Abrir o Sankofa, inserir CPF e senha. Sistema registra horário de entrada automaticamente.', tip: 'Sempre use autenticação de 2 fatores (SMS ou app).' },
                  { action: 'Ler o Log de Passagem de Turno', details: 'Rodrigo deixou anotações: "2 fraudes confirmadas às 03:00, conta laranja identificada."', tip: 'Tela: Auditoria > Log de Turno' },
                  { action: 'Verificar Dashboard Imediatamente', details: 'Olhar KPIs do turno noturno. Algum spike? Algum modelo offline?', tip: 'Foco nos números vermelhos primeiro.' },
                  { action: 'Checar Fila de Revisão Manual', details: 'Quantas transações estão esperando análise? SLA está sendo cumprido?', tip: 'Idealmente a fila deve ter < 20 itens.' },
                  { action: 'Confirmar Status dos Modelos de IA', details: 'Os 3 modelos (RF, GB, CB) devem estar "Online" e "Healthy".', tip: 'Se algum estiver offline, alerta TI.' }
                ]}
              />
            </div>
          </ManualSection>

          <ManualSection id="turno-resumo" title="📊 Resumo: Todos os Momentos de Atuação do Dia" icon={List}>
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
                    <td className="p-3">Investigação, HOT List</td>
                    <td className="p-3">Mapear contas laranjas, STR BACEN</td>
                    <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">Alta</span></td>
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
          </ManualSection>
        </>
      )}

      {/* ========== ABA: FEATURES DE IA ========== */}
      {activeTab === 'features' && (
        <>
          <ManualSection id="features-intro" title="🧠 Todas as 40+ Features de Inteligência Artificial" icon={Brain} defaultOpen={true}>
            <div className="space-y-6">
              <AlertBox type="info" title="O que são Features?">
                <p>
                  <strong>Features</strong> são as "perguntas" que a IA faz sobre cada transação. Imagine que você é um detetive 
                  investigando uma compra suspeita. Você perguntaria: "Quanto foi?", "Que horas?", "De onde veio?". 
                  A IA faz mais de 40 perguntas dessas, automaticamente, em menos de 1 segundo.
                </p>
              </AlertBox>
              
              <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl p-6">
                <h3 className="text-xl font-bold mb-2">Por que tantas features?</h3>
                <p className="text-purple-100">
                  Fraudadores são inteligentes. Se usássemos só o valor da transação, eles fariam valores normais.
                  Se usássemos só o horário, eles agiriam em horário comercial. Com 40+ features, criamos uma 
                  "impressão digital" única de cada transação que é muito difícil de falsificar.
                </p>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <DollarSign className="h-8 w-8 text-blue-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Dados da Transação</div>
                </div>
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <Zap className="h-8 w-8 text-orange-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Velocidade</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <Users className="h-8 w-8 text-green-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Comportamento</div>
                </div>
                <div className="bg-red-50 rounded-lg p-4 text-center">
                  <Target className="h-8 w-8 text-red-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Destinatário</div>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <Globe className="h-8 w-8 text-purple-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Device/Local</div>
                </div>
                <div className="bg-indigo-50 rounded-lg p-4 text-center">
                  <Clock className="h-8 w-8 text-indigo-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Temporal</div>
                </div>
                <div className="bg-teal-50 rounded-lg p-4 text-center">
                  <Network className="h-8 w-8 text-teal-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Rede</div>
                </div>
                <div className="bg-pink-50 rounded-lg p-4 text-center">
                  <Brain className="h-8 w-8 text-pink-600 mx-auto" />
                  <div className="font-bold text-gray-900 mt-2">5 Features</div>
                  <div className="text-xs text-gray-600">Derivadas IA</div>
                </div>
              </div>
            </div>
          </ManualSection>

          {Object.entries(allFeatures).map(([key, category]) => (
            <ManualSection key={key} id={`features-${key}`} title={`${category.category}`} icon={category.icon}>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {category.features.map((feature, i) => (
                    <FeatureCard key={i} feature={feature} category={category} />
                  ))}
                </div>
              </div>
            </ManualSection>
          ))}
        </>
      )}

      {/* ========== ABA: DATASETS ========== */}
      {activeTab === 'datasets' && (
        <>
          <ManualSection id="datasets-intro" title="📊 Os 3 DataSets que Treinam Nossa IA" icon={Database} defaultOpen={true}>
            <div className="space-y-6">
              <AlertBox type="info" title="O que são DataSets?">
                <p>
                  <strong>DataSets</strong> são conjuntos de dados usados para ensinar a IA. Imagine que você quer ensinar 
                  uma criança a reconhecer gatos. Você mostra milhares de fotos de gatos e "não-gatos". Depois de ver muitos 
                  exemplos, ela aprende. Nossa IA aprende da mesma forma, mas com transações financeiras.
                </p>
              </AlertBox>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <DatasetCard dataset={datasets.kaggle} />
                <DatasetCard dataset={datasets.production} />
                <DatasetCard dataset={datasets.feedback} />
              </div>
            </div>
          </ManualSection>

          <ManualSection id="dataset-kaggle" title="🌐 DataSet Kaggle - Base de Conhecimento Global" icon={Globe}>
            <div className="space-y-6">
              <div className="bg-blue-50 rounded-xl p-6">
                <h4 className="font-bold text-blue-800 mb-3">Por que usamos dados do Kaggle?</h4>
                <p className="text-gray-700">
                  O Kaggle é uma plataforma com competições de Machine Learning. O dataset de fraude de cartão de crédito 
                  é o mais famoso do mundo para este problema. Com 284.807 transações reais (anonimizadas), ele fornece 
                  uma base estatística robusta para o pré-treinamento inicial dos nossos modelos.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-3">Campos do Dataset</h4>
                  <div className="space-y-2">
                    {datasets.kaggle.features.map((f, i) => (
                      <div key={i} className="flex items-start gap-2 p-2 bg-gray-50 rounded">
                        <code className="text-xs bg-blue-100 px-2 py-1 rounded">{f.name}</code>
                        <span className="text-sm text-gray-600">{f.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-bold text-gray-900 mb-3">Pré-processamento</h4>
                  <ul className="space-y-2">
                    {datasets.kaggle.preprocessing.map((p, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                        {p}
                      </li>
                    ))}
                  </ul>
                  
                  <h4 className="font-bold text-gray-900 mb-3 mt-6">Limitações</h4>
                  <ul className="space-y-2">
                    {datasets.kaggle.limitations.map((l, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
                        {l}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </ManualSection>

          <ManualSection id="dataset-producao" title="🏦 DataSet de Produção - Dados Reais Brasileiros" icon={Server}>
            <div className="space-y-6">
              <div className="bg-green-50 rounded-xl p-6">
                <h4 className="font-bold text-green-800 mb-3">Dados Reais do Sistema</h4>
                <p className="text-gray-700">
                  Este é o dataset mais valioso: transações reais processadas pelo Sankofa. Inclui PIX, TED, cartões e boletos 
                  com contexto brasileiro. A IA aprende padrões específicos do nosso país, como horários de pico, 
                  comportamentos típicos de fraudadores brasileiros e características regionais.
                </p>
              </div>
              
              <h4 className="font-bold text-gray-900">Distribuição por Canal</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="p-3 text-left">Canal</th>
                      <th className="p-3 text-left">Transações</th>
                      <th className="p-3 text-left">Fraudes</th>
                      <th className="p-3 text-left">Taxa de Fraude</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.production.distribution.map((d, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="p-3 font-semibold">{d.channel}</td>
                        <td className="p-3">{d.count.toLocaleString()}</td>
                        <td className="p-3 text-red-600">{d.frauds.toLocaleString()}</td>
                        <td className="p-3">
                          <span className={`px-2 py-1 rounded text-xs ${
                            parseFloat(d.fraudRate) > 50 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                          }`}>
                            {d.fraudRate}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: TRANSFER LEARNING ========== */}
      {activeTab === 'transfer-learning' && (
        <>
          <ManualSection id="tl-intro" title="🔄 Transfer Learning - Como a IA Aprende e Evolui" icon={Layers} defaultOpen={true}>
            <div className="space-y-6">
              <AlertBox type="tip" title="O que é Transfer Learning?">
                <p>
                  <strong>Transfer Learning</strong> é como aprender a dirigir carro e depois usar esse conhecimento para 
                  dirigir caminhão. Você não começa do zero - aproveita o que já sabe. Nossa IA faz o mesmo: 
                  primeiro aprende com 284 mil transações do Kaggle, depois se adapta ao Brasil.
                </p>
              </AlertBox>
              
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">As 4 Fases do Aprendizado</h3>
                <div className="grid grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-4xl mb-2">📚</div>
                    <div className="font-semibold">1. Pré-Treino</div>
                    <div className="text-xs text-purple-200">Kaggle (284k tx)</div>
                  </div>
                  <div>
                    <div className="text-4xl mb-2">🇧🇷</div>
                    <div className="font-semibold">2. Adaptação</div>
                    <div className="text-xs text-purple-200">Brasil (4.4k tx)</div>
                  </div>
                  <div>
                    <div className="text-4xl mb-2">🗳️</div>
                    <div className="font-semibold">3. Ensemble</div>
                    <div className="text-xs text-purple-200">3 modelos votam</div>
                  </div>
                  <div>
                    <div className="text-4xl mb-2">🔁</div>
                    <div className="font-semibold">4. Contínuo</div>
                    <div className="text-xs text-purple-200">Feedback humano</div>
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>

          {transferLearning.phases.map((phase, i) => (
            <ManualSection key={i} id={`tl-fase-${phase.phase}`} title={`Fase ${phase.phase}: ${phase.name}`} icon={phase.icon}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <TransferLearningPhase phase={phase} />
                <div className="space-y-4">
                  <h4 className="font-bold text-gray-900">Explicação Detalhada</h4>
                  <p className="text-gray-700">{phase.description}</p>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-semibold text-gray-800 mb-2">O que acontece nesta fase:</h5>
                    <ul className="space-y-2">
                      {phase.details.map((d, j) => (
                        <li key={j} className="flex items-start gap-2 text-sm text-gray-600">
                          <ArrowRight className="h-4 w-4 text-blue-500 mt-0.5" />
                          {d}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </ManualSection>
          ))}

          <ManualSection id="tl-modelos" title="🤖 Os 3 Modelos de Machine Learning" icon={Cpu}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(transferLearning.models).map(([key, model]) => (
                <div key={key} className="bg-white rounded-xl shadow-lg overflow-hidden">
                  <div className="bg-gradient-to-r from-gray-700 to-gray-900 text-white p-4">
                    <model.icon className="h-8 w-8 mb-2" />
                    <h4 className="font-bold text-lg">{model.name}</h4>
                    <div className="text-sm text-gray-300">Peso: {(model.weight * 100).toFixed(0)}%</div>
                  </div>
                  <div className="p-4 space-y-4">
                    <div>
                      <h5 className="font-semibold text-green-700 text-sm mb-2">Pontos Fortes:</h5>
                      <ul className="text-xs space-y-1">
                        {model.strengths.map((s, i) => (
                          <li key={i} className="flex items-start gap-1">
                            <CheckCircle className="h-3 w-3 text-green-500 mt-0.5" /> {s}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-semibold text-orange-700 text-sm mb-2">Limitações:</h5>
                      <ul className="text-xs space-y-1">
                        {model.weaknesses.map((w, i) => (
                          <li key={i} className="flex items-start gap-1">
                            <AlertTriangle className="h-3 w-3 text-orange-500 mt-0.5" /> {w}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: TODAS AS TELAS ========== */}
      {activeTab === 'telas' && (
        <>
          <ManualSection id="telas-mapa" title="🗺️ Mapa de Todas as 16 Telas do Sistema" icon={Grid} defaultOpen={true}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { name: 'Dashboard', icon: BarChart3, desc: 'Visão geral e KPIs', color: 'blue' },
                { name: 'Transações', icon: FileText, desc: 'Busca e lista', color: 'green' },
                { name: 'Alertas', icon: Bell, desc: 'Notificações ativas', color: 'orange' },
                { name: 'Investigação', icon: Search, desc: 'Análise profunda', color: 'purple' },
                { name: 'Revisão Manual', icon: Eye, desc: 'Fila de análise', color: 'red' },
                { name: 'Calibração', icon: Settings, desc: 'Ajustar thresholds', color: 'indigo' },
                { name: 'Métricas', icon: Activity, desc: 'Tempo real', color: 'teal' },
                { name: 'Monitoramento', icon: Gauge, desc: 'Saúde do sistema', color: 'cyan' },
                { name: 'Hard Rules', icon: Shield, desc: 'Regras de negócio', color: 'gray' },
                { name: 'VIP List', icon: Star, desc: 'Whitelist', color: 'yellow' },
                { name: 'HOT List', icon: AlertTriangle, desc: 'Blacklist', color: 'red' },
                { name: 'Feedback', icon: ThumbsUp, desc: 'Treinar IA', color: 'green' },
                { name: 'Relatórios', icon: PieChart, desc: 'Análises e exports', color: 'blue' },
                { name: 'Auditoria', icon: FileText, desc: 'Logs e trail', color: 'gray' },
                { name: 'DataSets', icon: Database, desc: 'Catálogo de dados', color: 'purple' },
                { name: 'Configurações', icon: Settings, desc: 'Preferências', color: 'slate' }
              ].map((tela, i) => (
                <div key={i} className={`bg-${tela.color}-50 rounded-xl p-4 hover:shadow-md transition-shadow cursor-pointer`}>
                  <tela.icon className={`h-8 w-8 text-${tela.color}-600 mb-2`} />
                  <h4 className="font-bold text-gray-900">{tela.name}</h4>
                  <p className="text-xs text-gray-600">{tela.desc}</p>
                </div>
              ))}
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: COMPLIANCE ========== */}
      {activeTab === 'compliance' && (
        <>
          <ManualSection id="compliance-intro" title="🛡️ Compliance: LGPD, BACEN e PCI DSS" icon={Shield} defaultOpen={true}>
            <div className="space-y-6">
              <AlertBox type="warning" title="Por que Compliance é Importante?">
                <p>
                  O Sankofa lida com dados sensíveis de clientes e transações financeiras. Somos obrigados por lei 
                  a proteger esses dados. Três regulamentações principais nos guiam: LGPD (privacidade), BACEN 
                  (regulação bancária) e PCI DSS (cartões de crédito).
                </p>
              </AlertBox>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {Object.entries(compliance).map(([key, reg]) => (
                  <div key={key} className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className={`bg-${reg.color}-500 text-white p-4`}>
                      <reg.icon className="h-8 w-8 mb-2" />
                      <h4 className="font-bold">{reg.name}</h4>
                    </div>
                    <div className="p-4">
                      <p className="text-sm text-gray-600">
                        {key === 'lgpd' && 'Lei brasileira de proteção de dados pessoais.'}
                        {key === 'bacen' && 'Regulamentação do Banco Central do Brasil.'}
                        {key === 'pciDss' && 'Padrão global de segurança para cartões.'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ManualSection>

          <ManualSection id="lgpd-detalhes" title="📋 LGPD - Proteção de Dados Pessoais" icon={Lock}>
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-gray-900 mb-3">Artigos Relevantes</h4>
                  <div className="space-y-2">
                    {compliance.lgpd.articles.map((a, i) => (
                      <div key={i} className="bg-blue-50 rounded-lg p-3">
                        <div className="font-semibold text-blue-800">{a.number}: {a.title}</div>
                        <div className="text-sm text-gray-600">{a.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-bold text-gray-900 mb-3">Implementação no Sankofa</h4>
                  <div className="space-y-2">
                    {compliance.lgpd.implementation.map((impl, i) => (
                      <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span className="text-sm">{impl.feature}</span>
                        <span className="text-xs text-green-600">{impl.status}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: CENÁRIOS REAIS ========== */}
      {activeTab === 'cenarios' && (
        <>
          <ManualSection id="cenarios-intro" title="🎯 Cenários Reais de Fraude" icon={Target} defaultOpen={true}>
            <div className="space-y-6">
              <AlertBox type="info" title="Aprenda com Casos Reais">
                Cada cenário abaixo é baseado em fraudes reais detectadas pelo sistema. Acompanhe o passo a passo 
                de como a IA detectou, como o analista investigou, e qual foi o resultado.
              </AlertBox>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(allScenarios).map(([key, scenario]) => (
                  <div key={key} className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className={`p-4 ${scenario.outcomeType === 'success' ? 'bg-green-500' : 'bg-red-500'} text-white`}>
                      <h4 className="font-bold">{scenario.title}</h4>
                      <div className="flex gap-4 mt-2 text-sm">
                        <span>Dificuldade: {scenario.difficulty}</span>
                        <span>Tempo: {scenario.timeToResolve}</span>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="space-y-2">
                        {scenario.steps.slice(0, 3).map((step, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm">
                            <span className="text-gray-400">{step.time}</span>
                            <span className="text-gray-700">{step.title}</span>
                          </div>
                        ))}
                        <div className="text-sm text-blue-600">... e mais {scenario.steps.length - 3} passos</div>
                      </div>
                      <div className={`mt-4 p-3 rounded-lg ${scenario.outcomeType === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'}`}>
                        <strong>Resultado:</strong> {scenario.outcome}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ManualSection>
        </>
      )}

      {/* ========== ABA: GLOSSÁRIO ========== */}
      {activeTab === 'glossario' && (
        <>
          <ManualSection id="glossario" title="📖 Glossário Completo de Termos" icon={BookMarked} defaultOpen={true}>
            <div className="space-y-4">
              {[
                { term: 'AUC-ROC', definition: 'Área Sob a Curva ROC. Mede a qualidade do modelo. 1.0 = perfeito, 0.5 = aleatório. Nosso modelo tem 0.97.' },
                { term: 'Accuracy', definition: 'Porcentagem de previsões corretas. Se 98 de 100 transações foram classificadas corretamente, accuracy = 98%.' },
                { term: 'CatBoost', definition: 'Algoritmo de Gradient Boosting desenvolvido pelo Yandex, excelente para dados categóricos.' },
                { term: 'Conta Laranja', definition: 'Conta bancária usada para receber dinheiro de fraudes. Geralmente aberta com documentos falsos.' },
                { term: 'Ensemble', definition: 'Técnica que combina múltiplos modelos para uma decisão mais robusta.' },
                { term: 'False Positive', definition: 'Quando o sistema bloqueia uma transação legítima. Causa inconveniente ao cliente.' },
                { term: 'Feature', definition: 'Característica ou atributo de uma transação usado pelo modelo para fazer previsões.' },
                { term: 'Fine-tuning', definition: 'Ajustar um modelo pré-treinado para um novo contexto (ex: adaptar ao Brasil).' },
                { term: 'Gradient Boosting', definition: 'Técnica de ML onde modelos são treinados sequencialmente, cada um corrigindo erros do anterior.' },
                { term: 'Hard Rule', definition: 'Regra de negócio fixa que sempre bloqueia/aprova, independente do score da IA.' },
                { term: 'HOT List', definition: 'Lista negra de CPFs, devices ou IPs que SEMPRE são bloqueados.' },
                { term: 'LGPD', definition: 'Lei Geral de Proteção de Dados. Lei brasileira que protege dados pessoais.' },
                { term: 'Overfitting', definition: 'Quando o modelo decora os dados de treino mas não generaliza para dados novos.' },
                { term: 'PIX', definition: 'Sistema de pagamento instantâneo do Brasil. Funciona 24/7, transferência em segundos.' },
                { term: 'Precision', definition: 'Das transações que o modelo disse serem fraude, quantas realmente eram.' },
                { term: 'Random Forest', definition: 'Algoritmo que usa múltiplas árvores de decisão e vota na resposta final.' },
                { term: 'Recall', definition: 'De todas as fraudes reais, quantas o modelo conseguiu detectar.' },
                { term: 'SLA', definition: 'Service Level Agreement. Acordo de nível de serviço. Ex: latência < 50ms.' },
                { term: 'SMOTE', definition: 'Técnica para balancear datasets criando exemplos sintéticos da classe minoritária.' },
                { term: 'STR', definition: 'Suspicious Transaction Report. Relatório de transação suspeita enviado ao BACEN.' },
                { term: 'Threshold', definition: 'Limite de decisão. Se score > threshold, bloqueia. Se < threshold, aprova.' },
                { term: 'Transfer Learning', definition: 'Usar conhecimento de um problema para resolver outro relacionado.' },
                { term: 'VIP List', definition: 'Lista branca de clientes que NÃO devem ser bloqueados (baixo risco).' }
              ].map((item, i) => (
                <div key={i} className="flex gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100">
                  <div className="font-mono font-bold text-blue-600 min-w-[140px]">{item.term}</div>
                  <div className="text-gray-700">{item.definition}</div>
                </div>
              ))}
            </div>
          </ManualSection>
        </>
      )}
    </div>
  );
}

export default Manual;
