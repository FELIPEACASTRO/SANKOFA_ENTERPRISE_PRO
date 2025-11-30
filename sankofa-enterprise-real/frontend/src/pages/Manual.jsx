import { useState, useRef, useEffect } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Users, Target, Shield, AlertTriangle, Clock, Zap, Eye, Brain, Settings, FileText, BarChart3, Database, Bell, Lock, Star, CheckCircle, XCircle, TrendingUp, Phone, Building, HelpCircle, Search, Filter, Download, Upload, RefreshCw, Play, Pause, Edit, Trash2, Plus, ArrowRight, ArrowLeft, Info, MessageSquare, ThumbsUp, ThumbsDown, Activity, Cpu, Server, Globe, Calendar, DollarSign, Percent, Hash, List, Grid, PieChart, LineChart, Table, Map, Flag, Award, Bookmark, ExternalLink, Copy, Share, Mail, Send, Layers, GitBranch, Box, Terminal, Code, Workflow, Boxes, Network, Gauge, Timer, Sparkles, GraduationCap, Lightbulb, BookMarked, CircuitBoard, Home, ChevronRight, User, Fingerprint, CreditCard, Banknote, Smartphone, MapPin, AlertCircle, ShieldCheck, Scale, Gavel, FileCheck, ClipboardList, Monitor, Headphones, Coffee, Sunrise, Sun, Moon } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';

const personas = {
  ana: {
    name: 'Ana Paula',
    role: 'Lider de Prevencao a Fraudes',
    avatar: 'AP',
    experience: '8 anos',
    color: 'blue',
    intro: 'Ana supervisiona a equipe e toma decisoes criticas sobre fraudes de alto valor.'
  },
  carlos: {
    name: 'Carlos Roberto',
    role: 'Analista Senior',
    avatar: 'CR', 
    experience: '5 anos',
    color: 'green',
    intro: 'Carlos analisa transacoes diariamente e treina novos analistas.'
  },
  marina: {
    name: 'Marina',
    role: 'Compliance Officer',
    avatar: 'MF',
    experience: '10 anos',
    color: 'purple',
    intro: 'Marina garante que tudo esteja em conformidade com LGPD e BACEN.'
  },
  rodrigo: {
    name: 'Rodrigo',
    role: 'Analista Junior',
    avatar: 'RM',
    experience: '1 ano',
    color: 'orange',
    intro: 'Rodrigo esta aprendendo e faz o turno noturno.'
  }
};

const todasAsTelas = [
  {
    id: 'dashboard',
    nome: 'Dashboard',
    icone: BarChart3,
    caminho: 'Menu > Dashboard',
    rota: '/',
    modulo: 'Visao Geral',
    cor: 'blue',
    objetivo: 'Mostrar uma visao geral de tudo que esta acontecendo no sistema em tempo real. E como o painel de um aviao: todos os indicadores importantes em um so lugar.',
    quandoUsar: [
      'Logo ao chegar no trabalho, para ver como esta o dia',
      'Durante picos de transacoes, para monitorar volume',
      'Quando a gerencia pedir um resumo rapido',
      'Para identificar anomalias antes que virem problemas'
    ],
    elementos: [
      { nome: 'Total de Transacoes', tipo: 'KPI', desc: 'Numero total de transacoes processadas hoje' },
      { nome: 'Taxa de Fraude', tipo: 'KPI', desc: 'Porcentagem de transacoes marcadas como fraude' },
      { nome: 'Latencia Media', tipo: 'KPI', desc: 'Tempo medio de resposta do sistema (meta: <50ms)' },
      { nome: 'Alertas Pendentes', tipo: 'KPI', desc: 'Quantos alertas ainda nao foram analisados' },
      { nome: 'Grafico de Timeline', tipo: 'Grafico', desc: 'Mostra transacoes ao longo do dia' },
      { nome: 'Distribuicao por Canal', tipo: 'Grafico', desc: 'PIX, TED, Cartao - qual canal tem mais volume' }
    ],
    historia: 'Carlos chega as 06:00 e a primeira coisa que faz e abrir o Dashboard. Ele ve que durante a madrugada houve um pico de transacoes as 03:00 - algo incomum. Clica no grafico para investigar e descobre que foi um ataque de bot que o sistema bloqueou automaticamente. Ufa!',
    cuidados: [
      'Os dados atualizam a cada 30 segundos - nao precisa ficar apertando F5',
      'Se a latencia subir de 50ms, algo pode estar errado - avise a TI',
      'KPIs vermelhos precisam de atencao imediata'
    ],
    ascii: `
+--------------------------------------------------+
|  DASHBOARD                              [hoje]   |
+--------------------------------------------------+
|  +--------+  +--------+  +--------+  +--------+  |
|  | 4.467  |  | 69.7%  |  |  37ms  |  |   25   |  |
|  | Trans. |  | Fraude |  | Laten. |  | Alertas|  |
|  +--------+  +--------+  +--------+  +--------+  |
|                                                  |
|  [=====GRAFICO DE TIMELINE==================]    |
|  |    *                    *                |    |
|  |   * *     **           * *    ***        |    |
|  |  *   *   *  *         *   *  *   *       |    |
|  |------06h----09h----12h----15h----18h-----|    |
|                                                  |
|  PIX: 96%  |  TED: 2%  |  CARTAO: 2%            |
+--------------------------------------------------+
    `
  },
  {
    id: 'transactions',
    nome: 'Transacoes',
    icone: FileText,
    caminho: 'Menu > Transacoes',
    rota: '/transactions',
    modulo: 'Operacoes',
    cor: 'green',
    objetivo: 'Listar, buscar e visualizar todas as transacoes processadas pelo sistema. Pense como uma planilha gigante com todas as movimentacoes financeiras.',
    quandoUsar: [
      'Quando um cliente ligar perguntando sobre uma transacao especifica',
      'Para investigar um CPF ou conta suspeita',
      'Quando precisar de dados para um relatorio',
      'Para entender o historico de um cliente'
    ],
    elementos: [
      { nome: 'Campo de Busca', tipo: 'Input', desc: 'Digite CPF, ID da transacao ou valor para filtrar' },
      { nome: 'Filtro de Data', tipo: 'Seletor', desc: 'Escolha o periodo que deseja visualizar' },
      { nome: 'Filtro de Canal', tipo: 'Dropdown', desc: 'PIX, TED, CARTAO ou BOLETO' },
      { nome: 'Filtro de Status', tipo: 'Dropdown', desc: 'Aprovada, Bloqueada, Em Analise' },
      { nome: 'Tabela de Resultados', tipo: 'Grid', desc: 'Lista com todas as transacoes encontradas' },
      { nome: 'Botao Exportar', tipo: 'Acao', desc: 'Baixa os dados em CSV ou Excel' }
    ],
    historia: 'Um cliente ligou reclamando que sua TED de R$ 5.000 foi bloqueada. Ana abre a tela de Transacoes, digita o CPF do cliente no campo de busca, e rapidamente encontra a transacao. Ela ve que o score de risco foi 78 (alto) porque era a primeira vez que o cliente enviava dinheiro para aquele destinatario. Ana verifica que e legitimo e libera manualmente.',
    cuidados: [
      'Buscas muito amplas (sem filtros) podem demorar',
      'Dados sensiveis como CPF aparecem mascarados por LGPD',
      'Exportar muitos dados pode travar o navegador'
    ],
    ascii: `
+--------------------------------------------------+
|  TRANSACOES                                      |
+--------------------------------------------------+
|  Busca: [CPF, ID ou valor________] [Buscar]      |
|                                                  |
|  Filtros: [Hoje v] [PIX v] [Todas v]             |
+--------------------------------------------------+
|  ID       | VALOR    | CANAL | SCORE | STATUS   |
|-----------|----------|-------|-------|----------|
|  TX-001   | R$ 4.850 | PIX   |  87   | BLOQUEIO |
|  TX-002   | R$ 1.200 | TED   |  23   | APROVADA |
|  TX-003   | R$ 500   | PIX   |  15   | APROVADA |
|  TX-004   | R$ 8.900 | CART  |  92   | BLOQUEIO |
+--------------------------------------------------+
|  Mostrando 1-50 de 4.467 | [Exportar CSV]        |
+--------------------------------------------------+
    `
  },
  {
    id: 'alerts',
    nome: 'Alertas',
    icone: Bell,
    caminho: 'Menu > Alertas',
    rota: '/alerts',
    modulo: 'Operacoes',
    cor: 'orange',
    objetivo: 'Exibir todos os alertas gerados pelo sistema que precisam de atencao humana. E como uma caixa de entrada de emails urgentes - so aparecem coisas que precisam de acao.',
    quandoUsar: [
      'Varias vezes ao dia, para ver novos alertas',
      'Quando o badge de notificacao ficar vermelho',
      'Para priorizar quais casos analisar primeiro',
      'Para distribuir trabalho entre a equipe'
    ],
    elementos: [
      { nome: 'Badge de Contagem', tipo: 'Indicador', desc: 'Numero vermelho mostrando alertas pendentes' },
      { nome: 'Lista de Alertas', tipo: 'Lista', desc: 'Ordenada por prioridade e hora' },
      { nome: 'Filtro de Prioridade', tipo: 'Abas', desc: 'Critico, Alto, Medio, Baixo' },
      { nome: 'Botao Assumir', tipo: 'Acao', desc: 'Marca o alerta como "em analise por voce"' },
      { nome: 'Botao Resolver', tipo: 'Acao', desc: 'Fecha o alerta apos investigacao' }
    ],
    historia: 'Rodrigo esta no turno noturno e ve que tem 5 alertas criticos. Ele clica no primeiro - e um cliente fazendo 10 PIX de R$ 999 em 5 minutos (logo abaixo do limite de R$ 1.000). Padrao classico de fraude! Rodrigo assume o caso, bloqueia a conta, e escala para Ana revisar pela manha.',
    cuidados: [
      'Alertas criticos devem ser tratados em ate 5 minutos',
      'Sempre assuma o alerta antes de investigar (evita duplicidade)',
      'Se nao souber resolver, escale para um senior'
    ],
    ascii: `
+--------------------------------------------------+
|  ALERTAS                             [3] novos   |
+--------------------------------------------------+
|  [CRITICO] [ALTO] [MEDIO] [BAIXO]                |
+--------------------------------------------------+
|  ! CRITICO | 10:32 | Multiplos PIX abaixo limite |
|    CPF: ***.456.***-78 | Score: 94               |
|    [Assumir] [Ver Detalhes]                      |
|--------------------------------------------------|
|  ! ALTO    | 10:28 | Device novo + valor alto   |
|    CPF: ***.789.***-01 | Score: 78               |
|    [Assumir] [Ver Detalhes]                      |
|--------------------------------------------------|
|  ! MEDIO   | 10:15 | Horario incomum            |
|    CPF: ***.123.***-45 | Score: 62               |
|    [Assumir] [Ver Detalhes]                      |
+--------------------------------------------------+
    `
  },
  {
    id: 'investigation',
    nome: 'Investigacao',
    icone: Search,
    caminho: 'Menu > Investigacao',
    rota: '/investigation',
    modulo: 'Analise',
    cor: 'purple',
    objetivo: 'Tela para analise profunda de casos suspeitos. Aqui voce junta todas as pecas do quebra-cabeca: historico do cliente, rede de relacionamentos, padroes de comportamento.',
    quandoUsar: [
      'Quando um alerta precisa de investigacao detalhada',
      'Para mapear redes de fraude (varios CPFs conectados)',
      'Antes de decidir bloquear uma conta definitivamente',
      'Para preparar relatorios para o BACEN (STR)'
    ],
    elementos: [
      { nome: 'Perfil do Investigado', tipo: 'Card', desc: 'Dados do cliente sob investigacao' },
      { nome: 'Timeline de Atividades', tipo: 'Grafico', desc: 'Historico completo de transacoes' },
      { nome: 'Mapa de Rede', tipo: 'Visual', desc: 'Conexoes com outras contas' },
      { nome: 'Evidencias', tipo: 'Lista', desc: 'Prints, logs, registros relevantes' },
      { nome: 'Botao Gerar STR', tipo: 'Acao', desc: 'Cria relatorio para o Banco Central' }
    ],
    historia: 'Carlos esta investigando uma conta que recebeu dinheiro de 15 CPFs diferentes em uma semana. No mapa de rede, ele percebe que 12 desses CPFs tambem enviaram para uma segunda conta. Padrao classico de "laranja"! Carlos documenta tudo, gera o STR e envia para o BACEN.',
    cuidados: [
      'Investigacoes demoradas devem ser documentadas',
      'Nunca acuse um cliente sem evidencias solidas',
      'STR deve ser enviado em ate 24 horas apos confirmacao'
    ],
    ascii: `
+--------------------------------------------------+
|  INVESTIGACAO - CASO #2024-1234                  |
+--------------------------------------------------+
|  INVESTIGADO                                     |
|  +------------------------------------------+    |
|  | Nome: Joao *** Silva                     |    |
|  | CPF: ***.456.***-78                      |    |
|  | Conta ativa desde: 15/03/2024            |    |
|  | Total recebido: R$ 127.450,00            |    |
|  +------------------------------------------+    |
|                                                  |
|  MAPA DE REDE                                    |
|       [CPF-1]--\\                                 |
|       [CPF-2]--->[INVESTIGADO]--->[CPF-X]        |
|       [CPF-3]--/                                 |
|                                                  |
|  EVIDENCIAS: 3 documentos anexados               |
|                                                  |
|  [Adicionar Nota] [Gerar STR] [Concluir Caso]    |
+--------------------------------------------------+
    `
  },
  {
    id: 'manual-review',
    nome: 'Revisao Manual',
    icone: Eye,
    caminho: 'Menu > Revisao Manual',
    rota: '/manual-review',
    modulo: 'Operacoes',
    cor: 'red',
    objetivo: 'Fila de transacoes que a IA nao conseguiu decidir sozinha e precisa de um humano. O famoso "Human-in-the-Loop" - quando a maquina pede ajuda.',
    quandoUsar: [
      'Continuamente durante o expediente',
      'Quando a fila estiver crescendo',
      'Para transacoes com score entre 60 e 80 (zona cinza)',
      'Quando o cliente ligar pedindo liberacao urgente'
    ],
    elementos: [
      { nome: 'Fila de Pendentes', tipo: 'Lista', desc: 'Transacoes aguardando decisao humana' },
      { nome: 'Tempo na Fila', tipo: 'Indicador', desc: 'Quanto tempo cada item esta esperando' },
      { nome: 'Detalhes da Transacao', tipo: 'Painel', desc: 'Todas as informacoes para decidir' },
      { nome: 'Botao Aprovar', tipo: 'Acao', desc: 'Libera a transacao' },
      { nome: 'Botao Rejeitar', tipo: 'Acao', desc: 'Bloqueia a transacao' },
      { nome: 'Botao Escalar', tipo: 'Acao', desc: 'Envia para um senior decidir' }
    ],
    historia: 'Uma transacao de R$ 50.000 esta na fila ha 3 minutos. Rodrigo ve que e um empresario fazendo um pagamento para fornecedor. O score e 65 - zona cinza. Rodrigo liga para o cliente, confirma que e legitimo, aprova a transacao e adiciona o fornecedor na lista de destinatarios conhecidos.',
    cuidados: [
      'SLA maximo: 5 minutos na fila',
      'Sempre documente o motivo da aprovacao/rejeicao',
      'Na duvida, escale - nao adivinhe'
    ],
    ascii: `
+--------------------------------------------------+
|  REVISAO MANUAL                    [12 na fila]  |
+--------------------------------------------------+
|  TRANSACAO EM ANALISE                            |
|  +------------------------------------------+    |
|  | ID: TX-7891234                           |    |
|  | Valor: R$ 50.000,00                      |    |
|  | Canal: TED                               |    |
|  | Score: 65 (ZONA CINZA)                   |    |
|  | Motivo: Valor alto + destino novo        |    |
|  | Na fila ha: 3 min 22 seg                 |    |
|  +------------------------------------------+    |
|                                                  |
|  DECISAO:                                        |
|  [APROVAR]  [REJEITAR]  [ESCALAR]               |
|                                                  |
|  Justificativa: [_________________________]      |
+--------------------------------------------------+
    `
  },
  {
    id: 'calibration',
    nome: 'Calibracao',
    icone: Settings,
    caminho: 'Menu > Calibracao',
    rota: '/calibration',
    modulo: 'Configuracao',
    cor: 'indigo',
    objetivo: 'Ajustar os limites (thresholds) que definem quando uma transacao e aprovada, vai para revisao ou e bloqueada automaticamente. E como calibrar a sensibilidade de um alarme.',
    quandoUsar: [
      'Quando houver muitos falsos positivos (bloqueando clientes legitimos)',
      'Quando fraudes estiverem passando (falsos negativos)',
      'Apos mudancas significativas no perfil de clientes',
      'Mensalmente, como manutencao preventiva'
    ],
    elementos: [
      { nome: 'Slider de Threshold Baixo', tipo: 'Controle', desc: 'Abaixo disso = aprovacao automatica' },
      { nome: 'Slider de Threshold Alto', tipo: 'Controle', desc: 'Acima disso = bloqueio automatico' },
      { nome: 'Simulador', tipo: 'Ferramenta', desc: 'Mostra impacto antes de aplicar' },
      { nome: 'Historico de Mudancas', tipo: 'Log', desc: 'Quem mudou o que e quando' }
    ],
    historia: 'Ana percebe que o numero de revisoes manuais dobrou na ultima semana. Ela abre a tela de Calibracao e ve que o threshold de bloqueio esta muito baixo (60). Ela simula subir para 70 e ve que isso reduziria 30% das revisoes manuais sem aumentar fraudes. Aplica a mudanca.',
    cuidados: [
      'NUNCA mude thresholds sem simular antes',
      'Apenas lideres devem ter acesso a esta tela',
      'Mudancas afetam TODAS as transacoes - tenha cuidado'
    ],
    ascii: `
+--------------------------------------------------+
|  CALIBRACAO DE THRESHOLDS                        |
+--------------------------------------------------+
|                                                  |
|  APROVACAO AUTOMATICA (score abaixo de):         |
|  [====|====================] 30                  |
|                                                  |
|  REVISAO MANUAL (score entre):                   |
|  30 ----------[ZONA CINZA]----------- 70         |
|                                                  |
|  BLOQUEIO AUTOMATICO (score acima de):           |
|  [====================|====] 70                  |
|                                                  |
+--------------------------------------------------+
|  SIMULACAO COM NOVOS VALORES                     |
|  +------------------------------------------+    |
|  | Reducao de revisoes: -30%               |    |
|  | Impacto em fraudes: +0.02%              |    |
|  +------------------------------------------+    |
|                                                  |
|  [Simular] [Aplicar Mudancas] [Cancelar]         |
+--------------------------------------------------+
    `
  },
  {
    id: 'monitoring',
    nome: 'Monitoramento',
    icone: Activity,
    caminho: 'Menu > Monitoramento',
    rota: '/monitoring',
    modulo: 'Operacoes',
    cor: 'teal',
    objetivo: 'Monitorar a saude dos modelos de IA e do sistema como um todo. E como o check-up de um carro - verifica se tudo esta funcionando bem.',
    quandoUsar: [
      'Diariamente, como rotina de operacao',
      'Quando o sistema parecer lento',
      'Apos atualizacoes ou mudancas',
      'Quando houver suspeita de problemas'
    ],
    elementos: [
      { nome: 'Status dos Modelos', tipo: 'Indicadores', desc: 'RF, GB, CatBoost - Online/Offline' },
      { nome: 'Metricas de Performance', tipo: 'Graficos', desc: 'Accuracy, Precision, Recall' },
      { nome: 'Latencia por Endpoint', tipo: 'Tabela', desc: 'Tempo de resposta de cada API' },
      { nome: 'Alertas de Sistema', tipo: 'Lista', desc: 'Erros e avisos tecnicos' }
    ],
    historia: 'O Dashboard mostra latencia de 120ms - muito acima do normal. Ana abre o Monitoramento e ve que o modelo CatBoost esta respondendo lento. Ela reinicia o servico e a latencia volta para 40ms.',
    cuidados: [
      'Latencia > 100ms e critica - avise TI imediatamente',
      'Se um modelo ficar offline, o ensemble usa os outros dois',
      'Nunca ignore alertas vermelhos'
    ],
    ascii: `
+--------------------------------------------------+
|  MONITORAMENTO DO SISTEMA                        |
+--------------------------------------------------+
|  STATUS DOS MODELOS                              |
|  +----------------+----------------+             |
|  | Random Forest  | [ONLINE]       |             |
|  | Grad. Boosting | [ONLINE]       |             |
|  | CatBoost       | [ONLINE]       |             |
|  +----------------+----------------+             |
|                                                  |
|  METRICAS (ultimas 24h)                          |
|  +----------------+-------+                      |
|  | Accuracy       | 98.5% |                      |
|  | Precision      | 94.2% |                      |
|  | Recall         | 91.8% |                      |
|  | Latencia Media | 37ms  |                      |
|  +----------------+-------+                      |
|                                                  |
|  ALERTAS: Nenhum alerta ativo                    |
+--------------------------------------------------+
    `
  },
  {
    id: 'metrics',
    nome: 'Metricas',
    icone: Gauge,
    caminho: 'Menu > Metricas',
    rota: '/metrics',
    modulo: 'Observabilidade',
    cor: 'cyan',
    objetivo: 'Exibir metricas em tempo real do sistema. Contadores, tempos, volumes - tudo que pode ser medido aparece aqui em graficos atualizados constantemente.',
    quandoUsar: [
      'Para acompanhar picos de uso',
      'Quando precisar de dados para reunioes',
      'Para identificar tendencias',
      'Durante incidentes, para entender o impacto'
    ],
    elementos: [
      { nome: 'Contador de Transacoes', tipo: 'Metrica', desc: 'Total hoje, hora, minuto' },
      { nome: 'Taxa de Erro', tipo: 'Metrica', desc: 'Porcentagem de falhas' },
      { nome: 'Graficos Temporais', tipo: 'Visualizacao', desc: 'Evolucao ao longo do tempo' },
      { nome: 'Comparativo', tipo: 'Tabela', desc: 'Hoje vs ontem vs semana passada' }
    ],
    historia: 'Patricia precisa apresentar resultados para a diretoria. Ela abre Metricas, seleciona o periodo do mes, e exporta um grafico mostrando que as fraudes bloqueadas aumentaram 15% enquanto os falsos positivos cairam 8%.',
    cuidados: [
      'Metricas atualizam em tempo real - podem variar',
      'Para relatorios oficiais, use a tela de Relatorios',
      'Dados antigos podem ter leve diferenca por arredondamento'
    ],
    ascii: `
+--------------------------------------------------+
|  METRICAS EM TEMPO REAL                          |
+--------------------------------------------------+
|  CONTADORES                                      |
|  +----------+----------+----------+              |
|  | HOJE     | HORA     | MINUTO   |              |
|  | 4.467    | 312      | 5        |              |
|  +----------+----------+----------+              |
|                                                  |
|  PERFORMANCE                                     |
|  [=============================] 99.8% uptime    |
|  [===========================  ] 98.2% sucesso   |
|  [=                            ] 0.3% erros      |
|                                                  |
|  COMPARATIVO                                     |
|  Hoje: 4.467 | Ontem: 4.231 (+5.6%)             |
+--------------------------------------------------+
    `
  },
  {
    id: 'hard-rules',
    nome: 'Hard Rules',
    icone: Shield,
    caminho: 'Menu > Hard Rules',
    rota: '/hard-rules',
    modulo: 'Configuracao',
    cor: 'gray',
    objetivo: 'Regras de negocio que SEMPRE sao aplicadas, independente do que a IA diga. Sao como leis inquebrantaveis - nao tem excecao.',
    quandoUsar: [
      'Para criar bloqueios absolutos (ex: pais proibidos)',
      'Para definir limites de valor',
      'Para implementar regras regulatorias',
      'Para proteger contra ataques conhecidos'
    ],
    elementos: [
      { nome: 'Lista de Regras', tipo: 'Tabela', desc: 'Todas as regras ativas' },
      { nome: 'Editor de Regras', tipo: 'Formulario', desc: 'Criar ou editar uma regra' },
      { nome: 'Simulador', tipo: 'Ferramenta', desc: 'Testar regra antes de ativar' },
      { nome: 'Historico', tipo: 'Log', desc: 'Mudancas feitas nas regras' }
    ],
    historia: 'O BACEN emitiu nova regulamentacao: transacoes para certos paises precisam de aprovacao manual. Ana cria uma Hard Rule: "Se pais_destino in [lista], entao BLOQUEAR". A regra entra em vigor imediatamente.',
    cuidados: [
      'Hard Rules tem prioridade sobre a IA',
      'Uma regra mal configurada pode bloquear TUDO',
      'Sempre simule antes de ativar'
    ],
    ascii: `
+--------------------------------------------------+
|  HARD RULES - REGRAS DE NEGOCIO                  |
+--------------------------------------------------+
|  REGRAS ATIVAS                                   |
|  +----+----------------------------------+----+  |
|  | ID | DESCRICAO                        | ON |  |
|  +----+----------------------------------+----+  |
|  | 01 | PIX > R$50k = Revisao Manual     | ON |  |
|  | 02 | TED noturna > R$10k = Bloquear   | ON |  |
|  | 03 | Paises na lista OFAC = Bloquear  | ON |  |
|  +----+----------------------------------+----+  |
|                                                  |
|  [Nova Regra] [Editar] [Desativar]               |
+--------------------------------------------------+
    `
  },
  {
    id: 'vip-list',
    nome: 'VIP List',
    icone: Star,
    caminho: 'Menu > VIP List',
    rota: '/vip-list',
    modulo: 'Listas',
    cor: 'yellow',
    objetivo: 'Lista de clientes confiavels que NAO devem ser bloqueados. Sao os "clientes VIP" que tem historico impecavel e nao precisam de analise extra.',
    quandoUsar: [
      'Para adicionar clientes corporativos grandes',
      'Para evitar bloqueios recorrentes de clientes legitimos',
      'Apos analise profunda confirmar que cliente e seguro',
      'Para CEOs, diretores e pessoas publicas verificadas'
    ],
    elementos: [
      { nome: 'Lista de VIPs', tipo: 'Tabela', desc: 'CPFs/CNPJs na lista branca' },
      { nome: 'Motivo', tipo: 'Campo', desc: 'Por que esta pessoa e VIP' },
      { nome: 'Validade', tipo: 'Data', desc: 'Ate quando vale a excecao' },
      { nome: 'Responsavel', tipo: 'Campo', desc: 'Quem adicionou e quando' }
    ],
    historia: 'Um empresario faz TEDs de R$ 500k semanais para fornecedores. Toda semana o sistema bloqueia e ele reclama. Apos Ana verificar que e 100% legitimo, ela adiciona o CNPJ na VIP List com validade de 1 ano.',
    cuidados: [
      'VIPs ainda sao monitorados - apenas nao sao bloqueados',
      'Revisar a lista periodicamente',
      'Remover VIPs que mudarem de comportamento'
    ],
    ascii: `
+--------------------------------------------------+
|  VIP LIST - LISTA BRANCA                         |
+--------------------------------------------------+
|  CLIENTES VIP ATIVOS                             |
|  +------------+-------------------+----------+   |
|  | CPF/CNPJ   | MOTIVO            | VALIDADE |   |
|  +------------+-------------------+----------+   |
|  | **.***.***-01 | Empresario verificado | 12/2025 |
|  | **.***.***-02 | Diretor do banco   | 06/2025 |   |
|  +------------+-------------------+----------+   |
|                                                  |
|  [Adicionar VIP] [Remover] [Renovar]             |
+--------------------------------------------------+
    `
  },
  {
    id: 'hot-list',
    nome: 'HOT List',
    icone: AlertTriangle,
    caminho: 'Menu > HOT List',
    rota: '/hot-list',
    modulo: 'Listas',
    cor: 'red',
    objetivo: 'Lista negra de CPFs, devices ou IPs que SEMPRE sao bloqueados. Sao os "bandidos conhecidos" que ja foram confirmados como fraudadores.',
    quandoUsar: [
      'Apos confirmar uma fraude',
      'Para bloquear devices de bots',
      'Para barrar IPs de ataques',
      'Para incluir contas laranjas identificadas'
    ],
    elementos: [
      { nome: 'Lista de Bloqueados', tipo: 'Tabela', desc: 'CPFs, devices, IPs na lista negra' },
      { nome: 'Tipo', tipo: 'Campo', desc: 'CPF, Device ID, IP, Email' },
      { nome: 'Data de Inclusao', tipo: 'Data', desc: 'Quando foi adicionado' },
      { nome: 'Caso Relacionado', tipo: 'Link', desc: 'Investigacao que originou' }
    ],
    historia: 'Carlos identificou uma rede de 15 contas laranjas. Apos concluir a investigacao, ele adiciona todos os 15 CPFs na HOT List. Qualquer transacao futura desses CPFs sera bloqueada automaticamente.',
    cuidados: [
      'Verificar MUITO bem antes de adicionar',
      'Clientes na HOT List NAO conseguem fazer nada',
      'Mantenha o caso de investigacao vinculado'
    ],
    ascii: `
+--------------------------------------------------+
|  HOT LIST - LISTA NEGRA                          |
+--------------------------------------------------+
|  BLOQUEADOS PERMANENTEMENTE                      |
|  +------+------------+------------+----------+   |
|  | TIPO | VALOR      | DATA       | CASO     |   |
|  +------+------------+------------+----------+   |
|  | CPF  | ***456***  | 28/11/2024 | #2024-99 |   |
|  | DEV  | ABC123...  | 25/11/2024 | #2024-87 |   |
|  | IP   | 192.168... | 20/11/2024 | #2024-65 |   |
|  +------+------------+------------+----------+   |
|                                                  |
|  [Adicionar] [Remover] [Exportar]                |
+--------------------------------------------------+
    `
  },
  {
    id: 'feedback-analyst',
    nome: 'Feedback do Analista',
    icone: ThumbsUp,
    caminho: 'Menu > Feedback',
    rota: '/feedback-analyst',
    modulo: 'ML',
    cor: 'green',
    objetivo: 'Registrar feedback sobre as decisoes da IA. Quando voce corrige a IA, esse feedback e usado para treina-la e torna-la melhor.',
    quandoUsar: [
      'Apos aprovar uma transacao que a IA bloqueou (falso positivo)',
      'Apos rejeitar algo que a IA aprovou (falso negativo)',
      'Para registrar casos interessantes de aprendizado',
      'Diariamente, como parte da rotina'
    ],
    elementos: [
      { nome: 'Transacao Avaliada', tipo: 'Card', desc: 'Detalhes da transacao' },
      { nome: 'Decisao da IA', tipo: 'Indicador', desc: 'O que a IA disse' },
      { nome: 'Sua Decisao', tipo: 'Botoes', desc: 'Concordar ou Discordar' },
      { nome: 'Justificativa', tipo: 'Texto', desc: 'Por que voce discordou' },
      { nome: 'Confianca', tipo: 'Escala', desc: 'Quao certo voce esta' }
    ],
    historia: 'A IA bloqueou um PIX de R$ 2.000 porque o cliente nunca tinha feito PIX acima de R$ 500. Mas Ana verificou que ele acabou de receber o 13o salario. Ela marca como "falso positivo" e explica: "Primeira compra grande apos bonus salarial - padrao normal".',
    cuidados: [
      'Feedback de qualidade melhora a IA',
      'Feedback errado pode piorar a IA',
      'Seja especifico na justificativa'
    ],
    ascii: `
+--------------------------------------------------+
|  FEEDBACK DO ANALISTA                            |
+--------------------------------------------------+
|  TRANSACAO AVALIADA                              |
|  +------------------------------------------+    |
|  | ID: TX-5678                              |    |
|  | Valor: R$ 2.000                          |    |
|  | IA disse: BLOQUEAR (score 75)            |    |
|  +------------------------------------------+    |
|                                                  |
|  SUA AVALIACAO:                                  |
|  [CONCORDO] [DISCORDO - FALSO POSITIVO]          |
|                                                  |
|  Justificativa:                                  |
|  [Cliente recebeu 13o, padrao normal____]        |
|                                                  |
|  Confianca: [=======|===] 80%                    |
|                                                  |
|  [Enviar Feedback]                               |
+--------------------------------------------------+
    `
  },
  {
    id: 'reports',
    nome: 'Relatorios',
    icone: PieChart,
    caminho: 'Menu > Relatorios',
    rota: '/reports',
    modulo: 'Analise',
    cor: 'blue',
    objetivo: 'Gerar relatorios estruturados para apresentar a gestao, auditorias e orgaos reguladores. Sao documentos formais, nao dados brutos.',
    quandoUsar: [
      'Para reunioes de diretoria',
      'Para auditorias internas e externas',
      'Para enviar ao BACEN quando solicitado',
      'Para analise de performance mensal'
    ],
    elementos: [
      { nome: 'Tipo de Relatorio', tipo: 'Seletor', desc: 'Operacional, Gerencial, Regulatorio' },
      { nome: 'Periodo', tipo: 'Datas', desc: 'De quando ate quando' },
      { nome: 'Filtros', tipo: 'Opcoes', desc: 'Canal, status, score, etc.' },
      { nome: 'Botao Gerar', tipo: 'Acao', desc: 'Processa e cria o relatorio' },
      { nome: 'Formatos', tipo: 'Opcoes', desc: 'PDF, Excel, CSV' }
    ],
    historia: 'O BACEN solicitou um relatorio de todas as transacoes suspeitas do ultimo trimestre. Marina abre Relatorios, seleciona "Regulatorio", define as datas, e gera um PDF formatado com todas as informacoes exigidas.',
    cuidados: [
      'Relatorios regulatorios tem formato especifico',
      'Verifique os dados antes de enviar para fora',
      'Mantenha copia de todos os relatorios gerados'
    ],
    ascii: `
+--------------------------------------------------+
|  RELATORIOS                                      |
+--------------------------------------------------+
|  TIPO DE RELATORIO:                              |
|  ( ) Operacional  (x) Gerencial  ( ) Regulatorio |
|                                                  |
|  PERIODO:                                        |
|  De: [01/11/2024] Ate: [30/11/2024]             |
|                                                  |
|  FILTROS:                                        |
|  Canal: [Todos v]  Status: [Todos v]             |
|                                                  |
|  FORMATO:                                        |
|  (x) PDF  ( ) Excel  ( ) CSV                     |
|                                                  |
|  [Gerar Relatorio]                               |
+--------------------------------------------------+
    `
  },
  {
    id: 'audit',
    nome: 'Auditoria',
    icone: FileText,
    caminho: 'Menu > Auditoria',
    rota: '/audit',
    modulo: 'Compliance',
    cor: 'gray',
    objetivo: 'Registro de TUDO que aconteceu no sistema. Quem fez o que, quando e por que. E o "dedo-duro" do sistema - ninguem escapa.',
    quandoUsar: [
      'Para investigar acoes suspeitas de usuarios',
      'Quando houver duvida sobre quem fez algo',
      'Para auditorias de seguranca',
      'Para cumprir requisitos de LGPD'
    ],
    elementos: [
      { nome: 'Log de Eventos', tipo: 'Lista', desc: 'Todos os eventos registrados' },
      { nome: 'Filtro de Usuario', tipo: 'Seletor', desc: 'Ver acoes de pessoa especifica' },
      { nome: 'Filtro de Acao', tipo: 'Seletor', desc: 'Tipo de acao (login, aprovacao, etc.)' },
      { nome: 'Detalhes do Evento', tipo: 'Painel', desc: 'Informacoes completas' }
    ],
    historia: 'Uma transacao foi aprovada indevidamente e o cliente reclamou. Ana abre a Auditoria, filtra pelo ID da transacao, e descobre exatamente quem aprovou, quando, e qual justificativa foi dada.',
    cuidados: [
      'Logs de auditoria NAO podem ser apagados',
      'Guarde por no minimo 5 anos (exigencia legal)',
      'Use para aprender, nao para punir'
    ],
    ascii: `
+--------------------------------------------------+
|  AUDITORIA - REGISTRO DE EVENTOS                 |
+--------------------------------------------------+
|  FILTROS: Usuario [Todos v] Acao [Todas v]       |
|           Data [Hoje v]                          |
+--------------------------------------------------+
|  HORA   | USUARIO  | ACAO              | DETALHE |
|---------|----------|-------------------|---------|
|  10:45  | carlos   | Aprovar TX        | TX-123  |
|  10:32  | ana      | Alterar threshold | 60->70  |
|  10:15  | rodrigo  | Login             | -       |
|  09:58  | carlos   | Adicionar VIP     | CPF-*** |
+--------------------------------------------------+
|  Total: 38 eventos hoje                          |
+--------------------------------------------------+
    `
  },
  {
    id: 'datasets',
    nome: 'DataSets',
    icone: Database,
    caminho: 'Menu > DataSets',
    rota: '/datasets',
    modulo: 'ML',
    cor: 'purple',
    objetivo: 'Visualizar e gerenciar os conjuntos de dados usados para treinar a IA. E onde os dados brutos viram conhecimento.',
    quandoUsar: [
      'Para entender de onde vem o conhecimento da IA',
      'Antes de retreinar modelos',
      'Para verificar qualidade dos dados',
      'Para auditorias de ML'
    ],
    elementos: [
      { nome: 'Lista de Datasets', tipo: 'Cards', desc: 'Todos os datasets disponiveis' },
      { nome: 'Estatisticas', tipo: 'Metricas', desc: 'Quantidade, distribuicao, etc.' },
      { nome: 'Preview', tipo: 'Tabela', desc: 'Amostra dos dados' },
      { nome: 'Historico de Uso', tipo: 'Log', desc: 'Quando foi usado para treino' }
    ],
    historia: 'O time de ML quer retreinar o modelo. Antes, eles abrem a tela de DataSets para verificar se o dataset de producao esta atualizado e balanceado corretamente.',
    cuidados: [
      'Dados desbalanceados prejudicam o modelo',
      'Nunca use dados de producao em ambiente de teste',
      'Anonimize dados antes de exportar'
    ],
    ascii: `
+--------------------------------------------------+
|  DATASETS - CATALOGO DE DADOS                    |
+--------------------------------------------------+
|  +------------------+  +------------------+      |
|  | KAGGLE           |  | PRODUCAO         |      |
|  | 284.807 tx       |  | 4.467 tx         |      |
|  | Fraude: 0.17%    |  | Fraude: 69.7%    |      |
|  | [Ver Detalhes]   |  | [Ver Detalhes]   |      |
|  +------------------+  +------------------+      |
|                                                  |
|  +------------------+                            |
|  | FEEDBACK         |                            |
|  | ~50/dia          |                            |
|  | Usado: continuo  |                            |
|  | [Ver Detalhes]   |                            |
|  +------------------+                            |
+--------------------------------------------------+
    `
  },
  {
    id: 'settings',
    nome: 'Configuracoes',
    icone: Settings,
    caminho: 'Menu > Configuracoes',
    rota: '/settings',
    modulo: 'Sistema',
    cor: 'slate',
    objetivo: 'Ajustar preferencias do usuario e configuracoes gerais do sistema. Cada usuario pode personalizar sua experiencia.',
    quandoUsar: [
      'Para mudar idioma ou tema',
      'Para configurar notificacoes',
      'Para ajustar preferencias pessoais',
      'Para gerenciar integracao com outros sistemas'
    ],
    elementos: [
      { nome: 'Tema', tipo: 'Toggle', desc: 'Claro ou escuro' },
      { nome: 'Notificacoes', tipo: 'Checkboxes', desc: 'Quais alertas receber' },
      { nome: 'Idioma', tipo: 'Seletor', desc: 'Portugues, Ingles, etc.' },
      { nome: 'Integracao', tipo: 'Configuracao', desc: 'APIs externas' }
    ],
    historia: 'Rodrigo trabalha a noite e acha a tela muito clara. Ele abre Configuracoes e ativa o tema escuro. Tambem desativa notificacoes sonoras para nao incomodar os colegas.',
    cuidados: [
      'Algumas configuracoes sao pessoais, outras sao do sistema',
      'Configuracoes de sistema precisam de permissao de admin',
      'Mudancas de integracao podem afetar outros usuarios'
    ],
    ascii: `
+--------------------------------------------------+
|  CONFIGURACOES                                   |
+--------------------------------------------------+
|  APARENCIA                                       |
|  Tema: [Claro] [Escuro]                          |
|                                                  |
|  NOTIFICACOES                                    |
|  [x] Alertas criticos                            |
|  [x] Alertas altos                               |
|  [ ] Alertas medios                              |
|  [ ] Alertas baixos                              |
|  [ ] Som ativado                                 |
|                                                  |
|  IDIOMA                                          |
|  [Portugues (Brasil) v]                          |
|                                                  |
|  [Salvar Preferencias]                           |
+--------------------------------------------------+
    `
  }
];

const todasAsFeatures = {
  transacao: {
    categoria: 'Dados da Transacao',
    icone: DollarSign,
    cor: 'blue',
    descricao: 'Informacoes basicas sobre cada movimentacao financeira.',
    analogia: 'Pense como os dados de uma nota fiscal: valor, data, quem pagou, quem recebeu.',
    features: [
      { nome: 'amount', nomeAmigavel: 'Valor da Transacao', desc: 'Quanto dinheiro esta sendo movimentado', exemplo: 'R$ 4.850,00', importancia: 'Valores muito altos ou muito baixos podem indicar fraude. Fraudadores costumam testar com valores baixos primeiro.' },
      { nome: 'channel', nomeAmigavel: 'Canal', desc: 'Por onde a transacao foi feita', exemplo: 'PIX, TED, CARTAO, BOLETO', importancia: 'PIX e o canal mais arriscado porque e instantaneo e irreversivel. Cartao permite chargeback.' },
      { nome: 'transaction_hour', nomeAmigavel: 'Hora da Transacao', desc: 'Em que momento do dia aconteceu', exemplo: '03:42 (madrugada)', importancia: 'Transacoes de madrugada sao 4x mais suspeitas. Fraudadores agem quando a vitima dorme.' },
      { nome: 'day_of_week', nomeAmigavel: 'Dia da Semana', desc: 'Se e dia util ou fim de semana', exemplo: 'Sabado', importancia: 'Fins de semana tem menos supervisao bancaria, facilitando fraudes.' },
      { nome: 'is_weekend', nomeAmigavel: 'E Fim de Semana?', desc: 'Indicador binario', exemplo: 'Sim/Nao', importancia: 'Complementa a analise do dia, simplificando regras.' }
    ]
  },
  velocidade: {
    categoria: 'Velocidade e Frequencia',
    icone: Zap,
    cor: 'orange',
    descricao: 'Quantas transacoes em quanto tempo. Velocidade e inimiga do fraudador comum, mas amiga do bot.',
    analogia: 'E como contar quantas vezes alguem passou pelo mesmo pedagio. Uma vez e normal, 50 vezes em 1 hora e estranho.',
    features: [
      { nome: 'velocity_1h', nomeAmigavel: 'Transacoes na Ultima Hora', desc: 'Quantidade de transacoes nos ultimos 60 minutos', exemplo: '5 transacoes', importancia: 'Muitas transacoes em pouco tempo = bot ou pressa de fraudador.' },
      { nome: 'velocity_24h', nomeAmigavel: 'Transacoes em 24h', desc: 'Quantidade nas ultimas 24 horas', exemplo: '12 transacoes', importancia: 'Compara com o padrao historico. Se voce faz 3/dia e hoje ja fez 12, algo mudou.' },
      { nome: 'amount_velocity_1h', nomeAmigavel: 'Valor Movimentado em 1h', desc: 'Soma dos valores na ultima hora', exemplo: 'R$ 15.000', importancia: 'Mais importante que quantidade: quanto dinheiro saiu rapidamente.' },
      { nome: 'avg_time_between_tx', nomeAmigavel: 'Tempo Medio Entre Transacoes', desc: 'Intervalo tipico entre uma transacao e outra', exemplo: '2.5 minutos', importancia: 'Intervalos muito curtos (<1 min) indicam automacao ou pressa.' },
      { nome: 'velocity_ratio', nomeAmigavel: 'Razao de Velocidade', desc: 'Velocidade atual dividida pela velocidade media', exemplo: '8.5x', importancia: 'Se voce esta 8.5x mais rapido que o normal, algo muito diferente esta acontecendo.' }
    ]
  },
  comportamento: {
    categoria: 'Comportamento do Cliente',
    icone: Users,
    cor: 'green',
    descricao: 'O que e "normal" para este cliente? Cada pessoa tem um padrao.',
    analogia: 'E como conhecer os habitos de um vizinho. Voce sabe que ele sai as 7h todo dia. Se ele sair as 3h da manha, voce estranha.',
    features: [
      { nome: 'avg_transaction_amount', nomeAmigavel: 'Media de Valor', desc: 'Valor medio historico das transacoes do cliente', exemplo: 'R$ 1.200', importancia: 'Define o que e "normal" para este cliente. Transacao muito acima da media e suspeita.' },
      { nome: 'std_transaction_amount', nomeAmigavel: 'Desvio Padrao', desc: 'Variacao tipica dos valores', exemplo: 'R$ 500', importancia: 'Clientes com gastos regulares tem desvio baixo. Variacao alta pode indicar conta compartilhada.' },
      { nome: 'amount_deviation', nomeAmigavel: 'Quantos Desvios do Normal', desc: 'Quao diferente e esta transacao da media', exemplo: '3.5 desvios', importancia: 'Mais de 3 desvios e muito raro estatisticamente. Quase certamente algo anormal.' },
      { nome: 'days_since_last_tx', nomeAmigavel: 'Dias Desde Ultima Transacao', desc: 'Ha quanto tempo a conta estava inativa', exemplo: '45 dias', importancia: 'Conta parada ha 45 dias que de repente movimenta milhares = suspeito.' },
      { nome: 'account_age_days', nomeAmigavel: 'Idade da Conta', desc: 'Ha quantos dias a conta existe', exemplo: '30 dias', importancia: 'Contas muito novas (<30 dias) sao frequentemente usadas para fraude (contas laranja).' }
    ]
  },
  destinatario: {
    categoria: 'Destinatario',
    icone: Target,
    cor: 'red',
    descricao: 'Quem esta recebendo o dinheiro? O destino e tao importante quanto a origem.',
    analogia: 'E como investigar para quem voce esta transferindo. Voce conhece essa pessoa? Ja mandou dinheiro antes?',
    features: [
      { nome: 'recipient_is_new', nomeAmigavel: 'Destinatario Novo?', desc: 'Se e a primeira vez que envia para esta pessoa', exemplo: 'Sim', importancia: 'Em golpes, o destino e SEMPRE novo. A vitima nunca enviou dinheiro para o golpista antes.' },
      { nome: 'recipient_risk_score', nomeAmigavel: 'Score de Risco do Recebedor', desc: 'Quao arriscada e a conta de destino', exemplo: '85/100', importancia: 'Se a conta de destino ja recebeu dinheiro de outras fraudes, ela e de alto risco.' },
      { nome: 'recipient_account_age', nomeAmigavel: 'Idade da Conta Destino', desc: 'Ha quanto tempo a conta de destino existe', exemplo: '7 dias', importancia: 'Conta laranja tipica tem menos de 30 dias. Abrem, usam para fraude, e abandonam.' },
      { nome: 'recipient_tx_count', nomeAmigavel: 'Transacoes do Destinatario', desc: 'Quantas transacoes a conta destino ja recebeu', exemplo: '3', importancia: 'Conta que so recebe (nunca envia) e tem poucas transacoes = suspeita.' },
      { nome: 'is_known_merchant', nomeAmigavel: 'E Comerciante Conhecido?', desc: 'Se o destino e uma empresa verificada', exemplo: 'Nao', importancia: 'Pagamentos para Uber, iFood, Netflix sao seguros. Para CPFs desconhecidos, menos.' }
    ]
  },
  dispositivo: {
    categoria: 'Dispositivo e Localizacao',
    icone: Globe,
    cor: 'purple',
    descricao: 'De onde e de qual aparelho veio a transacao?',
    analogia: 'E como verificar se voce esta usando seu proprio celular de casa, ou um celular estranho de outro pais.',
    features: [
      { nome: 'device_is_new', nomeAmigavel: 'Dispositivo Novo?', desc: 'Se o aparelho nunca foi usado antes', exemplo: 'Sim', importancia: 'Voce sempre usa o mesmo celular. Se de repente usa outro, pode ser fraudador com sua senha.' },
      { nome: 'device_fingerprint_match', nomeAmigavel: 'Similaridade do Dispositivo', desc: 'Quao parecido e com dispositivos anteriores', exemplo: '0.2 (baixo)', importancia: 'Mesmo que troque de celular, ha padroes. Modelo, sistema, configuracoes.' },
      { nome: 'ip_is_vpn', nomeAmigavel: 'IP e de VPN?', desc: 'Se o endereco de internet e de uma VPN', exemplo: 'Sim', importancia: 'VPNs escondem localizacao real. Fraudadores usam para parecer que estao em outro lugar.' },
      { nome: 'geolocation_distance_km', nomeAmigavel: 'Distancia da Localizacao Habitual', desc: 'Quantos km do local normal', exemplo: '2500 km', importancia: 'Se voce esta em SP e a transacao vem do Nordeste, como voce viajou 2500km em 1 hora?' },
      { nome: 'location_risk_score', nomeAmigavel: 'Score de Risco da Regiao', desc: 'Quao arriscada e a regiao de origem', exemplo: '75', importancia: 'Algumas regioes tem mais fraudes que outras. Dado estatistico, nao preconceito.' }
    ]
  },
  temporal: {
    categoria: 'Padroes Temporais',
    icone: Clock,
    cor: 'indigo',
    descricao: 'Quando aconteceu e se bate com o padrao historico.',
    analogia: 'E como observar se alguem esta agindo em horario normal ou em horario suspeito.',
    features: [
      { nome: 'is_night_transaction', nomeAmigavel: 'E Transacao Noturna?', desc: 'Se aconteceu entre 00h e 06h', exemplo: 'Sim', importancia: 'Madrugada e horario preferido de fraudadores. Menos supervisao, vitima dormindo.' },
      { nome: 'is_rush_hour', nomeAmigavel: 'E Horario de Pico?', desc: 'Se e horario comercial movimentado', exemplo: 'Nao', importancia: 'Transacoes em horario comercial sao mais normais.' },
      { nome: 'time_since_account_login', nomeAmigavel: 'Minutos Desde o Login', desc: 'Quanto tempo passou desde que logou', exemplo: '2 minutos', importancia: 'Transacao logo apos login longo pode ser sessao hackeada.' },
      { nome: 'usual_hour_deviation', nomeAmigavel: 'Desvio do Horario Habitual', desc: 'Quao diferente e do horario normal', exemplo: '8.5 horas', importancia: 'Se voce sempre faz transacoes as 14h e agora faz as 3h, mudanca suspeita.' },
      { nome: 'days_since_password_change', nomeAmigavel: 'Dias Desde Troca de Senha', desc: 'Quando a senha foi trocada pela ultima vez', exemplo: '1 dia', importancia: 'Transacao grande logo apos troca de senha pode indicar que fraudador trocou.' }
    ]
  },
  rede: {
    categoria: 'Analise de Rede',
    icone: Network,
    cor: 'teal',
    descricao: 'Conexoes entre contas, dispositivos e IPs. Fraude nunca e isolada.',
    analogia: 'E como mapear amizades. Se seus amigos sao fraudadores, voce e mais suspeito tambem.',
    features: [
      { nome: 'shared_device_count', nomeAmigavel: 'Contas no Mesmo Device', desc: 'Quantas contas usam o mesmo aparelho', exemplo: '5', importancia: 'Se 5 CPFs diferentes usam o mesmo celular, provavelmente e uma quadrilha.' },
      { nome: 'shared_ip_count', nomeAmigavel: 'Contas no Mesmo IP', desc: 'Quantas contas usam o mesmo endereco de internet', exemplo: '8', importancia: 'Multiplas contas no mesmo IP = possivel operacao de fraude.' },
      { nome: 'network_fraud_rate', nomeAmigavel: 'Taxa de Fraude na Rede', desc: 'Porcentagem de fraude entre conexoes', exemplo: '15%', importancia: 'Se pessoas conectadas a voce cometeram fraude, voce e mais suspeito.' },
      { nome: 'degree_centrality', nomeAmigavel: 'Centralidade na Rede', desc: 'Quao conectada e a conta', exemplo: '0.85', importancia: 'Contas muito conectadas podem ser laranjas distribuindo dinheiro.' },
      { nome: 'community_risk', nomeAmigavel: 'Risco da Comunidade', desc: 'Risco medio das contas conectadas', exemplo: '0.4', importancia: 'Se voce transaciona com pessoas de alto risco, seu risco aumenta.' }
    ]
  },
  derivadas: {
    categoria: 'Features Derivadas por IA',
    icone: Brain,
    cor: 'pink',
    descricao: 'Calculadas automaticamente pela IA. Sao combinacoes complexas das outras features.',
    analogia: 'E como a nota final de uma prova, que junta todas as questoes em uma unica pontuacao.',
    features: [
      { nome: 'anomaly_score', nomeAmigavel: 'Score de Anomalia', desc: 'Quao diferente e esta transacao de todas as outras', exemplo: '0.92', importancia: 'Quanto mais perto de 1, mais anormal. 0.92 = muito fora do padrao.' },
      { nome: 'cluster_distance', nomeAmigavel: 'Distancia do Cluster', desc: 'Quao longe esta do grupo similar', exemplo: '3.5', importancia: 'Transacoes sao agrupadas por similaridade. Estar longe do grupo = diferente.' },
      { nome: 'fraud_probability', nomeAmigavel: 'Probabilidade de Fraude', desc: 'Chance de ser fraude (0-100%)', exemplo: '87%', importancia: 'Resultado final do modelo. 87% = alta chance de ser fraude.' },
      { nome: 'ensemble_agreement', nomeAmigavel: 'Concordancia do Ensemble', desc: 'Se os 3 modelos concordam', exemplo: '95%', importancia: 'Se os 3 modelos concordam, confiamos mais. Discordancia = incerteza.' },
      { nome: 'confidence_score', nomeAmigavel: 'Confianca da Predicao', desc: 'Quao certo o modelo esta', exemplo: '88%', importancia: 'Baixa confianca = revisar manualmente.' }
    ]
  }
};

const datasets = [
  {
    id: 'kaggle',
    nome: 'Credit Card Fraud Detection (Kaggle)',
    icone: Globe,
    cor: 'blue',
    registros: '284.807 transacoes',
    taxaFraude: '0.172%',
    origem: 'Kaggle - Machine Learning Group ULB (Belgica)',
    descricao: 'O dataset mais famoso do mundo para detecao de fraudes. Contem transacoes reais de cartao de credito de setembro de 2013 por titulares europeus.',
    campos: [
      { nome: 'V1 a V28', desc: 'Features transformadas por PCA para proteger privacidade. Nao sabemos o significado original.' },
      { nome: 'Time', desc: 'Segundos desde a primeira transacao do dataset.' },
      { nome: 'Amount', desc: 'Valor da transacao em euros.' },
      { nome: 'Class', desc: 'Se e fraude (1) ou nao (0).' }
    ],
    uso: 'Pre-treinamento inicial. Fornece base estatistica robusta antes de adaptar para o Brasil.',
    limitacoes: [
      'Dados de 2013 - padroes de fraude evoluiram',
      'Apenas cartao de credito (sem PIX)',
      'Contexto europeu (diferente do Brasil)',
      'Features anonimas (nao sabemos o que significam)'
    ]
  },
  {
    id: 'producao',
    nome: 'Transacoes de Producao',
    icone: Server,
    cor: 'green',
    registros: '4.467 transacoes',
    taxaFraude: '69.73%',
    origem: 'Sistema Sankofa - PostgreSQL',
    descricao: 'Dados reais processados pelo sistema em producao. Inclui PIX, TED, cartoes e boletos com contexto 100% brasileiro.',
    campos: [
      { nome: 'transaction_id', desc: 'Identificador unico (UUID).' },
      { nome: 'amount', desc: 'Valor em reais (R$).' },
      { nome: 'channel', desc: 'PIX, TED, CREDIT_CARD, DEBIT_CARD, BOLETO.' },
      { nome: 'risk_score', desc: 'Score calculado pelo modelo (0-100).' },
      { nome: 'is_fraud', desc: 'Se foi confirmada como fraude.' },
      { nome: 'created_at', desc: 'Data e hora da transacao.' }
    ],
    uso: 'Fine-tuning dos modelos. Adapta o conhecimento global para realidade brasileira.',
    distribuicao: [
      { canal: 'PIX', quantidade: 4285, fraudes: 3081, taxa: '71.9%' },
      { canal: 'TED', quantidade: 86, fraudes: 14, taxa: '16.3%' },
      { canal: 'BOLETO', quantidade: 88, fraudes: 14, taxa: '15.9%' }
    ]
  },
  {
    id: 'feedback',
    nome: 'Feedback dos Analistas',
    icone: MessageSquare,
    cor: 'purple',
    registros: '~50 feedbacks/dia',
    taxaFraude: 'N/A',
    origem: 'Human-in-the-Loop',
    descricao: 'Correcoes e confirmacoes feitas por analistas humanos. Fundamental para aprendizado continuo da IA.',
    campos: [
      { nome: 'transaction_id', desc: 'ID da transacao avaliada.' },
      { nome: 'analyst_decision', desc: 'FRAUD, LEGITIMATE, ou NEEDS_REVIEW.' },
      { nome: 'confidence', desc: 'Confianca do analista (1-5).' },
      { nome: 'reasoning', desc: 'Texto explicando a decisao.' },
      { nome: 'analyst_id', desc: 'Quem deu o feedback.' }
    ],
    uso: 'Continuous Learning. Modelo evolui diariamente com feedback humano.',
    fluxo: [
      'IA faz predicao inicial',
      'Analista revisa e confirma/corrige',
      'Feedback e armazenado',
      'Retraining diario as 04:00',
      'Modelo e atualizado'
    ]
  }
];

const transferLearningFases = [
  {
    fase: 1,
    nome: 'Pre-Treinamento',
    icone: Database,
    cor: 'blue',
    duracao: '~2 horas',
    descricao: 'O modelo aprende padroes GERAIS de fraude usando o dataset Kaggle com 284 mil transacoes.',
    analogia: 'E como estudar medicina geral antes de se especializar. Primeiro aprende o basico que vale para todo mundo.',
    detalhes: [
      'Carrega 284.807 transacoes do Kaggle',
      'Treina Random Forest com 100 arvores',
      'Treina Gradient Boosting com 100 estimadores',
      'Treina CatBoost com 500 iteracoes',
      'Valida com 15% dos dados reservados'
    ],
    metricas: { accuracy: '99.2%', auc: '0.97', precision: '85%', recall: '82%' },
    resultado: 'Modelo que sabe detectar fraude em geral, mas ainda nao conhece o Brasil.'
  },
  {
    fase: 2,
    nome: 'Adaptacao de Dominio',
    icone: RefreshCw,
    cor: 'green',
    duracao: '~30 minutos',
    descricao: 'O modelo se ADAPTA ao contexto brasileiro usando dados reais de producao.',
    analogia: 'E como um medico estrangeiro fazendo residencia no Brasil. Ele ja sabe medicina, mas aprende as doencas mais comuns aqui.',
    detalhes: [
      'Fine-tune com 4.467 transacoes brasileiras',
      'Adiciona features especificas de PIX',
      'Aprende padroes de TED brasileira',
      'Ajusta pesos para horarios locais',
      'Calibra thresholds para falsos positivos aceitaveis'
    ],
    metricas: { accuracy: '97.8%', auc: '0.94', precision: '92%', recall: '89%' },
    resultado: 'Modelo adaptado para Brasil. Conhece PIX, TED, horarios e padroes locais.'
  },
  {
    fase: 3,
    nome: 'Ensemble Voting',
    icone: Layers,
    cor: 'purple',
    duracao: 'Tempo real',
    descricao: 'Os 3 modelos VOTAM juntos para uma decisao mais robusta. E como um tribunal com 3 juizes.',
    analogia: 'Se 2 de 3 especialistas concordam, voce confia mais. Se os 3 discordam, precisa investigar melhor.',
    detalhes: [
      'Random Forest da probabilidade P1 (peso 30%)',
      'Gradient Boosting da probabilidade P2 (peso 35%)',
      'CatBoost da probabilidade P3 (peso 35%)',
      'Score final = media ponderada',
      'Discordancia alta = encaminha para analise manual'
    ],
    metricas: { accuracy: '98.5%', auc: '0.96', precision: '94%', recall: '91%' },
    resultado: 'Decisao mais confiavel que qualquer modelo individual.'
  },
  {
    fase: 4,
    nome: 'Aprendizado Continuo',
    icone: Brain,
    cor: 'pink',
    duracao: 'Continuo',
    descricao: 'O modelo EVOLUI diariamente com feedback dos analistas. Nunca para de aprender.',
    analogia: 'E como um medico que continua estudando a vida toda. Cada caso novo e uma licao.',
    detalhes: [
      'Coleta ~50 feedbacks/dia dos analistas',
      'Batch retraining as 04:00 (baixo volume)',
      'Incremental learning: nao descarta conhecimento anterior',
      'Detecta concept drift (mudanca de padroes)',
      'Rollback automatico se nova versao piorar'
    ],
    metricas: { accuracy: '99.1%', auc: '0.97', precision: '95%', recall: '93%' },
    resultado: 'Modelo que melhora constantemente e se adapta a novos tipos de fraude.'
  }
];

const faq = [
  { pergunta: 'Onde encontro o cadastro de clientes?', resposta: 'Este sistema nao tem cadastro de clientes direto. Os clientes vem do sistema bancario principal. Aqui voce ve apenas as transacoes que eles fazem.' },
  { pergunta: 'Como vejo o score de risco de uma transacao?', resposta: 'Va em Menu > Transacoes, busque pela transacao, e veja a coluna "Score". Quanto maior o numero (0-100), maior o risco.' },
  { pergunta: 'O que significa score 87?', resposta: 'Score 87 significa 87% de chance de ser fraude. Scores acima de 70 geralmente sao bloqueados automaticamente.' },
  { pergunta: 'Posso desfazer um bloqueio?', resposta: 'Sim! Va em Menu > Transacoes, encontre a transacao bloqueada, e clique em "Liberar". Voce precisara justificar a decisao.' },
  { pergunta: 'Como adiciono um cliente na VIP List?', resposta: 'Menu > VIP List > Adicionar VIP. Preencha o CPF/CNPJ, motivo, e defina a validade. Apenas lideres tem essa permissao.' },
  { pergunta: 'O que e a HOT List?', resposta: 'E a lista negra. CPFs, devices ou IPs que SEMPRE serao bloqueados, sem excecao. Use para fraudadores confirmados.' },
  { pergunta: 'Como a IA aprende?', resposta: 'Toda vez que voce da feedback (concordando ou discordando da IA), esse feedback treina o modelo. Quanto mais feedback, mais esperta ela fica.' },
  { pergunta: 'O que fazer se o sistema estiver lento?', resposta: 'Primeiro, verifique Menu > Monitoramento para ver se algum servico esta com problema. Se a latencia estiver alta, avise a TI.' },
  { pergunta: 'Posso exportar dados?', resposta: 'Sim, na maioria das telas tem um botao "Exportar". Voce pode baixar em CSV, Excel ou PDF dependendo da tela.' },
  { pergunta: 'O que e LGPD?', resposta: 'E a Lei Geral de Protecao de Dados. Por isso CPFs aparecem mascarados (***456***) e voce nao pode ver dados completos de clientes sem justificativa.' }
];

function CollapsibleSection({ title, icon: Icon, children, defaultOpen = false, id, color = 'blue' }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const sectionRef = useRef(null);

  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
    green: 'bg-green-50 border-green-200 text-green-800',
    purple: 'bg-purple-50 border-purple-200 text-purple-800',
    orange: 'bg-orange-50 border-orange-200 text-orange-800',
    red: 'bg-red-50 border-red-200 text-red-800'
  };

  return (
    <div id={id} ref={sectionRef} className="bg-white rounded-xl shadow-lg overflow-hidden mb-6">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full p-5 flex items-center justify-between hover:bg-gray-50 transition-colors ${colorClasses[color]}`}
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="h-6 w-6" />}
          <h2 className="text-xl font-bold">{title}</h2>
        </div>
        {isOpen ? <ChevronUp className="h-6 w-6" /> : <ChevronDown className="h-6 w-6" />}
      </button>
      {isOpen && <div className="p-6 border-t">{children}</div>}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const styles = {
    info: { bg: 'bg-blue-50', border: 'border-blue-300', icon: Info, color: 'text-blue-800' },
    success: { bg: 'bg-green-50', border: 'border-green-300', icon: CheckCircle, color: 'text-green-800' },
    warning: { bg: 'bg-yellow-50', border: 'border-yellow-300', icon: AlertTriangle, color: 'text-yellow-800' },
    danger: { bg: 'bg-red-50', border: 'border-red-300', icon: XCircle, color: 'text-red-800' },
    tip: { bg: 'bg-purple-50', border: 'border-purple-300', icon: Lightbulb, color: 'text-purple-800' }
  };
  const { bg, border, icon: IconComponent, color } = styles[type];

  return (
    <div className={`${bg} ${border} border-2 rounded-xl p-4 my-4`}>
      <div className="flex items-start gap-3">
        <IconComponent className={`h-6 w-6 ${color} mt-0.5 flex-shrink-0`} />
        <div>
          <h4 className={`font-bold ${color}`}>{title}</h4>
          <div className="mt-1 text-gray-700">{children}</div>
        </div>
      </div>
    </div>
  );
}

function TelaCard({ tela, onClick }) {
  return (
    <div 
      onClick={onClick}
      className="bg-white rounded-xl shadow-md hover:shadow-lg transition-all cursor-pointer p-4 border-l-4 border-blue-500"
    >
      <div className="flex items-center gap-3 mb-2">
        <tela.icone className="h-8 w-8 text-blue-600" />
        <div>
          <h4 className="font-bold text-gray-900">{tela.nome}</h4>
          <p className="text-xs text-gray-500">{tela.caminho}</p>
        </div>
      </div>
      <p className="text-sm text-gray-600 line-clamp-2">{tela.objetivo.substring(0, 100)}...</p>
    </div>
  );
}

function FeatureCard({ feature, categoria }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4 mb-3 hover:bg-gray-100 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div>
          <code className="text-sm font-mono bg-gray-200 px-2 py-1 rounded">{feature.nome}</code>
          <span className="ml-2 text-sm font-semibold text-gray-700">{feature.nomeAmigavel}</span>
        </div>
      </div>
      <p className="text-sm text-gray-600 mb-2">{feature.desc}</p>
      <p className="text-xs text-gray-500 mb-2"><strong>Exemplo:</strong> {feature.exemplo}</p>
      <div className="bg-blue-50 rounded p-2">
        <p className="text-xs text-blue-800"><strong>Por que e importante:</strong> {feature.importancia}</p>
      </div>
    </div>
  );
}

export function Manual() {
  const [activeSection, setActiveSection] = useState('bem-vindo');
  const [selectedTela, setSelectedTela] = useState(null);

  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-12">
      {/* HEADER PRINCIPAL */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-800 text-white p-8 rounded-b-3xl shadow-xl">
        <div className="flex items-center gap-4 mb-4">
          <div className="bg-white/20 p-4 rounded-xl">
            <BookOpen className="h-12 w-12" />
          </div>
          <div>
            <h1 className="text-4xl font-bold">Manual Completo do Sankofa</h1>
            <p className="text-xl text-blue-100">Sistema de Deteccao de Fraudes Bancarias - Guia Ultra-Didatico</p>
          </div>
        </div>
        
        <div className="bg-white/10 rounded-xl p-4 mt-6">
          <p className="text-blue-100 text-lg">
            Bem-vindo! Este manual foi criado para ser extremamente claro e didatico. 
            Aqui voce vai aprender TUDO sobre o sistema: cada tela, cada botao, como a IA funciona, 
            de onde vem os dados, e como tudo se conecta. Use este guia como sua referencia diaria.
          </p>
        </div>

        <div className="flex items-center gap-6 mt-6 text-sm flex-wrap">
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Clock className="h-4 w-4" /> Atualizado: 30/11/2025</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Users className="h-4 w-4" /> Para: Analistas de Fraude</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Grid className="h-4 w-4" /> 16 Telas Documentadas</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Brain className="h-4 w-4" /> 40+ Features de IA</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Database className="h-4 w-4" /> 3 DataSets</span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 mt-8">

        {/* SECAO 1: BEM-VINDO */}
        <CollapsibleSection id="bem-vindo" title="Bem-vindo ao Sistema - Visao Geral Didatica" icon={GraduationCap} defaultOpen={true} color="blue">
          <div className="space-y-6">
            <AlertBox type="tip" title="O que e o Sankofa?">
              <p className="text-lg">
                Pense no Sankofa como um <strong>guarda-costas digital</strong> para o banco. 
                Enquanto milhoes de transacoes acontecem por segundo, ele analisa CADA UMA em menos de 50 milissegundos 
                (isso e mais rapido que um piscar de olhos!) e decide se e segura ou suspeita.
              </p>
            </AlertBox>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 rounded-xl p-6 text-center">
                <Zap className="h-12 w-12 text-blue-600 mx-auto mb-3" />
                <h4 className="font-bold text-gray-900 text-lg">Super Rapido</h4>
                <p className="text-sm text-gray-600 mt-2">
                  Analisa cada transacao em <strong>37 milissegundos</strong>. 
                  O cliente nem percebe que foi verificado.
                </p>
              </div>
              <div className="bg-green-50 rounded-xl p-6 text-center">
                <Brain className="h-12 w-12 text-green-600 mx-auto mb-3" />
                <h4 className="font-bold text-gray-900 text-lg">Inteligente</h4>
                <p className="text-sm text-gray-600 mt-2">
                  <strong>3 modelos de IA</strong> trabalham juntos, analisando mais de 40 caracteristicas diferentes.
                </p>
              </div>
              <div className="bg-purple-50 rounded-xl p-6 text-center">
                <Users className="h-12 w-12 text-purple-600 mx-auto mb-3" />
                <h4 className="font-bold text-gray-900 text-lg">Humano + Maquina</h4>
                <p className="text-sm text-gray-600 mt-2">
                  Quando a IA tem duvida, <strong>voce decide</strong>. 
                  E seu feedback treina a IA para ser melhor.
                </p>
              </div>
            </div>

            <div className="bg-gray-900 text-white rounded-xl p-6 mt-6">
              <h4 className="font-bold text-xl mb-4 flex items-center gap-2">
                <Workflow className="h-6 w-6" /> Como Funciona em 6 Passos
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
                {[
                  { emoji: '1', icon: Smartphone, label: 'Cliente faz PIX' },
                  { emoji: '2', icon: Search, label: 'Sistema analisa 40+ features' },
                  { emoji: '3', icon: Brain, label: '3 IAs votam juntas' },
                  { emoji: '4', icon: Gauge, label: 'Gera score 0-100' },
                  { emoji: '5', icon: Scale, label: 'Decide: aprovar/bloquear' },
                  { emoji: '6', icon: CheckCircle, label: 'Tudo em 37ms!' }
                ].map((step, i) => (
                  <div key={i} className="bg-gray-800 rounded-lg p-3">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-2 font-bold">{step.emoji}</div>
                    <step.icon className="h-6 w-6 mx-auto mb-1 text-blue-400" />
                    <div className="text-xs text-gray-300">{step.label}</div>
                  </div>
                ))}
              </div>
            </div>

            <h4 className="font-bold text-xl mt-8 mb-4 flex items-center gap-2">
              <Users className="h-6 w-6 text-blue-600" /> Quem Usa o Sistema?
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.values(personas).map((persona, i) => (
                <div key={i} className={`bg-${persona.color}-50 rounded-xl p-4`}>
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`w-12 h-12 bg-${persona.color}-500 rounded-full flex items-center justify-center text-white font-bold`}>
                      {persona.avatar}
                    </div>
                    <div>
                      <div className="font-bold">{persona.name}</div>
                      <div className="text-xs text-gray-600">{persona.role}</div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">{persona.intro}</p>
                </div>
              ))}
            </div>
          </div>
        </CollapsibleSection>

        {/* SECAO 2: MAPA DAS TELAS */}
        <CollapsibleSection id="mapa-telas" title="Mapa Visual das Telas e Modulos" icon={Grid} color="green">
          <AlertBox type="info" title="Navegue pelo Sistema">
            <p>Abaixo esta o mapa completo de todas as 16 telas do sistema, organizadas por modulo. 
            Clique em qualquer tela para ver a documentacao detalhada.</p>
          </AlertBox>

          <div className="space-y-6">
            {[
              { modulo: 'Visao Geral', icone: BarChart3, telas: ['dashboard'] },
              { modulo: 'Operacoes', icone: Activity, telas: ['transactions', 'alerts', 'manual-review'] },
              { modulo: 'Analise', icone: Search, telas: ['investigation', 'reports'] },
              { modulo: 'Configuracao', icone: Settings, telas: ['calibration', 'hard-rules'] },
              { modulo: 'Listas', icone: List, telas: ['vip-list', 'hot-list'] },
              { modulo: 'ML/Inteligencia', icone: Brain, telas: ['datasets', 'feedback-analyst'] },
              { modulo: 'Observabilidade', icone: Gauge, telas: ['monitoring', 'metrics'] },
              { modulo: 'Compliance', icone: Shield, telas: ['audit'] },
              { modulo: 'Sistema', icone: Settings, telas: ['settings'] }
            ].map((grupo, i) => (
              <div key={i}>
                <h4 className="font-bold text-lg mb-3 flex items-center gap-2">
                  <grupo.icone className="h-5 w-5 text-blue-600" />
                  {grupo.modulo}
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {grupo.telas.map(telaId => {
                    const tela = todasAsTelas.find(t => t.id === telaId);
                    if (!tela) return null;
                    return (
                      <TelaCard 
                        key={tela.id} 
                        tela={tela}
                        onClick={() => scrollToSection(`tela-${tela.id}`)}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* SECAO 3: MANUAL DETALHADO POR TELA */}
        <CollapsibleSection id="manual-telas" title="Manual Detalhado por Tela (Todas as 16)" icon={FileText} color="purple">
          <AlertBox type="tip" title="Como ler esta secao">
            <p>Cada tela segue o mesmo formato: Nome, Caminho, Ilustracao Visual, Objetivo, Quando Usar, 
            Elementos, Historia de Uso e Cuidados. Isso facilita encontrar qualquer informacao rapidamente.</p>
          </AlertBox>

          {todasAsTelas.map((tela, index) => (
            <div key={tela.id} id={`tela-${tela.id}`} className="bg-white border-2 border-gray-200 rounded-xl p-6 mb-8 scroll-mt-20">
              <div className="flex items-center gap-4 mb-4">
                <div className={`bg-${tela.cor}-100 p-3 rounded-xl`}>
                  <tela.icone className={`h-8 w-8 text-${tela.cor}-600`} />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">{index + 1}. {tela.nome}</h3>
                  <p className="text-gray-500 flex items-center gap-2">
                    <ChevronRight className="h-4 w-4" /> {tela.caminho}
                  </p>
                </div>
              </div>

              {/* ASCII Art */}
              <div className="bg-gray-900 text-green-400 font-mono text-xs p-4 rounded-lg mb-6 overflow-x-auto">
                <pre>{tela.ascii}</pre>
              </div>

              {/* Objetivo */}
              <div className="mb-6">
                <h4 className="font-bold text-lg text-gray-900 mb-2 flex items-center gap-2">
                  <Target className="h-5 w-5 text-blue-600" /> Objetivo da Tela
                </h4>
                <p className="text-gray-700">{tela.objetivo}</p>
              </div>

              {/* Quando Usar */}
              <div className="mb-6">
                <h4 className="font-bold text-lg text-gray-900 mb-2 flex items-center gap-2">
                  <Clock className="h-5 w-5 text-green-600" /> Quando Usar Esta Tela?
                </h4>
                <ul className="space-y-2">
                  {tela.quandoUsar.map((uso, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{uso}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Elementos */}
              <div className="mb-6">
                <h4 className="font-bold text-lg text-gray-900 mb-2 flex items-center gap-2">
                  <Boxes className="h-5 w-5 text-purple-600" /> Elementos Principais
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {tela.elementos.map((elem, i) => (
                    <div key={i} className="bg-gray-50 rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded">{elem.tipo}</span>
                        <span className="font-semibold text-gray-900">{elem.nome}</span>
                      </div>
                      <p className="text-sm text-gray-600">{elem.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Historia */}
              <div className="mb-6">
                <h4 className="font-bold text-lg text-gray-900 mb-2 flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-orange-600" /> Historia de Uso (Mini-Cenario)
                </h4>
                <div className="bg-orange-50 border-l-4 border-orange-400 p-4 rounded-r-lg">
                  <p className="text-gray-700 italic">"{tela.historia}"</p>
                </div>
              </div>

              {/* Cuidados */}
              <div>
                <h4 className="font-bold text-lg text-gray-900 mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-600" /> Cuidados Importantes
                </h4>
                <div className="space-y-2">
                  {tela.cuidados.map((cuidado, i) => (
                    <div key={i} className="flex items-start gap-2 bg-red-50 p-3 rounded-lg">
                      <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{cuidado}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </CollapsibleSection>

        {/* SECAO 4: FEATURES DE ML */}
        <CollapsibleSection id="features-ml" title="Features de Machine Learning (40+ Caracteristicas)" icon={Brain} color="orange">
          <AlertBox type="info" title="O que sao Features?">
            <p className="text-lg">
              <strong>Features</strong> sao as "perguntas" que a IA faz sobre cada transacao. 
              Assim como um medico analisa sintomas (febre, pressao, dor) para diagnosticar uma doenca, 
              nossa IA analisa features (valor, horario, destinatario, dispositivo) para detectar fraude.
            </p>
          </AlertBox>

          {Object.entries(todasAsFeatures).map(([key, categoria]) => (
            <div key={key} className="mb-8">
              <div className={`bg-${categoria.cor}-50 rounded-xl p-4 mb-4`}>
                <div className="flex items-center gap-3 mb-2">
                  <categoria.icone className={`h-8 w-8 text-${categoria.cor}-600`} />
                  <h4 className="text-xl font-bold text-gray-900">{categoria.categoria}</h4>
                </div>
                <p className="text-gray-700">{categoria.descricao}</p>
                <p className="text-sm text-gray-500 mt-2 italic">Analogia: {categoria.analogia}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {categoria.features.map((feature, i) => (
                  <FeatureCard key={i} feature={feature} categoria={categoria} />
                ))}
              </div>
            </div>
          ))}
        </CollapsibleSection>

        {/* SECAO 5: DATASETS */}
        <CollapsibleSection id="datasets" title="DataSets Utilizados (Origem dos Dados)" icon={Database} color="green">
          <AlertBox type="info" title="O que sao DataSets?">
            <p className="text-lg">
              <strong>DataSets</strong> sao grandes colecoes de dados organizados, como uma planilha gigante. 
              E deles que a IA aprende. Pense em um fichario com milhoes de fichas - cada ficha e uma transacao, 
              e cada campo da ficha e uma informacao (valor, hora, cliente, etc.).
            </p>
          </AlertBox>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {datasets.map((dataset) => (
              <div key={dataset.id} className={`bg-white rounded-xl shadow-lg overflow-hidden border-t-4 border-${dataset.cor}-500`}>
                <div className={`bg-${dataset.cor}-50 p-4`}>
                  <div className="flex items-center gap-3">
                    <dataset.icone className={`h-10 w-10 text-${dataset.cor}-600`} />
                    <div>
                      <h4 className="font-bold text-lg">{dataset.nome}</h4>
                      <p className="text-sm text-gray-600">{dataset.origem}</p>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                      <div className="text-lg font-bold text-gray-900">{dataset.registros}</div>
                      <div className="text-xs text-gray-500">Registros</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 text-center">
                      <div className="text-lg font-bold text-red-600">{dataset.taxaFraude}</div>
                      <div className="text-xs text-gray-500">Taxa Fraude</div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 mb-4">{dataset.descricao}</p>
                  <h5 className="font-semibold text-gray-900 mb-2">Campos Principais:</h5>
                  <ul className="text-xs text-gray-600 space-y-1">
                    {dataset.campos.slice(0, 4).map((campo, i) => (
                      <li key={i}><code className="bg-gray-100 px-1 rounded">{campo.nome}</code>: {campo.desc}</li>
                    ))}
                  </ul>
                  <div className="mt-4 bg-blue-50 p-3 rounded-lg">
                    <p className="text-xs text-blue-800"><strong>Uso:</strong> {dataset.uso}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* SECAO 6: TRANSFER LEARNING */}
        <CollapsibleSection id="transfer-learning" title="Transfer Learning Explicado" icon={Layers} color="purple">
          <AlertBox type="tip" title="O que e Transfer Learning?">
            <p className="text-lg">
              <strong>Transfer Learning</strong> e como aprender a dirigir carro e depois usar esse conhecimento 
              para aprender a dirigir caminhao mais rapido. Voce nao comeca do zero - aproveita o que ja sabe. 
              Nossa IA faz o mesmo: primeiro aprende com dados globais, depois se adapta ao Brasil.
            </p>
          </AlertBox>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {transferLearningFases.map((fase) => (
              <div key={fase.fase} className={`bg-white rounded-xl shadow-lg overflow-hidden border-l-4 border-${fase.cor}-500`}>
                <div className={`bg-${fase.cor}-50 p-4`}>
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 bg-${fase.cor}-500 text-white rounded-full flex items-center justify-center text-xl font-bold`}>
                      {fase.fase}
                    </div>
                    <div>
                      <h4 className="font-bold text-lg">{fase.nome}</h4>
                      <p className="text-sm text-gray-600">Duracao: {fase.duracao}</p>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <p className="text-gray-700 mb-3">{fase.descricao}</p>
                  <p className="text-sm text-gray-500 italic mb-4">Analogia: {fase.analogia}</p>
                  
                  <h5 className="font-semibold text-gray-900 mb-2">O que acontece:</h5>
                  <ul className="text-sm text-gray-600 space-y-1 mb-4">
                    {fase.detalhes.map((d, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <ArrowRight className="h-4 w-4 text-gray-400 mt-0.5" />
                        {d}
                      </li>
                    ))}
                  </ul>

                  <div className="grid grid-cols-4 gap-2">
                    {Object.entries(fase.metricas).map(([key, val]) => (
                      <div key={key} className="bg-gray-50 rounded p-2 text-center">
                        <div className="text-sm font-bold text-gray-900">{val}</div>
                        <div className="text-[10px] text-gray-500">{key}</div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 bg-green-50 p-3 rounded-lg">
                    <p className="text-xs text-green-800"><strong>Resultado:</strong> {fase.resultado}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

        {/* SECAO 7: FLUXO PONTA A PONTA */}
        <CollapsibleSection id="fluxo-ponta-ponta" title="Fluxo Ponta a Ponta: Do Dado a Decisao" icon={Workflow} color="blue">
          <AlertBox type="info" title="Como tudo se conecta">
            <p>Veja como os dados entram no sistema, passam pela IA, aparecem nas telas e geram decisoes.</p>
          </AlertBox>

          <div className="bg-gray-900 text-white rounded-xl p-6 font-mono text-sm overflow-x-auto">
            <pre>{`
+==================================================================================+
|                           FLUXO PONTA A PONTA                                    |
+==================================================================================+

    ENTRADA                    PROCESSAMENTO                         SAIDA
    ═══════                    ═════════════                         ═════

  +-----------+            +------------------+                 +-------------+
  |  Cliente  |            |  40+ FEATURES    |                 |   DECISAO   |
  |  faz PIX  | ---------> |  Valor, hora,    | --------------> |   Score     |
  |  no App   |    5ms     |  device, local   |      25ms       |   0-100     |
  +-----------+            +------------------+                 +-------------+
       |                           |                                  |
       |                           v                                  |
       |                   +------------------+                       |
       |                   |  3 MODELOS IA    |                       |
       |                   |  RF + GB + CB    |                       |
       |                   |  votam juntos    |                       |
       |                   +------------------+                       |
       |                           |                                  |
       v                           v                                  v
  +-----------+            +------------------+                 +-------------+
  |  DATASET  |            |  SCORE < 30      | --------------> |   APROVAR   |
  |  Producao |            |  Baixo risco     |                 |   (verde)   |
  |  4.467 tx |            +------------------+                 +-------------+
  +-----------+            +------------------+                 +-------------+
       |                   |  SCORE 30-70     | --------------> |   REVISAR   |
       |                   |  Zona cinza      |                 |   (amarelo) |
       v                   +------------------+                 +-------------+
  +-----------+            +------------------+                 +-------------+
  |  FEEDBACK |            |  SCORE > 70      | --------------> |   BLOQUEAR  |
  |  Analista |            |  Alto risco      |                 |   (verm.)   |
  |  ~50/dia  |            +------------------+                 +-------------+
  +-----------+                    |                                  |
       |                           |                                  |
       +---------------------------+----------------------------------+
                                   |
                                   v
                          +------------------+
                          |  TELAS DO        |
                          |  SISTEMA         |
                          |  Dashboard,      |
                          |  Alertas, etc    |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  ANALISTA        |
                          |  TOMA DECISAO    |
                          |  Humano decide   |
                          +------------------+

                     TEMPO TOTAL: 37 milissegundos
            `}</pre>
          </div>

          <div className="mt-6">
            <h4 className="font-bold text-lg mb-4">Historia do Fluxo (Exemplo Real)</h4>
            <div className="bg-blue-50 rounded-xl p-6">
              <p className="text-gray-700 leading-relaxed">
                <strong>14:32:01</strong> - Joao abre o app do banco e inicia um PIX de R$ 4.850 para um CPF que nunca usou antes.
                <br/><br/>
                <strong>14:32:02</strong> - O sistema captura a transacao e extrai 40+ features: valor (R$ 4.850), hora (14h), 
                dispositivo (mesmo celular de sempre), destinatario (novo!), localizacao (Sao Paulo).
                <br/><br/>
                <strong>14:32:03</strong> - Os 3 modelos de IA analisam. Random Forest: 85%, Gradient Boosting: 88%, CatBoost: 86%. 
                Media ponderada: Score 87/100.
                <br/><br/>
                <strong>14:32:04</strong> - Score 87 {'>'} 70, portanto BLOQUEAR automaticamente. 
                Joao recebe SMS: "Sua transacao esta em analise de seguranca."
                <br/><br/>
                <strong>14:35:00</strong> - Carlos, analista, ve o alerta na fila. Abre a investigacao, 
                verifica historico, liga para Joao.
                <br/><br/>
                <strong>14:40:00</strong> - Joao confirma que NAO fez a transacao. Era golpe do WhatsApp! 
                Carlos confirma fraude, bloqueia definitivamente, adiciona CPF destino na HOT List.
                <br/><br/>
                <strong>Resultado:</strong> R$ 4.850 salvos. Cliente protegido. IA aprendeu com o feedback.
              </p>
            </div>
          </div>
        </CollapsibleSection>

        {/* SECAO 8: DICAS E BOAS PRATICAS */}
        <CollapsibleSection id="dicas" title="Dicas de Uso, Boas Praticas e Cuidados" icon={Lightbulb} color="orange">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-lg mb-4 flex items-center gap-2 text-green-700">
                <ThumbsUp className="h-5 w-5" /> Boas Praticas
              </h4>
              <ul className="space-y-3">
                {[
                  'Sempre comece o dia verificando o Dashboard',
                  'Assume alertas antes de investigar (evita duplicidade)',
                  'De feedback para a IA - ela aprende com voce',
                  'Use filtros nas buscas para ser mais rapido',
                  'Documente suas decisoes - auditoria agradece',
                  'Na duvida, escale para um senior',
                  'Simule antes de mudar thresholds',
                  'Revise a VIP List periodicamente'
                ].map((dica, i) => (
                  <li key={i} className="flex items-start gap-2 bg-green-50 p-3 rounded-lg">
                    <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-700">{dica}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-lg mb-4 flex items-center gap-2 text-red-700">
                <AlertTriangle className="h-5 w-5" /> Cuidados Importantes
              </h4>
              <ul className="space-y-3">
                {[
                  'NUNCA mude thresholds sem simular o impacto',
                  'NAO adicione na HOT List sem investigacao completa',
                  'Latencia > 100ms e critica - avise TI imediatamente',
                  'Dados incorretos prejudicam a IA',
                  'Nao ignore alertas vermelhos',
                  'CPFs mascarados sao protecao LGPD - respeite',
                  'Logs de auditoria NAO podem ser apagados',
                  'Feedback errado pode piorar a IA'
                ].map((cuidado, i) => (
                  <li key={i} className="flex items-start gap-2 bg-red-50 p-3 rounded-lg">
                    <XCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
                    <span className="text-gray-700">{cuidado}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </CollapsibleSection>

        {/* SECAO 9: FAQ */}
        <CollapsibleSection id="faq" title="FAQ - Perguntas Frequentes" icon={HelpCircle} color="blue">
          <div className="space-y-4">
            {faq.map((item, i) => (
              <div key={i} className="border border-gray-200 rounded-lg overflow-hidden">
                <div className="bg-gray-50 p-4">
                  <h4 className="font-bold text-gray-900 flex items-center gap-2">
                    <HelpCircle className="h-5 w-5 text-blue-500" />
                    {item.pergunta}
                  </h4>
                </div>
                <div className="p-4 bg-white">
                  <p className="text-gray-700">{item.resposta}</p>
                </div>
              </div>
            ))}
          </div>
        </CollapsibleSection>

      </div>
    </div>
  );
}

export default Manual;
