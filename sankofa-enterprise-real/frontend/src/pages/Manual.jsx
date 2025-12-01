import { useState, useRef, useEffect } from 'react';
import { ChevronDown, ChevronUp, BookOpen, Users, Target, Shield, AlertTriangle, Clock, Zap, Eye, Brain, Settings, FileText, BarChart3, Database, Bell, Lock, Star, CheckCircle, XCircle, TrendingUp, Phone, Building, HelpCircle, Search, Filter, Download, Upload, RefreshCw, Play, Pause, Edit, Trash2, Plus, ArrowRight, ArrowLeft, Info, MessageSquare, ThumbsUp, ThumbsDown, Activity, Cpu, Server, Globe, Calendar, DollarSign, Percent, Hash, List, Grid, PieChart, LineChart, Table, Map, Flag, Award, Bookmark, ExternalLink, Copy, Share, Mail, Send, Layers, GitBranch, Box, Terminal, Code, Workflow, Boxes, Network, Gauge, Timer, Sparkles, GraduationCap, Lightbulb, BookMarked, CircuitBoard, Home, ChevronRight, User, Fingerprint, CreditCard, Banknote, Smartphone, MapPin, AlertCircle, ShieldCheck, Scale, Gavel, FileCheck, ClipboardList, Monitor, Headphones, Coffee, Sunrise, Sun, Moon, History } from 'lucide-react';
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
    id: 'research-ml',
    nome: 'Modulos de Pesquisa ML',
    icone: Brain,
    caminho: 'Menu > Documentacao > Modulos ML',
    rota: '/documentation',
    modulo: 'ML Avancado',
    cor: 'indigo',
    objetivo: '4 modulos avancados de ML baseados em pesquisas academicas: Bahnsen Feature Engineering, PIX Fraud Taxonomy, NLP Social Engineering, e Transfer Learning.',
    quandoUsar: [
      'Para entender como a IA detecta fraudes sofisticadas',
      'Para analisar transacoes PIX de alto risco',
      'Para detectar golpes de engenharia social (phishing, smishing)',
      'Para treinar modelos com datasets externos'
    ],
    elementos: [
      { nome: 'Bahnsen Engine', tipo: 'Modulo', desc: 'Gera 62+ features temporais por transacao' },
      { nome: 'PIX Taxonomy', tipo: 'Modulo', desc: 'Detecta 10+ tipos de fraude PIX brasileira' },
      { nome: 'NLP Detector', tipo: 'Modulo', desc: 'Identifica phishing, smishing e manipulacao' },
      { nome: 'Transfer Learning', tipo: 'Pipeline', desc: 'Suporte a 4 datasets externos (17M+ transacoes)' }
    ],
    historia: 'Carlos recebe um alerta de transacao PIX suspeita. O sistema mostra que foi detectado software de acesso remoto (Mao Fantasma) com 100% de probabilidade de fraude. O NLP tambem detectou mensagem de phishing anterior. Acao recomendada: BLOQUEAR.',
    cuidados: [
      'Modulos geram explicacoes compativeis com LGPD',
      'Scores altos (>80%) indicam alto risco - revisar imediatamente',
      'NLP funciona melhor com textos em portugues brasileiro'
    ],
    ascii: `
+--------------------------------------------------+
|  MODULOS DE PESQUISA ML v2.0                     |
+--------------------------------------------------+
|  +------------------+  +------------------+      |
|  | BAHNSEN          |  | PIX TAXONOMY     |      |
|  | 62+ features     |  | 10+ tipos fraude |      |
|  | Temporal/Ciclico |  | Mao Fantasma     |      |
|  | [ATIVO]          |  | Clone WhatsApp   |      |
|  +------------------+  +------------------+      |
|                                                  |
|  +------------------+  +------------------+      |
|  | NLP DETECTOR     |  | TRANSFER LEARN.  |      |
|  | Phishing: 67%    |  | 4 datasets       |      |
|  | Smishing: 71%    |  | 17M+ transacoes  |      |
|  | [ATIVO]          |  | [CONFIGURAVEL]   |      |
|  +------------------+  +------------------+      |
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
    red: 'bg-red-50 border-red-200 text-red-800',
    indigo: 'bg-indigo-50 border-indigo-200 text-indigo-800'
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
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Shield className="h-4 w-4" /> Compliance: LGPD/BACEN/PCI DSS</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Brain className="h-4 w-4" /> 40+ Features de IA</span>
          <span className="flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full"><Database className="h-4 w-4" /> 3 DataSets Explicados</span>
        </div>

        {/* NAVEGACAO RAPIDA */}
        <div className="flex flex-wrap gap-2 mt-6">
          {[
            { id: 'bem-vindo', label: 'Inicio', icon: Home },
            { id: 'fluxo-ponta-ponta', label: 'Dia a Dia', icon: Calendar },
            { id: 'features-ml', label: 'Features de IA', icon: Brain },
            { id: 'datasets', label: 'DataSets', icon: Database },
            { id: 'transfer-learning', label: 'Transfer Learning', icon: Layers },
            { id: 'manual-telas', label: 'Todas as Telas', icon: Grid },
            { id: 'dicas', label: 'Compliance', icon: Shield },
            { id: 'cenarios-reais', label: 'Cenarios Reais', icon: Flag },
            { id: 'faq', label: 'Glossario', icon: BookOpen }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => scrollToSection(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                activeSection === tab.id
                  ? 'bg-white text-blue-700 shadow-lg'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
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

        {/* SECAO 9: JORNADA COMPLETA DA REQUISICAO */}
        <CollapsibleSection id="jornada-requisicao" title="Jornada da Requisicao: Do JSON ao Veredito" icon={Code} color="indigo">
          
          {/* INTRODUCAO */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-6">
            <h3 className="text-xl font-bold text-indigo-800 mb-3 flex items-center gap-2">
              <Terminal className="h-6 w-6" /> O Que Voce Vai Aprender Nesta Secao
            </h3>
            <p className="text-gray-700 mb-4">
              Aqui vamos abrir a "caixa preta" do sistema e mostrar EXATAMENTE o que acontece quando uma transacao 
              chega para ser analisada. Voce vai entender:
            </p>
            <ul className="space-y-2">
              {[
                'Como e o JSON que chega no sistema (todos os campos explicados)',
                'O caminho completo da requisicao dentro do motor de decisao',
                'Como cada campo influencia na decisao final',
                'O JSON de resposta e o que cada campo significa',
                'Exemplos reais de FRAUDE, SUSPEITA e APROVADO'
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-2 text-gray-700">
                  <CheckCircle className="h-4 w-4 text-indigo-500" />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          {/* JSON DE ENTRADA */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <ArrowRight className="h-6 w-6 text-blue-500" /> 1. JSON de ENTRADA (O Que Chega no Sistema)
            </h3>
            
            <div className="bg-gray-900 rounded-xl p-4 mb-4 overflow-x-auto">
              <pre className="text-green-400 text-sm font-mono whitespace-pre">{`{
  "transactions": [
    {
      "transaction_id": "TXN123456789",
      "customer_id": "CPF***456***",
      "amount": 4850.00,
      "channel": "PIX",
      "hour": 14,
      "merchant_id": "MERC001",
      "merchant_category": "transferencia_pf",
      "device_id": "DEV-ABC123",
      "ip_address": "189.***.***.45",
      "latitude": -23.5505,
      "longitude": -46.6333,
      "is_new_device": false,
      "is_new_recipient": true,
      "velocity_score": 0.3,
      "avg_amount_30d": 850.00,
      "transaction_count_24h": 2
    }
  ],
  "include_explanation": true,
  "fast_mode": true
}`}</pre>
            </div>

            <h4 className="font-bold text-lg mb-4">Explicacao de CADA Campo:</h4>
            <div className="space-y-4">
              {[
                {
                  campo: 'transaction_id',
                  nome: 'ID da Transacao',
                  tipo: 'string',
                  peso: 'Identificacao',
                  desc: 'Codigo unico que identifica esta transacao. Serve para rastrear e auditar.',
                  exemplo: 'TXN123456789',
                  importancia: 'Essencial para rastreamento'
                },
                {
                  campo: 'customer_id',
                  nome: 'ID do Cliente (CPF)',
                  tipo: 'string',
                  peso: 'Alto',
                  desc: 'CPF mascarado do cliente. Usado para buscar historico e comportamento.',
                  exemplo: 'CPF***456***',
                  importancia: 'Historico do cliente afeta decisao'
                },
                {
                  campo: 'amount',
                  nome: 'Valor da Transacao',
                  tipo: 'number',
                  peso: 'Muito Alto',
                  desc: 'Valor em reais. Valores muito altos ou fora do padrao aumentam o risco.',
                  exemplo: '4850.00',
                  importancia: 'Campo mais importante na decisao'
                },
                {
                  campo: 'channel',
                  nome: 'Canal',
                  tipo: 'string (PIX, TED, CARTAO, BOLETO)',
                  peso: 'Alto',
                  desc: 'Por onde a transacao foi feita. PIX tem regras diferentes de cartao.',
                  exemplo: 'PIX',
                  importancia: 'PIX exige latencia menor 50ms'
                },
                {
                  campo: 'hour',
                  nome: 'Hora da Transacao',
                  tipo: 'number (0-23)',
                  peso: 'Alto',
                  desc: 'Hora do dia. Transacoes de madrugada (0-5h) sao mais suspeitas.',
                  exemplo: '14 (2 da tarde)',
                  importancia: 'Horario incomum aumenta risco'
                },
                {
                  campo: 'merchant_id',
                  nome: 'ID do Comerciante',
                  tipo: 'string',
                  peso: 'Medio',
                  desc: 'Identificador do destino/loja. Usado para verificar se e destino conhecido.',
                  exemplo: 'MERC001',
                  importancia: 'Destino novo = mais risco'
                },
                {
                  campo: 'device_id',
                  nome: 'ID do Dispositivo',
                  tipo: 'string',
                  peso: 'Alto',
                  desc: 'Identificador unico do celular/computador. Dispositivo novo e suspeito.',
                  exemplo: 'DEV-ABC123',
                  importancia: 'Dispositivo desconhecido = alerta'
                },
                {
                  campo: 'ip_address',
                  nome: 'Endereco IP',
                  tipo: 'string',
                  peso: 'Alto',
                  desc: 'IP de origem. IPs de VPN, Tor ou paises estranhos aumentam risco.',
                  exemplo: '189.***.***.45',
                  importancia: 'IP diferente do usual = suspeito'
                },
                {
                  campo: 'latitude/longitude',
                  nome: 'Geolocalizacao',
                  tipo: 'number',
                  peso: 'Medio',
                  desc: 'Coordenadas geograficas. Usadas para verificar se faz sentido.',
                  exemplo: '-23.55, -46.63 (Sao Paulo)',
                  importancia: 'Localizacao impossivel = fraude'
                },
                {
                  campo: 'is_new_device',
                  nome: 'Dispositivo Novo?',
                  tipo: 'boolean',
                  peso: 'Alto',
                  desc: 'Se e a primeira vez que o cliente usa este dispositivo.',
                  exemplo: 'false (ja usou antes)',
                  importancia: 'true = aumenta score de risco'
                },
                {
                  campo: 'is_new_recipient',
                  nome: 'Destinatario Novo?',
                  tipo: 'boolean',
                  peso: 'Alto',
                  desc: 'Se e a primeira vez que envia para este destino.',
                  exemplo: 'true (primeira vez)',
                  importancia: 'true = muito mais risco'
                },
                {
                  campo: 'velocity_score',
                  nome: 'Score de Velocidade',
                  tipo: 'number (0-1)',
                  peso: 'Alto',
                  desc: 'Indica se o cliente esta fazendo muitas transacoes rapido.',
                  exemplo: '0.3 (normal)',
                  importancia: 'Maior 0.7 = possivel ataque'
                },
                {
                  campo: 'avg_amount_30d',
                  nome: 'Media 30 Dias',
                  tipo: 'number',
                  peso: 'Muito Alto',
                  desc: 'Media de valor das transacoes nos ultimos 30 dias.',
                  exemplo: 'R$ 850,00',
                  importancia: 'Valor muito acima = suspeito'
                },
                {
                  campo: 'transaction_count_24h',
                  nome: 'Transacoes 24h',
                  tipo: 'number',
                  peso: 'Medio',
                  desc: 'Quantas transacoes o cliente fez nas ultimas 24 horas.',
                  exemplo: '2 transacoes',
                  importancia: 'Muitas = possivel fraude'
                }
              ].map((item, i) => (
                <div key={i} className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <code className="bg-gray-100 px-2 py-1 rounded text-blue-600 font-mono">{item.campo}</code>
                    <span className="text-gray-500">|</span>
                    <span className="font-medium">{item.nome}</span>
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      item.peso === 'Muito Alto' ? 'bg-red-100 text-red-700' :
                      item.peso === 'Alto' ? 'bg-orange-100 text-orange-700' :
                      item.peso === 'Medio' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      Peso: {item.peso}
                    </span>
                  </div>
                  <p className="text-gray-700 mb-2">{item.desc}</p>
                  <div className="flex flex-wrap gap-4 text-sm">
                    <span className="text-gray-500">Tipo: <code className="text-purple-600">{item.tipo}</code></span>
                    <span className="text-gray-500">Exemplo: <code className="text-green-600">{item.exemplo}</code></span>
                  </div>
                  <p className="text-indigo-600 text-sm mt-2 font-medium">{item.importancia}</p>
                </div>
              ))}
            </div>
          </div>

          {/* DIAGRAMA DA JORNADA */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Workflow className="h-6 w-6 text-purple-500" /> 2. Jornada Completa da Requisicao
            </h3>

            <div className="bg-gray-100 rounded-xl p-4 mb-4 overflow-x-auto">
              <pre className="text-gray-800 text-sm font-mono whitespace-pre">{`
     JORNADA COMPLETA: DO JSON AO VEREDITO (37ms)
     =============================================

     +------------------+
     |  JSON DE ENTRADA |  <-- Transacao chega do app/web do cliente
     |  (request body)  |
     +--------+---------+
              |
              v  [1-2ms]
     +------------------+
     |   CONTROLLER     |  <-- Endpoint /api/fraud/predict recebe
     |   Flask API      |      Valida se JSON esta correto
     +--------+---------+
              |
              v  [1ms]
     +------------------+
     |   VALIDACAO      |  <-- Verifica campos obrigatorios
     |   Cerberus       |      Valida tipos de dados
     +--------+---------+
              |
              v  [2-3ms]
     +------------------+
     |  CACHE CHECK     |  <-- Verifica se ja analisou recentemente
     |  SimpleCache     |      Cache hit = resposta instantanea!
     +--------+---------+
              |
       [cache miss]
              |
              v  [3-5ms]
     +------------------+
     |  ENRIQUECIMENTO  |  <-- Busca dados extras do cliente:
     |  PostgreSQL      |      - Historico de transacoes
     +------------------+      - Media de valores
                               - Transacoes recentes
                               - Flags de fraude anterior
              |
              v  [5-8ms]
     +------------------+
     |  FEATURE         |  <-- Cria 16 features derivadas:
     |  ENGINEERING     |      - amount_log, is_night
     |                  |      - suspicious_combo, etc
     +--------+---------+
              |
              v  [10-15ms]
     +------------------+
     |  MOTOR DE IA     |  <-- 3 modelos votam em paralelo:
     |  Ensemble Vote   |      
     |                  |      Random Forest (40%)   --> 0.82
     |  RF + GB + LR    |      Gradient Boost (40%)  --> 0.78
     |                  |      Logistic Reg (20%)    --> 0.75
     +--------+---------+      
              |               Media Ponderada = 0.79
              v  [2ms]
     +------------------+
     |  REGRAS HARD     |  <-- Aplica regras de negocio:
     |  BusinessRules   |      - VIP List? (whitelist)
     +------------------+      - HOT List? (blacklist)
                               - Hard Rules customizadas
              |
              v  [1ms]
     +------------------+
     |  CALIBRACAO      |  <-- Aplica threshold atual:
     |  Threshold Check |      Score >= 70? BLOQUEAR
     +--------+---------+      Score 30-69? REVISAR
              |               Score < 30? APROVAR
              v  [2ms]
     +------------------+
     |  SALVAR NO DB    |  <-- Persiste no PostgreSQL:
     |  PostgreSQL      |      - Transacao completa
     +------------------+      - Decisao tomada
                               - Timestamp
              |
              v  [1ms]
     +------------------+
     |  JSON DE SAIDA   |  <-- Monta resposta final
     |  (response)      |      com todos os detalhes
     +------------------+

     TEMPO TOTAL: ~37ms (SLA PIX: <50ms) ✅
`}</pre>
            </div>

            <h4 className="font-bold text-lg mb-4">Explicacao Passo a Passo:</h4>
            <div className="space-y-4">
              {[
                {
                  passo: '1. Chegada do JSON',
                  tempo: '1-2ms',
                  desc: 'O aplicativo do banco envia a transacao como JSON para o endpoint /api/fraud/predict. O Flask recebe e faz um parse inicial.',
                  detalhe: 'Se o JSON estiver mal formatado, retorna erro 400 imediatamente.'
                },
                {
                  passo: '2. Validacao de Campos',
                  tempo: '1ms',
                  desc: 'O sistema verifica se todos os campos obrigatorios estao presentes e se os tipos estao corretos.',
                  detalhe: 'Campo "amount" precisa ser numero, "channel" precisa ser PIX/TED/CARTAO/BOLETO.'
                },
                {
                  passo: '3. Verificacao de Cache',
                  tempo: '2-3ms',
                  desc: 'Antes de processar, verifica se essa mesma transacao ja foi analisada recentemente (ultimos 30 segundos).',
                  detalhe: 'Se encontrar no cache, retorna resultado imediatamente. Isso economiza 90% do tempo!'
                },
                {
                  passo: '4. Enriquecimento de Dados',
                  tempo: '3-5ms',
                  desc: 'Busca no PostgreSQL o historico completo do cliente: transacoes anteriores, media de valores, ultima transacao, etc.',
                  detalhe: 'Esses dados extras sao cruciais para entender se o comportamento e normal.'
                },
                {
                  passo: '5. Engenharia de Features',
                  tempo: '5-8ms',
                  desc: 'Cria 16 novas caracteristicas derivadas dos dados originais.',
                  detalhe: 'Exemplos: amount_log (log do valor), is_night (se e madrugada), suspicious_combo (valor alto + horario estranho).'
                },
                {
                  passo: '6. Motor de IA (Ensemble)',
                  tempo: '10-15ms',
                  desc: 'Os 3 modelos de Machine Learning analisam a transacao em paralelo e votam.',
                  detalhe: 'Random Forest (40% do peso), Gradient Boosting (40%), Logistic Regression (20%). A media ponderada gera o score final.'
                },
                {
                  passo: '7. Regras de Negocio',
                  tempo: '2ms',
                  desc: 'Verifica as Hard Rules, VIP List e HOT List. Pode sobrescrever a decisao da IA.',
                  detalhe: 'Se o CPF esta na HOT List = fraude automatica. Se esta na VIP List = aprovado automatico.'
                },
                {
                  passo: '8. Calibracao e Decisao',
                  tempo: '1ms',
                  desc: 'Compara o score com os thresholds configurados para decidir: APROVAR, REVISAR ou BLOQUEAR.',
                  detalhe: 'Os thresholds podem ser ajustados na tela de Calibracao.'
                },
                {
                  passo: '9. Persistencia',
                  tempo: '2ms',
                  desc: 'Salva a transacao e a decisao no PostgreSQL para auditoria e aprendizado futuro.',
                  detalhe: 'Para PIX, usa fila assincrona para nao atrasar a resposta.'
                },
                {
                  passo: '10. Resposta Final',
                  tempo: '1ms',
                  desc: 'Monta o JSON de resposta com todos os detalhes e envia de volta.',
                  detalhe: 'Inclui score, decisao, razoes e explicacao LGPD-compliant.'
                }
              ].map((item, i) => (
                <div key={i} className="flex gap-4 bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex-shrink-0 w-16">
                    <div className="bg-purple-100 text-purple-700 rounded-full h-10 w-10 flex items-center justify-center font-bold">
                      {i + 1}
                    </div>
                    <div className="text-xs text-gray-500 mt-1 text-center">{item.tempo}</div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900">{item.passo}</h5>
                    <p className="text-gray-700">{item.desc}</p>
                    <p className="text-gray-500 text-sm mt-1 italic">{item.detalhe}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* JSON DE SAIDA */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <ArrowLeft className="h-6 w-6 text-green-500" /> 3. JSON de SAIDA (O Que o Sistema Responde)
            </h3>

            <div className="bg-gray-900 rounded-xl p-4 mb-4 overflow-x-auto">
              <pre className="text-green-400 text-sm font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [
      {
        "transaction_id": "TXN123456789",
        "is_fraud": true,
        "fraud_probability": 0.87,
        "risk_score": 87.0,
        "risk_level": "HIGH",
        "confidence": 0.92,
        "processing_time_ms": 37.5,
        "model_version": "1.0.0",
        "detection_reason": [
          "Valor 5.7x maior que media do cliente",
          "Destinatario nunca utilizado",
          "Combinacao de alto valor + horario atipico"
        ],
        "timestamp": "2025-11-30T14:32:04.123Z",
        "explanation": {
          "risk_level": "HIGH",
          "explanation_text": "Transacao bloqueada por apresentar multiplos indicadores de risco.",
          "top_risk_factors": [
            {"factor": "amount_deviation", "impact": 0.35, "description": "Valor muito acima do padrao"},
            {"factor": "is_new_recipient", "impact": 0.28, "description": "Primeiro envio para este destino"},
            {"factor": "suspicious_combo", "impact": 0.18, "description": "Combinacao hora + valor"}
          ],
          "top_protective_factors": [
            {"factor": "known_device", "impact": -0.12, "description": "Dispositivo conhecido"}
          ],
          "lgpd_compliant": true
        }
      }
    ],
    "summary": {
      "total": 1,
      "frauds_detected": 1,
      "avg_risk_score": 0.87,
      "model_version": "1.0.0",
      "explanations_included": true
    }
  }
}`}</pre>
            </div>

            <h4 className="font-bold text-lg mb-4">Explicacao de CADA Campo da Resposta:</h4>
            <div className="space-y-3">
              {[
                {
                  campo: 'success',
                  desc: 'Indica se a requisicao foi processada com sucesso. True = tudo ok, False = erro.',
                  valores: 'true ou false'
                },
                {
                  campo: 'is_fraud',
                  desc: 'Decisao final: a transacao e fraude ou nao? Este e o campo mais importante!',
                  valores: 'true (bloquear) ou false (liberar)'
                },
                {
                  campo: 'fraud_probability',
                  desc: 'Probabilidade de fraude calculada pela IA (0 a 1). Quanto maior, mais suspeito.',
                  valores: '0.00 a 1.00 (0.87 = 87% de chance de fraude)'
                },
                {
                  campo: 'risk_score',
                  desc: 'Score de risco de 0 a 100. E a probabilidade multiplicada por 100 para facilitar leitura.',
                  valores: '0 a 100 (87 = alto risco)'
                },
                {
                  campo: 'risk_level',
                  desc: 'Classificacao em texto do nivel de risco.',
                  valores: 'LOW (0-29), MEDIUM (30-69), HIGH (70-100)'
                },
                {
                  campo: 'confidence',
                  desc: 'Quanto o modelo esta confiante na decisao (0 a 1). Confianca baixa = revisar manualmente.',
                  valores: '0.00 a 1.00 (0.92 = 92% de certeza)'
                },
                {
                  campo: 'processing_time_ms',
                  desc: 'Tempo que o sistema levou para processar, em milissegundos. SLA PIX: menor 50ms.',
                  valores: 'Numero em ms (37.5 = muito bom!)'
                },
                {
                  campo: 'detection_reason',
                  desc: 'Lista de razoes que levaram a decisao. Util para o analista entender o "porque".',
                  valores: 'Array de strings explicativas'
                },
                {
                  campo: 'explanation.top_risk_factors',
                  desc: 'Os 3 principais fatores que AUMENTARAM o risco, com impacto numerico.',
                  valores: 'Array com factor, impact e description'
                },
                {
                  campo: 'explanation.top_protective_factors',
                  desc: 'Fatores que DIMINUIRAM o risco (ex: dispositivo conhecido).',
                  valores: 'Array com factor, impact e description'
                },
                {
                  campo: 'lgpd_compliant',
                  desc: 'Confirma que a explicacao esta em conformidade com a LGPD (sem dados sensiveis expostos).',
                  valores: 'true (sempre)'
                }
              ].map((item, i) => (
                <div key={i} className="bg-white border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <code className="bg-green-100 px-2 py-1 rounded text-green-700 font-mono text-sm">{item.campo}</code>
                  </div>
                  <p className="text-gray-700 text-sm">{item.desc}</p>
                  <p className="text-gray-500 text-xs mt-1">Valores: {item.valores}</p>
                </div>
              ))}
            </div>
          </div>

          {/* EXEMPLOS COMPLETOS */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <FileText className="h-6 w-6 text-orange-500" /> 4. Exemplos Completos de Cenarios Reais
            </h3>

            {/* EXEMPLO 1: FRAUDE */}
            <div className="border-2 border-red-200 rounded-xl mb-6 overflow-hidden">
              <div className="bg-red-500 text-white p-4">
                <h4 className="text-lg font-bold flex items-center gap-2">
                  <XCircle className="h-5 w-5" /> CENARIO 1: FRAUDE DETECTADA
                </h4>
                <p className="text-red-100 text-sm">Cartao de credito via web, IP diferente, dispositivo desconhecido</p>
              </div>
              <div className="p-4 bg-white">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Entrada:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono">{`{
  "transactions": [{
    "transaction_id": "TXN-FRAUD-001",
    "customer_id": "CPF***789***",
    "amount": 12500.00,
    "channel": "CARTAO",
    "hour": 3,
    "device_id": "DEV-UNKNOWN-999",
    "ip_address": "45.***.***.12",
    "is_new_device": true,
    "is_new_recipient": true,
    "velocity_score": 0.85,
    "avg_amount_30d": 320.00
  }]
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Resposta:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-red-400 text-xs font-mono">{`{
  "predictions": [{
    "is_fraud": true,
    "risk_score": 94.2,
    "risk_level": "HIGH",
    "detection_reason": [
      "Valor 39x maior que media",
      "Dispositivo desconhecido",
      "Transacao as 03h (madrugada)",
      "IP de VPN detectado",
      "Muitas tentativas recentes"
    ]
  }]
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="mt-4 bg-red-50 rounded-lg p-4">
                  <h5 className="font-bold text-red-800 mb-2">Por Que Foi Bloqueada?</h5>
                  <ul className="space-y-1 text-red-700 text-sm">
                    <li>• Valor R$ 12.500 e 39x maior que a media do cliente (R$ 320)</li>
                    <li>• Feita as 3h da madrugada (horario atipico)</li>
                    <li>• Dispositivo nunca usado antes (is_new_device = true)</li>
                    <li>• IP diferente do habitual (possivel VPN)</li>
                    <li>• Score de velocidade 0.85 = muitas tentativas recentes</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* EXEMPLO 2: SUSPEITA */}
            <div className="border-2 border-yellow-200 rounded-xl mb-6 overflow-hidden">
              <div className="bg-yellow-500 text-white p-4">
                <h4 className="text-lg font-bold flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" /> CENARIO 2: SUSPEITA (REVISAR)
                </h4>
                <p className="text-yellow-100 text-sm">PIX de valor medio, dispositivo novo, IP conhecido</p>
              </div>
              <div className="p-4 bg-white">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Entrada:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono">{`{
  "transactions": [{
    "transaction_id": "TXN-REVIEW-002",
    "customer_id": "CPF***123***",
    "amount": 2800.00,
    "channel": "PIX",
    "hour": 22,
    "device_id": "DEV-NEW-456",
    "ip_address": "189.***.***.78",
    "is_new_device": true,
    "is_new_recipient": false,
    "velocity_score": 0.35,
    "avg_amount_30d": 1200.00
  }]
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Resposta:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-yellow-400 text-xs font-mono">{`{
  "predictions": [{
    "is_fraud": false,
    "risk_score": 52.8,
    "risk_level": "MEDIUM",
    "detection_reason": [
      "Dispositivo novo detectado",
      "Valor 2.3x acima da media",
      "Horario noturno (22h)"
    ],
    "review_recommended": true
  }]
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="mt-4 bg-yellow-50 rounded-lg p-4">
                  <h5 className="font-bold text-yellow-800 mb-2">Por Que Precisa de Revisao?</h5>
                  <ul className="space-y-1 text-yellow-700 text-sm">
                    <li>• Dispositivo novo, mas IP e conhecido (cliente pode ter trocado celular)</li>
                    <li>• Valor acima da media, mas nao absurdamente alto</li>
                    <li>• Destinatario ja recebeu antes (is_new_recipient = false) = bom sinal</li>
                    <li>• Horario noturno, mas nao madrugada</li>
                    <li>• Recomendacao: Analista deve ligar para confirmar</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* EXEMPLO 3: APROVADO */}
            <div className="border-2 border-green-200 rounded-xl mb-6 overflow-hidden">
              <div className="bg-green-500 text-white p-4">
                <h4 className="text-lg font-bold flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" /> CENARIO 3: APROVADO
                </h4>
                <p className="text-green-100 text-sm">Cliente recorrente, dispositivo conhecido, valor tipico</p>
              </div>
              <div className="p-4 bg-white">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Entrada:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono">{`{
  "transactions": [{
    "transaction_id": "TXN-APPROVE-003",
    "customer_id": "CPF***456***",
    "amount": 450.00,
    "channel": "PIX",
    "hour": 10,
    "device_id": "DEV-KNOWN-123",
    "ip_address": "189.***.***.22",
    "is_new_device": false,
    "is_new_recipient": false,
    "velocity_score": 0.12,
    "avg_amount_30d": 520.00
  }]
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2">JSON de Resposta:</h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono">{`{
  "predictions": [{
    "is_fraud": false,
    "risk_score": 12.4,
    "risk_level": "LOW",
    "detection_reason": [
      "Cliente recorrente",
      "Dispositivo conhecido",
      "Valor dentro do padrao",
      "Destinatario habitual"
    ],
    "processing_time_ms": 28.3
  }]
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="mt-4 bg-green-50 rounded-lg p-4">
                  <h5 className="font-bold text-green-800 mb-2">Por Que Foi Aprovada Automaticamente?</h5>
                  <ul className="space-y-1 text-green-700 text-sm">
                    <li>• Valor R$ 450 esta ABAIXO da media do cliente (R$ 520)</li>
                    <li>• Horario comercial (10h da manha)</li>
                    <li>• Dispositivo ja conhecido (is_new_device = false)</li>
                    <li>• Destinatario ja recebeu antes (is_new_recipient = false)</li>
                    <li>• Velocidade normal (0.12 = poucas transacoes recentes)</li>
                    <li>• Processamento ultra-rapido: 28ms (SLA PIX: menor 50ms)</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* EXEMPLOS ADICIONAIS */}
            <h4 className="font-bold text-lg mb-4">Mais Cenarios Importantes:</h4>
            <div className="grid md:grid-cols-2 gap-4">
              {[
                {
                  titulo: 'PIX de Alto Valor as 4h',
                  score: 89,
                  decisao: 'BLOQUEAR',
                  cor: 'red',
                  motivos: ['R$ 45.000 as 04:00', 'Destinatario novo', 'Valor 50x maior que media']
                },
                {
                  titulo: 'TED para Conta Conhecida',
                  score: 18,
                  decisao: 'APROVAR',
                  cor: 'green',
                  motivos: ['Destino ja recebeu 12x', 'Horario comercial', 'Valor dentro do padrao']
                },
                {
                  titulo: 'Cartao em Maquininha Nova',
                  score: 45,
                  decisao: 'REVISAR',
                  cor: 'yellow',
                  motivos: ['POS nunca usado', 'Cliente viajando', 'Valor compativel']
                },
                {
                  titulo: 'CPF na HOT List',
                  score: 100,
                  decisao: 'BLOQUEAR',
                  cor: 'red',
                  motivos: ['CPF marcado como fraude', 'Regra absoluta', 'Bloqueio imediato']
                },
                {
                  titulo: 'Dispositivo com Historico de Fraude',
                  score: 95,
                  decisao: 'BLOQUEAR',
                  cor: 'red',
                  motivos: ['Device ID ja usado em fraude', 'Risco maximo', 'Notificar seguranca']
                },
                {
                  titulo: 'Cliente VIP Fazendo PIX Alto',
                  score: 5,
                  decisao: 'APROVAR',
                  cor: 'green',
                  motivos: ['CPF na VIP List', 'Cliente verificado', 'Aprovacao automatica']
                }
              ].map((cenario, i) => (
                <div key={i} className={`border-2 rounded-lg p-4 ${
                  cenario.cor === 'red' ? 'border-red-200 bg-red-50' :
                  cenario.cor === 'green' ? 'border-green-200 bg-green-50' :
                  'border-yellow-200 bg-yellow-50'
                }`}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-gray-900">{cenario.titulo}</span>
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      cenario.cor === 'red' ? 'bg-red-500 text-white' :
                      cenario.cor === 'green' ? 'bg-green-500 text-white' :
                      'bg-yellow-500 text-white'
                    }`}>
                      Score: {cenario.score} | {cenario.decisao}
                    </span>
                  </div>
                  <ul className="text-sm text-gray-700 space-y-1">
                    {cenario.motivos.map((m, j) => (
                      <li key={j}>• {m}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* SECAO 5: COMBINACOES COMPLETAS COM JSONS */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Grid className="h-6 w-6 text-indigo-500" /> 5. Tabelas de Combinacoes Completas
            </h3>
            
            <p className="text-gray-700 mb-6">
              Esta secao mostra TODAS as combinacoes possiveis de variaveis que afetam a decisao do sistema.
              Para cada combinacao, voce vera o JSON de entrada, o processamento e o JSON de saida.
            </p>

            {/* TABELA 5.1: COMBINACOES DE CANAL */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-3 flex items-center gap-2 text-blue-800">
                <CreditCard className="h-5 w-5" /> 5.1 Combinacoes por CANAL (PIX vs Credito vs Debito)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse bg-white rounded-lg overflow-hidden shadow">
                  <thead className="bg-blue-600 text-white">
                    <tr>
                      <th className="p-3 text-left">Canal</th>
                      <th className="p-3 text-left">SLA</th>
                      <th className="p-3 text-left">Peso Base</th>
                      <th className="p-3 text-left">Regras Especiais</th>
                      <th className="p-3 text-left">Risco Inerente</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-green-600">PIX</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">{'<'}50ms</span></td>
                      <td className="p-3">+15 pontos</td>
                      <td className="p-3 text-sm">Fast mode obrigatorio, cache agressivo, sem explicacao por default</td>
                      <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">MEDIO-ALTO</span></td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-purple-600">CARTAO CREDITO</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">{'<'}2000ms</span></td>
                      <td className="p-3">+10 pontos</td>
                      <td className="p-3 text-sm">Verificacao CVV, limite de credito, historico de compras</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs">ALTO</span></td>
                    </tr>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-blue-600">CARTAO DEBITO</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">{'<'}2000ms</span></td>
                      <td className="p-3">+5 pontos</td>
                      <td className="p-3 text-sm">Verificacao de saldo, limite diario, POS validation</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs">MEDIO</span></td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-gray-600">TED</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">{'<'}5000ms</span></td>
                      <td className="p-3">+3 pontos</td>
                      <td className="p-3 text-sm">Horario bancario, valor maximo, confirmacao dupla</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs">BAIXO</span></td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-bold text-gray-600">BOLETO</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">{'<'}5000ms</span></td>
                      <td className="p-3">+2 pontos</td>
                      <td className="p-3 text-sm">Validacao de codigo de barras, vencimento, beneficiario</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs">BAIXO</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* TABELA 5.2: COMBINACOES DE INTERFACE */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-3 flex items-center gap-2 text-green-800">
                <Monitor className="h-5 w-5" /> 5.2 Combinacoes por INTERFACE (WEB vs POS vs APP)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse bg-white rounded-lg overflow-hidden shadow">
                  <thead className="bg-green-600 text-white">
                    <tr>
                      <th className="p-3 text-left">Interface</th>
                      <th className="p-3 text-left">Peso Risco</th>
                      <th className="p-3 text-left">Verificacoes</th>
                      <th className="p-3 text-left">Pontos Fracos</th>
                      <th className="p-3 text-left">Mitigacao</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold">WEB (Browser)</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+20</span></td>
                      <td className="p-3 text-sm">User-agent, cookies, fingerprint JS</td>
                      <td className="p-3 text-sm text-red-600">VPNs, proxies, extensoes maliciosas</td>
                      <td className="p-3 text-sm">Behavioral biometrics, CAPTCHA</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold">POS (Maquininha)</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">+12</span></td>
                      <td className="p-3 text-sm">Serial do terminal, merchant ID, geo</td>
                      <td className="p-3 text-sm text-orange-600">Maquininha clonada, visor quebrado</td>
                      <td className="p-3 text-sm">Validacao EMV, PIN obrigatorio</td>
                    </tr>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold">APP (Mobile)</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">+5</span></td>
                      <td className="p-3 text-sm">Device ID, biometria, push token</td>
                      <td className="p-3 text-sm text-yellow-600">Root/jailbreak, emuladores</td>
                      <td className="p-3 text-sm">Attestation API, root detection</td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-bold">API (Sistema)</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-bold">+8</span></td>
                      <td className="p-3 text-sm">API key, certificate, IP whitelist</td>
                      <td className="p-3 text-sm text-blue-600">Chave vazada, man-in-middle</td>
                      <td className="p-3 text-sm">mTLS, rate limiting, rotation</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* TABELA 5.3: COMBINACOES DE IP */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-3 flex items-center gap-2 text-purple-800">
                <Globe className="h-5 w-5" /> 5.3 Combinacoes por IP (Conhecido vs Diferente vs Suspeito vs Internacional)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse bg-white rounded-lg overflow-hidden shadow">
                  <thead className="bg-purple-600 text-white">
                    <tr>
                      <th className="p-3 text-left">Tipo IP</th>
                      <th className="p-3 text-left">Impacto Score</th>
                      <th className="p-3 text-left">Exemplos</th>
                      <th className="p-3 text-left">Acao do Sistema</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-green-600">IP Conhecido</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">-10 pontos</span></td>
                      <td className="p-3 text-sm">Mesmo IP das ultimas 10 transacoes</td>
                      <td className="p-3 text-sm text-green-600">Bonus de confianca</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-yellow-600">IP Diferente</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">+15 pontos</span></td>
                      <td className="p-3 text-sm">IP nunca visto, mas mesmo ISP/regiao</td>
                      <td className="p-3 text-sm text-yellow-600">Verificacao adicional</td>
                    </tr>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-orange-600">IP Suspeito</td>
                      <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-bold">+35 pontos</span></td>
                      <td className="p-3 text-sm">VPN, Tor, proxy anonimo, datacenter</td>
                      <td className="p-3 text-sm text-orange-600">Alerta + revisao manual</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-red-600">IP Internacional</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+45 pontos</span></td>
                      <td className="p-3 text-sm">Pais diferente, timezone impossivel</td>
                      <td className="p-3 text-sm text-red-600">Bloqueio preventivo</td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-bold text-red-800">IP na HOT List</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+100 pontos</span></td>
                      <td className="p-3 text-sm">IP ja usado em fraude confirmada</td>
                      <td className="p-3 text-sm text-red-800">BLOQUEIO IMEDIATO</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* TABELA 5.4: COMBINACOES DE DISPOSITIVO */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-3 flex items-center gap-2 text-orange-800">
                <Smartphone className="h-5 w-5" /> 5.4 Combinacoes por DISPOSITIVO (Conhecido vs Desconhecido vs Suspeito)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse bg-white rounded-lg overflow-hidden shadow">
                  <thead className="bg-orange-600 text-white">
                    <tr>
                      <th className="p-3 text-left">Tipo Device</th>
                      <th className="p-3 text-left">Impacto Score</th>
                      <th className="p-3 text-left">Sinais</th>
                      <th className="p-3 text-left">Acao do Sistema</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-green-600">Device Conhecido</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">-15 pontos</span></td>
                      <td className="p-3 text-sm">Mesmo device_id das ultimas transacoes, biometria OK</td>
                      <td className="p-3 text-sm text-green-600">Fast track approval</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-yellow-600">Device Novo</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">+20 pontos</span></td>
                      <td className="p-3 text-sm">Primeira vez que cliente usa este device</td>
                      <td className="p-3 text-sm text-yellow-600">SMS/push de confirmacao</td>
                    </tr>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-orange-600">Device Emulador</td>
                      <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-bold">+40 pontos</span></td>
                      <td className="p-3 text-sm">Bluestacks, Nox, sinais de virtualizacao</td>
                      <td className="p-3 text-sm text-orange-600">Bloqueio + investigacao</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-red-600">Device Root/JB</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+30 pontos</span></td>
                      <td className="p-3 text-sm">Root Android, jailbreak iOS detectado</td>
                      <td className="p-3 text-sm text-red-600">Revisao obrigatoria</td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-bold text-red-800">Device na HOT List</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+100 pontos</span></td>
                      <td className="p-3 text-sm">Device ja usado em fraude confirmada</td>
                      <td className="p-3 text-sm text-red-800">BLOQUEIO IMEDIATO</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* TABELA 5.5: COMBINACOES DE HISTORICO */}
            <div className="mb-6">
              <h4 className="font-bold text-lg mb-3 flex items-center gap-2 text-red-800">
                <History className="h-5 w-5" /> 5.5 Combinacoes por HISTORICO DO CLIENTE
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse bg-white rounded-lg overflow-hidden shadow">
                  <thead className="bg-red-600 text-white">
                    <tr>
                      <th className="p-3 text-left">Tipo Historico</th>
                      <th className="p-3 text-left">Impacto Score</th>
                      <th className="p-3 text-left">Indicadores</th>
                      <th className="p-3 text-left">Tratamento</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-green-600">Historico Normal</td>
                      <td className="p-3"><span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">-20 pontos</span></td>
                      <td className="p-3 text-sm">Padroes consistentes, nenhuma fraude anterior, conta antiga</td>
                      <td className="p-3 text-sm text-green-600">Cliente confiavel</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-blue-600">Cliente VIP</td>
                      <td className="p-3"><span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-bold">-50 pontos</span></td>
                      <td className="p-3 text-sm">Na VIP List, verificado manualmente, alto patrimonio</td>
                      <td className="p-3 text-sm text-blue-600">Aprovacao automatica</td>
                    </tr>
                    <tr className="border-b border-gray-200">
                      <td className="p-3 font-bold text-yellow-600">Historico Inconsistente</td>
                      <td className="p-3"><span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs font-bold">+25 pontos</span></td>
                      <td className="p-3 text-sm">Padroes variados, algumas disputas, conta recente</td>
                      <td className="p-3 text-sm text-yellow-600">Monitoramento ativo</td>
                    </tr>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <td className="p-3 font-bold text-orange-600">Conta Nova (menos 30d)</td>
                      <td className="p-3"><span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-bold">+30 pontos</span></td>
                      <td className="p-3 text-sm">Conta criada recentemente, pouco historico</td>
                      <td className="p-3 text-sm text-orange-600">Limites reduzidos</td>
                    </tr>
                    <tr className="bg-gray-50">
                      <td className="p-3 font-bold text-red-800">Historico Fraudulento</td>
                      <td className="p-3"><span className="bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-bold">+100 pontos</span></td>
                      <td className="p-3 text-sm">Fraude anterior confirmada, chargebacks, na HOT List</td>
                      <td className="p-3 text-sm text-red-800">BLOQUEIO TOTAL</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* SECAO 6: EXEMPLOS COMBINADOS COM JSON COMPLETO */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Code className="h-6 w-6 text-blue-500" /> 6. Exemplos Combinados Completos (JSON Entrada + Saida)
            </h3>
            
            <p className="text-gray-700 mb-6">
              Cada exemplo abaixo mostra o JSON COMPLETO de entrada, o que acontece no motor, e o JSON COMPLETO de saida.
              Use estes exemplos como referencia para entender qualquer cenario que encontrar.
            </p>

            {/* EXEMPLO 6.1: PIX + Device Desconhecido + IP Novo + Valor Alto */}
            <div className="border-2 border-red-300 rounded-xl mb-6 overflow-hidden">
              <div className="bg-red-600 text-white p-4">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-bold">6.1 PIX + Dispositivo Desconhecido + IP Novo + Valor Alto</h4>
                  <span className="bg-white/20 px-3 py-1 rounded-full text-sm">Score: 89 | BLOQUEAR</span>
                </div>
              </div>
              <div className="p-4 bg-white">
                <div className="grid lg:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-blue-500" /> JSON de ENTRADA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "transactions": [{
    "transaction_id": "TXN-PIX-HIGH-001",
    "customer_id": "CPF***234***",
    "amount": 25000.00,
    "channel": "PIX",
    "hour": 2,
    "merchant_id": "UNKNOWN-RECEIVER",
    "merchant_category": "transferencia_pf",
    "device_id": "DEV-NEVER-SEEN-999",
    "ip_address": "45.***.***.128",
    "latitude": -23.5505,
    "longitude": -46.6333,
    "is_new_device": true,
    "is_new_recipient": true,
    "velocity_score": 0.72,
    "avg_amount_30d": 1200.00,
    "transaction_count_24h": 0
  }],
  "include_explanation": true,
  "fast_mode": true
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowLeft className="h-4 w-4 text-green-500" /> JSON de SAIDA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-red-400 text-xs font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [{
      "transaction_id": "TXN-PIX-HIGH-001",
      "is_fraud": true,
      "fraud_probability": 0.89,
      "risk_score": 89.0,
      "risk_level": "HIGH",
      "confidence": 0.94,
      "processing_time_ms": 42.1,
      "model_version": "1.0.0",
      "detection_reason": [
        "Valor 20.8x maior que media",
        "Dispositivo NUNCA visto antes",
        "IP em faixa de VPN suspeita",
        "Transacao as 02h (madrugada)",
        "Destinatario desconhecido",
        "Velocidade de tentativas alta"
      ],
      "timestamp": "2025-11-30T02:15:33.456Z",
      "explanation": {
        "risk_level": "HIGH",
        "explanation_text": "Transacao bloqueada 
por multiplos indicadores de alto risco.",
        "top_risk_factors": [
          {"factor": "amount_deviation", 
           "impact": 0.42, 
           "description": "Valor muito acima"},
          {"factor": "new_device", 
           "impact": 0.28, 
           "description": "Device novo"},
          {"factor": "suspicious_hour", 
           "impact": 0.19, 
           "description": "Madrugada"}
        ],
        "lgpd_compliant": true
      }
    }],
    "summary": {
      "total": 1,
      "frauds_detected": 1,
      "avg_risk_score": 0.89
    }
  }
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="bg-red-50 rounded-lg p-4">
                  <h5 className="font-bold text-red-800 mb-2">O Que Aconteceu no Motor:</h5>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">1. Cache</div>
                      <p className="text-gray-600">Cache MISS - transacao nova, precisa processar</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">2. PostgreSQL</div>
                      <p className="text-gray-600">Buscou historico: media R$1.200, ultima txn ha 3 dias, 0 fraudes</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">3. Ensemble ML</div>
                      <p className="text-gray-600">RF: 0.91 | GB: 0.88 | LR: 0.85 = Media: 0.89</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* EXEMPLO 6.2: Credito + Maquininha + Merchant Suspeito */}
            <div className="border-2 border-orange-300 rounded-xl mb-6 overflow-hidden">
              <div className="bg-orange-600 text-white p-4">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-bold">6.2 Credito + Maquininha (POS) + Merchant Suspeito</h4>
                  <span className="bg-white/20 px-3 py-1 rounded-full text-sm">Score: 72 | BLOQUEAR</span>
                </div>
              </div>
              <div className="p-4 bg-white">
                <div className="grid lg:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-blue-500" /> JSON de ENTRADA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "transactions": [{
    "transaction_id": "TXN-CC-POS-002",
    "customer_id": "CPF***567***",
    "amount": 4999.99,
    "channel": "CARTAO",
    "card_type": "CREDIT",
    "hour": 15,
    "merchant_id": "MERC-SUSPECT-123",
    "merchant_category": "eletronicos",
    "merchant_risk_level": "HIGH",
    "device_id": "POS-TERM-456",
    "terminal_type": "POS",
    "ip_address": null,
    "latitude": -22.9068,
    "longitude": -43.1729,
    "is_new_device": false,
    "is_new_recipient": true,
    "velocity_score": 0.45,
    "avg_amount_30d": 890.00,
    "transaction_count_24h": 2
  }],
  "include_explanation": true
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowLeft className="h-4 w-4 text-green-500" /> JSON de SAIDA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-orange-400 text-xs font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [{
      "transaction_id": "TXN-CC-POS-002",
      "is_fraud": true,
      "fraud_probability": 0.72,
      "risk_score": 72.0,
      "risk_level": "HIGH",
      "confidence": 0.88,
      "processing_time_ms": 156.3,
      "model_version": "1.0.0",
      "detection_reason": [
        "Merchant com historico de fraude",
        "Valor 5.6x maior que media",
        "Valor R$ 4999,99 (evitando limite)",
        "Categoria alto risco: eletronicos",
        "POS em regiao diferente do habitual"
      ],
      "timestamp": "2025-11-30T15:22:18.789Z",
      "explanation": {
        "risk_level": "HIGH",
        "top_risk_factors": [
          {"factor": "merchant_risk", 
           "impact": 0.35, 
           "description": "Loja suspeita"},
          {"factor": "amount_pattern", 
           "impact": 0.22, 
           "description": "Valor quebrado"},
          {"factor": "category_risk", 
           "impact": 0.15, 
           "description": "Eletronicos"}
        ],
        "lgpd_compliant": true
      }
    }]
  }
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="bg-orange-50 rounded-lg p-4">
                  <h5 className="font-bold text-orange-800 mb-2">O Que Aconteceu no Motor:</h5>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">1. Merchant Check</div>
                      <p className="text-gray-600">MERC-SUSPECT-123 tem 5 chargebacks nos ultimos 30 dias</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">2. Pattern Detection</div>
                      <p className="text-gray-600">R$ 4999,99 = tecnica comum para evitar limite de R$ 5000</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">3. Hard Rules</div>
                      <p className="text-gray-600">Regra ativada: merchant_risk + high_value + eletronicos</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* EXEMPLO 6.3: Debito + WEB + 3 Tentativas Rapidas */}
            <div className="border-2 border-yellow-300 rounded-xl mb-6 overflow-hidden">
              <div className="bg-yellow-500 text-white p-4">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-bold">6.3 Debito + WEB + 3 Tentativas Rapidas</h4>
                  <span className="bg-white/20 px-3 py-1 rounded-full text-sm">Score: 58 | REVISAR</span>
                </div>
              </div>
              <div className="p-4 bg-white">
                <div className="grid lg:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-blue-500" /> JSON de ENTRADA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "transactions": [{
    "transaction_id": "TXN-DEB-WEB-003",
    "customer_id": "CPF***890***",
    "amount": 1500.00,
    "channel": "CARTAO",
    "card_type": "DEBIT",
    "hour": 19,
    "merchant_id": "MERC-LOJA-789",
    "merchant_category": "vestuario",
    "device_id": "DEV-BROWSER-ABC",
    "ip_address": "189.***.***.55",
    "latitude": -23.5505,
    "longitude": -46.6333,
    "is_new_device": true,
    "is_new_recipient": false,
    "velocity_score": 0.68,
    "avg_amount_30d": 650.00,
    "transaction_count_24h": 3,
    "failed_attempts_1h": 2,
    "user_agent": "Mozilla/5.0..."
  }],
  "include_explanation": true
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowLeft className="h-4 w-4 text-green-500" /> JSON de SAIDA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-yellow-400 text-xs font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [{
      "transaction_id": "TXN-DEB-WEB-003",
      "is_fraud": false,
      "fraud_probability": 0.58,
      "risk_score": 58.0,
      "risk_level": "MEDIUM",
      "confidence": 0.82,
      "processing_time_ms": 89.7,
      "model_version": "1.0.0",
      "detection_reason": [
        "Dispositivo novo detectado",
        "Valor 2.3x maior que media",
        "2 tentativas falhas recentes",
        "Velocidade de transacoes alta"
      ],
      "timestamp": "2025-11-30T19:45:22.123Z",
      "review_recommended": true,
      "explanation": {
        "risk_level": "MEDIUM",
        "explanation_text": "Transacao com 
indicadores mistos - revisao sugerida.",
        "top_risk_factors": [
          {"factor": "new_device", 
           "impact": 0.22},
          {"factor": "failed_attempts", 
           "impact": 0.18},
          {"factor": "velocity", 
           "impact": 0.15}
        ],
        "top_protective_factors": [
          {"factor": "known_merchant", 
           "impact": -0.12},
          {"factor": "known_ip_range", 
           "impact": -0.08}
        ],
        "lgpd_compliant": true
      }
    }]
  }
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="bg-yellow-50 rounded-lg p-4">
                  <h5 className="font-bold text-yellow-800 mb-2">O Que Aconteceu no Motor:</h5>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">1. Velocity Check</div>
                      <p className="text-gray-600">3 txns em 24h + 2 falhas = padrao de teste de cartao?</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">2. Balanceamento</div>
                      <p className="text-gray-600">Fatores negativos (device novo) vs positivos (merchant ok)</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">3. Decisao</div>
                      <p className="text-gray-600">Score 58 = zona cinza, enviar para analista</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* EXEMPLO 6.4: PIX Recorrente + Mesmo Device + Novo IP */}
            <div className="border-2 border-green-300 rounded-xl mb-6 overflow-hidden">
              <div className="bg-green-600 text-white p-4">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-bold">6.4 PIX Recorrente + Mesmo Device + Novo IP (Viagem)</h4>
                  <span className="bg-white/20 px-3 py-1 rounded-full text-sm">Score: 22 | APROVAR</span>
                </div>
              </div>
              <div className="p-4 bg-white">
                <div className="grid lg:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-blue-500" /> JSON de ENTRADA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "transactions": [{
    "transaction_id": "TXN-PIX-TRAVEL-004",
    "customer_id": "CPF***111***",
    "amount": 850.00,
    "channel": "PIX",
    "hour": 12,
    "merchant_id": "PIX-FAMILIA-001",
    "merchant_category": "transferencia_pf",
    "device_id": "DEV-KNOWN-IPHONE",
    "ip_address": "201.***.***.99",
    "latitude": -25.4284,
    "longitude": -49.2733,
    "is_new_device": false,
    "is_new_recipient": false,
    "velocity_score": 0.15,
    "avg_amount_30d": 920.00,
    "transaction_count_24h": 1,
    "location_city": "Curitiba",
    "usual_city": "Sao Paulo"
  }],
  "include_explanation": true,
  "fast_mode": true
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowLeft className="h-4 w-4 text-green-500" /> JSON de SAIDA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [{
      "transaction_id": "TXN-PIX-TRAVEL-004",
      "is_fraud": false,
      "fraud_probability": 0.22,
      "risk_score": 22.0,
      "risk_level": "LOW",
      "confidence": 0.91,
      "processing_time_ms": 31.2,
      "model_version": "1.0.0",
      "detection_reason": [
        "Cliente recorrente",
        "Dispositivo conhecido",
        "Valor dentro do padrao",
        "Destinatario ja recebeu 15x antes",
        "Padrao de viagem detectado"
      ],
      "timestamp": "2025-11-30T12:30:45.678Z",
      "explanation": {
        "risk_level": "LOW",
        "explanation_text": "Transacao aprovada.
Padrao consistente com viagem.",
        "top_protective_factors": [
          {"factor": "known_device", 
           "impact": -0.25, 
           "description": "Device confiavel"},
          {"factor": "known_recipient", 
           "impact": -0.20, 
           "description": "Destino habitual"},
          {"factor": "amount_normal", 
           "impact": -0.15, 
           "description": "Valor tipico"}
        ],
        "top_risk_factors": [
          {"factor": "new_location", 
           "impact": 0.12, 
           "description": "Cidade diferente"}
        ],
        "lgpd_compliant": true
      }
    }],
    "summary": {
      "total": 1,
      "frauds_detected": 0,
      "avg_risk_score": 0.22
    }
  }
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <h5 className="font-bold text-green-800 mb-2">O Que Aconteceu no Motor:</h5>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">1. Device Trust</div>
                      <p className="text-gray-600">iPhone do cliente usado ha 2 anos, biometria OK</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">2. Travel Pattern</div>
                      <p className="text-gray-600">SP para Curitiba = 400km, tempo compativel com aviao</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-gray-700 mb-1">3. Recipient History</div>
                      <p className="text-gray-600">PIX para "mae" - 15 transferencias anteriores</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* EXEMPLO 6.5: Credito Internacional + GeoLocation Divergente */}
            <div className="border-2 border-red-300 rounded-xl mb-6 overflow-hidden">
              <div className="bg-red-700 text-white p-4">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-bold">6.5 Credito Internacional + GeoLocation Divergente (Teletransporte)</h4>
                  <span className="bg-white/20 px-3 py-1 rounded-full text-sm">Score: 97 | BLOQUEAR</span>
                </div>
              </div>
              <div className="p-4 bg-white">
                <div className="grid lg:grid-cols-2 gap-4 mb-4">
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowRight className="h-4 w-4 text-blue-500" /> JSON de ENTRADA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre">{`{
  "transactions": [{
    "transaction_id": "TXN-CC-INTL-005",
    "customer_id": "CPF***333***",
    "amount": 8500.00,
    "currency": "USD",
    "amount_brl": 42500.00,
    "channel": "CARTAO",
    "card_type": "CREDIT",
    "hour": 4,
    "merchant_id": "MERC-INTL-AMAZON-US",
    "merchant_category": "ecommerce",
    "merchant_country": "US",
    "device_id": "DEV-UNKNOWN-XYZ",
    "ip_address": "104.***.***.55",
    "ip_country": "US",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "last_transaction_city": "Sao Paulo",
    "last_transaction_time": "2025-11-30T03:55:00Z",
    "is_new_device": true,
    "is_new_recipient": true,
    "velocity_score": 0.88
  }],
  "include_explanation": true
}`}</pre>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
                      <ArrowLeft className="h-4 w-4 text-green-500" /> JSON de SAIDA:
                    </h5>
                    <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                      <pre className="text-red-400 text-xs font-mono whitespace-pre">{`{
  "success": true,
  "data": {
    "predictions": [{
      "transaction_id": "TXN-CC-INTL-005",
      "is_fraud": true,
      "fraud_probability": 0.97,
      "risk_score": 97.0,
      "risk_level": "CRITICAL",
      "confidence": 0.98,
      "processing_time_ms": 234.5,
      "model_version": "1.0.0",
      "detection_reason": [
        "IMPOSSIBILIDADE FISICA DETECTADA",
        "SP -> San Francisco em 9 minutos",
        "Distancia: 10.500km impossivel",
        "Dispositivo completamente novo",
        "IP de outro pais",
        "Valor R$ 42.500 muito alto",
        "Transacao as 04h (madrugada)"
      ],
      "timestamp": "2025-11-30T04:04:12.999Z",
      "alert_level": "CRITICAL",
      "security_action": "BLOCK_AND_NOTIFY",
      "explanation": {
        "risk_level": "CRITICAL",
        "explanation_text": "Transacao bloqueada.
Impossibilidade fisica detectada.",
        "top_risk_factors": [
          {"factor": "geo_impossibility", 
           "impact": 0.50, 
           "description": "Teletransporte"},
          {"factor": "international", 
           "impact": 0.25, 
           "description": "Pais diferente"},
          {"factor": "high_amount", 
           "impact": 0.15, 
           "description": "Valor extremo"}
        ],
        "lgpd_compliant": true
      }
    }]
  }
}`}</pre>
                    </div>
                  </div>
                </div>
                <div className="bg-red-50 rounded-lg p-4">
                  <h5 className="font-bold text-red-800 mb-2">O Que Aconteceu no Motor:</h5>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-red-700 mb-1">1. Geo Check</div>
                      <p className="text-gray-600">SP 03:55 → SF 04:04 = 9min para 10.500km = IMPOSSIVEL</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-red-700 mb-1">2. Hard Rule</div>
                      <p className="text-gray-600">Regra "TELETRANSPORTE" ativada = bloqueio automatico</p>
                    </div>
                    <div className="bg-white rounded p-3">
                      <div className="font-bold text-red-700 mb-1">3. Alerta</div>
                      <p className="text-gray-600">Notificacao enviada para equipe de seguranca</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* RESUMO VISUAL */}
          <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-xl p-6">
            <h3 className="text-xl font-bold text-indigo-900 mb-4 flex items-center gap-2">
              <Sparkles className="h-6 w-6" /> Resumo: Como Interpretar os Resultados
            </h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg p-4 text-center">
                <div className="text-4xl font-bold text-green-500 mb-2">0-29</div>
                <div className="text-lg font-bold text-green-700">LOW</div>
                <div className="text-sm text-gray-600">Aprovar automaticamente</div>
              </div>
              <div className="bg-white rounded-lg p-4 text-center">
                <div className="text-4xl font-bold text-yellow-500 mb-2">30-69</div>
                <div className="text-lg font-bold text-yellow-700">MEDIUM</div>
                <div className="text-sm text-gray-600">Revisar manualmente</div>
              </div>
              <div className="bg-white rounded-lg p-4 text-center">
                <div className="text-4xl font-bold text-red-500 mb-2">70-100</div>
                <div className="text-lg font-bold text-red-700">HIGH</div>
                <div className="text-sm text-gray-600">Bloquear automaticamente</div>
              </div>
            </div>
          </div>

        </CollapsibleSection>

        {/* SECAO 10: CENARIOS REAIS COMPLETOS */}
        <CollapsibleSection id="cenarios-reais" title="Catalogo Completo de Cenarios Reais de Fraude" icon={Flag} color="red">
          
          {/* INTRODUCAO */}
          <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-xl p-6 mb-6">
            <h3 className="text-xl font-bold text-red-800 mb-3 flex items-center gap-2">
              <AlertTriangle className="h-6 w-6" /> Por Que Este Catalogo e Importante?
            </h3>
            <p className="text-gray-700 mb-4">
              Este catalogo contem TODOS os padroes de fraude que o sistema sabe reconhecer. 
              Cada cenario foi aprendido atraves de datasets reais, Transfer Learning e feedback de analistas.
              Use este catalogo como referencia para entender por que o sistema toma cada decisao.
            </p>
            <div className="grid md:grid-cols-4 gap-4 mt-4">
              <div className="bg-white rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-red-600">40+</div>
                <div className="text-xs text-gray-600">Cenarios Mapeados</div>
              </div>
              <div className="bg-white rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-orange-600">7</div>
                <div className="text-xs text-gray-600">Categorias de Features</div>
              </div>
              <div className="bg-white rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-yellow-600">3</div>
                <div className="text-xs text-gray-600">Datasets Analisados</div>
              </div>
              <div className="bg-white rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-purple-600">4</div>
                <div className="text-xs text-gray-600">Fases de Transfer Learning</div>
              </div>
            </div>
          </div>

          {/* SECAO 1: CENARIOS BASEADOS EM FEATURES */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Brain className="h-6 w-6 text-blue-500" /> 1. Cenarios Baseados em Features (Variaveis do Modelo)
            </h3>

            {/* 1.1 VALOR */}
            <div className="bg-blue-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-blue-800 mb-3 flex items-center gap-2">
                <DollarSign className="h-5 w-5" /> 1.1 Cenarios de VALOR (amount)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'Valor muito alto fora do padrao historico', score: 85, decisao: 'BLOQUEAR', exemplo: 'Cliente gastou R$ 15.000, media historica R$ 400', cor: 'red' },
                  { cenario: 'Valor medio repetido varias vezes', score: 72, decisao: 'BLOQUEAR', exemplo: '5 transacoes de R$ 999 em 10 minutos (evitar limite)', cor: 'red' },
                  { cenario: 'Valor baixo em frequencia anormal', score: 65, decisao: 'REVISAR', exemplo: '20 compras de R$ 50 em lojas diferentes', cor: 'yellow' },
                  { cenario: 'Valor alto em horario comercial', score: 35, decisao: 'REVISAR', exemplo: 'R$ 5.000 as 14h, cliente executivo', cor: 'yellow' },
                  { cenario: 'Valor incompativel com renda estimada', score: 78, decisao: 'BLOQUEAR', exemplo: 'Renda R$ 2.000, compra R$ 8.000', cor: 'red' },
                  { cenario: 'Valor dentro do padrao habitual', score: 12, decisao: 'APROVAR', exemplo: 'R$ 350, media R$ 400, mesmo comerciante', cor: 'green' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.2 CANAL */}
            <div className="bg-green-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-green-800 mb-3 flex items-center gap-2">
                <Globe className="h-5 w-5" /> 1.2 Cenarios de CANAL (channel)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'WEB: Primeira transacao em navegador desconhecido', score: 68, decisao: 'REVISAR', exemplo: 'Chrome novo, sem cookies, JS desabilitado', cor: 'yellow' },
                  { cenario: 'WEB: IP compativel com historico', score: 15, decisao: 'APROVAR', exemplo: 'Mesmo IP dos ultimos 30 dias', cor: 'green' },
                  { cenario: 'POS: Varias tentativas negadas em sequencia', score: 88, decisao: 'BLOQUEAR', exemplo: '5 tentativas recusadas, 6a aprovada', cor: 'red' },
                  { cenario: 'POS: Troca de maquina/merchant inesperada', score: 72, decisao: 'BLOQUEAR', exemplo: 'Sempre usa maquina A, aparece em maquina B', cor: 'red' },
                  { cenario: 'APP: Biometria recusada antes da transacao', score: 82, decisao: 'BLOQUEAR', exemplo: '3 tentativas de digital falharam', cor: 'red' },
                  { cenario: 'APP: Dispositivo conhecido, biometria OK', score: 8, decisao: 'APROVAR', exemplo: 'Celular habitual, Face ID confirmado', cor: 'green' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.3 TIPO DE PAGAMENTO */}
            <div className="bg-purple-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-purple-800 mb-3 flex items-center gap-2">
                <CreditCard className="h-5 w-5" /> 1.3 Cenarios de PAGAMENTO (paymentType)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'PIX: Alto valor, transferencia entre estados', score: 75, decisao: 'BLOQUEAR', exemplo: 'R$ 25.000 de SP para BA, primeira vez', cor: 'red' },
                  { cenario: 'PIX: Recorrente entre mesmos destinatarios', score: 10, decisao: 'APROVAR', exemplo: 'Todo mes R$ 1.500 para mae', cor: 'green' },
                  { cenario: 'CREDITO: Valor atipico a vista', score: 70, decisao: 'BLOQUEAR', exemplo: 'R$ 12.000 a vista, limite R$ 15.000', cor: 'red' },
                  { cenario: 'CREDITO: Parcelado fora do padrao', score: 55, decisao: 'REVISAR', exemplo: '12x de R$ 800, nunca parcelou antes', cor: 'yellow' },
                  { cenario: 'DEBITO: Pequeno valor repetido muitas vezes', score: 78, decisao: 'BLOQUEAR', exemplo: '15 debitos de R$ 10 em 5 minutos', cor: 'red' },
                  { cenario: 'DEBITO: Internacional com moeda incomum', score: 85, decisao: 'BLOQUEAR', exemplo: 'Debito em rublos russos, cliente nunca viajou', cor: 'red' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.4 IP */}
            <div className="bg-orange-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-orange-800 mb-3 flex items-center gap-2">
                <Network className="h-5 w-5" /> 1.4 Cenarios de IP (ipAddress)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'IP conhecido do cliente', score: 8, decisao: 'APROVAR', exemplo: 'Mesmo IP residencial dos ultimos 6 meses', cor: 'green' },
                  { cenario: 'IP desconhecido mas mesmo estado', score: 35, decisao: 'REVISAR', exemplo: 'IP novo, mas ainda em Sao Paulo', cor: 'yellow' },
                  { cenario: 'IP de outro pais', score: 82, decisao: 'BLOQUEAR', exemplo: 'Cliente em SP, IP da Nigeria', cor: 'red' },
                  { cenario: 'IP ja apareceu em fraudes anteriores', score: 95, decisao: 'BLOQUEAR', exemplo: 'IP na HOT List por fraude confirmada', cor: 'red' },
                  { cenario: 'IP de VPN/Proxy/Tor detectado', score: 78, decisao: 'BLOQUEAR', exemplo: 'IP identificado como NordVPN', cor: 'red' },
                  { cenario: 'IP empresarial (corporativo)', score: 20, decisao: 'APROVAR', exemplo: 'IP do escritorio, horario comercial', cor: 'green' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.5 DISPOSITIVO */}
            <div className="bg-pink-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-pink-800 mb-3 flex items-center gap-2">
                <Smartphone className="h-5 w-5" /> 1.5 Cenarios de DISPOSITIVO (deviceId)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'Dispositivo conhecido sem historico negativo', score: 5, decisao: 'APROVAR', exemplo: 'iPhone do cliente ha 2 anos', cor: 'green' },
                  { cenario: 'Dispositivo novo nunca visto', score: 55, decisao: 'REVISAR', exemplo: 'Primeiro acesso deste Android', cor: 'yellow' },
                  { cenario: 'Dispositivo ja participou de fraude', score: 98, decisao: 'BLOQUEAR', exemplo: 'Device ID na HOT List', cor: 'red' },
                  { cenario: 'Dispositivo em 2 cidades no mesmo dia', score: 92, decisao: 'BLOQUEAR', exemplo: 'SP as 10h, RJ as 10h30 (impossivel)', cor: 'red' },
                  { cenario: 'Dispositivo troca de IP 5x em 10 min', score: 85, decisao: 'BLOQUEAR', exemplo: 'Comportamento tipico de emulador/bot', cor: 'red' },
                  { cenario: 'Dispositivo novo mas biometria confirmada', score: 25, decisao: 'APROVAR', exemplo: 'Celular novo, Face ID do cliente OK', cor: 'green' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.6 GEOLOCALIZACAO */}
            <div className="bg-teal-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-teal-800 mb-3 flex items-center gap-2">
                <MapPin className="h-5 w-5" /> 1.6 Cenarios de GEOLOCALIZACAO (geoLocation)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: 'Localizacao habitual do cliente', score: 5, decisao: 'APROVAR', exemplo: 'Sempre compra em SP, comprando em SP', cor: 'green' },
                  { cenario: 'Cliente viajando (padrao conhecido)', score: 18, decisao: 'APROVAR', exemplo: 'Historico de viagens, RJ no fim de semana', cor: 'green' },
                  { cenario: 'Compra em pais que nunca visitou', score: 88, decisao: 'BLOQUEAR', exemplo: 'Nunca viajou, compra na Russia', cor: 'red' },
                  { cenario: 'Teletransporte digital (impossivel)', score: 99, decisao: 'BLOQUEAR', exemplo: 'SP as 10h, Londres as 10h05', cor: 'red' },
                  { cenario: 'Cidade nao bate com IP', score: 75, decisao: 'BLOQUEAR', exemplo: 'GPS mostra RJ, IP e de SP', cor: 'red' },
                  { cenario: 'Cidade diferente do endereco cadastrado', score: 45, decisao: 'REVISAR', exemplo: 'Mora em SP, compra em MG (viagem?)', cor: 'yellow' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 1.7 FREQUENCIA */}
            <div className="bg-indigo-50 rounded-xl p-4 mb-4">
              <h4 className="font-bold text-lg text-indigo-800 mb-3 flex items-center gap-2">
                <Activity className="h-5 w-5" /> 1.7 Cenarios de FREQUENCIA (velocity)
              </h4>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { cenario: '10 transacoes em menos de 1 minuto', score: 95, decisao: 'BLOQUEAR', exemplo: 'Ataque automatizado detectado', cor: 'red' },
                  { cenario: 'Alta frequencia de tentativas recusadas', score: 88, decisao: 'BLOQUEAR', exemplo: '8 tentativas recusadas, testando cartao', cor: 'red' },
                  { cenario: 'Inatividade por meses, comportamento repentino', score: 72, decisao: 'BLOQUEAR', exemplo: 'Sem uso por 4 meses, 10 compras hoje', cor: 'red' },
                  { cenario: 'Aumento dramatico de gastos', score: 80, decisao: 'BLOQUEAR', exemplo: 'R$ 200/mes vira R$ 8.000 em 2 dias', cor: 'red' },
                  { cenario: 'Frequencia normal, valor normal', score: 8, decisao: 'APROVAR', exemplo: '2-3 transacoes por semana, como sempre', cor: 'green' },
                  { cenario: 'Pico esperado (Black Friday, Natal)', score: 22, decisao: 'APROVAR', exemplo: 'Mais compras que o normal em novembro', cor: 'green' }
                ].map((item, i) => (
                  <div key={i} className={`bg-white rounded-lg p-3 border-l-4 ${
                    item.cor === 'red' ? 'border-red-500' : item.cor === 'yellow' ? 'border-yellow-500' : 'border-green-500'
                  }`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-gray-900 text-sm">{item.cenario}</span>
                      <span className={`text-xs px-2 py-1 rounded font-bold ${
                        item.cor === 'red' ? 'bg-red-100 text-red-700' : 
                        item.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                        'bg-green-100 text-green-700'
                      }`}>{item.score}</span>
                    </div>
                    <p className="text-xs text-gray-600 italic">{item.exemplo}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SECAO 2: CENARIOS BASEADOS EM DATASETS */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Database className="h-6 w-6 text-green-500" /> 2. Cenarios Baseados em Datasets (Historico)
            </h3>

            <div className="grid md:grid-cols-3 gap-4">
              {/* Historico de Fraudes */}
              <div className="bg-red-50 rounded-xl p-4">
                <h4 className="font-bold text-red-800 mb-3 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" /> 2.1 Historico de Fraudes
                </h4>
                <ul className="space-y-2">
                  {[
                    'Cliente sem historico + merchant com historico ruim',
                    'Cliente com fraude anterior tentando nova transacao',
                    'Mesmo IP usado em fraude confirmada',
                    'Mesmo dispositivo de fraude anterior',
                    'Mesmo padrao de horario de fraudes passadas',
                    'Valor identico a fraudes anteriores'
                  ].map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Historico de Cliente */}
              <div className="bg-blue-50 rounded-xl p-4">
                <h4 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
                  <User className="h-5 w-5" /> 2.2 Historico do Cliente
                </h4>
                <ul className="space-y-2">
                  {[
                    'Cliente recorrente sempre no mesmo horario',
                    'Padrao de gastos mensal consistente',
                    'Cliente que viaja com frequencia (padrao movel)',
                    'Cliente que raramente usa POS (so APP)',
                    'Cliente que so usa PIX para familia',
                    'Cliente VIP com limite especial'
                  ].map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Historico de Merchant */}
              <div className="bg-purple-50 rounded-xl p-4">
                <h4 className="font-bold text-purple-800 mb-3 flex items-center gap-2">
                  <Building className="h-5 w-5" /> 2.3 Historico do Merchant
                </h4>
                <ul className="space-y-2">
                  {[
                    'Merchant conhecido e confiavel',
                    'Merchant suspeito em auditorias',
                    'Merchant novo nunca visto no sistema',
                    'Merchant com volume repentino alto',
                    'Merchant com taxa de chargeback alta',
                    'Merchant em categoria de alto risco'
                  ].map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                      <Building className="h-4 w-4 text-purple-500 mt-0.5 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* SECAO 3: CENARIOS DE TRANSFER LEARNING */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Layers className="h-6 w-6 text-purple-500" /> 3. Cenarios de Transfer Learning (Padroes Aprendidos)
            </h3>

            <div className="grid md:grid-cols-2 gap-4">
              {/* Padroes que o modelo ja viu */}
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl p-4">
                <h4 className="font-bold text-purple-800 mb-3">3.1 Padroes que o Modelo "Ja Viu"</h4>
                <p className="text-sm text-gray-600 mb-3">
                  Aprendidos do dataset Kaggle (284k transacoes) e adaptados ao Brasil:
                </p>
                <div className="space-y-2">
                  {[
                    { padrao: 'Comportamento tipico de BOT', desc: 'Tentativas em intervalos exatos, sem variacao humana', score: 92 },
                    { padrao: 'IP com padroes de ataque automatizado', desc: 'Muitas requisicoes sequenciais do mesmo IP', score: 88 },
                    { padrao: 'Valores quebrados repetitivos', desc: 'R$ 999,99 varias vezes (evitar limite)', score: 75 },
                    { padrao: 'Sequencias de tentativas em horarios especificos', desc: 'Sempre as 03h-04h (fraude automatizada)', score: 85 }
                  ].map((item, i) => (
                    <div key={i} className="bg-white rounded-lg p-3">
                      <div className="flex justify-between items-start">
                        <span className="font-medium text-gray-900 text-sm">{item.padrao}</span>
                        <span className="text-xs px-2 py-1 rounded bg-red-100 text-red-700 font-bold">{item.score}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Padroes sem existir no dataset local */}
              <div className="bg-gradient-to-br from-orange-50 to-yellow-50 rounded-xl p-4">
                <h4 className="font-bold text-orange-800 mb-3">3.2 Padroes Reconhecidos de Outros Dominios</h4>
                <p className="text-sm text-gray-600 mb-3">
                  Padroes internacionais que o modelo reconhece mesmo sem ver no Brasil:
                </p>
                <div className="space-y-2">
                  {[
                    { padrao: 'Perfis de ataque internacional', desc: 'Padroes de Nigeria, Russia, etc.', score: 90 },
                    { padrao: 'Testes de cartao (small-test big-test)', desc: 'Compra de R$ 1, depois R$ 5.000', score: 82 },
                    { padrao: 'Deslocamento simultaneo IP + Device', desc: 'Ambos mudam ao mesmo tempo (clone)', score: 94 },
                    { padrao: 'Encadeamento de micro-transacoes', desc: 'Varias pequenas para testar limite', score: 78 }
                  ].map((item, i) => (
                    <div key={i} className="bg-white rounded-lg p-3">
                      <div className="flex justify-between items-start">
                        <span className="font-medium text-gray-900 text-sm">{item.padrao}</span>
                        <span className="text-xs px-2 py-1 rounded bg-orange-100 text-orange-700 font-bold">{item.score}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* SECAO 4: MATRIZ COMPLETA DE CENARIOS COMBINADOS */}
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Grid className="h-6 w-6 text-indigo-500" /> 4. Matriz Completa: Cenarios Combinados
            </h3>

            <div className="grid md:grid-cols-2 gap-4">
              {[
                {
                  titulo: '4.1 FRAUDE CLASSICA',
                  cor: 'red',
                  score: 94,
                  caracteristicas: ['Credito', 'Web', 'IP desconhecido', 'Dispositivo novo', 'Valor alto', 'Merchant nunca usado', 'Horario 03h', 'Localizacao diferente'],
                  json: '{"channel": "CARTAO", "amount": 12500, "hour": 3, "is_new_device": true, "ip_risk": "high"}'
                },
                {
                  titulo: '4.2 FRAUDE ORGANIZADA',
                  cor: 'red',
                  score: 97,
                  caracteristicas: ['Multiplos cartoes do mesmo cliente', 'Tentativas rapidas', 'Merchants diferentes em minutos', 'PIX para contas suspeitas', 'Device conhecido mas IP via VPN'],
                  json: '{"velocity_score": 0.95, "merchant_count_1h": 5, "ip_is_vpn": true}'
                },
                {
                  titulo: '4.3 SUSPEITA (REVISAR)',
                  cor: 'yellow',
                  score: 55,
                  caracteristicas: ['Debito', 'POS', 'Dispositivo novo', 'IP normal', 'Valor dentro do padrao', 'Horario incomum (04h)', 'Cliente com historico moderado'],
                  json: '{"channel": "DEBITO", "is_new_device": true, "hour": 4, "amount_deviation": 1.2}'
                },
                {
                  titulo: '4.4 APROVADO AUTOMATICO',
                  cor: 'green',
                  score: 12,
                  caracteristicas: ['Cliente recorrente', 'Dispositivo conhecido', 'IP conhecido', 'Valor habitual', 'Merchant dentro do padrao', 'Horario usual'],
                  json: '{"is_known_device": true, "is_known_ip": true, "amount_deviation": 0.8}'
                },
                {
                  titulo: '4.5 PIX ALTO RISCO',
                  cor: 'red',
                  score: 88,
                  caracteristicas: ['PIX alto valor', 'Beneficiario desconhecido', 'Conta criada recentemente', 'IP divergente', 'Dispositivo novo', 'Primeiro PIX para este CPF'],
                  json: '{"channel": "PIX", "amount": 25000, "recipient_is_new": true, "recipient_account_age_days": 3}'
                },
                {
                  titulo: '4.6 DEBITO SUSPEITO',
                  cor: 'yellow',
                  score: 68,
                  caracteristicas: ['Tres debitos seguidos recusados', 'Merchant incomum', 'IP e Device sempre mudando', 'Valor ligeiramente alto'],
                  json: '{"failed_attempts_1h": 3, "device_changes_24h": 4, "ip_changes_24h": 5}'
                },
                {
                  titulo: '4.7 TELETRANSPORTE DIGITAL',
                  cor: 'red',
                  score: 99,
                  caracteristicas: ['SP as 10h', 'RJ as 10h05', 'Mexico as 10h10', 'Londres as 10h15', 'DeviceID igual', 'IPs impossiveis'],
                  json: '{"geo_impossibility": true, "locations_10min": ["SP", "RJ", "MX", "UK"]}'
                },
                {
                  titulo: '4.8 ATAQUE DE BOT',
                  cor: 'red',
                  score: 96,
                  caracteristicas: ['Tentativas a cada milissegundo', 'Padrao repetitivo perfeito', 'Sem variacao humana', 'Valores identicos', 'IPs em sequencia'],
                  json: '{"is_bot": true, "requests_per_second": 50, "pattern_regularity": 0.99}'
                },
                {
                  titulo: '4.9 USO LEGITIMO EXCEPCIONAL',
                  cor: 'green',
                  score: 25,
                  caracteristicas: ['Cliente viaja (IP muda mas device permanece)', 'Black Friday (compras maiores)', 'PIX incomum mas para pessoa conhecida', 'Horario diferente mas valor normal'],
                  json: '{"is_known_device": true, "is_travel_pattern": true, "seasonal_event": true}'
                },
                {
                  titulo: '4.10 DADOS INCOMPLETOS/ERRO',
                  cor: 'yellow',
                  score: 50,
                  caracteristicas: ['Campos faltando', 'GeoLocation inconsistente', 'DeviceID em formato errado', 'Currency invalida', 'Timestamp futuro'],
                  json: '{"data_quality_score": 0.3, "missing_fields": ["ip", "device_id"]}'
                }
              ].map((cenario, i) => (
                <div key={i} className={`rounded-xl overflow-hidden border-2 ${
                  cenario.cor === 'red' ? 'border-red-300' : 
                  cenario.cor === 'yellow' ? 'border-yellow-300' : 
                  'border-green-300'
                }`}>
                  <div className={`p-3 ${
                    cenario.cor === 'red' ? 'bg-red-500 text-white' : 
                    cenario.cor === 'yellow' ? 'bg-yellow-500 text-white' : 
                    'bg-green-500 text-white'
                  }`}>
                    <div className="flex justify-between items-center">
                      <span className="font-bold">{cenario.titulo}</span>
                      <span className="bg-white/20 px-2 py-1 rounded text-sm">Score: {cenario.score}</span>
                    </div>
                  </div>
                  <div className="p-3 bg-white">
                    <div className="flex flex-wrap gap-1 mb-3">
                      {cenario.caracteristicas.map((c, j) => (
                        <span key={j} className={`text-xs px-2 py-1 rounded ${
                          cenario.cor === 'red' ? 'bg-red-100 text-red-700' : 
                          cenario.cor === 'yellow' ? 'bg-yellow-100 text-yellow-700' : 
                          'bg-green-100 text-green-700'
                        }`}>{c}</span>
                      ))}
                    </div>
                    <div className="bg-gray-900 rounded p-2 overflow-x-auto">
                      <code className="text-xs text-green-400 font-mono">{cenario.json}</code>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* RESUMO FINAL */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-xl p-6 text-white">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Award className="h-6 w-6 text-yellow-400" /> Resumo: Como Usar Este Catalogo
            </h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-bold text-yellow-400 mb-2">Para Analistas</h4>
                <p className="text-sm text-gray-300">
                  Use este catalogo para entender por que o sistema bloqueou ou aprovou uma transacao. 
                  Compare as caracteristicas da transacao com os cenarios listados.
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-bold text-blue-400 mb-2">Para Calibracao</h4>
                <p className="text-sm text-gray-300">
                  Se muitos cenarios de um tipo estao sendo mal classificados, 
                  ajuste os thresholds na tela de Calibracao.
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-bold text-green-400 mb-2">Para Feedback</h4>
                <p className="text-sm text-gray-300">
                  Quando der feedback, pense em qual cenario a transacao se encaixa. 
                  Isso ajuda a IA a aprender melhor.
                </p>
              </div>
            </div>
          </div>

        </CollapsibleSection>

        {/* SECAO 11: FAQ */}
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
