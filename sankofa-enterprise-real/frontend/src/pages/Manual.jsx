import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';
import { Button } from '@/components/ui/Button.jsx';

const manualSections = [
  {
    id: 'introducao',
    title: '🎯 Bem-vindo ao Sankofa Enterprise Pro',
    content: `Sankofa é seu assistente inteligente para detecção de fraudes bancárias em tempo real. O nome "Sankofa" vem de um provérbio africano que significa "voltar para buscar", simbolizando a ideia de aprender com o passado para construir um futuro melhor.

    **O que faz o Sankofa?**
    Analisa milhões de transações bancárias (PIX, TED, BOLETO, Cartão) e identifica automaticamente quais são suspeitas de fraude, protegendo seus clientes e sua instituição.

    **Como funciona?**
    O sistema usa Inteligência Artificial (modelos de Machine Learning) que aprendem padrões de fraude. Quando uma transação suspeita chega, o Sankofa avalia mais de 40 características diferentes e dá um "diagnóstico" em milissegundos.

    **Você precisa fazer algo?**
    Não! A maioria das decisões é automática. Mas o Sankofa também oferece várias telas para você analista entender, validar e até corrigir as decisões quando necessário.`,
    isOpen: true
  },
  {
    id: 'visao-geral',
    title: '🗺️ Visão Geral das Áreas Principais',
    content: `O Sankofa é organizado em 4 grandes áreas:

    **1. ANÁLISE EM TEMPO REAL**
    - Dashboard: seu painel de controle com estatísticas
    - Transações: lista de todas as transações processadas
    - Alertas: notificações de fraudes detectadas

    **2. INVESTIGAÇÃO E REVISÃO**
    - Investigação: análise profunda de uma fraude específica
    - Revisão Manual: validar decisões do sistema (alguém realmente fez essa transação?)
    - Feedback Analista: treinar o modelo com sua experiência

    **3. CONFIGURAÇÃO DO SISTEMA**
    - Calibragem: ajustar a "agressividade" na detecção
    - Hard Rules: criar regras rígidas de bloqueio
    - Lista VIP: clientes que a gente já conhece e confia
    - Lista HOT: CPFs/contas que estão comprometidas

    **4. OBSERVABILIDADE**
    - Métricas: números do sistema em tempo real
    - Monitoramento: saúde dos algoritmos de IA
    - Relatórios: análises para relatórios gerenciais
    - Auditoria: registro de TUDO que você fez (compliance LGPD)

    **Dica:** Comece no Dashboard para ter uma visão geral, depois navegue para as outras telas conforme necessário.`
  },
  {
    id: 'dashboard',
    title: '📊 Dashboard - Seu Painel de Controle',
    content: `**Aonde encontrar?**
    Menu > Dashboard (primeiro item)

    **Para que serve?**
    O Dashboard é como o painel de bordo de um avião. Você vê rapidamente:
    - Quantas transações entraram hoje
    - Quantas eram fraudes
    - Quanto dinheiro foi protegido
    - Quais canais (PIX, Cartão, TED) tiveram mais problemas
    - Como está a saúde do sistema

    **O que você vê?**

    **Seção 1: Indicadores Principais (KPIs)**
    - Total de Transações: quantas transações chegaram
    - Fraudes Detectadas: quantas foram suspeitas
    - Taxa de Fraude: percentual (ex: 70%)
    - Valor Protegido: quanto dinheiro foi salvo em reais

    **Seção 2: Gráfico de Série Temporal**
    Uma linha que mostra ao longo do dia como evoluiu o número de fraudes. Se sobe muito de repente, pode indicar um ataque.

    **Seção 3: Análise por Canal**
    Um gráfico de pizza mostrando:
    - PIX: transferências instantâneas (maior volume de fraude)
    - Cartão: compras e saques com cartão
    - TED: transferências tradicionalmente lentas
    - Boleto: boletos emitidos

    **Seção 4: Alertas Recentes**
    Uma lista dos últimos alertas críticos. Se alguma coisa estranha aconteceu há poucos minutos, você vê aqui.

    **Seção 5: Status dos Modelos de IA**
    Mostra a saúde de cada algoritmo que está rodando.

    **O que fazer aqui?**
    - Monitorar: abra o Dashboard de manhã para ver como foi a noite
    - Investigar: se vê algo estranho (pico repentino), clique em um alerta para investigar
    - Atualizar: clique no botão 🔄 (Atualizar) para ver dados novos

    **Dica importante:**
    O Dashboard se atualiza a cada 30 segundos automaticamente. Não é necessário atualizar manualmente, mas você pode se quiser dados mais recentes.`
  },
  {
    id: 'transacoes',
    title: '💳 Transações - Lista de Todas as Operações',
    content: `**Aonde encontrar?**
    Menu > Transações

    **Para que serve?**
    Você precisa encontrar uma transação específica? Aqui você busca por qualquer critério:
    - CPF do cliente
    - Data/horário
    - Valor
    - Status (bloqueada, analisada, suspeita)
    - Canal (PIX, Cartão, etc.)

    **O que você vê?**

    **Filtros no topo:**
    - Buscar por CPF: mascarado por privacidade (ex: ***.***.789-01)
    - Filtro por Data: período que quer analisar
    - Filtro por Status: fraude, legítima, pendente
    - Filtro por Canal: escolha PIX, Cartão, TED ou Boleto

    **Tabela de Transações:**
    Cada linha é uma transação com:
    - ID: identificador único
    - CPF: do cliente (mascarado)
    - Valor: em reais
    - Hora: quando aconteceu
    - Canal: tipo de transação
    - Status: resultado (fraude, legítima, etc.)
    - Score de Risco: 0-100, quanto maior mais suspeita
    - Ações: botão para ver detalhes

    **O que fazer aqui?**
    1. Buscar uma transação específica que alguém reclamou
    2. Clicar em "Ver Detalhes" para entender por que foi bloqueada/aprovada
    3. Se discordar da decisão, ir para "Feedback Analista" para ensinar ao sistema

    **Dica:**
    Se quer ver transações que foram FRAUDE, deixe o filtro "Status" em "Fraude" e clique em Buscar. Assim você treina seu olho no que o Sankofa considera suspeito.`
  },
  {
    id: 'investigacao',
    title: '🔍 Investigação - Análise Profunda de Fraudes',
    content: `**Aonde encontrar?**
    Menu > Investigação

    **Para que serve?**
    Quando você precisa ENTENDER por que uma transação foi marcada como fraude. O Sankofa explica os motivos, de forma que até auditores e compliance entendem.

    **O que você vê?**

    **Seção 1: Dados da Transação**
    - CPF, valor, canal, hora
    - Dados da origem e destino (mascarados por privacidade)

    **Seção 2: Explicabilidade (LGPD)**
    Uma lista dos motivos técnicos por que foi marcada como fraude:
    - "Transação noturna em PIX (alto risco)"
    - "Destinatário novo não cadastrado"
    - "Valor acima da média histórica do cliente"
    - "Múltiplas transações em 1 hora (padrão de teste de cartão)"

    Cada motivo tem uma "força": quanto maior, mais contribuiu para a decisão.

    **Seção 3: Score e Probability**
    - Score final: 0-100 (quanto maior, mais fraude)
    - Confiança do modelo: percentual (ex: 92% confiante que é fraude)

    **Seção 4: Histórico do Cliente**
    Como o cliente normalmente se comporta:
    - Média de transações por dia
    - Horários normais
    - Canais preferidos
    - Desvios (o que está diferente hoje)

    **O que fazer aqui?**
    1. Ler os motivos e validar se fazem sentido
    2. Se discordar, marcar como "Feedback" (ensinar ao modelo)
    3. Usar as informações para relatórios ou justificativas ao cliente

    **Dica:**
    Este é o "raio-X" da fraude. Use quando tiver que explicar a um cliente ou em uma auditoria por que a transação foi bloqueada.`
  },
  {
    id: 'revisao-manual',
    title: '👁️ Revisão Manual - Human-in-the-Loop',
    content: `**Aonde encontrar?**
    Menu > Revisão Manual

    **Para que serve?**
    O Sankofa identifica as transações mais INCERTAS (não é 100% seguro se é fraude). Você, como expert, revisa essas e valida a decisão.

    Exemplo: uma transação tem score 55 (bem no meio). O sistema não tem certeza. Você olha e diz "isso é legítimo" ou "não, é fraude".

    **O que você vê?**

    **Fila de Revisão:**
    Uma lista de transações esperando sua análise (as mais incertas primeiro).

    **Para cada transação:**
    - Dados dela
    - Score do sistema
    - Motivos de suspeita
    - Botões: "Validar como Legítima" ou "Confirmar Fraude"

    **O que fazer aqui?**
    1. Revisar a transação baseado em seu conhecimento
    2. Clicar em um dos botões de decisão
    3. Opcionalmente, deixar um comentário (ex: "cliente confirmou, é legítimo")

    **Por que é importante?**
    Cada validação você faz TREINA o modelo. O Sankofa aprende com você e fica mais inteligente!

    **Dica:**
    Essa tela é crítica para a qualidade. Se você revisar 100 transações por dia, em uma semana o sistema fica muito mais preciso.`
  },
  {
    id: 'calibragem',
    title: '⚙️ Calibragem - Ajuste da Sensibilidade',
    content: `**Aonde encontrar?**
    Menu > Calibragem

    **Para que serve?**
    Você pode "controlar" o quanto agressivo o Sankofa é. Quer detectar MAIS fraudes (mais rigoroso) ou MENOS fraudes (mais permissivo)?

    **Analogia:**
    É como o volume de um alarme. Volume alto = detecta mais (mas pode falsar). Volume baixo = quieto (mas pode perder ataques).

    **O que você vê?**

    **Seção 1: Slider de Threshold**
    Um controle de 0 a 100:
    - 0-30: muito permissivo (deixa passar fraudes)
    - 30-50: balanço (recomendado)
    - 50-100: muito rigoroso (bloqueia muito, até clientes bons)

    **Seção 2: Impacto Esperado**
    Mostra o que acontece se você mudar:
    - Quantas fraudes vai deixar passar
    - Quantos clientes legítimos vai bloquear
    - Valor em reais afetado

    **Seção 3: Histórico de Calibragens**
    Quando foi mudado, quem mudou, qual era o threshold anterior.

    **O que fazer aqui?**
    1. Revisar o threshold atual
    2. Se estiver deixando fraudes passar: aumentar (< 50)
    3. Se estiver bloqueando muitos clientes bons: diminuir (> 50)
    4. Clicar "Aplicar" e monitorar os resultados

    **Quando mexer?**
    - Após grandes mudanças no comportamento de fraude
    - Se receber feedback dos clientes (bloqueios em massa)
    - Uma vez por semana em análise de performance

    **Dica importante:**
    Não mude drasticamente. Faça ajustes pequenos (de 5 em 5 pontos) e monitore por um dia. Depois ajusta novamente se necessário.`
  },
  {
    id: 'alertas',
    title: '🚨 Alertas - Notificações Críticas',
    content: `**Aonde encontrar?**
    Menu > Alertas

    **Para que serve?**
    Alertas são AVISOS críticos do sistema. Quando algo fora do normal acontece, você recebe um alerta aqui.

    **Exemplos de alertas:**
    - "Spike de fraudes no PIX (50% acima da média)"
    - "Modelo offline por mais de 1 hora"
    - "Taxa de fraude subindo anormalmente"
    - "Muitos falsos positivos detectados"

    **O que você vê?**

    **Seção 1: Alertas Ativos**
    Alertas que AINDA ESTÃO ACONTECENDO:
    - Ícone de severidade (critério/aviso/info)
    - Descrição do problema
    - Quando começou
    - Status (ativo/resolvido)

    **Seção 2: Alertas Históricos**
    Alertas já resolvidos, para você aprender com o passado.

    **O que fazer aqui?**
    1. Receber o alerta (automático na tela)
    2. Clicar para ler detalhes
    3. Investigar o que causou
    4. Tomar ação (aumentar calibragem, conferir sistema, etc.)
    5. Marcar como "Resolvido" quando acabar

    **Dica:**
    Se um alerta aparecer, não ignore! Geralmente significa que algo mudou no comportamento de fraudes.`
  },
  {
    id: 'hard-rules',
    title: '🔒 Regras Rígidas (Hard Rules) - Bloqueio Automático',
    content: `**Aonde encontrar?**
    Menu > Regras Duras

    **Para que serve?**
    Hard Rules são decisões 100% automáticas: "SE isso acontecer, bloqueia SEMPRE".

    **Exemplos de hard rules:**
    - "SE valor > R$ 50.000 E horário entre 23h e 5h: BLOQUEIO"
    - "SE transação vem de IP fora do Brasil: BLOQUEIO"
    - "SE CPF está na lista de fraude confirmada: BLOQUEIO"

    **O que você vê?**

    **Seção 1: Lista de Regras Ativas**
    Cada regra que está funcionando agora:
    - Condições (SE)
    - Ação (BLOQUEIO ou ALERTA)
    - Data de criação
    - Quem criou
    - Botões: Editar, Desativar, Deletar

    **Seção 2: Criar Nova Regra**
    Formulário simples:
    - Nome da regra
    - Condições (campo + operador + valor)
    - Ação (blocar ou alertar)

    **Seção 3: Histórico de Mudanças**
    Quando regras foram ativadas, desativadas, modificadas.

    **O que fazer aqui?**
    1. Revisar as regras ativas (alguma desatualizada?)
    2. Criar novas regras conforme aprenda novos padrões de fraude
    3. Desativar regras que não estão mais funcionando

    **Exemplo prático:**
    Você percebeu que toda fraude de cartão clonado começa em São Paulo entre 14h-16h. Cria uma regra:
    "SE Cartão + São Paulo + 14h-16h + valor > R$ 1000: ALERTAR gerente"

    **Dica:**
    Hard Rules são úteis, mas use com moderação. Regras muito rígidas podem bloquear clientes legítimos.`
  },
  {
    id: 'vip-list',
    title: '✨ Lista VIP - Aprovação Direta',
    content: `**Aonde encontrar?**
    Menu > Lista VIP

    **Para que serve?**
    Uma whitelist: CPFs de clientes que você CONFIA 100%, que passam automático.

    Exemplo: seus gerentes, diretores, clientes premium que você conhece bem.

    **Analogia:**
    VIP é como passar direto na segurança do aeroporto (fast lane).

    **O que você vê?**

    **Seção 1: Lista de VIPs**
    - CPF (mascarado para privacidade)
    - Nome do cliente
    - Data de adição à VIP
    - Quem adicionou
    - Razão (opcional)
    - Botão: Remover

    **Seção 2: Adicionar Nova VIP**
    - Campo para CPF
    - Campo para razão (opcional, ex: "Diretor")
    - Botão: Adicionar

    **O que fazer aqui?**
    1. Verificar a VIP list regularmente (toda semana)
    2. Remover clientes que não são mais confiáveis
    3. Adicionar novos clientes internos/confiáveis conforme necessário

    **Cuidado:**
    Se adiciona uma VIP por engano e ela ficar fraudadora, o Sankofa não vai detectar! Revise regularmente.

    **Dica:**
    Mantenha a VIP list pequena. Para < 50 clientes. Se tiver 1000 clientes, não vale a pena.`
  },
  {
    id: 'hot-list',
    title: '❌ Lista HOT - Bloqueio Direto',
    content: `**Aonde encontrar?**
    Menu > Lista HOT

    **Para que serve?**
    Uma blacklist: CPFs/contas que você SABE que são problemáticas, que têm que ser bloqueadas sempre.

    Exemplo: uma conta que você detectou como fraude, documento clonado, etc.

    **Analogia:**
    HOT é como a lista VIP, mas ao contrário. É a "lista negra".

    **O que você vê?**

    **Seção 1: Lista de HOTs**
    - CPF/conta (mascarado)
    - Status (Ativo/Inativo)
    - Data de adição
    - Razão do bloqueio (ex: "Conta clonada", "Fraude confirmada")
    - Botão: Remover ou Desativar

    **Seção 2: Adicionar Nova HOT**
    - Campo para CPF/conta
    - Campo para razão
    - Botão: Adicionar

    **Seção 3: Importar em Massa**
    Se você tiver uma lista grande de bloqueios, pode fazer upload de um arquivo.

    **O que fazer aqui?**
    1. Adicionar CPFs de clientes que tiveram fraude confirmada
    2. Revisar regularmente (alguns foram resolvidos? Remover)
    3. Comunicar com compliance/legal sobre novas adições

    **Importante:**
    Uma conta na HOT lista é SEMPRE bloqueada. Sem exceção. Então adicione apenas com certeza.

    **Dica:**
    Documente bem o motivo de cada adição. Depois você pode precisar justificar.`
  },
  {
    id: 'metricas',
    title: '📈 Métricas - Números em Tempo Real',
    content: `**Aonde encontrar?**
    Menu > Métricas

    **Para que serve?**
    Você quer saber AGORA como está a performance do sistema? Métricas em tempo real:
    - Transações por segundo processadas
    - Taxa de fraude AGORA
    - Latência média das decisões
    - Disponibilidade do sistema

    **O que você vê?**

    **Seção 1: Contadores em Tempo Real**
    - TPS (Transactions Per Second): quantas transações/segundo
    - Taxa de Fraude: percentual agora
    - Latência P95: 95% das decisões saem em quanto tempo
    - Uptime: percentual de tempo online

    **Seção 2: Gráficos Históricos**
    Evolução das métricas nas últimas 6 horas (atualiza a cada 1 minuto).

    **Seção 3: Limites e Alertas**
    Se alguma métrica sai do normal, você vê aqui.

    **O que fazer aqui?**
    1. Monitorar de manhã/noite de operações críticas
    2. Se latência sobe muito (> 100ms), investigar
    3. Se taxa de fraude sobe, conferir se há ataque
    4. Usar para relatórios de SLA e performance

    **Dica técnica:**
    Latência P95 < 50ms é o SLA do Sankofa. Se sair disso, significa que há problema.`
  },
  {
    id: 'monitoramento',
    title: '🏥 Monitoramento - Saúde dos Modelos de IA',
    content: `**Aonde encontrar?**
    Menu > Monitoramento

    **Para que serve?**
    Os algoritmos de IA conseguem "ficar doentes". Este painel mostra a saúde de cada modelo rodando.

    **Analogia:**
    É como um check-up médico: você quer saber se os órgãos (modelos) estão funcionando bem.

    **O que você vê?**

    **Seção 1: Status dos Modelos**
    Sankofa usa 3 algoritmos em paralelo:
    - Random Forest (modelo de árvores)
    - Gradient Boosting (modelo iterativo)
    - CatBoost (modelo de categorical boosting)

    Cada um mostra:
    - Status (online/offline)
    - Acurácia: percentual de acertos
    - Latência: tempo de resposta
    - Versão: qual versão está rodando
    - Última atualização: quando foi retreinado

    **Seção 2: Data Drift (Desvio de Dados)**
    Se os dados MUDARAM muito (novas fraudes, novo padrão de clientes), o modelo pode ficar menos preciso.

    **Seção 3: Concept Drift (Desvio de Conceito)**
    Se o SIGNIFICADO dos dados mudou (fraude mudou de padrão), precisa retreinar.

    **O que fazer aqui?**
    1. Verificar se todos estão online antes de abrir o sistema
    2. Se acurácia cair < 85%, chamar data science
    3. Se data drift > 0.1, considerar retreinar
    4. Usar para comprovar SLA com compliance

    **Dica:**
    Se um modelo ficar offline, o Sankofa usa os outros dois. Não é crítico, mas avise ao time DevOps.`
  },
  {
    id: 'feedback',
    title: '💬 Feedback Analista - Treinar o Modelo',
    content: `**Aonde encontrar?**
    Menu > Feedback Analista

    **Para que serve?**
    VOCÊ ensina ao Sankofa. Quando discorda de uma decisão, marca aqui e o modelo aprende.

    **Exemplo:**
    - Sankofa bloqueou uma transação como fraude
    - Você revisa e vê que é legítima
    - Deixa feedback: "Isso é legítimo, modelo errou"
    - Com 100 feedbacks assim, o Sankofa melhora

    **O que você vê?**

    **Seção 1: Feedback Pendente**
    Transações que você quer deixar feedback:
    - Dados da transação
    - Decisão que o Sankofa deu
    - Opções: "Discordo - era legítima" ou "Discordo - era fraude"
    - Campo opcional para comentário

    **Seção 2: Histórico de Feedbacks**
    Feedbacks que você deixou, o impacto no modelo.

    **O que fazer aqui?**
    1. Quando revisar uma transação em "Revisão Manual" e discordar, deixar feedback
    2. Ou vir aqui e selecionar transações para dar feedback
    3. Quanto mais feedback preciso você deixa, melhor o modelo fica

    **Por que é importante?**
    O feedback é o "combustível" do ML. Sem feedback dos analistas, o modelo não aprende e fica estagnado.

    **Dica:**
    Deixe feedback detalhado. Um comentário como "Esse cliente estava viajando, por isso a transação foi normal" ajuda o Sankofa a entender o contexto.`
  },
  {
    id: 'datasets',
    title: '📚 Datasets - Catálogo de Dados',
    content: `**Aonde encontrar?**
    Menu > Datasets

    **Para que serve?**
    Aqui você vê quais datasets estão disponíveis para análise e treino do Sankofa.

    **O que você vê?**

    **Seção 1: Datasets Disponíveis**
    - Nome do dataset
    - Descrição
    - Número de registros
    - Data de criação
    - Tamanho em GB
    - Status (online/offline)

    **Exemplos:**
    - "Histórico de Fraudes 2024": 50 mil transações fraudulentas
    - "Transações Legítimas PIX": 1 milhão de transações
    - "Padrões de Clientes": dados agregados

    **Seção 2: Usar um Dataset**
    Botão para utilizar um dataset em:
    - Análises customizadas
    - Retreinar o modelo
    - Exportar para relatórios

    **O que fazer aqui?**
    1. Explorar o que temos disponível
    2. Usar dados para criar relatórios
    3. Solicitar novos datasets se precisar

    **Dica:**
    Se você for fazer uma análise e precisa de dados específicos, vem aqui ver se existe. Senão, solicita ao time de data science.`
  },
  {
    id: 'relatorios',
    title: '📋 Relatórios - Análises Gerenciais',
    content: `**Aonde encontrar?**
    Menu > Relatórios

    **Para que serve?**
    Gerar relatórios para a gerência, compliance, auditoria.

    **Tipos de relatórios disponíveis:**
    - Performance do Sankofa (taxa de detecção, latência, etc.)
    - Fraudes por período (diário, semanal, mensal)
    - Fraudes por canal (qual canal teve mais)
    - Fraudes por razão (motivos mais comuns)
    - SLA Compliance (atendimento de SLAs)

    **O que você vê?**

    **Seção 1: Templates de Relatórios**
    Lista de modelos prontos. Basta clicar e customizar as datas.

    **Seção 2: Criar Relatório Customizado**
    Escolher:
    - Período
    - Filtros (canal, status, etc.)
    - Formato (PDF, Excel, CSV)
    - Agregar por (dia, semana, mês)

    **Seção 3: Histórico de Relatórios**
    Relatórios que você já gerou anteriormente.

    **O que fazer aqui?**
    1. Toda segunda-feira: gerar relatório da semana anterior para o gerente
    2. Final do mês: relatório mensal para compliance
    3. Quando pedido: criar relatório customizado para auditoria

    **Dica:**
    Salve relatórios importantes. Depois pode ser que você precise justificar uma decisão.`
  },
  {
    id: 'auditoria',
    title: '📜 Auditoria - Trilha de Tudo',
    content: `**Aonde encontrar?**
    Menu > Auditoria

    **Para que serve?**
    LGPD (lei de proteção de dados) exige que você registre TUDO. Auditoria é o histórico completo de:
    - Quem acessou o sistema e quando
    - Quais dados foram consultados
    - Quem fez calibragens/bloqueios
    - Qualquer ação importante

    **Analogia:**
    É como uma câmera de segurança do sistema, só que registra ações.

    **O que você vê?**

    **Seção 1: Log de Auditoria**
    Uma tabela com todas as ações:
    - Data/hora exata
    - Usuário que fez
    - Ação (consultou, bloqueou, calibrou, etc.)
    - Dados afetados (mascarados)
    - Resultado

    **Seção 2: Filtrar Logs**
    - Por usuário
    - Por tipo de ação
    - Por período
    - Por dados afetados

    **O que fazer aqui?**
    1. Se compliance pede: "quem acessou dados do cliente X?", você filtra aqui
    2. Investigar: se suspeitar que alguém fez algo indevido
    3. Relatórios: gerar relatório de auditoria para compliance

    **Importante (LGPD):**
    Qualquer consulta a dados pessoais é registrada. Se um cliente solicitar "acesso aos meus dados", você vem aqui e vê quem acessou.

    **Dica:**
    Nunca delete logs de auditoria. São obrigatórios por lei.`
  },
  {
    id: 'configuracoes',
    title: '⚙️ Configurações - Ajustes do Sistema',
    content: `**Aonde encontrar?**
    Menu > Configurações

    **Para que serve?**
    Configurações gerais da aplicação:
    - Preferências visuais (tema claro/escuro)
    - Notificações (quais alertas quer receber por email)
    - Perfil do usuário
    - Permissões (se você é admin)

    **O que você vê?**

    **Seção 1: Preferências Visuais**
    - Tema (claro ou escuro)
    - Idioma (português, inglês, etc.)
    - Notificações sonoras

    **Seção 2: Configurações de Notificação**
    - Receber email em caso de:
      - Spike de fraudes
      - Modelo offline
      - Novo alerta crítico

    **Seção 3: Perfil**
    - Nome de usuário
    - Email
    - Último login
    - Botão: Trocar Senha

    **Seção 4: Permissões e Papéis (se admin)**
    - Ver quais permissões você tem
    - Gerenciar outros usuários (se admin)

    **O que fazer aqui?**
    1. Configurar suas preferências na primeira vez
    2. Ativar notificações que são importantes para você
    3. Trocar senha regularmente (segurança)

    **Dica:**
    Não compartilhe sua conta. Cada pessoa deve ter uma login própria para auditoria.`
  },
  {
    id: 'dicas-praticas',
    title: '💡 Dicas e Boas Práticas',
    content: `**1. Rotina Diária Recomendada**

    Começo do dia:
    - 5 minutos no Dashboard para entender o cenário
    - Revisar alertas da noite
    - Conferir métricas (latência, taxa de fraude)

    Durante o dia:
    - Revisar transações que clientes reclamaram
    - Validar decisões do sistema em "Revisão Manual"
    - Deixar feedback quando discordar

    Final do dia:
    - Gerar relatório da jornada
    - Revisar se há anomalias
    - Se necesário, fazer ajustes em calibragem

    **2. Quando Calibrar**

    Aumente threshold (< 50) se:
    - Taxa de fraude subindo muito
    - Modelo deixando fraudes passar

    Diminua threshold (> 50) se:
    - Recebendo muitos bloqueios de clientes legítimos
    - Taxa de falsos positivos muito alta

    **3. Usar Hard Rules com Sabedoria**

    Regras são ótimas para fraudes ÓBVIAS:
    - Valor > R$ 100.000 noturno: BLOQUEIO
    - CPF na HOT lista: BLOQUEIO

    Mas não para decisões complexas. Use o modelo para isso.

    **4. Feedback é Ouro**

    Sempre que revisar uma transação, deixe feedback:
    - Fraude confirmada
    - Era legítima (modelo errou)
    - Contexto (cliente viajando, etc.)

    **5. Auditoria é Obrigação LGPD**

    Lembre-se: você está trabalhando com dados pessoais. Tudo é rastreado:
    - Toda consulta é registrada
    - Toda ação é registrada
    - Cliente pode solicitar "quem viu meus dados"

    **6. Escalação de Alertas**

    Se algo muito crítico acontecer:
    1. Documentar em comentário de alerta
    2. Notificar gerente imediatamente
    3. Colocar em "Investigação" para análise profunda

    **7. Não Confie 100% no Sistema**

    Sankofa é inteligente, MAS pode errar. Sempre:
    - Revisar decisões críticas
    - Considerar contexto (cliente em viagem, mudança de conta)
    - Usar seu julgamento humano

    **8. Performance Matters**

    Latência < 50ms é o SLA. Se sair disso:
    - Não deixe de reportar
    - Pode indicar problema maior
    - Compliance pode questionar

    **9. Segurança da Senha**

    Sua senha acessa dados pessoais de milhões. Portanto:
    - Senha forte (letras, números, símbolos)
    - Não compartilhe
    - Troque a cada 3 meses

    **10. Dúvidas? Documente!**

    Se não sabe como usar uma tela:
    1. Volte ao Manual (essa página)
    2. Procure a tela no índice
    3. Leia a explicação
    4. Se ainda não entender, fale com seu gerente`
  },
  {
    id: 'faq',
    title: '❓ Perguntas Frequentes',
    content: `**P: Por que uma transação foi bloqueada?**
    R: Vai em "Investigação", busca a transação e lê os motivos (explicabilidade). Se discordar, deixe feedback.

    **P: Como adiciono um cliente à VIP lista?**
    R: Menu > Lista VIP > Adicionar Nova VIP > CPF + Motivo > Salvar.

    **P: Qual é a latência de uma decisão?**
    R: Consulte "Métricas". Deve estar entre 37-72ms se estiver em cache, ou ~700ms se é primeira vez.

    **P: Como crio uma Hard Rule?**
    R: Menu > Regras Duras > Criar Nova Regra > Defina as condições e ação > Salvar.

    **P: Posso mudar a calibragem à noite?**
    R: Pode sim, mas cuidado. Pequenos ajustes (5 pontos). Depois monitore o resultado no dia seguinte.

    **P: Como deixo feedback de uma transação?**
    R: Menu > Feedback Analista > Selecione a transação > Marque "Legítima" ou "Fraude" > Deixe comentário > Enviar.

    **P: Onde vejo o histórico de fraudes?**
    R: Menu > Transações > Filtro Status "Fraude" > Buscar. Ou Menu > Relatórios para análises agregadas.

    **P: O que é Data Drift?**
    R: Quando os dados de entrada MUDAM (novas fraudes, novo padrão). Veja em "Monitoramento".

    **P: Por que o Dashboard se atualiza devagar?**
    R: Primeira requisição traz dados do banco (700ms). Segunda em diante usa cache (37-72ms). Aguarde.

    **P: Posso excluir um alerta resolvido?**
    R: Não! Alertas históricos são obrigação LGPD. Pode apenas marcar como "Resolvido".

    **P: Como sei se estou dentro da LGPD?**
    R: Auditoria registra tudo. Veja em Menu > Auditoria. Comply verifica regularmente.

    **P: Quantas transações o sistema processa por segundo?**
    R: Varie conforme a carga. Veja em Menu > Métricas > TPS (Transactions Per Second).

    **P: E se quiser sair de um alerta?**
    R: Clique no "X" ou botão Fechar. Ele volta se a condição persistir. Não é deletado.`
  }
];

export function Manual() {
  const [expandedSections, setExpandedSections] = useState(
    manualSections.reduce((acc, section) => ({
      ...acc,
      [section.id]: section.isOpen || false
    }), {})
  );

  const toggleSection = (id) => {
    setExpandedSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 rounded-lg p-8 text-white">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="h-8 w-8" />
          <h1 className="text-4xl font-bold">Manual do Sankofa</h1>
        </div>
        <p className="text-lg opacity-90">
          Guia completo para entender e usar o sistema de detecção de fraudes
        </p>
        <p className="text-sm opacity-75 mt-2">
          Última atualização: 30 de Novembro de 2025 | Versão 1.0
        </p>
      </div>

      {/* Índice */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">📋 Índice de Conteúdos</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {manualSections.map((section) => (
              <button
                key={section.id}
                onClick={() => {
                  const element = document.getElementById(`section-${section.id}`);
                  if (element) {
                    element.scrollIntoView({ behavior: 'smooth' });
                    setTimeout(() => {
                      if (!expandedSections[section.id]) {
                        toggleSection(section.id);
                      }
                    }, 100);
                  }
                }}
                className="text-left p-2 rounded hover:bg-blue-50 hover:text-blue-700 transition-colors text-sm"
              >
                {section.title}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Conteúdo das Seções */}
      <div className="space-y-3">
        {manualSections.map((section) => (
          <Card key={section.id} id={`section-${section.id}`}>
            <button
              onClick={() => toggleSection(section.id)}
              className="w-full"
            >
              <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg text-left">
                    {section.title}
                  </CardTitle>
                  {expandedSections[section.id] ? (
                    <ChevronUp className="h-5 w-5 text-blue-600" />
                  ) : (
                    <ChevronDown className="h-5 w-5 text-gray-400" />
                  )}
                </div>
              </CardHeader>
            </button>

            {expandedSections[section.id] && (
              <CardContent>
                <div className="prose prose-sm max-w-none whitespace-pre-wrap text-gray-700 leading-relaxed">
                  {section.content}
                </div>
              </CardContent>
            )}
          </Card>
        ))}
      </div>

      {/* Footer */}
      <div className="bg-gray-50 rounded-lg p-6 text-center text-sm text-gray-600">
        <p>📞 Dúvidas não respondidas? Fale com seu gerente ou time de suporte.</p>
        <p className="mt-2">🔐 Lembre-se: Todos os dados aqui são confidenciais e monitorados por auditoria LGPD.</p>
      </div>
    </div>
  );
}
