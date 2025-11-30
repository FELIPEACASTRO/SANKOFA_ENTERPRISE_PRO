import { useState } from 'react';
import { ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card.jsx';

export function Manual() {
  const [expandedSections, setExpandedSections] = useState({
    intro: true,
    dashboard: false,
    transactions: false,
  });

  const toggleSection = (id) => {
    setExpandedSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 rounded-lg p-8 text-white shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="h-10 w-10" />
          <h1 className="text-4xl font-bold">📘 Manual do Sankofa v1.0</h1>
        </div>
        <p className="text-lg opacity-95 mb-2">
          Guia Completo e Profissional para Entender e Usar o Sistema de Detecção de Fraudes
        </p>
        <div className="text-sm opacity-80 mt-4">
          ✅ Última atualização: 30 de Novembro de 2025
        </div>
      </div>

      {/* Seção Introdução */}
      <Card>
        <button onClick={() => toggleSection('intro')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>🎯 Bem-vindo ao Manual do Sankofa</CardTitle>
              {expandedSections.intro ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.intro && (
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="font-bold text-lg mb-2">O que é o Sankofa?</h3>
                <p className="text-sm text-gray-700 mb-3">
                  Sankofa significa "voltar para buscar" em um provérbio africano. É exatamente o que fazemos: analisamos padrões históricos de fraude para proteger suas transações AGORA.
                </p>
                <p className="text-sm text-gray-700 mb-3">
                  🚀 Detecta fraudes bancárias em milissegundos usando Inteligência Artificial
                </p>
                <p className="text-sm text-gray-700">
                  Processa: PIX • CARTÃO • TED • BOLETO
                </p>
              </div>

              <div>
                <h3 className="font-bold text-lg mb-2">Como Funciona?</h3>
                <div className="bg-blue-50 p-4 rounded text-xs font-mono text-gray-700 space-y-1">
                  <p>Transação Chega</p>
                  <p>       ↓</p>
                  <p>Sankofa Analisa 40+ características</p>
                  <p>       ↓</p>
                  <p>Resultado: FRAUDE? ou LEGÍTIMA?</p>
                </div>
              </div>

              <div>
                <h3 className="font-bold text-lg mb-2">🗺️ As 4 Áreas Principais</h3>
                <div className="bg-gray-50 p-4 rounded space-y-2 text-sm">
                  <p><strong>📊 Análise em Tempo Real:</strong> Dashboard, Transações, Alertas</p>
                  <p><strong>🔍 Investigação:</strong> Investigação, Revisão Manual, Feedback</p>
                  <p><strong>⚙️ Configuração:</strong> Calibragem, Regras Duras, Listas VIP/HOT</p>
                  <p><strong>📈 Observabilidade:</strong> Métricas, Monitoramento, Relatórios, Auditoria</p>
                </div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Dashboard */}
      <Card>
        <button onClick={() => toggleSection('dashboard')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>📊 Dashboard - Painel de Controle</CardTitle>
              {expandedSections.dashboard ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.dashboard && (
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-bold mb-2">Aonde Encontrar?</h3>
                <p className="text-sm text-gray-700">Menu principal (primeiro item)</p>
              </div>

              <div>
                <h3 className="font-bold mb-2">Para Que Serve?</h3>
                <p className="text-sm text-gray-700">
                  É como o painel de bordo de um avião. Você vê TUDO em uma tela: quantas fraudes entraram, qual foi a hora de pico, se há anomalia, se os algoritmos estão OK.
                </p>
              </div>

              <div>
                <h3 className="font-bold mb-2">O Que Você Vê?</h3>
                <ul className="text-sm text-gray-700 space-y-2">
                  <li><strong>🔢 KPIs:</strong> Total de Transações, Fraudes Detectadas, Taxa de Fraude, Valor Protegido</li>
                  <li><strong>📈 Série Temporal:</strong> Gráfico mostrando evolução de fraudes ao longo do dia</li>
                  <li><strong>🍰 Distribuição por Canal:</strong> PIX, TED, BOLETO (mostra qual teve mais fraude)</li>
                  <li><strong>🚨 Alertas Recentes:</strong> Últimas fraudes críticas</li>
                  <li><strong>🤖 Status dos Modelos:</strong> Se Random Forest, Gradient Boosting e CatBoost estão online</li>
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Quando Usar?</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>✅ Começo do turno: saber o cenário da noite</li>
                  <li>✅ Operações críticas: verificar SLA em tempo real</li>
                  <li>✅ Antes de tomar decisões: validar contexto do sistema</li>
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Dica de Ouro 💡</h3>
                <p className="text-sm text-gray-700 bg-blue-50 p-3 rounded">
                  O Dashboard atualiza a cada 30 segundos automaticamente. Ideal para deixar em um monitor durante todo o turno!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Transações */}
      <Card>
        <button onClick={() => toggleSection('transactions')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>💳 Transações - Busque Qualquer Operação</CardTitle>
              {expandedSections.transactions ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.transactions && (
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-bold mb-2">Aonde Encontrar?</h3>
                <p className="text-sm text-gray-700">Menu > Transações</p>
              </div>

              <div>
                <h3 className="font-bold mb-2">Para Que Serve?</h3>
                <p className="text-sm text-gray-700">
                  Encontrar uma transação específica que um cliente reclamou, ou buscar padrões. É seu "banco de dados visual".
                </p>
              </div>

              <div>
                <h3 className="font-bold mb-2">Quando Usar?</h3>
                <ul className="text-sm text-gray-700 space-y-2">
                  <li><strong>Cenário 1:</strong> Cliente ligou reclamando → Busca por CPF + data/hora</li>
                  <li><strong>Cenário 2:</strong> Quer estudar fraudes → Filtra por Status=Fraude, Canal=Cartão</li>
                  <li><strong>Cenário 3:</strong> Validar decisão → Busca pelo ID da transação</li>
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Elementos Principais</h3>
                <div className="bg-gray-50 p-3 rounded text-sm space-y-2 text-gray-700">
                  <p><strong>🔎 Busca por CPF:</strong> Digite os dígitos do CPF (mascarado automaticamente)</p>
                  <p><strong>📅 Filtro de Data:</strong> Últimas 24h, 7 dias, 30 dias, todo período</p>
                  <p><strong>⚡ Filtro de Canal:</strong> PIX, Cartão, TED, Boleto</p>
                  <p><strong>📊 Filtro de Status:</strong> Todas, Legítimas ✅, Suspeitas ⚠️, Fraudes ❌</p>
                  <p><strong>📋 Tabela:</strong> ID, CPF, Valor, Data, Canal, Status, Score (0-100)</p>
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2">Como Usar</h3>
                <ol className="text-sm text-gray-700 space-y-1">
                  <li>1. Filtre o que procura (CPF, data, canal, status)</li>
                  <li>2. Clique "Buscar"</li>
                  <li>3. Revise os resultados</li>
                  <li>4. Se precisa entender uma, clique "Ver Detalhes" → vai para Investigação</li>
                  <li>5. Se quer levar dados para fora, clique "Exportar"</li>
                </ol>
              </div>

              <div className="bg-blue-50 p-3 rounded">
                <p className="text-sm font-bold text-blue-900">💡 Dica:</p>
                <p className="text-sm text-blue-800">
                  Combine filtros! Exemplo: "PIX + Últimas 24h + Fraudes" = mostra todas as fraudes de PIX de hoje. Excelente para entender padrões!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Investigação */}
      <Card>
        <button onClick={() => toggleSection('investigation')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>🔍 Investigação - Análise Profunda</CardTitle>
              {expandedSections.investigation ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.investigation && (
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-700">
                <strong>Para que serve?</strong> Entender POR QUE uma transação foi bloqueada. Você vira um "detetive" armado com dados.
              </p>
              <p className="text-sm text-gray-700">
                <strong>Aonde encontrar?</strong> Menu > Investigação (ou clique "Ver Detalhes" em Transações)
              </p>

              <div>
                <h3 className="font-bold mb-2">Seções Principais</h3>
                <div className="bg-gray-50 p-3 rounded space-y-2 text-sm text-gray-700">
                  <p><strong>1. Dados da Transação:</strong> CPF, valor, hora, localização, canal</p>
                  <p><strong>2. Explicabilidade (LGPD):</strong> Motivos de suspeita com peso de cada um</p>
                  <p><strong>3. Score e Confiança:</strong> 0-100 score, % de confiança do modelo</p>
                  <p><strong>4. Histórico do Cliente:</strong> Padrão normal, comportamento típico</p>
                  <p><strong>5. Ações:</strong> Confirmar Fraude, Discordo, Deixar Feedback</p>
                </div>
              </div>

              <div className="bg-blue-50 p-3 rounded">
                <p className="text-sm font-bold text-blue-900">💡 Dica de Ouro:</p>
                <p className="text-sm text-blue-800">
                  A Explicabilidade é uma VANTAGEM competitiva. Use para ganhar confiança dos clientes explicando por que foi bloqueado!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Revisão Manual */}
      <Card>
        <button onClick={() => toggleSection('review')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>👁️ Revisão Manual - Human-in-the-Loop</CardTitle>
              {expandedSections.review ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.review && (
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-700">
                <strong>Para que serve?</strong> O Sankofa tem decisões INCERTAS (score ~ 50%). Aqui você, como especialista, revisa essas transações e valida.
              </p>

              <div>
                <h3 className="font-bold mb-2">Como Funciona</h3>
                <div className="bg-gray-50 p-3 rounded text-sm text-gray-700">
                  <p>Casos claros → Sankofa bloqueia/aprova</p>
                  <p>Casos duvidosos (score 40-60) → VAI PARA VOCÊ revisar</p>
                  <p>Você valida → "É fraude" ou "É legítima"</p>
                  <p>Resultado → Modelo aprende com seu feedback!</p>
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2">Botões Disponíveis</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li><strong>✅ Legítima:</strong> Isso é transação normal, modelo errou</li>
                  <li><strong>❌ Fraude:</strong> Realmente é fraude, modelo acertou</li>
                  <li><strong>💬 Comentário:</strong> Deixar contexto (ex: "cliente confirmou ao telefone")</li>
                </ul>
              </div>

              <div className="bg-green-50 p-3 rounded">
                <p className="text-sm font-bold text-green-900">✨ Por que é importante?</p>
                <p className="text-sm text-green-800">
                  Cada validação que você faz TREINA o modelo. Mais feedback = modelo mais inteligente!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Calibragem */}
      <Card>
        <button onClick={() => toggleSection('calibration')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>⚙️ Calibragem - Ajuste de Sensibilidade</CardTitle>
              {expandedSections.calibration ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.calibration && (
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-700">
                <strong>Para que serve?</strong> Você controla o quanto "rigoroso" ou "permissivo" o Sankofa é.
              </p>

              <div>
                <h3 className="font-bold mb-2">Escala de Calibragem (0-100)</h3>
                <div className="bg-gray-50 p-3 rounded space-y-2 text-sm">
                  <p>0-25: Extremamente permissivo (quase nada bloqueia)</p>
                  <p>25-50: Permissivo (RECOMENDADO: 40-50)</p>
                  <p>50-75: Rigoroso</p>
                  <p>75-100: Extremamente rigoroso (bloqueia quase tudo)</p>
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2">Quando Mexer?</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li><strong>Clientes reclamando?</strong> DIMINUA (45 → 35)</li>
                  <li><strong>Fraudes passando?</strong> AUMENTE (45 → 55)</li>
                  <li><strong>Otimizar performance?</strong> Veja gráfico histórico</li>
                </ul>
              </div>

              <div className="bg-yellow-50 p-3 rounded">
                <p className="text-sm font-bold text-yellow-900">⚠️ Estratégia Recomendada:</p>
                <p className="text-sm text-yellow-800">
                  Mudanças PEQUENAS (5 pontos), espere 1 dia de monitoramento, depois ajusta novamente. Assim evita grandes erros!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Alertas */}
      <Card>
        <button onClick={() => toggleSection('alerts')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>🚨 Alertas - Notificações Críticas</CardTitle>
              {expandedSections.alerts ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.alerts && (
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-700">
                <strong>Para que serve?</strong> Quando algo FORA DO NORMAL acontece, você recebe um ALERTA. É como uma sirene: algo merece atenção AGORA.
              </p>

              <div>
                <h3 className="font-bold mb-2">Níveis de Severidade</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li><strong>🔴 CRÍTICO:</strong> Ação urgente necessária</li>
                  <li><strong>🟠 AVISO:</strong> Atenção recomendada</li>
                  <li><strong>🟡 INFORMAÇÃO:</strong> Para conhecimento</li>
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Exemplos de Alertas</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>🔴 "Spike de fraudes de cartão: +250% vs média"</li>
                  <li>🟠 "Modelo offline há 15 minutos"</li>
                  <li>🟡 "Recalibração automática recomendada"</li>
                </ul>
              </div>

              <div className="bg-red-50 p-3 rounded">
                <p className="text-sm font-bold text-red-900">⚠️ Cuidado:</p>
                <p className="text-sm text-red-800">
                  Não ignore alertas vermelhos! Podem indicar ataque em progresso ou sistema falhando. Se não pode resolver, chame seu gerente!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Regras Duras */}
      <Card>
        <button onClick={() => toggleSection('hardrules')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>🔒 Regras Duras - Bloqueio Automático</CardTitle>
              {expandedSections.hardrules ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.hardrules && (
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-700">
                <strong>Para que serve?</strong> Hard Rules são decisões AUTOMÁTICAS: "SE [condição], ENTÃO [ação]"
              </p>

              <div>
                <h3 className="font-bold mb-2">Exemplos</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>• "SE CPF na lista negra ENTÃO BLOQUEIO"</li>
                  <li>• "SE valor &gt; R$ 100k E horário 23h-5h ENTÃO ALERTA"</li>
                  <li>• "SE IP do Exterior ENTÃO INVESTIGAR"</li>
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Ações Possíveis</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li><strong>BLOQUEIO:</strong> Nega a transação permanentemente</li>
                  <li><strong>ALERTA:</strong> Notifica, deixa passar para análise</li>
                  <li><strong>INVESTIGAÇÃO:</strong> Trata como suspeita</li>
                  <li><strong>PERMITIR:</strong> Aprova automaticamente</li>
                </ul>
              </div>

              <div className="bg-blue-50 p-3 rounded">
                <p className="text-sm font-bold text-blue-900">💡 Dica:</p>
                <p className="text-sm text-blue-800">
                  Crie regras baseada em PADRÕES OBSERVADOS. Sempre analise Transações primeiro, confirme padrão, depois cria regra!
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Listas VIP e HOT */}
      <Card>
        <button onClick={() => toggleSection('lists')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>✨ Listas VIP e HOT - Whitelist e Blacklist</CardTitle>
              {expandedSections.lists ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.lists && (
          <CardContent>
            <div className="space-y-4">
              <div>
                <h3 className="font-bold mb-2">📝 VIP (Whitelist)</h3>
                <p className="text-sm text-gray-700">
                  Clientes que CONFIO 100%. Deixa passar automático. Exemplos: Diretores, clientes top tier, contas internas.
                </p>
              </div>

              <div>
                <h3 className="font-bold mb-2">📝 HOT (Blacklist)</h3>
                <p className="text-sm text-gray-700">
                  Contas que SÃO PROBLEMÁTICAS. Bloqueia TUDO. Exemplos: Fraude confirmada, documentos clonados, contas comprometidas.
                </p>
              </div>

              <div>
                <h3 className="font-bold mb-2">Como Usar</h3>
                <div className="bg-gray-50 p-3 rounded text-sm text-gray-700 space-y-2">
                  <p><strong>Adicionar VIP:</strong> Menu > Lista VIP > Adicionar CPF + Motivo</p>
                  <p><strong>Remover VIP:</strong> Encontre e clique "Remover"</p>
                  <p><strong>Adicionar HOT:</strong> Menu > Lista HOT > Adicionar CPF + Motivo</p>
                  <p><strong>Remover HOT:</strong> ANTES revise por que foi adicionado, depois remove</p>
                </div>
              </div>

              <div className="bg-blue-50 p-3 rounded">
                <p className="text-sm font-bold text-blue-900">💡 Dica:</p>
                <p className="text-sm text-blue-800">
                  Revise regularmente! VIP que se torna fraudadora = problema. HOT removida rápido = cliente sofre.
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Outras Telas */}
      <Card>
        <button onClick={() => toggleSection('outros')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>📚 Outras Telas Importantes</CardTitle>
              {expandedSections.outros ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.outros && (
          <CardContent>
            <div className="space-y-3 text-sm text-gray-700">
              <div>
                <p className="font-bold">💬 Feedback Analista</p>
                <p>Deixe feedback para o modelo aprender. Quando discorda de uma decisão, marca "Legítima" ou "Fraude".</p>
              </div>
              <div>
                <p className="font-bold">📊 Monitoramento &amp; Métricas</p>
                <p>Saúde do sistema em tempo real: TPS (tx/segundo), Latência, Taxa de Fraude, Status dos modelos de IA.</p>
              </div>
              <div>
                <p className="font-bold">📋 Relatórios</p>
                <p>Gera análises para gerência/compliance: Performance, Fraudes por período, Fraudes por canal.</p>
              </div>
              <div>
                <p className="font-bold">📚 Datasets</p>
                <p>Catálogo de dados disponíveis para análise e criação de relatórios customizados.</p>
              </div>
              <div>
                <p className="font-bold">📜 Auditoria</p>
                <p>Registro LGPD de TUDO: quem acessou dados, quem fez ações, quando. Necessário para compliance.</p>
              </div>
              <div>
                <p className="font-bold">⚙️ Configurações</p>
                <p>Preferências pessoais: tema, notificações, trocar senha, permissões.</p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Rotina Diária */}
      <Card>
        <button onClick={() => toggleSection('rotina')} className="w-full">
          <CardHeader className="cursor-pointer hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle>⏰ Sua Rotina Diária Recomendada</CardTitle>
              {expandedSections.rotina ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
        </button>
        {expandedSections.rotina && (
          <CardContent>
            <div className="space-y-4 text-sm">
              <div className="bg-blue-50 p-3 rounded">
                <p className="font-bold text-blue-900">🌅 Início do Turno (5 minutos)</p>
                <ul className="text-blue-800 mt-2 space-y-1">
                  <li>1. Abra Alertas (há algo vermelho/laranja?)</li>
                  <li>2. Abra Dashboard (veja KPIs e gráficos)</li>
                  <li>3. Veja status dos modelos (todos online?)</li>
                </ul>
              </div>

              <div className="bg-green-50 p-3 rounded">
                <p className="font-bold text-green-900">💼 Durante o Turno (6 horas)</p>
                <ul className="text-green-800 mt-2 space-y-1">
                  <li>• Responda reclamações de clientes</li>
                  <li>• Valide decisões (Revisão Manual)</li>
                  <li>• Monitore alertas</li>
                  <li>• Deixe feedback (Feedback Analista)</li>
                </ul>
              </div>

              <div className="bg-yellow-50 p-3 rounded">
                <p className="font-bold text-yellow-900">🌆 Encerramento (10 minutos)</p>
                <ul className="text-yellow-800 mt-2 space-y-1">
                  <li>1. Revise todos os alertas abertos</li>
                  <li>2. Resolva ou passe para colega</li>
                  <li>3. Gere relatório do dia</li>
                  <li>4. Documente ações importantes</li>
                  <li>5. Passe informações ao próximo turno</li>
                </ul>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Footer */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100">
        <CardContent className="pt-6">
          <div className="text-center space-y-3">
            <p className="text-sm font-medium text-gray-900">
              📞 Dúvidas não respondidas neste manual?
            </p>
            <p className="text-xs text-gray-600">
              Fale com seu gerente ou time de suporte imediatamente.
            </p>
            <p className="text-xs text-blue-700 font-semibold">
              🔐 Lembre-se: Todos os dados aqui são confidenciais e monitorados por auditoria LGPD (Art. 20).
            </p>
            <p className="text-xs text-gray-500 pt-3 border-t border-blue-200">
              Sankofa Enterprise Pro v1.0 | Manual Didático Completo | 30 de Novembro de 2025
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
