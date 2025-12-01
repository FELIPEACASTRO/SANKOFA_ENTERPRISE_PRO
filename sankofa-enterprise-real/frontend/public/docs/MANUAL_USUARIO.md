# Manual do Usuario - Sankofa Enterprise Pro v1.0

## Guia Completo para Analistas de Fraude

![Dashboard Analista](images/dashboard_interface_analista.png)

**Versao:** 1.0  
**Ultima Atualizacao:** 30 de Novembro de 2025  
**Status:** ✅ PRONTO PARA PRODUCAO  
**Publico:** Analistas de Fraude, Gerentes de Operacoes, Compliance Officers

---

## Bem-vindo ao Sankofa!

Este manual vai te guiar passo a passo no uso do sistema de deteccao de fraudes. Nao se preocupe se voce nao e tecnico - este guia foi feito pensando em voce!

```
+==================================================================+
|                    MAPA DO MANUAL                                 |
+==================================================================+
|                                                                   |
|  ┌───────────────────────────────────────────────────────────┐   |
|  │  1. PRIMEIROS PASSOS                                       │   |
|  │     • Como acessar                                         │   |
|  │     • O que voce vai ver                                   │   |
|  └────────────────────────────┬──────────────────────────────┘   |
|                               ▼                                   |
|  ┌───────────────────────────────────────────────────────────┐   |
|  │  2. CONHECENDO O DASHBOARD                                 │   |
|  │     • O que significam os numeros                          │   |
|  │     • As cores dos indicadores                             │   |
|  └────────────────────────────┬──────────────────────────────┘   |
|                               ▼                                   |
|  ┌───────────────────────────────────────────────────────────┐   |
|  │  3-5. TRABALHANDO COM TRANSACOES                           │   |
|  │     • Analisando                                           │   |
|  │     • Investigando                                         │   |
|  │     • Revisao manual                                       │   |
|  └────────────────────────────┬──────────────────────────────┘   |
|                               ▼                                   |
|  ┌───────────────────────────────────────────────────────────┐   |
|  │  6-9. RECURSOS AVANCADOS                                   │   |
|  │     • Explicacoes do sistema                               │   |
|  │     • Monitoramento                                        │   |
|  │     • Relatorios e alertas                                 │   |
|  └───────────────────────────────────────────────────────────┘   |
|                                                                   |
+==================================================================+
```

---

## 1. Primeiros Passos

### 1.1 Como Acessar

```
+==============================================================================+
|                         PASSO A PASSO                                         |
+==============================================================================+
|                                                                               |
|  PASSO 1: Abra seu navegador                                                  |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                  |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   Recomendamos estes navegadores:                                        │ |
|  │                                                                          │ |
|  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ |
|  │   │          │  │          │  │          │  │          │               │ |
|  │   │  Chrome  │  │ Firefox  │  │   Edge   │  │  Safari  │               │ |
|  │   │    ✅    │  │    ✅    │  │    ✅    │  │    ✅    │               │ |
|  │   │          │  │          │  │          │  │          │               │ |
|  │   │   90+    │  │   88+    │  │   90+    │  │   14+    │               │ |
|  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘               │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  PASSO 2: Digite o endereco                                                   |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                   |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │  🔒 https://sankofa.seubanco.com.br                             │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   DICA: Verifique sempre o cadeado 🔒 - significa conexao segura        │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  PASSO 3: Voce chegou!                                                        |
|  ━━━━━━━━━━━━━━━━━━━━━                                                        |
|                                                                               |
|  Voce vera o Dashboard Executivo (proxima secao)                              |
|                                                                               |
+==============================================================================+
```

### 1.2 O Que Voce Vai Ver

```
+==============================================================================+
|                         TELA INICIAL                                          |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  Sankofa   [Buscar...]                              [Alertas] [Usuario] │ |
|  ├────────────┬────────────────────────────────────────────────────────────┤ |
|  │            │                                                             │ |
|  │  Dashboard │              Dashboard Executivo                            │ |
|  │    ◀───    │                                                             │ |
|  │            │         Sistema Online   1 Algoritmo Ativo                  │ |
|  │  Transacoes│                                                             │ |
|  │            │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │ |
|  │  Calibragem│   │   518   │  │   23    │  │  95.6%  │  │ 28.0ms  │       │ |
|  │            │   │Transac. │  │ Fraudes │  │Aprovacao│  │Latencia │       │ |
|  │  Investig. │   └─────────┘  └─────────┘  └─────────┘  └─────────┘       │ |
|  │            │                                                             │ |
|  │  Revisao   │   ┌─────────────────────────────────────────────────────┐  │ |
|  │            │   │                   GRAFICO DE ATIVIDADE               │  │ |
|  │  Monit.    │   │                                                      │  │ |
|  │            │   │     ▄▄▄▄                                             │  │ |
|  │  Relatorios│   │    ▄████▄▄                      ▄▄                   │  │ |
|  │            │   │   ▄██████████▄▄▄▄          ▄▄▄▄████▄                 │  │ |
|  │  Metricas  │   │  ▄█████████████████▄▄▄▄▄▄████████████▄▄              │  │ |
|  │            │   │  0h    4h    8h   12h   16h   20h   24h              │  │ |
|  │  Alertas   │   │                                                      │  │ |
|  │            │   └─────────────────────────────────────────────────────┘  │ |
|  │            │                                                             │ |
|  │ Sankofa    │                                                             │ |
|  │  v12.0     │                                                             │ |
|  └────────────┴─────────────────────────────────────────────────────────────┤ |
|                                                                               |
|  LEGENDA:                                                                     |
|  ━━━━━━━━                                                                     |
|                                                                               |
|  ┌────────────┐  Menu lateral - navegue pelas funcoes                        |
|  │ Dashboard  │                                                              |
|  └────────────┘                                                              |
|                                                                               |
|  ┌─────────┐     Cartoes de resumo - visao rapida                           |
|  │   518   │                                                                 |
|  └─────────┘                                                                 |
|                                                                               |
|  ┌───────────────────┐  Graficos - evolucao ao longo do tempo               |
|  │  ▄▄▄▄████▄▄▄▄     │                                                       |
|  └───────────────────┘                                                       |
|                                                                               |
+==============================================================================+
```

---

## 2. Conhecendo o Dashboard

### 2.1 O Que Significam os Numeros?

```
+==============================================================================+
|                    ENTENDENDO OS INDICADORES                                  |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ┌─────────────────────────────────────────────────────────────────┐   │ |
|  │   │                    TRANSACOES HOJE                               │   │ |
|  │   │                                                                  │   │ |
|  │   │         ┌───────────────────┐                                   │   │ |
|  │   │         │       518         │                                   │   │ |
|  │   │         └───────────────────┘                                   │   │ |
|  │   │                                                                  │   │ |
|  │   │   O QUE E: Numero de transacoes que passaram pelo sistema hoje  │   │ |
|  │   │                                                                  │   │ |
|  │   │   NORMAL: Varia conforme o dia, geralmente 10.000-50.000        │   │ |
|  │   │                                                                  │   │ |
|  │   │   PREOCUPANTE: Se cair muito (sistema fora do ar?)              │   │ |
|  │   │                 Se subir muito (ataque?)                         │   │ |
|  │   │                                                                  │   │ |
|  │   └─────────────────────────────────────────────────────────────────┘   │ |
|  │                                                                          │ |
|  │   ┌─────────────────────────────────────────────────────────────────┐   │ |
|  │   │                    FRAUDES DETECTADAS                            │   │ |
|  │   │                                                                  │   │ |
|  │   │         ┌───────────────────┐                                   │   │ |
|  │   │         │        23         │                                   │   │ |
|  │   │         └───────────────────┘                                   │   │ |
|  │   │                                                                  │   │ |
|  │   │   O QUE E: Transacoes que o sistema identificou como suspeitas  │   │ |
|  │   │                                                                  │   │ |
|  │   │   NORMAL: Geralmente 2-5% do total de transacoes                │   │ |
|  │   │                                                                  │   │ |
|  │   │   PREOCUPANTE: Aumento repentino (novo tipo de fraude?)         │   │ |
|  │   │                 Zero (sistema com problema?)                     │   │ |
|  │   │                                                                  │   │ |
|  │   └─────────────────────────────────────────────────────────────────┘   │ |
|  │                                                                          │ |
|  │   ┌─────────────────────────────────────────────────────────────────┐   │ |
|  │   │                    TAXA DE APROVACAO                             │   │ |
|  │   │                                                                  │   │ |
|  │   │         ┌───────────────────┐                                   │   │ |
|  │   │         │      95.6%        │                                   │   │ |
|  │   │         └───────────────────┘                                   │   │ |
|  │   │                                                                  │   │ |
|  │   │   O QUE E: Percentual de transacoes aprovadas automaticamente   │   │ |
|  │   │                                                                  │   │ |
|  │   │   NORMAL: Deve ficar acima de 95%                               │   │ |
|  │   │                                                                  │   │ |
|  │   │   PREOCUPANTE: Abaixo de 90% (muitos falsos positivos)          │   │ |
|  │   │                                                                  │   │ |
|  │   └─────────────────────────────────────────────────────────────────┘   │ |
|  │                                                                          │ |
|  │   ┌─────────────────────────────────────────────────────────────────┐   │ |
|  │   │                    LATENCIA MEDIA                                │   │ |
|  │   │                                                                  │   │ |
|  │   │         ┌───────────────────┐                                   │   │ |
|  │   │         │      28.0ms       │                                   │   │ |
|  │   │         └───────────────────┘                                   │   │ |
|  │   │                                                                  │   │ |
|  │   │   O QUE E: Quanto tempo o sistema leva para analisar            │   │ |
|  │   │                                                                  │   │ |
|  │   │   NORMAL: Menos de 50ms (0.05 segundos)                         │   │ |
|  │   │                                                                  │   │ |
|  │   │   PREOCUPANTE: Acima de 200ms (sistema lento)                   │   │ |
|  │   │                                                                  │   │ |
|  │   │   NOVO: Agora monitorado em tempo real!                         │   │ |
|  │   │                                                                  │   │ |
|  │   └─────────────────────────────────────────────────────────────────┘   │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

### 2.2 As Cores dos Indicadores

```
+==============================================================================+
|                    SISTEMA DE CORES                                           |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   🟢 VERDE - TUDO NORMAL                                                 │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━━━                                                 │ |
|  │                                                                          │ |
|  │   ┌──────────────────────────────────────────────────────────────────┐  │ |
|  │   │                                                                   │  │ |
|  │   │  O que significa:                                                 │  │ |
|  │   │  • Sistema funcionando perfeitamente                              │  │ |
|  │   │  • Metricas dentro do esperado                                    │  │ |
|  │   │  • Nenhuma acao necessaria                                        │  │ |
|  │   │                                                                   │  │ |
|  │   │  O que fazer:                                                     │  │ |
|  │   │  • Continue trabalhando normalmente                               │  │ |
|  │   │  • Monitore periodicamente                                        │  │ |
|  │   │                                                                   │  │ |
|  │   └──────────────────────────────────────────────────────────────────┘  │ |
|  │                                                                          │ |
|  │   🟡 AMARELO - ATENCAO                                                   │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━                                                   │ |
|  │                                                                          │ |
|  │   ┌──────────────────────────────────────────────────────────────────┐  │ |
|  │   │                                                                   │  │ |
|  │   │  O que significa:                                                 │  │ |
|  │   │  • Alguma metrica saiu do padrao                                  │  │ |
|  │   │  • Ainda nao e critico                                            │  │ |
|  │   │  • Precisa de atencao                                             │  │ |
|  │   │                                                                   │  │ |
|  │   │  O que fazer:                                                     │  │ |
|  │   │  • Monitore a cada 15 minutos                                     │  │ |
|  │   │  • Verifique se esta piorando                                     │  │ |
|  │   │  • Avise o supervisor se continuar                                │  │ |
|  │   │                                                                   │  │ |
|  │   └──────────────────────────────────────────────────────────────────┘  │ |
|  │                                                                          │ |
|  │   🔴 VERMELHO - PROBLEMA                                                 │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━━━                                                 │ |
|  │                                                                          │ |
|  │   ┌──────────────────────────────────────────────────────────────────┐  │ |
|  │   │                                                                   │  │ |
|  │   │  O que significa:                                                 │  │ |
|  │   │  • Problema serio identificado                                    │  │ |
|  │   │  • Requer acao imediata                                           │  │ |
|  │   │  • Pode estar afetando clientes                                   │  │ |
|  │   │                                                                   │  │ |
|  │   │  O que fazer:                                                     │  │ |
|  │   │  • Investigue imediatamente                                       │  │ |
|  │   │  • Acione o supervisor                                            │  │ |
|  │   │  • Siga o procedimento de incidente                               │  │ |
|  │   │                                                                   │  │ |
|  │   └──────────────────────────────────────────────────────────────────┘  │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 3. Analisando Transacoes

### 3.1 Acessando a Lista

```
+==============================================================================+
|                    TELA DE TRANSACOES                                         |
+==============================================================================+
|                                                                               |
|  1. Clique em "Transacoes" no menu lateral                                    |
|  2. Voce vera a lista de transacoes                                           |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │  Sankofa                                               [Alertas] [User] │ |
|  ├────────────┬────────────────────────────────────────────────────────────┤ |
|  │            │                                                             │ |
|  │  Dashboard │                    Transacoes                               │ |
|  │            │     Lista e busca de transacoes processadas                 │ |
|  │  Transacoes│                                                             │ |
|  │    ◀───    │  ┌──────────────────────────────────────────────────────┐  │ |
|  │            │  │  FILTROS                                              │  │ |
|  │  Calibragem│  │                                                       │  │ |
|  │            │  │  [Buscar ID, CPF...] [Status ▼] [Tipo ▼] [Data ▼]    │  │ |
|  │  Investig. │  │                                                       │  │ |
|  │            │  └──────────────────────────────────────────────────────┘  │ |
|  │  Revisao   │                                                             │ |
|  │            │  Mostrando 50 de 518 transacoes                             │ |
|  │  Monitor.  │                                                             │ |
|  │            │  ┌──────────────────────────────────────────────────────┐  │ |
|  │  Relat.    │  │ ID               │ VALOR    │ TIPO   │ STATUS │ DATA │  │ |
|  │            │  ├──────────────────┼──────────┼────────┼────────┼──────┤  │ |
|  │  Metricas  │  │ TXN176425488...  │ R$ 1.234 │  PIX   │ ✅ OK  │ 14:48│  │ |
|  │            │  ├──────────────────┼──────────┼────────┼────────┼──────┤  │ |
|  │  Alertas   │  │ TXN176425488...  │ R$ 5.000 │CREDITO │ 🔴BLOCK│ 14:45│  │ |
|  │            │  ├──────────────────┼──────────┼────────┼────────┼──────┤  │ |
|  │            │  │ TXN176425488...  │ R$ 250   │  TED   │ ✅ OK  │ 14:42│  │ |
|  │            │  ├──────────────────┼──────────┼────────┼────────┼──────┤  │ |
|  │ v12.0      │  │ TXN176425488...  │ R$ 800   │ DEBITO │ ⚠️ REV │ 14:40│  │ |
|  │            │  └──────────────────┴──────────┴────────┴────────┴──────┘  │ |
|  └────────────┴─────────────────────────────────────────────────────────────┤ |
|                                                                               |
+==============================================================================+
```

### 3.2 Tipos de Transacao

```
+==============================================================================+
|                    TIPOS DE TRANSACAO NO SISTEMA                              |
+==============================================================================+
|                                                                               |
|  O sistema analisa diferentes tipos de transacao. Cada um tem                 |
|  caracteristicas proprias que influenciam a avaliacao de risco:               |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   💳 PIX                                                                 │ |
|  │   ━━━━━                                                                  │ |
|  │   • Transferencia instantanea (24/7)                                    │ |
|  │   • Risco BASE: ALTO (dificil reverter)                                 │ |
|  │   • Fraudes comuns: Golpe do falso funcionario, sequestro               │ |
|  │   • O sistema analisa: valor, horario, destinatario, velocidade         │ |
|  │                                                                          │ |
|  │   Sinais de alerta:                                                      │ |
|  │   [!] PIX de madrugada para desconhecido                                │ |
|  │   [!] Valor muito acima do padrao do cliente                            │ |
|  │   [!] Multiplos PIX em sequencia rapida                                 │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   💳 CREDITO (Cartao de Credito)                                         │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                         │ |
|  │   • Compra com limite pre-aprovado                                       │ |
|  │   • Risco BASE: MEDIO (pode haver chargeback)                           │ |
|  │   • Fraudes comuns: Clonagem, compra online com dados vazados           │ |
|  │   • O sistema analisa: local, comerciante, padrao de compra             │ |
|  │                                                                          │ |
|  │   Sinais de alerta:                                                      │ |
|  │   [!] Compra internacional sem aviso de viagem                          │ |
|  │   [!] Varias compras em lojas diferentes em minutos                     │ |
|  │   [!] Primeiro uso online apos meses sem usar                           │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   💳 DEBITO (Cartao de Debito)                                           │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━━━━━━━                                           │ |
|  │   • Desconto direto do saldo da conta                                    │ |
|  │   • Risco BASE: BAIXO (requer senha + cartao fisico)                    │ |
|  │   • Fraudes comuns: Clonagem em maquininha, troca de cartao             │ |
|  │   • O sistema analisa: localizacao, ATM usado, horario                  │ |
|  │                                                                          │ |
|  │   Sinais de alerta:                                                      │ |
|  │   [!] Saques em ATMs de cidades diferentes em minutos                   │ |
|  │   [!] Compra presencial + saque logo depois                             │ |
|  │   [!] Uso em estabelecimento de alto risco                              │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   💳 TED/DOC                                                             │ |
|  │   ━━━━━━━━                                                               │ |
|  │   • Transferencia bancaria tradicional                                   │ |
|  │   • Risco BASE: MEDIO (pode ser revertido em alguns casos)              │ |
|  │   • Fraudes comuns: Golpe do boleto, engenharia social                  │ |
|  │   • O sistema analisa: valor, destinatario, frequencia                  │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  RESUMO DE RISCO POR TIPO:                                                    |
|  ┌────────────┬───────────────────────────────────────────────────────────┐ |
|  │ TIPO       │ RISCO BASE │ REVERSIVEL? │ FRAUDE MAIS COMUM             │ |
|  ├────────────┼────────────┼─────────────┼───────────────────────────────┤ |
|  │ PIX        │    ALTO    │    NAO      │ Engenharia social            │ |
|  │ CREDITO    │    MEDIO   │    SIM      │ Clonagem / dados vazados     │ |
|  │ DEBITO     │    BAIXO   │    NAO      │ Clonagem fisica              │ |
|  │ TED        │    MEDIO   │    TALVEZ   │ Golpe do falso boleto        │ |
|  └────────────┴────────────┴─────────────┴───────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

### 3.3 Entendendo os Status

```
+==============================================================================+
|                    STATUS DAS TRANSACOES                                      |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ✅ OK (APROVADA)                                                       │ |
|  │   ━━━━━━━━━━━━━━━━                                                       │ |
|  │                                                                          │ |
|  │   O que significa:                                                       │ |
|  │   • Sistema avaliou como segura (score < 30)                            │ |
|  │   • Foi aprovada automaticamente                                         │ |
|  │   • Cliente pode usar normalmente                                        │ |
|  │                                                                          │ |
|  │   Cor: Verde                                                             │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ⚠️ REV (EM REVISAO)                                                    │ |
|  │   ━━━━━━━━━━━━━━━━━━━                                                    │ |
|  │                                                                          │ |
|  │   O que significa:                                                       │ |
|  │   • Sistema tem duvida (score entre 30-85)                              │ |
|  │   • Esta na fila de revisao manual                                       │ |
|  │   • Precisa de decisao humana                                            │ |
|  │                                                                          │ |
|  │   Cor: Amarelo                                                           │ |
|  │   Acao: Clique para revisar                                              │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   🔴 BLOCK (BLOQUEADA)                                                   │ |
|  │   ━━━━━━━━━━━━━━━━━━━━                                                   │ |
|  │                                                                          │ |
|  │   O que significa:                                                       │ |
|  │   • Sistema identificou como fraude (score > 85)                        │ |
|  │   • Foi bloqueada automaticamente                                        │ |
|  │   • Precisa de investigacao                                              │ |
|  │                                                                          │ |
|  │   Cor: Vermelho                                                          │ |
|  │   Acao: Clique para investigar                                           │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 4. Investigando Fraudes

### 4.1 Central de Investigacao

```
+==============================================================================+
|                    CENTRAL DE INVESTIGACAO                                    |
+==============================================================================+
|                                                                               |
|  Acesse: Menu > Investigacao                                                  |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │  ┌────────────────────────────────────────────────────────────────────┐ │ |
|  │  │                    DETALHES DA TRANSACAO                           │ │ |
|  │  │                                                                     │ │ |
|  │  │  ID: TXN1764254880604000                                           │ │ |
|  │  │                                                                     │ │ |
|  │  │  ┌──────────────────┬──────────────────┬──────────────────┐        │ │ |
|  │  │  │      VALOR       │     HORARIO      │      LOCAL       │        │ │ |
|  │  │  │                  │                  │                  │        │ │ |
|  │  │  │   R$ 5.000,00    │      03:15       │   Rio de Janeiro │        │ │ |
|  │  │  │                  │                  │                  │        │ │ |
|  │  │  │  ⚠️ Alto valor   │  ⚠️ Madrugada   │   Cliente mora   │        │ │ |
|  │  │  │                  │                  │   em SP          │        │ │ |
|  │  │  └──────────────────┴──────────────────┴──────────────────┘        │ │ |
|  │  │                                                                     │ │ |
|  │  │  ┌────────────────────────────────────────────────────────────┐    │ │ |
|  │  │  │                    SCORE DE RISCO                          │    │ │ |
|  │  │  │                                                             │    │ │ |
|  │  │  │    0        30                    85         87.5    100   │    │ │ |
|  │  │  │    │─────────│──────────────────────│──────────●─────────│    │ │ |
|  │  │  │    │  BAIXO  │        MEDIO         │         ALTO        │    │ │ |
|  │  │  │                                                             │    │ │ |
|  │  │  │                     SCORE: 87.5 - ALTO RISCO               │    │ │ |
|  │  │  └────────────────────────────────────────────────────────────┘    │ │ |
|  │  │                                                                     │ │ |
|  │  └────────────────────────────────────────────────────────────────────┘ │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

### 4.2 O Que Analisar

```
+==============================================================================+
|                    CHECKLIST DE INVESTIGACAO                                  |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   PERGUNTE-SE:                                                           │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   1. VALOR                                                      │    │ |
|  │   │      ┌─────────────────────────────────────────────────────┐   │    │ |
|  │   │      │  E compativel com o perfil do cliente?              │   │    │ |
|  │   │      │                                                      │   │    │ |
|  │   │      │  Cliente costuma fazer transacoes deste valor?      │   │    │ |
|  │   │      │  Se nao, pode ser fraude OU uma compra especial     │   │    │ |
|  │   │      └─────────────────────────────────────────────────────┘   │    │ |
|  │   │                                                                 │    │ |
|  │   │   2. HORARIO                                                    │    │ |
|  │   │      ┌─────────────────────────────────────────────────────┐   │    │ |
|  │   │      │  O cliente costuma transacionar neste horario?      │   │    │ |
|  │   │      │                                                      │   │    │ |
|  │   │      │  Transacao as 03h da manha de pessoa que trabalha   │   │    │ |
|  │   │      │  em horario comercial e suspeita                     │   │    │ |
|  │   │      └─────────────────────────────────────────────────────┘   │    │ |
|  │   │                                                                 │    │ |
|  │   │   3. LOCAL                                                      │    │ |
|  │   │      ┌─────────────────────────────────────────────────────┐   │    │ |
|  │   │      │  A transacao foi feita de onde o cliente mora?      │   │    │ |
|  │   │      │                                                      │   │    │ |
|  │   │      │  Cliente de SP com transacao no RJ em 30min         │   │    │ |
|  │   │      │  e impossivel fisicamente                            │   │    │ |
|  │   │      └─────────────────────────────────────────────────────┘   │    │ |
|  │   │                                                                 │    │ |
|  │   │   4. HISTORICO                                                  │    │ |
|  │   │      ┌─────────────────────────────────────────────────────┐   │    │ |
|  │   │      │  O cliente ja fez transacoes similares?             │   │    │ |
|  │   │      │                                                      │   │    │ |
|  │   │      │  Verifique ultimas 10 transacoes                     │   │    │ |
|  │   │      │  Existe padrao? Esta fora do comum?                  │   │    │ |
|  │   │      └─────────────────────────────────────────────────────┘   │    │ |
|  │   │                                                                 │    │ |
|  │   │   5. EXPLICACAO DO SISTEMA (NOVO!)                              │    │ |
|  │   │      ┌─────────────────────────────────────────────────────┐   │    │ |
|  │   │      │  Leia a explicacao que o sistema gerou              │   │    │ |
|  │   │      │                                                      │   │    │ |
|  │   │      │  O sistema explica EXATAMENTE por que flagrou       │   │    │ |
|  │   │      │  Isso ajuda a tomar decisao mais rapida              │   │    │ |
|  │   │      └─────────────────────────────────────────────────────┘   │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 5. Revisao Manual

### 5.1 Prioridades

```
+==============================================================================+
|                    FILA DE REVISAO MANUAL                                     |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   NIVEIS DE PRIORIDADE                                                   │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   🔴 CRITICO (SLA: 1 minuto)                                   │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━━━━━━━                                    │    │ |
|  │   │                                                                 │    │ |
|  │   │   • Transacoes de valor muito alto                             │    │ |
|  │   │   • Score proximo de 100                                       │    │ |
|  │   │   • Cliente VIP                                                 │    │ |
|  │   │                                                                 │    │ |
|  │   │   ACAO: Resolver IMEDIATAMENTE!                                │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   🟠 ALTO (SLA: 5 minutos)                                     │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━━━━━                                      │    │ |
|  │   │                                                                 │    │ |
|  │   │   • Score entre 75-85                                          │    │ |
|  │   │   • Valor significativo                                        │    │ |
|  │   │   • Multiplos indicadores                                      │    │ |
|  │   │                                                                 │    │ |
|  │   │   ACAO: Priorizar apos criticos                                │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   🟡 MEDIO (SLA: 15 minutos)                                   │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━━━━━━                                     │    │ |
|  │   │                                                                 │    │ |
|  │   │   • Score entre 50-75                                          │    │ |
|  │   │   • Valor moderado                                             │    │ |
|  │   │   • Alguns indicadores                                         │    │ |
|  │   │                                                                 │    │ |
|  │   │   ACAO: Resolver quando possivel                               │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   🟢 BAIXO (SLA: 30 minutos)                                   │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━━━━━━━                                    │    │ |
|  │   │                                                                 │    │ |
|  │   │   • Score entre 30-50                                          │    │ |
|  │   │   • Valor baixo                                                │    │ |
|  │   │   • Poucos indicadores                                         │    │ |
|  │   │                                                                 │    │ |
|  │   │   ACAO: Pode aguardar                                          │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 6. Entendendo as Explicacoes (NOVO)

### 6.1 O Que Sao as Explicacoes?

```
+==============================================================================+
|                    EXPLICACOES DO SISTEMA                                     |
+==============================================================================+
|                                                                               |
|  O QUE E?                                                                     |
|  ━━━━━━━━                                                                     |
|  Cada transacao flagrada agora vem com uma explicacao em texto simples       |
|  de por que foi considerada suspeita.                                         |
|                                                                               |
|  POR QUE ISSO E IMPORTANTE?                                                   |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━                                                   |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ANTES (sem explicacao):                                                │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │  Score: 87.5                                                    │    │ |
|  │   │  Status: BLOQUEADA                                              │    │ |
|  │   │                                                                  │    │ |
|  │   │  🤔 "Por que foi bloqueada? Tenho que adivinhar?"               │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   AGORA (com explicacao):                                                │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │  Score: 87.5                                                    │    │ |
|  │   │  Status: BLOQUEADA                                              │    │ |
|  │   │                                                                  │    │ |
|  │   │  EXPLICACAO:                                                    │    │ |
|  │   │  ┌──────────────────────────────────────────────────────────┐  │    │ |
|  │   │  │  "Transacao de alto valor (R$ 15.000) em horario         │  │    │ |
|  │   │  │   noturno (03:00) com velocidade de transacoes           │  │    │ |
|  │   │  │   acima do padrao do cliente"                            │  │    │ |
|  │   │  └──────────────────────────────────────────────────────────┘  │    │ |
|  │   │                                                                  │    │ |
|  │   │  FATORES DE RISCO:                                              │    │ |
|  │   │  • Valor: +45% de impacto                                       │    │ |
|  │   │  • Horario: +32% de impacto                                     │    │ |
|  │   │                                                                  │    │ |
|  │   │  FATORES DE PROTECAO:                                           │    │ |
|  │   │  • Dispositivo conhecido: -15%                                  │    │ |
|  │   │                                                                  │    │ |
|  │   │  ✅ "Agora sei exatamente o que aconteceu!"                     │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

### 6.2 Como Usar as Explicacoes

```
+==============================================================================+
|                    USANDO AS EXPLICACOES                                      |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │  EXEMPLO PRATICO                                                         │ |
|  │                                                                          │ |
|  │  ┌────────────────────────────────────────────────────────────────────┐ │ |
|  │  │                                                                     │ │ |
|  │  │  TRANSACAO: TXN-2025-001                                           │ │ |
|  │  │  VALOR: R$ 15.000                                                   │ │ |
|  │  │  HORARIO: 03:00                                                     │ │ |
|  │  │  LOCAL: Rio de Janeiro                                              │ │ |
|  │  │                                                                     │ │ |
|  │  │  ┌───────────────────────────────────────────────────────────────┐ │ │ |
|  │  │  │               EXPLICACAO DO SISTEMA                           │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  │  "Transacao de alto valor (R$ 15.000) em horario noturno      │ │ │ |
|  │  │  │   (03:00) com velocidade de transacoes acima do padrao        │ │ │ |
|  │  │  │   do cliente"                                                  │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  │  FATORES DE RISCO:                                            │ │ │ |
|  │  │  │  ┌─────────────────────────────────────────────────────────┐  │ │ │ |
|  │  │  │  │ amount_normalized  ████████████████████████░░░░  45%    │  │ │ │ |
|  │  │  │  │ is_night           ████████████████░░░░░░░░░░░░  32%    │  │ │ │ |
|  │  │  │  │ velocity_1h        ████████░░░░░░░░░░░░░░░░░░░░  15%    │  │ │ │ |
|  │  │  │  └─────────────────────────────────────────────────────────┘  │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  │  FATORES DE PROTECAO:                                         │ │ │ |
|  │  │  │  ┌─────────────────────────────────────────────────────────┐  │ │ │ |
|  │  │  │  │ device_trust       ░░░░░░░░░░░░░░░░░░░░░░░░░████  -15%  │  │ │ │ |
|  │  │  │  └─────────────────────────────────────────────────────────┘  │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  └───────────────────────────────────────────────────────────────┘ │ │ |
|  │  │                                                                     │ │ |
|  │  │  MINHA ANALISE:                                                     │ │ |
|  │  │  ┌───────────────────────────────────────────────────────────────┐ │ │ |
|  │  │  │  O sistema flagrou por 3 motivos:                             │ │ │ |
|  │  │  │  1. Valor alto (R$ 15.000)                                    │ │ │ |
|  │  │  │  2. Horario incomum (03:00 da manha)                          │ │ │ |
|  │  │  │  3. Muitas transacoes em pouco tempo                          │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  │  Porem, o dispositivo e conhecido (-15%)                      │ │ │ |
|  │  │  │                                                                │ │ │ |
|  │  │  │  Vou verificar:                                               │ │ │ |
|  │  │  │  • Cliente viaja muito? (pode estar em fuso horario diferente)│ │ │ |
|  │  │  │  • Historico de compras de alto valor?                        │ │ │ |
|  │  │  │  • Entrar em contato para confirmar?                          │ │ │ |
|  │  │  └───────────────────────────────────────────────────────────────┘ │ │ |
|  │  │                                                                     │ │ |
|  │  └────────────────────────────────────────────────────────────────────┘ │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 7. Monitorando a Saude

![Metricas](images/metricas_performance_dashboard.png)

```
+==============================================================================+
|                    TELA DE MONITORAMENTO                                      |
+==============================================================================+
|                                                                               |
|  Acesse: Menu > Monitoramento                                                 |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │              PAINEL DE SAUDE DO SISTEMA                                  │ |
|  │                                                                          │ |
|  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │ |
|  │   │  LATENCIA P50 │  │  LATENCIA P95 │  │  LATENCIA P99 │               │ |
|  │   │               │  │               │  │               │               │ |
|  │   │    28 ms      │  │   300 ms      │  │   311 ms      │               │ |
|  │   │               │  │               │  │               │               │ |
|  │   │   ✅ BOM      │  │   ⚠️ ATENCAO  │  │   ⚠️ ATENCAO  │               │ |
|  │   └───────────────┘  └───────────────┘  └───────────────┘               │ |
|  │                                                                          │ |
|  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │ |
|  │   │  THROUGHPUT   │  │  ERROR RATE   │  │    UPTIME     │               │ |
|  │   │               │  │               │  │               │               │ |
|  │   │  33.88 TPS    │  │    0.0%       │  │    99.9%      │               │ |
|  │   │               │  │               │  │               │               │ |
|  │   │   ✅ OTIMO    │  │   ✅ PERFEITO │  │   ✅ EXCELENTE│               │ |
|  │   └───────────────┘  └───────────────┘  └───────────────┘               │ |
|  │                                                                          │ |
|  │   O QUE SIGNIFICAM ESSES NUMEROS?                                        │ |
|  │   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │ |
|  │                                                                          │ |
|  │   • LATENCIA: Quanto tempo o sistema demora para responder               │ |
|  │     - P50: Metade das requisicoes (28ms = OTIMO)                        │ |
|  │     - P95: 95% das requisicoes (300ms = OK)                             │ |
|  │     - P99: 99% das requisicoes (311ms = OK)                             │ |
|  │                                                                          │ |
|  │   • THROUGHPUT: Quantas transacoes por segundo                          │ |
|  │     - 33.88 TPS = Sistema suporta bem a carga atual                     │ |
|  │                                                                          │ |
|  │   • ERROR RATE: Taxa de erros                                           │ |
|  │     - 0.0% = Nenhum erro (perfeito!)                                    │ |
|  │                                                                          │ |
|  │   • UPTIME: Tempo que o sistema ficou no ar                             │ |
|  │     - 99.9% = Quase 100% disponivel (excelente!)                        │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 8. Gerando Relatorios

```
+==============================================================================+
|                    GERANDO RELATORIOS                                         |
+==============================================================================+
|                                                                               |
|  Acesse: Menu > Relatorios                                                    |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   TIPOS DE RELATORIOS DISPONIVEIS                                        │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   📊 RELATORIO DIARIO                                          │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━                                          │    │ |
|  │   │   • Total de transacoes                                         │    │ |
|  │   │   • Fraudes detectadas                                          │    │ |
|  │   │   • Taxa de aprovacao                                           │    │ |
|  │   │   • Tempo medio de resposta                                     │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   📈 RELATORIO SEMANAL                                         │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━━━━                                         │    │ |
|  │   │   • Tendencias de fraude                                        │    │ |
|  │   │   • Comparativo com semana anterior                             │    │ |
|  │   │   • Top tipos de fraude                                         │    │ |
|  │   │   • Performance do modelo ML                                    │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  │   ┌────────────────────────────────────────────────────────────────┐    │ |
|  │   │                                                                 │    │ |
|  │   │   📋 RELATORIO LGPD                                            │    │ |
|  │   │   ━━━━━━━━━━━━━━━━━                                            │    │ |
|  │   │   • Decisoes automatizadas                                      │    │ |
|  │   │   • Explicacoes fornecidas                                      │    │ |
|  │   │   • Solicitacoes de revisao                                     │    │ |
|  │   │   • Compliance status                                           │    │ |
|  │   │                                                                 │    │ |
|  │   └────────────────────────────────────────────────────────────────┘    │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 9. Entendendo os Alertas

![Alertas](images/fluxo_alertas_monitoramento.png)

```
+==============================================================================+
|                    TIPOS DE ALERTAS                                           |
+==============================================================================+
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   🔴 ALERTA CRITICO                                                      │ |
|  │   ━━━━━━━━━━━━━━━━━                                                      │ |
|  │                                                                          │ |
|  │   Quando aparece:                                                        │ |
|  │   • Fraude de alto valor detectada                                       │ |
|  │   • Sistema caiu                                                         │ |
|  │   • Taxa de erro muito alta                                              │ |
|  │                                                                          │ |
|  │   O que fazer:                                                           │ |
|  │   • PARE o que esta fazendo                                              │ |
|  │   • Investigue imediatamente                                             │ |
|  │   • Avise o supervisor                                                   │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   🟠 ALERTA ALTO                                                         │ |
|  │   ━━━━━━━━━━━━━                                                          │ |
|  │                                                                          │ |
|  │   Quando aparece:                                                        │ |
|  │   • Transacao suspeita de valor significativo                           │ |
|  │   • Fila de revisao crescendo                                           │ |
|  │   • Latencia aumentando                                                  │ |
|  │                                                                          │ |
|  │   O que fazer:                                                           │ |
|  │   • Priorize quando terminar tarefa atual                               │ |
|  │   • Monitore evolucao                                                    │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   🟡 ALERTA MEDIO                                                        │ |
|  │   ━━━━━━━━━━━━━                                                          │ |
|  │                                                                          │ |
|  │   Quando aparece:                                                        │ |
|  │   • Transacao na zona cinza                                             │ |
|  │   • Pequeno aumento em metricas                                         │ |
|  │                                                                          │ |
|  │   O que fazer:                                                           │ |
|  │   • Anote para verificar depois                                         │ |
|  │   • Nao precisa parar o que esta fazendo                                │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────┐ |
|  │                                                                          │ |
|  │   ℹ️ INFORMATIVO                                                         │ |
|  │   ━━━━━━━━━━━━━                                                          │ |
|  │                                                                          │ |
|  │   Quando aparece:                                                        │ |
|  │   • Modelo foi atualizado                                               │ |
|  │   • Configuracao alterada                                                │ |
|  │   • Manutencao programada                                               │ |
|  │                                                                          │ |
|  │   O que fazer:                                                           │ |
|  │   • Apenas tome conhecimento                                             │ |
|  │   • Nenhuma acao necessaria                                              │ |
|  │                                                                          │ |
|  └─────────────────────────────────────────────────────────────────────────┘ |
|                                                                               |
+==============================================================================+
```

---

## 10. Perguntas Frequentes

```
+==============================================================================+
|                    PERGUNTAS FREQUENTES (FAQ)                                 |
+==============================================================================+
|                                                                               |
|  ❓ O que fazer se o sistema ficar lento?                                     |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                       |
|  1. Verifique o painel de Monitoramento                                       |
|  2. Veja se a latencia esta acima de 500ms                                    |
|  3. Avise a equipe de TI se continuar                                         |
|                                                                               |
|  ❓ Como sei se uma transacao e realmente fraude?                              |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                  |
|  1. Leia a explicacao do sistema (NOVO!)                                      |
|  2. Verifique o historico do cliente                                          |
|  3. Compare com o padrao de comportamento                                     |
|  4. Na duvida, escale para o supervisor                                       |
|                                                                               |
|  ❓ O que significa "score de risco"?                                          |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                          |
|  E uma nota de 0 a 100 que indica a probabilidade de fraude:                  |
|  • 0-30: Provavelmente legitima (verde)                                       |
|  • 30-85: Precisa de analise humana (amarelo)                                 |
|  • 85-100: Provavelmente fraude (vermelho)                                    |
|                                                                               |
|  ❓ Por que o sistema bloqueou uma transacao legitima?                         |
|  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                             |
|  Isso e chamado "falso positivo". Acontece quando:                            |
|  • Cliente fez algo fora do comum                                             |
|  • Transacao muito diferente do historico                                     |
|  • Modelo precisa de ajuste                                                   |
|                                                                               |
|  Nesse caso:                                                                  |
|  1. Marque como "Falso Positivo"                                              |
|  2. Libere a transacao do cliente                                             |
|  3. O sistema aprende com seu feedback!                                       |
|                                                                               |
+==============================================================================+
```

---

## 11. Glossario

```
+==============================================================================+
|                    GLOSSARIO DE TERMOS                                        |
+==============================================================================+
|                                                                               |
|  TERMO              SIGNIFICADO                                               |
|  ━━━━━━             ━━━━━━━━━━━                                               |
|                                                                               |
|  Score              Nota de 0 a 100 que indica risco de fraude               |
|                                                                               |
|  Falso Positivo     Transacao legitima que foi marcada como fraude           |
|                                                                               |
|  Falso Negativo     Fraude que nao foi detectada (o pior caso!)              |
|                                                                               |
|  Threshold          Limite que define aprovacao/revisao/bloqueio             |
|                                                                               |
|  Latencia           Tempo que o sistema leva para responder                  |
|                                                                               |
|  TPS                Transacoes por segundo                                    |
|                                                                               |
|  SLA                Acordo de nivel de servico (tempo maximo)                |
|                                                                               |
|  ML/IA              Machine Learning / Inteligencia Artificial               |
|                                                                               |
|  LGPD               Lei Geral de Protecao de Dados                           |
|                                                                               |
|  Feature            Caracteristica analisada (valor, horario, etc)           |
|                                                                               |
|  Ensemble           Combinacao de varios modelos de ML                       |
|                                                                               |
|  Batch              Processamento de varias transacoes de uma vez            |
|                                                                               |
|  Dashboard          Painel visual com informacoes do sistema                 |
|                                                                               |
|  Uptime             Tempo que o sistema ficou disponivel                     |
|                                                                               |
+==============================================================================+
```

---

*Manual do Usuario atualizado em 27 de Novembro de 2025*  
*Sankofa Enterprise Pro v12.0*  
*Total: 15+ diagramas e ilustracoes visuais*
