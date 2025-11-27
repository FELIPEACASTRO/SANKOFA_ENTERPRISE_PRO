# Manual do Usuario - Sankofa Enterprise Pro v12.0
## Guia Completo para Analistas de Fraude

**Versao:** 12.0  
**Ultima Atualizacao:** 27 de Novembro de 2025  
**Publico:** Analistas de Fraude, Gerentes de Operacoes, Compliance Officers

---

## Bem-vindo ao Sankofa!

Este manual vai te guiar passo a passo no uso do sistema de deteccao de fraudes. Nao se preocupe se voce nao e tecnico - este guia foi feito pensando em voce!

---

## Indice

1. [Primeiros Passos](#1-primeiros-passos)
2. [Conhecendo o Dashboard](#2-conhecendo-o-dashboard)
3. [Analisando Transacoes](#3-analisando-transacoes)
4. [Investigando Fraudes](#4-investigando-fraudes)
5. [Revisao Manual](#5-revisao-manual)
6. [Entendendo as Explicacoes (NOVO)](#6-entendendo-as-explicacoes)
7. [Monitorando a Saude](#7-monitorando-a-saude)
8. [Gerando Relatorios](#8-gerando-relatorios)
9. [Entendendo os Alertas](#9-entendendo-os-alertas)
10. [Perguntas Frequentes](#10-perguntas-frequentes)
11. [Glossario](#11-glossario)

---

## 1. Primeiros Passos

### 1.1 Como Acessar

1. Abra seu navegador (Chrome, Firefox, Edge ou Safari)
2. Digite o endereco do sistema na barra de enderecos
3. Voce vera a tela inicial do Sankofa

### 1.2 Navegadores Suportados

| Navegador | Versao Minima | Recomendado |
|-----------|---------------|-------------|
| Chrome | 90+ | Sim |
| Firefox | 88+ | Sim |
| Edge | 90+ | Sim |
| Safari | 14+ | OK |

### 1.3 Primeira Coisa que Voce Vera

Ao acessar, voce caira direto no **Dashboard Executivo**:

```
+-------------------------------------------------------------------------+
|  Sankofa   [Buscar...]                              [Alertas] [Usuario] |
+----------------+--------------------------------------------------------+
|                |                                                         |
|  Dashboard     |              Dashboard Executivo                        |
|  (selecionado) |                                                         |
|                |         Sistema Online   1 Algoritmo Ativo              |
|  Transacoes    |                                                         |
|                |   +---------+  +---------+  +---------+  +---------+   |
|  Calibragem    |   |   518   |  |   23    |  |  95.6%  |  | 28.0ms  |   |
|                |   |Transacoes|  | Fraudes |  |Aprovacao|  |Latencia |   |
|  Investigacao  |   +---------+  +---------+  +---------+  +---------+   |
|                |                                                         |
|  Revisao       |                                                         |
|                |                                                         |
|  Monitoramento |                                                         |
|                |                                                         |
|  Relatorios    |                                                         |
|                |                                                         |
|  Metricas      |                                                         |
|                |                                                         |
|  Alertas       |                                                         |
|                |                                                         |
|  Sankofa v12.0 |                                                         |
+----------------+---------------------------------------------------------+
```

---

## 2. Conhecendo o Dashboard

### 2.1 O Que Significam os Numeros?

**Transacoes Hoje:** Quantas transacoes passaram pelo sistema hoje.
- Normal: Varia conforme o dia, geralmente entre 10.000-50.000.

**Fraudes Detectadas:** Quantas transacoes o sistema identificou como suspeitas.
- Normal: Geralmente 2-5% do total de transacoes.

**Taxa de Aprovacao:** Percentual de transacoes aprovadas automaticamente.
- Normal: Deve ficar acima de 95%.

**Latencia Media:** Quanto tempo o sistema leva para analisar uma transacao.
- Normal: Menos de 50ms (0.05 segundos).
- NOVO: Agora monitorado em tempo real!

### 2.2 As Cores dos Indicadores

| Cor | Significado | Acao |
|-----|-------------|------|
| Verde | Tudo normal | Nenhuma |
| Amarelo | Atencao | Monitorar |
| Vermelho | Problema | Investigar imediatamente |

---

## 3. Analisando Transacoes

### 3.1 Acessando a Lista

1. Clique em **Transacoes** no menu lateral
2. Voce vera uma lista com todas as transacoes do dia

### 3.2 Entendendo a Lista

```
+-------------------------------------------------------------------------+
|                            Transacoes                                    |
|        Lista e busca de transacoes processadas em tempo real             |
+-------------------------------------------------------------------------+
|  Filtros                                                                 |
|  [Buscar: ID, CPF, cidade...]   [Status: Todos]   [Tipo: Todos]         |
+-------------------------------------------------------------------------+
|  Mostrando 50 de 250 transacoes                                          |
+-------------------------------------------------------------------------+
|  ID                    | Valor      | Tipo   | Canal | Local   | Data   |
+------------------------+------------+--------+-------+---------+--------+
|  TXN1764254880868000   | R$ 1.234   |  PIX   |  TED  |Sao Paulo| 14:48  |
|  TXN1764254880604000   | -R$ 100    |CREDITO |  PIX  |Rio de J.| 14:48  |
+-------------------------------------------------------------------------+
```

### 3.3 Os Tipos de Transacao

| Tipo | O que e |
|------|---------|
| **PIX** | Pagamento instantaneo (mais comum hoje) |
| **TED** | Transferencia bancaria tradicional |
| **CREDITO** | Compra no cartao de credito |
| **DEBITO** | Compra no cartao de debito |

---

## 4. Investigando Fraudes

### 4.1 Quando Investigar?

Voce deve investigar quando:
- Receber um alerta de fraude
- Ver uma transacao com score alto
- Cliente reclamar de bloqueio indevido

### 4.2 Acessando a Central de Investigacao

1. Clique em **Investigacao** no menu
2. Voce vera os casos que precisam de atencao

### 4.3 O Que Analisar em um Caso

1. **Valor da transacao:** E compativel com o perfil do cliente?
2. **Horario:** O cliente costuma transacionar nesse horario?
3. **Local:** A transacao foi feita de onde o cliente mora?
4. **Historico:** O cliente ja fez transacoes similares?
5. **Explicacao do Sistema (NOVO):** Por que o sistema flagrou?

### 4.4 Tomando uma Decisao

| Decisao | Quando Usar | O Que Acontece |
|---------|-------------|----------------|
| **Confirmar Fraude** | Quando tem certeza que e fraude | Transacao e bloqueada |
| **Falso Positivo** | Quando a transacao e legitima | Libera o cliente |
| **Escalar** | Quando tem duvida | Vai para supervisor |

---

## 5. Revisao Manual

### 5.1 O Que e a Revisao Manual?

Algumas transacoes ficam na "zona cinza" - nao sao claramente fraude nem claramente legitimas. Essas vao para a fila de revisao manual.

### 5.2 Acessando a Fila

1. Clique em **Revisao Manual** no menu
2. Voce vera todas as transacoes aguardando

### 5.3 Prioridades

| Cor | Prioridade | SLA | O Que Fazer |
|-----|------------|-----|-------------|
| Vermelho | CRITICO | 1 min | Resolver imediatamente! |
| Laranja | ALTO | 5 min | Priorizar |
| Amarelo | MEDIO | 15 min | Resolver quando possivel |
| Verde | BAIXO | 30 min | Pode aguardar |

### 5.4 Como Revisar

1. Clique na transacao para ver detalhes
2. **NOVO:** Leia a explicacao do sistema (por que foi flagrada)
3. Analise as informacoes apresentadas
4. Clique em **Aprovar** ou **Rejeitar**
5. Digite uma justificativa (obrigatorio)
6. Confirme sua decisao

---

## 6. Entendendo as Explicacoes (NOVO)

### 6.1 O Que Sao as Explicacoes?

Cada transacao flagrada agora vem com uma explicacao em texto simples de por que foi considerada suspeita. Isso ajuda voce a tomar decisoes mais rapidas e seguras.

### 6.2 Exemplo de Explicacao

```
+-------------------------------------------------------------------------+
|  EXPLICACAO DA ANALISE                                                   |
+-------------------------------------------------------------------------+
|                                                                          |
|  "Transacao de alto valor (R$ 15.000) em horario noturno (03:00)        |
|   com velocidade de transacoes acima do padrao do cliente"               |
|                                                                          |
|  FATORES DE RISCO:                                                       |
|  - Valor muito alto para o perfil                                        |
|  - Horario incomum (madrugada)                                           |
|  - Muitas transacoes em pouco tempo                                      |
|                                                                          |
|  FATORES DE PROTECAO:                                                    |
|  - Dispositivo conhecido                                                 |
|  - Localizacao habitual                                                  |
|                                                                          |
+-------------------------------------------------------------------------+
```

### 6.3 Por Que Isso Importa?

1. **Mais Rapido:** Voce entende o problema sem precisar analisar dezenas de dados
2. **Mais Seguro:** Sabe exatamente o que o sistema viu
3. **LGPD:** O cliente tem direito a saber por que foi bloqueado

### 6.4 Como Usar as Explicacoes

| Se a explicacao diz... | Provavelmente... |
|------------------------|------------------|
| "Horario incomum" | Verifique se o cliente costuma usar a noite |
| "Valor muito alto" | Compare com transacoes anteriores do cliente |
| "Localizacao diferente" | O cliente pode estar viajando |
| "Dispositivo novo" | O cliente trocou de celular? |

---

## 7. Monitorando a Saude

### 7.1 Para Que Serve?

A pagina de Monitoramento mostra se o sistema esta funcionando bem.

### 7.2 Acessando o Monitor

1. Clique em **Monitoramento** no menu
2. Voce vera o status de todos os componentes

### 7.3 O Que Cada Indicador Significa (ATUALIZADO)

| Indicador | Bom | Atencao | Critico |
|-----------|-----|---------|---------|
| **Status Geral** | Saudavel | Degradado | Critico |
| **TPS** | >30 | 10-30 | <10 |
| **Latencia p95** | <100ms | 100-300ms | >300ms |
| **Error Rate** | 0% | <1% | >1% |
| **SLA** | Compliant | - | Violacao |

### 7.4 Metricas em Tempo Real (NOVO)

O sistema agora mostra metricas em tempo real:
- **TPS:** Transacoes processadas por segundo
- **Latencia p50/p95/p99:** Tempo de resposta (percentis)
- **Error Rate:** Percentual de erros
- **SLA Status:** Se os acordos de nivel de servico estao sendo cumpridos

---

## 8. Gerando Relatorios

### 8.1 Tipos de Relatorio

| Relatorio | Para Que | Tempo |
|-----------|----------|-------|
| **Mensal de Fraudes** | Resumo do mes para diretoria | 5-10 min |
| **Performance Trimestral** | Avaliacao de performance | 3-5 min |
| **Analise de Tendencias** | Identificar padroes | 7-12 min |
| **Impacto Financeiro** | Calcular economia | 4-8 min |

### 8.2 Como Gerar

1. Clique em **Relatorios** no menu
2. Escolha o template desejado
3. Configure o periodo
4. Clique em **Gerar Relatorio**
5. Aguarde a conclusao
6. Faca download do arquivo

---

## 9. Entendendo os Alertas

### 9.1 Tipos de Alerta

| Tipo | Icone | Descricao |
|------|-------|-----------|
| **Fraude Detectada** | Vermelho | Transacao bloqueada |
| **Revisao Necessaria** | Amarelo | Precisa analise humana |
| **Sistema** | Azul | Informacoes tecnicas |
| **SLA (NOVO)** | Laranja | Alerta de performance |

### 9.2 O Que Fazer com Cada Tipo

| Alerta | Acao |
|--------|------|
| Fraude Detectada | Verificar se e fraude real |
| Revisao Necessaria | Acessar fila de revisao |
| Sistema | Informar TI se persistir |
| SLA | Monitorar, informar TI se critico |

---

## 10. Perguntas Frequentes

### "O sistema bloqueou um cliente legitimo. O que faco?"

1. Acesse a Central de Investigacao
2. Encontre a transacao
3. Marque como "Falso Positivo"
4. O sistema aprendera com isso!

### "A latencia esta alta. O que significa?"

Significa que o sistema esta demorando para responder. Normalmente se resolve sozinho. Se persistir por mais de 30 minutos, avise a TI.

### "O que sao os 'fatores de risco' na explicacao?"

Sao os motivos que fizeram o sistema considerar a transacao suspeita. Por exemplo: "valor alto" ou "horario incomum".

### "Como sei se o sistema esta funcionando bem?"

Acesse a pagina de Monitoramento. Se o status for "Saudavel" (verde), esta tudo bem. O sistema agora monitora SLAs automaticamente.

---

## 11. Glossario

| Termo | Definicao |
|-------|-----------|
| **Threshold** | Limite de corte para decisao |
| **Score** | Pontuacao de risco (0-100) |
| **TPS** | Transacoes por segundo |
| **SLA** | Acordo de nivel de servico |
| **Latencia** | Tempo de resposta do sistema |
| **Falso Positivo** | Transacao legitima bloqueada por engano |
| **LGPD** | Lei de protecao de dados (exige explicacoes) |
| **Fator de Risco** | Motivo que aumenta suspeita |
| **Fator de Protecao** | Motivo que diminui suspeita |

---

## Contato e Suporte

Para duvidas ou problemas, entre em contato com a equipe de suporte.

---

*Manual do Usuario - Sankofa Enterprise Pro v12.0*  
*Ultima atualizacao: 27 de Novembro de 2025*
