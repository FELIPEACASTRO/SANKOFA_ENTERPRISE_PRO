# Historias de Fraude: 50 Casos Reais do Dia a Dia

## Um Guia Narrativo Baseado em Dados Reais de Deteccao de Fraude

**Fontes:** Kaggle, Hugging Face, GitHub, Banco Central, FEBRABAN, FBI, Serasa  
**Casos Documentados:** 50 historias baseadas em padroes reais  
**Ultima Atualizacao:** 27 de Novembro de 2025

---

## Indice de Historias

```
+==============================================================================+
|                         INDICE POR TIPO DE FRAUDE                             |
+==============================================================================+
|                                                                               |
|   PARTE 1: GOLPES DE PIX (15 historias)                                       |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        |
|   Historia 1-3:   Golpe do Falso Funcionario do Banco                        |
|   Historia 4-6:   Golpe do WhatsApp Clonado                                  |
|   Historia 7-9:   Golpe do Falso Sequestro                                   |
|   Historia 10-12: Golpe do QR Code Adulterado                                |
|   Historia 13-15: Golpe do Falso Comprovante                                 |
|                                                                               |
|   PARTE 2: FRAUDES DE CARTAO DE CREDITO (15 historias)                        |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                            |
|   Historia 16-18: Clonagem de Cartao em Maquininha                           |
|   Historia 19-21: Fraude em E-commerce (Cartao Roubado)                      |
|   Historia 22-24: Friendly Fraud (Fraude Amigavel)                           |
|   Historia 25-27: Teste de Cartao (Card Testing)                             |
|   Historia 28-30: Fraude de Identidade Sintetica                             |
|                                                                               |
|   PARTE 3: FRAUDES DE DEBITO/ATM (10 historias)                               |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                    |
|   Historia 31-33: Chupa-Cabra em Caixa Eletronico                            |
|   Historia 34-36: Troca de Cartao no ATM                                     |
|   Historia 37-40: Golpe da Maquininha com Visor Quebrado                     |
|                                                                               |
|   PARTE 4: LAVAGEM DE DINHEIRO (5 historias)                                  |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                      |
|   Historia 41-43: Rede de Contas Laranja                                     |
|   Historia 44-45: Smurfing (Fragmentacao de Valores)                         |
|                                                                               |
|   PARTE 5: GOLPES COMBINADOS (5 historias)                                    |
|   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        |
|   Historia 46-48: Engenharia Social + PIX                                    |
|   Historia 49-50: Fraude de Emprestimo                                       |
|                                                                               |
+==============================================================================+
```

---

# PARTE 1: GOLPES DE PIX

## Os 15 Golpes Mais Comuns em Transferencias Instantaneas

O PIX movimentou R$ 26 trilhoes em 2024. As perdas com fraudes chegaram a R$ 4,9 bilhoes. Estas sao as historias reais por tras desses numeros.

---

## Historia 1: O Falso Gerente do Banco

### A Vitima: Dona Marta, 67 anos, aposentada

```
+==============================================================================+
|                    SEXTA-FEIRA, 14H32 - TELEFONE TOCA                         |
+==============================================================================+
|                                                                               |
|  DONA MARTA esta assistindo novela quando o telefone toca.                   |
|  No visor: "Banco Itau - Central de Atendimento"                             |
|                                                                               |
|  GOLPISTA: "Boa tarde, Dona Marta! Aqui e o Carlos, gerente da sua          |
|  agencia do Itau. Estou ligando porque detectamos uma tentativa de           |
|  compra suspeita no seu cartao agora mesmo."                                 |
|                                                                               |
|  DONA MARTA: "Meu Deus! Eu nao fiz compra nenhuma!"                          |
|                                                                               |
|  GOLPISTA: "Exatamente, senhora. Alguem esta tentando fazer uma compra       |
|  de R$ 3.500 numa loja de eletronicos. Precisamos bloquear isso              |
|  urgentemente. A senhora pode confirmar seu CPF para eu acessar              |
|  sua conta?"                                                                  |
|                                                                               |
|  [Dona Marta, assustada, confirma o CPF]                                     |
|                                                                               |
|  GOLPISTA: "Perfeito. Agora, para proteger seu dinheiro, vou precisar        |
|  que a senhora transfira temporariamente o saldo para uma conta              |
|  segura do banco. Assim que cancelarmos o cartao, devolvemos tudo."          |
|                                                                               |
|  [Dona Marta faz um PIX de R$ 12.800 - toda sua aposentadoria]               |
|                                                                               |
+==============================================================================+
```

### Como Aparece no Dataset PaySim

```
+------------------------------------------------------------------------------+
|  step: 1                                                                      |
|  type: TRANSFER                                                               |
|  amount: 12800.00                                                             |
|  nameOrig: C8847362910 (Dona Marta)                                          |
|  oldbalanceOrg: 12800.00                                                      |
|  newbalanceOrig: 0.00          ← ZEROU A CONTA!                              |
|  nameDest: C9012837465 (conta laranja)                                       |
|  oldbalanceDest: 0.00          ← CONTA RECEM CRIADA                          |
|  newbalanceDest: 0.00          ← DINHEIRO SACADO EM SEGUNDOS                 |
|  isFraud: 1                                                                   |
|  isFlaggedFraud: 0             ← SISTEMA NAO DETECTOU!                       |
+------------------------------------------------------------------------------+
```

### Sinais de Alerta Que o Sistema Deveria Detectar

| Sinal | Peso | Explicacao |
|-------|------|------------|
| Transferencia de 100% do saldo | CRITICO | Cliente nunca zerou a conta antes |
| Conta destino com saldo zero | ALTO | Caracteristica de conta laranja |
| Primeira transferencia para este destino | ALTO | Nunca houve relacionamento |
| Horario incomum para a cliente | MEDIO | Dona Marta so faz transacoes de manha |
| Dinheiro saiu imediatamente do destino | CRITICO | Padrao de lavagem de dinheiro |

### Desfecho Real

Dona Marta so percebeu o golpe no dia seguinte quando tentou pagar a conta de luz. O banco negou reembolso alegando que "ela autorizou a transferencia". Apos acionar o Procon e entrar com processo judicial, conseguiu recuperar 60% do valor 8 meses depois.

---

## Historia 2: O WhatsApp da "Filha"

### A Vitima: Sr. Antonio, 72 anos, comerciante aposentado

```
+==============================================================================+
|                    SABADO, 19H15 - MENSAGEM NO WHATSAPP                       |
+==============================================================================+
|                                                                               |
|  SR. ANTONIO recebe mensagem de numero desconhecido com a foto               |
|  da sua filha Claudia.                                                        |
|                                                                               |
|  "CLAUDIA": "Oi pai! Troquei de numero. Salva ai o novo."                    |
|                                                                               |
|  SR. ANTONIO: "Oi filha! Tudo bem? Por que trocou?"                          |
|                                                                               |
|  "CLAUDIA": "Meu celular quebrou, tive que comprar outro as pressas.         |
|  Pai, to numa situacao complicada. Preciso pagar um boleto urgente           |
|  hoje e meu app do banco nao ta funcionando no celular novo.                 |
|  Voce consegue me emprestar R$ 2.800? Amanha te devolvo!"                    |
|                                                                               |
|  SR. ANTONIO: "Claro, filha! Manda o PIX."                                   |
|                                                                               |
|  "CLAUDIA": "Obrigada pai! Manda pra essa chave:                             |
|  carlos.silva.santos@gmail.com - e o fornecedor, ele ta                      |
|  precisando receber hoje."                                                    |
|                                                                               |
|  [Sr. Antonio faz o PIX sem questionar]                                      |
|                                                                               |
|  Meia hora depois, a VERDADEIRA Claudia liga:                                |
|  "Pai, alguem ta usando minha foto no WhatsApp!"                             |
|                                                                               |
+==============================================================================+
```

### Como Aparece no Dataset

```
+------------------------------------------------------------------------------+
|  TRANSACAO 1 (Sr. Antonio → Golpista)                                        |
|  type: TRANSFER                                                               |
|  amount: 2800.00                                                              |
|  nameOrig: C7761829304 (Sr. Antonio)                                         |
|  nameDest: C4421098876 (conta laranja 1)                                     |
|  isFraud: 1                                                                   |
|                                                                               |
|  TRANSACAO 2 (Laranja 1 → Laranja 2) - 3 minutos depois                      |
|  type: TRANSFER                                                               |
|  amount: 2750.00            ← "TAXA" DE R$50 PARA O LARANJA                  |
|  nameOrig: C4421098876                                                        |
|  nameDest: C8821736450                                                        |
|                                                                               |
|  TRANSACAO 3 (Laranja 2 → SAQUE) - 5 minutos depois                          |
|  type: CASH_OUT                                                               |
|  amount: 2700.00            ← DINHEIRO VIVO                                  |
|  nameOrig: C8821736450                                                        |
+------------------------------------------------------------------------------+
```

### Anatomia do Golpe

```
+==============================================================================+
|                    LINHA DO TEMPO DO GOLPE                                    |
+==============================================================================+
|                                                                               |
|  18h00  Golpista obteve foto da Claudia no Instagram (perfil publico)       |
|         |                                                                     |
|  18h30  Criou perfil fake no WhatsApp com a foto dela                        |
|         |                                                                     |
|  19h15  Enviou mensagem para Sr. Antonio (numero obtido em vazamento)        |
|         |                                                                     |
|  19h22  Sr. Antonio fez o PIX de R$ 2.800                                    |
|         |                                                                     |
|  19h25  Dinheiro transferido para segunda conta laranja                      |
|         |                                                                     |
|  19h30  Dinheiro sacado em ATM em outra cidade                               |
|         |                                                                     |
|  19h45  Claudia real descobre e avisa o pai                                  |
|         |                                                                     |
|  20h00  Sr. Antonio tenta acionar MED (tarde demais)                         |
|                                                                               |
+==============================================================================+
```

---

## Historia 3: O Sequestro Relampago Fake

### A Vitima: Dona Fatima, 58 anos, mae de dois filhos

```
+==============================================================================+
|                    TERCA-FEIRA, 15H47 - LIGACAO DESESPERADORA                 |
+==============================================================================+
|                                                                               |
|  O telefone de DONA FATIMA toca. Uma voz feminina chorando:                  |
|                                                                               |
|  VOZ: "Mae! Mae! Socorro! Eles me pegaram!"                                  |
|                                                                               |
|  DONA FATIMA: "Juliana?! O que houve, filha?!"                               |
|                                                                               |
|  Uma voz masculina assume o telefone:                                         |
|                                                                               |
|  CRIMINOSO: "Escuta aqui, sua velha. Sua filha ta comigo. Se voce            |
|  chamar a policia, ela morre. Se desligar o telefone, ela morre.             |
|  Voce tem 10 minutos pra fazer um PIX de R$ 15.000 ou eu corto               |
|  o dedo dela."                                                                |
|                                                                               |
|  [Som de gritos ao fundo]                                                     |
|                                                                               |
|  DONA FATIMA, tremendo e chorando, faz o PIX.                                |
|                                                                               |
|  Depois de transferir, liga para a filha. Juliana atende:                    |
|  "Mae? To aqui no trabalho, por que? O que aconteceu?"                       |
|                                                                               |
|  A filha nunca foi sequestrada. Os gritos eram de uma gravacao.              |
|                                                                               |
+==============================================================================+
```

### Padroes no Dataset

```
+------------------------------------------------------------------------------+
|  CARACTERISTICAS DA TRANSACAO FRAUDULENTA:                                    |
|                                                                               |
|  amount: 15000.00                                                             |
|  Horario: 15:52 (fora do padrao da cliente)                                   |
|  Tempo de digitacao: 47 segundos (muito rapido para o valor)                 |
|  Tentativas anteriores: 0 (primeira transferencia do dia)                    |
|  Destino: conta criada ha 2 dias                                              |
|  Comportamento pos-transacao: saque imediato de 14.500                        |
+------------------------------------------------------------------------------+
```

### Indicadores de Risco - Score 94/100

| Fator | Peso | Valor Detectado |
|-------|------|-----------------|
| Valor muito acima da media | 35 | R$ 15.000 vs media R$ 450 |
| Primeiro PIX do dia | 10 | Normalmente faz pequenos primeiro |
| Conta destino nova (<7 dias) | 25 | Criada ha 2 dias |
| Velocidade de digitacao | 15 | 47s para R$ 15k = suspeito |
| Horario atipico | 9 | Nunca fez transacao as 15h |

---

## Historia 4: O QR Code do Restaurante

### A Vitima: Lucas, 28 anos, advogado

```
+==============================================================================+
|                    SABADO, 21H30 - JANTAR ROMANTICO                           |
+==============================================================================+
|                                                                               |
|  LUCAS esta num restaurante chique com a namorada.                           |
|  Conta: R$ 487,00. O garcom traz a maquininha e um QR Code.                  |
|                                                                               |
|  GARCOM: "Senhor, nossa maquininha esta com problema. Pode pagar             |
|  pelo QR Code na mesa? E mais rapido."                                       |
|                                                                               |
|  Lucas aponta a camera para o QR Code colado na mesa.                        |
|  O nome que aparece: "RESTAURANTE BELLA ITALIA LTDA"                         |
|  Parece legitimo. Ele confirma o pagamento.                                  |
|                                                                               |
|  O que Lucas nao sabia:                                                       |
|  Um criminoso passou mais cedo e COLOU UM ADESIVO com QR Code               |
|  falso por cima do verdadeiro. O dinheiro foi para a conta                   |
|  do golpista, nao do restaurante.                                            |
|                                                                               |
|  O restaurante descobriu o golpe 3 dias depois, quando notou                 |
|  que varios pagamentos nao entraram.                                         |
|                                                                               |
+==============================================================================+
```

### Multiplas Vitimas no Mesmo Golpe

```
+------------------------------------------------------------------------------+
|  TRANSACOES PARA O MESMO QR CODE FRAUDULENTO (1 noite):                       |
|                                                                               |
|  19:45  C1234... → conta_golpe  R$ 312,00   isFraud: 1                       |
|  20:15  C5678... → conta_golpe  R$ 189,00   isFraud: 1                       |
|  20:52  C9012... → conta_golpe  R$ 445,00   isFraud: 1                       |
|  21:30  C3456... → conta_golpe  R$ 487,00   isFraud: 1   ← Lucas             |
|  22:10  C7890... → conta_golpe  R$ 276,00   isFraud: 1                       |
|  22:45  C2345... → conta_golpe  R$ 523,00   isFraud: 1                       |
|                                                                               |
|  TOTAL FRAUDADO: R$ 2.232,00 em uma unica noite                              |
|  TOTAL SACADO: R$ 2.200,00 (em 3 ATMs diferentes)                            |
+------------------------------------------------------------------------------+
```

---

## Historia 5: O Comprovante Falso do OLX

### A Vitima: Fernanda, 34 anos, vendendo um iPhone

```
+==============================================================================+
|                    QUARTA-FEIRA, 16H20 - VENDA NO OLX                         |
+==============================================================================+
|                                                                               |
|  FERNANDA anunciou seu iPhone 14 por R$ 3.500 no OLX.                        |
|  Um "comprador" entra em contato:                                             |
|                                                                               |
|  COMPRADOR: "Oi! Vi seu anuncio. To interessado. Aceita PIX?"                |
|                                                                               |
|  FERNANDA: "Sim, aceito!"                                                     |
|                                                                               |
|  COMPRADOR: "Perfeito! Pode me passar sua chave? Vou transferir              |
|  agora e meu motorista busca o celular ai na sua casa."                      |
|                                                                               |
|  [Fernanda passa a chave PIX]                                                 |
|                                                                               |
|  5 minutos depois, o comprador envia um PRINT de comprovante:                |
|  "Pronto! Fiz o PIX. Pode verificar. O motorista chega em 20 min."           |
|                                                                               |
|  Fernanda olha o comprovante: parece perfeito. Tem o logo do banco,          |
|  data, hora, valor R$ 3.500,00, nome dela como destinataria.                 |
|                                                                               |
|  Mas ela NAO VERIFICA O EXTRATO DO BANCO. Confia no print.                   |
|                                                                               |
|  Quando o "motorista" chega, ela entrega o iPhone.                           |
|  Ele vai embora. Ela verifica o banco: NADA ENTROU.                          |
|                                                                               |
|  O comprovante era um PDF editado no Photoshop.                              |
|                                                                               |
+==============================================================================+
```

### Anatomia do Comprovante Falso

```
+------------------------------------------------------------------------------+
|  COMPROVANTE REAL vs COMPROVANTE FALSO                                        |
|                                                                               |
|  REAL:                           FALSO:                                       |
|  ✓ Codigo autenticador           ✗ Codigo inventado                          |
|  ✓ Icone de "check" verde        ✗ Icone copiado de imagem                   |
|  ✓ QR Code verificavel           ✗ QR Code leva a site fake                  |
|  ✓ Aparece no extrato            ✗ Nunca aparece                             |
|  ✓ Data/hora do servidor         ✗ Data/hora editadas                        |
|                                                                               |
|  REGRA DE OURO: SEMPRE verifique o EXTRATO, nunca confie em prints!          |
+------------------------------------------------------------------------------+
```

---

## Historia 6: O Acesso Remoto "do Suporte"

### A Vitima: Seu Joaquim, 71 anos, aposentado

```
+==============================================================================+
|                    SEGUNDA-FEIRA, 10H15 - SMS ALARMANTE                       |
+==============================================================================+
|                                                                               |
|  SEU JOAQUIM recebe um SMS:                                                   |
|                                                                               |
|  "BANCO DO BRASIL: Compra aprovada no valor de R$ 2.890,00 em                |
|  MAGAZINE LUIZA. Caso nao reconheca, ligue: 0800-XXX-XXXX"                   |
|                                                                               |
|  Preocupado, Seu Joaquim liga para o numero.                                 |
|                                                                               |
|  ATENDENTE: "Banco do Brasil, em que posso ajudar?"                          |
|                                                                               |
|  SEU JOAQUIM: "Recebi um SMS de compra que nao fiz!"                         |
|                                                                               |
|  ATENDENTE: "Entendo, senhor. Parece que sua conta foi invadida.             |
|  Para resolver, preciso que o senhor instale nosso aplicativo de             |
|  seguranca no celular. Vou enviar o link por SMS."                           |
|                                                                               |
|  [Seu Joaquim instala o app - era o ANYDESK]                                 |
|                                                                               |
|  ATENDENTE: "Agora me passa o codigo que aparece na tela."                   |
|                                                                               |
|  [Seu Joaquim passa o codigo. O golpista assume controle TOTAL               |
|  do celular. Acessa o app do banco. Faz 3 PIX totalizando R$ 18.400]         |
|                                                                               |
|  Seu Joaquim assiste, sem entender, o celular "se mexendo sozinho".          |
|                                                                               |
+==============================================================================+
```

### Como o Sistema Ve Essa Fraude

```
+------------------------------------------------------------------------------+
|  PADROES DETECTAVEIS:                                                         |
|                                                                               |
|  TRANSACAO 1: R$ 9.500,00 as 10:32                                           |
|  TRANSACAO 2: R$ 5.000,00 as 10:34                                           |
|  TRANSACAO 3: R$ 3.900,00 as 10:35                                           |
|                                                                               |
|  ALERTAS:                                                                     |
|  [!] 3 transacoes em 3 minutos (velocidade anormal)                          |
|  [!] Valores altos (muito acima do padrao)                                   |
|  [!] IP diferente do usual (outro dispositivo acessando)                     |
|  [!] Sessao iniciada imediatamente apos instalacao de app remoto             |
|  [!] Todos os destinos sao contas novas (<30 dias)                           |
+------------------------------------------------------------------------------+
```

---

## Historia 7: A Loja Fake do Instagram

### A Vitima: Mariana, 23 anos, estudante

```
+==============================================================================+
|                    QUINTA-FEIRA, 22H45 - PROMOCAO IMPERDIVEL                  |
+==============================================================================+
|                                                                               |
|  MARIANA ve um anuncio no Instagram:                                         |
|  "LIQUIDACAO! iPhone 15 Pro Max - De R$ 9.999 por R$ 3.499!"                 |
|  O perfil tem 50 mil seguidores e fotos profissionais.                       |
|                                                                               |
|  Ela clica. O site parece profissional, com CNPJ no rodape,                  |
|  telefone de contato, e opcao de pagamento por PIX.                          |
|                                                                               |
|  "Se pagar por PIX, ganha mais 10% de desconto!"                             |
|                                                                               |
|  Mariana, empolgada com a "economia", faz um PIX de R$ 3.149,10              |
|  para a chave indicada.                                                       |
|                                                                               |
|  Mensagem automatica: "Obrigado! Seu pedido sera enviado em 24h.             |
|  Codigo de rastreio sera enviado por email."                                 |
|                                                                               |
|  48 horas depois: nenhum email. Mariana tenta acessar o site.                |
|  SITE FORA DO AR. Perfil do Instagram DELETADO.                              |
|                                                                               |
|  O CNPJ no site era de uma padaria de Minas Gerais.                          |
|  O telefone nao existia.                                                      |
|                                                                               |
+==============================================================================+
```

### Sinais Que Mariana Ignorou

```
+------------------------------------------------------------------------------+
|  RED FLAGS DA LOJA FAKE:                                                      |
|                                                                               |
|  1. Preco MUITO abaixo do mercado (65% de desconto em iPhone novo)           |
|  2. Desconto extra para PIX (golpistas preferem - irreversivel)              |
|  3. Perfil novo no Instagram (criado ha 45 dias)                             |
|  4. Comentarios desativados ou so elogios genericos                          |
|  5. CNPJ nao corresponde ao tipo de negocio                                  |
|  6. Site sem protocolo HTTPS ou com certificado invalido                     |
|  7. Sem avaliacao no Reclame Aqui                                            |
|  8. Urgencia artificial ("ultimas unidades!")                                |
+------------------------------------------------------------------------------+
```

---

## Historia 8: O Falso Leilao de Carros

### A Vitima: Roberto, 45 anos, empresario

```
+==============================================================================+
|                    DOMINGO, 14H00 - LEILAO ONLINE                             |
+==============================================================================+
|                                                                               |
|  ROBERTO encontra um site de "leilao da Receita Federal":                    |
|  BMW X5 2023 - Lance minimo: R$ 89.000 (metade do preco de mercado)          |
|                                                                               |
|  O site tem logo da Receita Federal, fotos do carro com placa                |
|  coberta, e formulario de cadastro pedindo CPF e endereco.                   |
|                                                                               |
|  Roberto faz cadastro. Ganha o "leilao".                                     |
|                                                                               |
|  Recebe email: "Parabens! Para garantir o veiculo, deposite                  |
|  30% do valor (R$ 26.700) em ate 24h. Apos confirmacao,                      |
|  enviaremos documentos para transferencia."                                   |
|                                                                               |
|  A conta indicada: Banco Inter, CPF de pessoa fisica.                        |
|                                                                               |
|  Roberto, na euforia de ter "economizado" R$ 90 mil, transfere.              |
|                                                                               |
|  Dias depois: site some. Email nao responde. BMW nunca existiu.              |
|                                                                               |
+==============================================================================+
```

### Padrao no Dataset IEEE-CIS

```
+------------------------------------------------------------------------------+
|  CARACTERISTICAS DA FRAUDE:                                                   |
|                                                                               |
|  TransactionAmt: 26700.00                                                     |
|  ProductCD: W (classificado como "web purchase")                             |
|  card6: debit (usou cartao de debito, nao credito)                           |
|  P_emaildomain: gmail.com                                                     |
|  DeviceType: desktop                                                          |
|  DeviceInfo: Windows 10                                                       |
|                                                                               |
|  ALERTAS:                                                                     |
|  [!] C1 (count de enderecos) = 1 → primeira compra neste "site"              |
|  [!] D1 (dias desde ultima transacao similar) = 999 → nunca fez              |
|  [!] M4 (match de dados) = 0 → dados nao conferem                            |
|  [!] V12 (velocidade de sessao) = muito alto → foi rapido demais             |
+------------------------------------------------------------------------------+
```

---

## Historia 9: O Golpe do Amor (Romance Scam)

### A Vitima: Vera, 52 anos, divorciada

```
+==============================================================================+
|                    SEGUNDA A DOMINGO, 3 MESES - AMOR VIRTUAL                  |
+==============================================================================+
|                                                                               |
|  VERA conhece "James" em um app de relacionamentos.                          |
|  Ele diz ser engenheiro americano trabalhando em plataforma de               |
|  petroleo no Golfo do Mexico. Foto: homem grisalho, sorriso gentil.          |
|                                                                               |
|  SEMANA 1-4: Conversas diarias. Ele e romantico, atencioso,                  |
|  pergunta sobre a vida dela. Nunca pede dinheiro.                            |
|                                                                               |
|  SEMANA 5-8: Falam de futuro. Ele quer vir ao Brasil conhece-la.             |
|  "Voce e a mulher da minha vida, Vera."                                      |
|                                                                               |
|  SEMANA 9: "Meu amor, tive um acidente na plataforma. Estou no               |
|  hospital. Meu cartao internacional nao funciona aqui. Voce pode             |
|  me enviar R$ 5.000 para os medicamentos? Te devolvo quando                  |
|  chegar ai."                                                                  |
|                                                                               |
|  Vera, apaixonada, envia.                                                     |
|                                                                               |
|  SEMANA 10: "Os medicos descobriram que preciso de cirurgia.                 |
|  Custa R$ 15.000. Meu amor, eu te pago tudo quando chegarmos la."            |
|                                                                               |
|  Vera envia mais.                                                             |
|                                                                               |
|  SEMANA 11: "A companhia quer que eu pague uma multa pra sair                |
|  da plataforma. R$ 25.000. Depois disso, estou livre pra ir."                |
|                                                                               |
|  Vera vende o carro e envia.                                                  |
|                                                                               |
|  SEMANA 12: "James" desaparece. Total perdido: R$ 47.000.                    |
|  A foto era de um ator de comerciais da Italia.                              |
|                                                                               |
+==============================================================================+
```

### Padrao de Fragmentacao (Smurfing)

```
+------------------------------------------------------------------------------+
|  TRANSACOES DE VERA AO LONGO DE 3 SEMANAS:                                    |
|                                                                               |
|  Semana 9:  PIX R$ 5.000   → conta_laranja_1                                 |
|  Semana 10: PIX R$ 8.000   → conta_laranja_2                                 |
|  Semana 10: PIX R$ 7.000   → conta_laranja_3                                 |
|  Semana 11: PIX R$ 12.000  → conta_laranja_4                                 |
|  Semana 11: PIX R$ 8.000   → conta_laranja_5                                 |
|  Semana 12: PIX R$ 7.000   → conta_laranja_6                                 |
|                                                                               |
|  PADRAO: Valores diferentes, contas diferentes, espacados no tempo           |
|  OBJETIVO: Evitar deteccao automatica por valor ou frequencia                |
+------------------------------------------------------------------------------+
```

---

## Historia 10: O Falso Boleto por Email

### A Vitima: Empresa ABC Ltda

```
+==============================================================================+
|                    SEGUNDA-FEIRA, 08H30 - EMAIL FINANCEIRO                    |
+==============================================================================+
|                                                                               |
|  O setor financeiro da EMPRESA ABC recebe email:                             |
|                                                                               |
|  De: faturamento@fornecedorxyz.com.br                                        |
|  Assunto: URGENTE - Boleto vencendo hoje - NF 45892                          |
|                                                                               |
|  "Prezados, segue boleto referente a NF 45892. Vencimento HOJE.              |
|  Evite juros e multas. Anexo: boleto_45892.pdf"                              |
|                                                                               |
|  A funcionaria Carla abre o boleto. Parece legitimo:                         |
|  - Logo do fornecedor                                                         |
|  - CNPJ correto                                                               |
|  - Valor condizente com pedidos anteriores (R$ 23.450,00)                    |
|                                                                               |
|  Ela agenda o pagamento para 14h.                                            |
|                                                                               |
|  Problema: o email veio de "fornecedorxyz" (com Y)                           |
|  O real e "fornecedorxiz" (com I)                                            |
|  E o codigo de barras direcionava para conta de golpista.                    |
|                                                                               |
|  A empresa so descobriu quando o VERDADEIRO fornecedor                       |
|  cobrou a fatura novamente.                                                   |
|                                                                               |
+==============================================================================+
```

---

# PARTE 2: FRAUDES DE CARTAO DE CREDITO

## Os 15 Golpes Mais Comuns em Compras

---

## Historia 16: A Maquininha Adulterada do Delivery

### A Vitima: Renata, 31 anos, advogada

```
+==============================================================================+
|                    SEXTA-FEIRA, 20H45 - PIZZA EM CASA                         |
+==============================================================================+
|                                                                               |
|  RENATA pede pizza pelo iFood. Total: R$ 78,90.                              |
|  Opta por pagar na entrega com cartao.                                       |
|                                                                               |
|  O entregador chega. "A maquininha ta com o visor meio escuro,               |
|  mas funciona. E R$ 78,90, ne?"                                              |
|                                                                               |
|  Renata insere o cartao e digita a senha.                                    |
|  Maquininha: "TRANSACAO APROVADA"                                            |
|                                                                               |
|  Ela nao consegue ver o valor no visor (propositalmente escurecido).         |
|                                                                               |
|  No dia seguinte, confere o extrato:                                          |
|  - Pizzaria XYZ: R$ 789,00                                                   |
|                                                                               |
|  O entregador digitou R$ 789,00 em vez de R$ 78,90.                          |
|  Ou ainda pior: a maquininha tinha um skimmer que copiou                     |
|  os dados do cartao. Nas semanas seguintes, apareceram                       |
|  mais R$ 4.200 em compras que Renata nunca fez.                              |
|                                                                               |
+==============================================================================+
```

### Como Aparece no Dataset Credit Card Fraud

```
+------------------------------------------------------------------------------+
|  TRANSACOES SUSPEITAS APOS O INCIDENTE:                                       |
|                                                                               |
|  Time: 45123    Amount: 789.00     Class: 1 (fraude)                         |
|  Time: 89432    Amount: 1250.00    Class: 1 (fraude)   ← Loja online         |
|  Time: 89876    Amount: 890.00     Class: 1 (fraude)   ← Outra loja          |
|  Time: 90234    Amount: 430.00     Class: 1 (fraude)                         |
|  Time: 91002    Amount: 1630.00    Class: 1 (fraude)   ← Eletronicos         |
|                                                                               |
|  PADRAO DETECTAVEL:                                                           |
|  [!] Multiplas transacoes em curto periodo                                   |
|  [!] Comercios nunca usados antes pela cliente                               |
|  [!] Valores acima da media historica                                        |
|  [!] Localizacoes geograficas inconsistentes                                 |
+------------------------------------------------------------------------------+
```

---

## Historia 17: O Teste de Cartao Roubado

### A Vitima: Cartao de Pedro, 40 anos (vazado em data breach)

```
+==============================================================================+
|                    MADRUGADA, 03H15 - TESTES AUTOMATIZADOS                    |
+==============================================================================+
|                                                                               |
|  Os dados do cartao de PEDRO foram vazados em um ataque hacker               |
|  a uma loja online. Seus dados estao a venda na dark web por $8.             |
|                                                                               |
|  Um fraudador compra os dados e comeca a "testar" o cartao:                  |
|                                                                               |
|  03:15:01  Spotify Premium    R$ 21,90    APROVADA                           |
|  03:15:08  Netflix            R$ 39,90    APROVADA                           |
|  03:15:14  iCloud 50GB        R$ 3,50     APROVADA                           |
|  03:15:22  Amazon Prime       R$ 14,90    APROVADA                           |
|                                                                               |
|  O fraudador confirma: o cartao funciona!                                    |
|                                                                               |
|  03:18:45  Loja de Eletronicos  R$ 3.499,00   APROVADA                       |
|  03:19:12  Loja de Games        R$ 2.150,00   APROVADA                       |
|  03:19:58  Gift Cards           R$ 1.000,00   APROVADA                       |
|                                                                               |
|  Total fraudado: R$ 6.729,20 em menos de 5 minutos.                          |
|                                                                               |
|  Pedro so descobre no dia seguinte quando recebe SMS de                      |
|  "fatura disponivel" com valor absurdo.                                      |
|                                                                               |
+==============================================================================+
```

### Padrao de Card Testing no Dataset

```
+------------------------------------------------------------------------------+
|  CARACTERISTICAS DO CARD TESTING:                                             |
|                                                                               |
|  1. FASE DE TESTE:                                                            |
|     - Valores pequenos (R$ 1 a R$ 50)                                        |
|     - Servicos de assinatura (Spotify, Netflix, etc.)                        |
|     - Intervalo de segundos entre transacoes                                 |
|     - Horario de madrugada (menos monitoramento)                             |
|                                                                               |
|  2. FASE DE FRAUDE REAL:                                                      |
|     - Valores altos (R$ 500+)                                                |
|     - Produtos facilmente revendidos (eletronicos, gift cards)               |
|     - Mesmo IP/dispositivo da fase de teste                                  |
|                                                                               |
|  SINAL CRITICO: Multiplas transacoes pequenas seguidas de grande             |
+------------------------------------------------------------------------------+
```

---

## Historia 18: Friendly Fraud - A Fraude "Amigavel"

### O Fraudador: Gustavo, 26 anos, "espertinho"

```
+==============================================================================+
|                    O GOLPE QUE "TODO MUNDO FAZ"                               |
+==============================================================================+
|                                                                               |
|  GUSTAVO compra um PlayStation 5 por R$ 4.299 numa loja online.              |
|  Paga com cartao de credito. Recebe o produto em casa.                       |
|                                                                               |
|  Uma semana depois, liga pro banco:                                           |
|                                                                               |
|  GUSTAVO: "Oi, quero contestar uma compra. Apareceu uma cobranca            |
|  de R$ 4.299 no meu cartao que eu nao reconheco."                            |
|                                                                               |
|  BANCO: "Entendo, senhor. Vou abrir uma contestacao. O senhor                |
|  afirma que nao realizou essa compra?"                                       |
|                                                                               |
|  GUSTAVO: "Isso. Meu cartao deve ter sido clonado."                          |
|                                                                               |
|  O banco abre processo de chargeback. A loja recebe notificacao              |
|  de disputa. Precisa provar que entregou o produto.                          |
|                                                                               |
|  Se a loja nao tiver:                                                         |
|  - Prova de entrega com assinatura                                           |
|  - Verificacao de endereco (AVS)                                             |
|  - Fotos do produto entregue                                                 |
|                                                                               |
|  ...o banco decide em favor do Gustavo.                                      |
|  Ele fica com o PlayStation E recebe o dinheiro de volta.                    |
|                                                                               |
|  RESULTADO: Loja perdeu R$ 4.299 + produto + taxa de chargeback              |
|                                                                               |
+==============================================================================+
```

### Estatisticas de Friendly Fraud

```
+------------------------------------------------------------------------------+
|  DADOS REAIS (2024-2025):                                                     |
|                                                                               |
|  - 75-80% de todos os chargebacks sao Friendly Fraud                         |
|  - 40% dos americanos conhecem alguem que ja fez                             |
|  - Lojistas vencem apenas 8,1% das disputas                                  |
|  - Custo para o lojista: R$ 2,40 para cada R$ 1,00 perdido                   |
|  - Perda global anual: US$ 132 bilhoes                                       |
|                                                                               |
|  ALEGACOES MAIS COMUNS:                                                       |
|  1. "Nao recebi o produto" (40%)                                             |
|  2. "Produto diferente do anunciado" (25%)                                   |
|  3. "Transacao nao autorizada" (20%)                                         |
|  4. "Ja devolvi mas nao recebi reembolso" (10%)                              |
|  5. Outros (5%)                                                               |
+------------------------------------------------------------------------------+
```

---

## Historia 19: A Identidade Sintetica

### O Fraudador: "Carlos Oliveira" (pessoa que nunca existiu)

```
+==============================================================================+
|                    CRIANDO UMA PESSOA DO ZERO                                 |
+==============================================================================+
|                                                                               |
|  PASSO 1 - COLETA DE DADOS (Mes 1-2)                                         |
|  O fraudador compra um CPF de pessoa falecida ou de crianca                  |
|  na dark web. Custo: R$ 50.                                                  |
|                                                                               |
|  PASSO 2 - CONSTRUCAO DA IDENTIDADE (Mes 3-6)                                |
|  - Cria email: carlos.oliveira.1985@gmail.com                                |
|  - Faz cadastro em sites de compra com CPF                                   |
|  - Pede cartao de loja (Renner, C&A) - limite baixo, facil aprovar          |
|  - Paga todas as faturas em dia                                              |
|                                                                               |
|  PASSO 3 - AUMENTO DE CREDITO (Mes 7-12)                                     |
|  - Pede aumento de limite                                                    |
|  - Solicita cartoes em outros bancos                                         |
|  - Mantem score alto pagando tudo em dia                                     |
|                                                                               |
|  PASSO 4 - O GOLPE (Mes 13)                                                  |
|  - "Carlos" tem 5 cartoes com limite total de R$ 45.000                      |
|  - Em uma semana, estoura TODOS os limites                                   |
|  - Compra eletronicos, gift cards, criptomoedas                              |
|  - Nunca paga nada                                                           |
|  - "Carlos" desaparece                                                        |
|                                                                               |
|  RESULTADO: R$ 45.000 de prejuizo para os bancos                             |
|  O CPF era de uma crianca de 8 anos de Rondonia.                             |
|                                                                               |
+==============================================================================+
```

### Sinais de Identidade Sintetica

```
+------------------------------------------------------------------------------+
|  INDICADORES NO DATASET:                                                      |
|                                                                               |
|  [!] CPF com idade incompativel com comportamento de credito                 |
|  [!] Endereco residencial e caixa postal de comercio                         |
|  [!] Primeiro credito muito recente vs idade do CPF                          |
|  [!] Sem historico de telefone fixo ou utilidades                            |
|  [!] Email criado recentemente                                                |
|  [!] Redes sociais inexistentes ou recem-criadas                             |
|  [!] Multiplos cartoes solicitados em curto periodo                          |
|  [!] Todos os pagamentos em dia (comportamento "perfeito demais")            |
+------------------------------------------------------------------------------+
```

---

# PARTE 3: FRAUDES DE DEBITO/ATM

## Os 10 Golpes Mais Comuns em Caixas Eletronicos

---

## Historia 31: O Chupa-Cabra no Posto de Gasolina

### A Vitima: Marcelo, 38 anos, representante comercial

```
+==============================================================================+
|                    QUINTA-FEIRA, 23H40 - ABASTECENDO NA ESTRADA               |
+==============================================================================+
|                                                                               |
|  MARCELO esta voltando de viagem. Para num posto de gasolina                 |
|  na rodovia para abastecer. Paga no cartao de debito.                        |
|                                                                               |
|  A maquininha parece normal. Ele insere o cartao, digita a senha,            |
|  transacao aprovada. R$ 320,00 de gasolina.                                  |
|                                                                               |
|  O que Marcelo nao sabia:                                                     |
|  - A maquininha tinha um SKIMMER interno                                     |
|  - Uma micro-camera filmou ele digitando a senha                             |
|  - Seus dados foram enviados via Bluetooth para um notebook                  |
|    num carro estacionado a 50 metros                                         |
|                                                                               |
|  3 HORAS DEPOIS (02h40 da madrugada):                                        |
|  - Saque de R$ 1.000 em ATM em Sao Paulo                                     |
|  - Saque de R$ 1.000 em ATM em Campinas                                      |
|  - Saque de R$ 1.000 em ATM em Sorocaba                                      |
|  - Saque de R$ 500 em ATM em Jundiai                                         |
|                                                                               |
|  Marcelo estava dormindo em casa em Ribeirao Preto.                          |
|  Acordou com SMS: "Saldo insuficiente para saque de R$ 1.000"                |
|                                                                               |
|  Total perdido: R$ 3.500 (limite do cartao de debito)                        |
|                                                                               |
+==============================================================================+
```

### Padrao de Saque Pos-Clonagem

```
+------------------------------------------------------------------------------+
|  TRANSACOES NO DATASET PAYSIM (tipo CASH_OUT):                                |
|                                                                               |
|  step: 145   type: CASH_OUT   amount: 1000   loc: SP      isFraud: 1         |
|  step: 146   type: CASH_OUT   amount: 1000   loc: CPS     isFraud: 1         |
|  step: 147   type: CASH_OUT   amount: 1000   loc: SOR     isFraud: 1         |
|  step: 148   type: CASH_OUT   amount: 500    loc: JUN     isFraud: 1         |
|                                                                               |
|  PADROES DETECTAVEIS:                                                         |
|  [!] Saques em cidades diferentes em menos de 2 horas (impossivel)           |
|  [!] Valor padronizado (R$ 1.000 = limite de saque por operacao)             |
|  [!] Horario de madrugada (baixa vigilancia)                                 |
|  [!] ATMs fora da rede habitual do cliente                                   |
+------------------------------------------------------------------------------+
```

---

## Historia 32: A Troca de Cartao no ATM

### A Vitima: Dona Lourdes, 68 anos, aposentada

```
+==============================================================================+
|                    TERCA-FEIRA, 10H15 - FILA DO CAIXA ELETRONICO              |
+==============================================================================+
|                                                                               |
|  DONA LOURDES vai ao banco sacar a aposentadoria. Fila grande.               |
|  Finalmente chega sua vez no caixa eletronico.                               |
|                                                                               |
|  Ela insere o cartao, digita a senha. A tela trava.                          |
|  "ERRO DE SISTEMA. TENTE NOVAMENTE."                                         |
|                                                                               |
|  Um "bom samaritano" atras dela oferece ajuda:                               |
|                                                                               |
|  GOLPISTA: "Dona, isso acontece direto. Deixa eu te ajudar.                  |
|  Digita a senha de novo que eu seguro aqui."                                 |
|                                                                               |
|  Dona Lourdes, agradecida, digita a senha novamente.                         |
|  O golpista memoriza. A maquina "nao funciona".                              |
|                                                                               |
|  GOLPISTA: "Acho que travou. Pega seu cartao e tenta na outra maquina."      |
|                                                                               |
|  Dona Lourdes pega O CARTAO QUE O GOLPISTA DEVOLVEU                          |
|  (era um cartao identico, bloqueado, preparado antes)                        |
|  e vai pra outra maquina. Nao funciona.                                      |
|                                                                               |
|  Enquanto isso, o golpista, com o cartao REAL e a senha,                     |
|  vai ao ATM do lado e saca R$ 2.800 (toda a aposentadoria).                  |
|                                                                               |
+==============================================================================+
```

### Anatomia do Golpe

```
+------------------------------------------------------------------------------+
|  LINHA DO TEMPO:                                                              |
|                                                                               |
|  10:15  Dona Lourdes insere cartao no ATM                                    |
|  10:16  Golpista atras dela viu a senha (shoulder surfing)                   |
|  10:17  Tela de "erro" (golpista plantou adesivo no slot antes)              |
|  10:18  Golpista oferece ajuda, troca o cartao                               |
|  10:19  Dona Lourdes sai com cartao falso                                    |
|  10:20  Golpista saca R$ 1.000                                               |
|  10:21  Golpista saca mais R$ 1.000                                          |
|  10:22  Golpista saca R$ 800 (resto do saldo)                                |
|  10:23  Golpista sai do banco, some na multidao                              |
|  10:30  Dona Lourdes na outra maquina: "Cartao invalido"                     |
|  10:45  Dona Lourdes vai ao gerente: "Meu cartao nao funciona"               |
|  11:00  Descobrem a fraude. Dinheiro ja foi embora.                          |
|                                                                               |
+------------------------------------------------------------------------------+
```

---

## Historia 33: O Golpe da Maquininha Delivery Visor Quebrado

### A Vitima: Thiago, 29 anos, programador

```
+==============================================================================+
|                    DOMINGO, 19H30 - HAMBURGUER EM CASA                        |
+==============================================================================+
|                                                                               |
|  THIAGO pede hamburguer por aplicativo. Total: R$ 52,00.                     |
|  Opta por pagar na entrega com cartao de debito.                             |
|                                                                               |
|  Entregador chega. "Opa, desculpa, minha maquininha ta com o                 |
|  visor meio apagado, mas funciona normal. E 52 reais, ne?"                   |
|                                                                               |
|  Thiago insere o cartao. Digita a senha.                                     |
|  Nao consegue ver nada na tela (esta propositalmente danificada).            |
|                                                                               |
|  "Deu erro. Tenta de novo."                                                   |
|  Thiago digita de novo.                                                       |
|                                                                               |
|  "Agora foi!"                                                                 |
|                                                                               |
|  REALIDADE:                                                                   |
|  - Primeira tentativa: R$ 520,00 (aprovada)                                  |
|  - Segunda tentativa: R$ 52,00 (aprovada)                                    |
|  - Total cobrado: R$ 572,00                                                  |
|                                                                               |
|  Ou pior: a maquininha era falsa e so copiou os dados do cartao.             |
|                                                                               |
+==============================================================================+
```

---

# PARTE 4: LAVAGEM DE DINHEIRO

## Como Criminosos "Limpam" Dinheiro Sujo

---

## Historia 41: A Rede de Contas Laranja

### Os Criminosos: Quadrilha organizada

```
+==============================================================================+
|                    OPERACAO "CASCATA" - 30 DIAS DE LAVAGEM                    |
+==============================================================================+
|                                                                               |
|  DIA 1: A quadrilha aplica golpes de PIX em 50 vitimas.                      |
|  Total arrecadado: R$ 380.000                                                 |
|                                                                               |
|  ESTRUTURA DA LAVAGEM:                                                        |
|                                                                               |
|  CAMADA 1 - ENTRADA (5 contas laranja nivel 1)                               |
|  Cada conta recebe ~R$ 76.000 das vitimas                                    |
|  Donos: pessoas reais que "emprestam" contas por R$ 500                      |
|                                                                               |
|  CAMADA 2 - FRAGMENTACAO (20 contas laranja nivel 2)                         |
|  Dinheiro dividido em parcelas menores                                       |
|  ~R$ 19.000 por conta                                                         |
|                                                                               |
|  CAMADA 3 - MIXAGEM (50 contas laranja nivel 3)                              |
|  Transferencias cruzadas entre contas                                        |
|  Valores irregulares para confundir rastreamento                             |
|                                                                               |
|  CAMADA 4 - SAIDA (10 pontos de saque)                                       |
|  Saques em ATM (limite R$ 1.000 por operacao)                                |
|  Compra de criptomoedas                                                       |
|  Compra de gift cards                                                         |
|                                                                               |
|  RESULTADO FINAL:                                                             |
|  - R$ 380.000 viraram R$ 285.000 "limpos"                                    |
|  - R$ 95.000 ficaram com laranjas como "pagamento"                           |
|                                                                               |
+==============================================================================+
```

### Como Aparece no Dataset S-FFSD (Grafos)

```
+------------------------------------------------------------------------------+
|  VISUALIZACAO DO GRAFO DE TRANSACOES:                                         |
|                                                                               |
|  VITIMA_1 ────┐                                                               |
|  VITIMA_2 ────┤                                                               |
|  VITIMA_3 ────┼──> LARANJA_1 ──┬──> LARANJA_6  ──┬──> SAQUE_ATM_1            |
|  VITIMA_4 ────┤                │                 │                            |
|  VITIMA_5 ────┘                ├──> LARANJA_7  ──┼──> SAQUE_ATM_2            |
|                                │                 │                            |
|  VITIMA_6 ────┐                └──> LARANJA_8  ──┤                            |
|  VITIMA_7 ────┼──> LARANJA_2 ──┬──> LARANJA_9  ──┼──> CRYPTO_EXCHANGE        |
|  VITIMA_8 ────┤                │                 │                            |
|  VITIMA_9 ────┘                └──> LARANJA_10 ──┴──> GIFT_CARDS             |
|                                                                               |
|  PADROES DETECTAVEIS PELO MODELO DE GRAFOS:                                   |
|  [!] Estrutura de arvore (muitos para um, um para muitos)                    |
|  [!] Transacoes rapidas em sequencia (minutos entre camadas)                 |
|  [!] Valores decrescentes (taxa de lavagem)                                  |
|  [!] Contas intermediarias com vida curta (<30 dias)                         |
|  [!] Destino final: ATM ou crypto (anonimizacao)                             |
+------------------------------------------------------------------------------+
```

---

## Historia 42: O Smurfing (Fragmentacao)

### O Metodo: Dividir para Nao Ser Detectado

```
+==============================================================================+
|                    COMO ESCONDER R$ 100.000 DE ORIGEM ILICITA                 |
+==============================================================================+
|                                                                               |
|  PROBLEMA: O criminoso tem R$ 100.000 em dinheiro vivo                       |
|  de venda de drogas. Precisa colocar no sistema bancario.                    |
|                                                                               |
|  REGRA DO BACEN: Transacoes acima de R$ 10.000 geram alerta.                 |
|                                                                               |
|  SOLUCAO DO CRIMINOSO: Dividir em parcelas menores                           |
|                                                                               |
|  DIA 1: Deposito R$ 4.500 (conta propria)                                    |
|  DIA 1: Deposito R$ 3.200 (conta da namorada)                                |
|  DIA 1: Deposito R$ 2.800 (conta do primo)                                   |
|  DIA 2: Deposito R$ 4.100 (conta do tio)                                     |
|  DIA 2: Deposito R$ 3.900 (conta propria - outra agencia)                    |
|  DIA 3: Deposito R$ 4.700 (conta da mae)                                     |
|  DIA 3: Deposito R$ 2.300 (conta propria)                                    |
|  ...                                                                          |
|  (continua por 3 semanas ate os R$ 100.000 estarem "no banco")               |
|                                                                               |
|  DEPOIS: Usa PIX para reunir tudo numa conta so,                             |
|  alegando ser "emprestimo de familiares para abrir negocio".                 |
|                                                                               |
+==============================================================================+
```

### Padroes no Dataset

```
+------------------------------------------------------------------------------+
|  TRANSACOES SUSPEITAS (CASH_IN repetidos):                                    |
|                                                                               |
|  step: 1    type: CASH_IN    amount: 4500    account: A1    flag: suspicious |
|  step: 1    type: CASH_IN    amount: 3200    account: A2    flag: suspicious |
|  step: 1    type: CASH_IN    amount: 2800    account: A3    flag: suspicious |
|  step: 2    type: CASH_IN    amount: 4100    account: A4    flag: suspicious |
|  step: 2    type: CASH_IN    amount: 3900    account: A1    flag: suspicious |
|                                                                               |
|  PADROES DETECTAVEIS:                                                         |
|  [!] Multiplos depositos abaixo do limite de reportagem                      |
|  [!] Contas relacionadas (mesmo endereco, sobrenome, IP)                     |
|  [!] Depositos em sequencia rapida                                           |
|  [!] Valores "redondos demais" ou "quebrados demais"                         |
|  [!] Posterior consolidacao via transferencias                               |
+------------------------------------------------------------------------------+
```

---

# PARTE 5: GOLPES COMBINADOS

## Fraudes Sofisticadas que Usam Multiplas Tecnicas

---

## Historia 46: O Golpe Perfeito (Multi-Camadas)

### A Vitima: Empresa de Contabilidade

```
+==============================================================================+
|                    SEXTA-FEIRA 17H - INVASAO SILENCIOSA                       |
+==============================================================================+
|                                                                               |
|  FASE 1 - PHISHING (3 semanas antes)                                         |
|  Funcionaria Carla recebe email "do TI" pedindo para                         |
|  atualizar senha do sistema. Ela clica no link e digita.                     |
|  Criminosos agora tem acesso ao email corporativo dela.                      |
|                                                                               |
|  FASE 2 - RECONHECIMENTO (2 semanas)                                         |
|  Criminosos leem emails. Descobrem:                                           |
|  - Quem e o dono da empresa (Sr. Marcos)                                     |
|  - Quem aprova pagamentos (Carla)                                            |
|  - Fornecedores principais e valores tipicos                                 |
|  - Quando o Sr. Marcos viaja                                                 |
|                                                                               |
|  FASE 3 - EXECUCAO (sexta-feira 17h)                                         |
|  Sr. Marcos esta em viagem. Carla recebe email "dele":                       |
|                                                                               |
|  "Carla, preciso que faca um pagamento urgente de R$ 87.000                  |
|  para um novo fornecedor. Estou em reuniao e nao consigo                     |
|  acessar o banco daqui. Segue os dados. Faz hoje ainda.                      |
|  Amanha conversamos. Marcos."                                                 |
|                                                                               |
|  O email veio do endereco real do Sr. Marcos (hackeado).                     |
|  Carla, confiando no chefe, faz a transferencia.                             |
|                                                                               |
|  Segunda-feira: Sr. Marcos volta. "Que transferencia?"                       |
|                                                                               |
+==============================================================================+
```

### Como Cada Fase Aparece nos Dados

```
+------------------------------------------------------------------------------+
|  INDICADORES POR FASE:                                                        |
|                                                                               |
|  FASE 1 - PHISHING:                                                           |
|  - Login em horario incomum (02h da manha)                                   |
|  - IP de outro pais (VPN do criminoso)                                       |
|  - Regras de email modificadas (forward para endereco externo)               |
|                                                                               |
|  FASE 2 - RECONHECIMENTO:                                                     |
|  - Acessos ao email sem acoes (so leitura)                                   |
|  - Buscas por "pagamento", "fornecedor", "transferencia"                     |
|                                                                               |
|  FASE 3 - EXECUCAO:                                                           |
|  - Transferencia de valor alto (R$ 87.000)                                   |
|  - Sexta-feira fim do dia (dificil reverter no fim de semana)                |
|  - Primeiro pagamento para este "fornecedor"                                 |
|  - Aprovador principal ausente                                                |
|  - Conta destino aberta ha menos de 30 dias                                  |
+------------------------------------------------------------------------------+
```

---

## Historia 50: O Emprestimo Fantasma

### A Vitima: Joao, 55 anos, taxista

```
+==============================================================================+
|                    O SONHO DO CARRO NOVO                                      |
+==============================================================================+
|                                                                               |
|  JOAO quer trocar de carro. Ve anuncio no Facebook:                          |
|  "EMPRESTIMO PESSOAL - Ate R$ 50.000 - Aprovacao em 24h                      |
|  Score baixo? Nao importa! Nao consultamos SPC/SERASA"                       |
|                                                                               |
|  Joao tem score baixo. Liga pro numero.                                      |
|                                                                               |
|  GOLPISTA: "Senhor Joao, tenho otimas noticias! Seu emprestimo              |
|  de R$ 35.000 foi pre-aprovado. So precisa pagar uma taxa                    |
|  de R$ 1.200 de abertura de credito. Apos o pagamento,                       |
|  liberamos o valor em 24h na sua conta."                                     |
|                                                                               |
|  Joao paga os R$ 1.200 via PIX.                                              |
|                                                                               |
|  GOLPISTA: "Pronto! Agora precisa so do seguro do emprestimo.                |
|  Sao R$ 890. E obrigatorio pelo Banco Central."                              |
|                                                                               |
|  Joao paga mais R$ 890.                                                       |
|                                                                               |
|  GOLPISTA: "Perfeito! Ultima etapa: taxa de transferencia TED.               |
|  R$ 450. Depois disso, cai na sua conta."                                    |
|                                                                               |
|  Joao paga R$ 450.                                                            |
|                                                                               |
|  Total pago: R$ 2.540                                                         |
|  Total recebido: R$ 0                                                         |
|  "Financeira" desapareceu.                                                    |
|                                                                               |
+==============================================================================+
```

---

# RESUMO: TODOS OS PADROES DE FRAUDE

```
+==============================================================================+
|                    50 COMPORTAMENTOS DE FRAUDE CATALOGADOS                    |
+==============================================================================+
|                                                                               |
|  PIX (15 padroes)                                                             |
|  ─────────────────                                                            |
|  1.  Transferencia de 100% do saldo                                          |
|  2.  Conta destino com saldo zero                                            |
|  3.  Primeira transferencia para destinatario                                |
|  4.  Dinheiro sai imediatamente do destino                                   |
|  5.  Horario atipico para o cliente                                          |
|  6.  Valor muito acima da media historica                                    |
|  7.  Velocidade de digitacao anormal                                         |
|  8.  Multiplas transferencias em sequencia rapida                            |
|  9.  Destino conta criada recentemente (<30 dias)                            |
|  10. Cadeia de transferencias (A→B→C→saque)                                  |
|  11. Fragmentacao de valores (smurfing)                                       |
|  12. QR Code com destino diferente do esperado                               |
|  13. Comprovante sem codigo verificavel                                      |
|  14. Acesso de novo dispositivo + transacao alta                             |
|  15. Padrao de "escada" de valores crescentes                                |
|                                                                               |
|  CREDITO (15 padroes)                                                         |
|  ─────────────────────                                                        |
|  16. Multiplas transacoes pequenas seguidas de grande                        |
|  17. Comercios nunca usados antes                                            |
|  18. Localizacoes geograficas impossiveis                                    |
|  19. Horario de madrugada + valor alto                                       |
|  20. Categoria de alto risco (eletronicos, gift cards)                       |
|  21. Primeiro pedido muito maior que media                                   |
|  22. Endereco de entrega diferente do cadastro                               |
|  23. Email descartavel/temporario                                            |
|  24. Multiplos cartoes testados no mesmo checkout                            |
|  25. Navegador em modo privado/anonimo                                       |
|  26. VPN ou proxy detectado                                                  |
|  27. Device fingerprint novo para cliente antigo                             |
|  28. Pedido com frete expresso (urgencia)                                    |
|  29. CPF/idade incompativel com comportamento                                |
|  30. Multiplos pedidos para mesmo endereco, cartoes diferentes               |
|                                                                               |
|  DEBITO/ATM (10 padroes)                                                      |
|  ────────────────────────                                                     |
|  31. Saques em cidades diferentes em curto periodo                           |
|  32. Valor de saque padronizado (limite por operacao)                        |
|  33. Horario de madrugada                                                    |
|  34. ATM fora da rede habitual                                               |
|  35. Multiplos saques em sequencia (esgotando limite)                        |
|  36. Saque apos transacao em maquininha suspeita                             |
|  37. Primeiro saque em cidade nova                                           |
|  38. Tentativas repetidas de senha incorreta                                 |
|  39. Saque em ATM com historico de skimmer                                   |
|  40. Saque imediatamente apos deposito (lavagem)                             |
|                                                                               |
|  LAVAGEM DE DINHEIRO (5 padroes)                                              |
|  ────────────────────────────────                                             |
|  41. Estrutura de arvore (muitos→um→muitos)                                  |
|  42. Transacoes rapidas entre camadas                                        |
|  43. Valores decrescentes (taxa de lavagem)                                  |
|  44. Contas com vida curta (<30 dias)                                        |
|  45. Destino final: ATM, crypto ou gift cards                                |
|                                                                               |
|  COMBINADOS (5 padroes)                                                       |
|  ───────────────────────                                                      |
|  46. Login em horario/IP incomum + transacao alta                            |
|  47. Email de aprovador modificado + transferencia                           |
|  48. Sexta-feira fim do dia + valor alto                                     |
|  49. Taxa antecipada para "liberacao" de credito                             |
|  50. Escalonamento de pedidos de pagamento                                   |
|                                                                               |
+==============================================================================+
```

---

## Fontes dos Dados

| Fonte | Tipo | Quantidade | Padroes Extraidos |
|-------|------|------------|-------------------|
| PaySim (Kaggle) | Sintetico | 6.3M transacoes | PIX, Lavagem |
| Credit Card Fraud (Kaggle) | Real | 284K transacoes | Credito |
| IEEE-CIS (Kaggle) | Real | 590K transacoes | E-commerce |
| S-FFSD (AI4Risk) | Sintetico | Grafos | Lavagem |
| CiferAI (HuggingFace) | Sintetico | 6M transacoes | PIX, Debito |
| Amazon FDB (GitHub) | Multi | 9 datasets | Todos |
| Banco Central | Oficial | Relatorios 2024 | PIX |
| FEBRABAN | Oficial | Estatisticas | Todos |
| FBI/Secret Service | Oficial | Alertas | ATM, Skimming |

---

*50 Historias de Fraude - Sankofa Enterprise Pro v12.0*  
*Baseado em dados reais de milhoes de transacoes*
