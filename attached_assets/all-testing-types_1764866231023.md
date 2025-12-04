# Enciclopédia Completa de Tipos de Testes de Software

## Sumário Executivo

Este documento consolida **todos os tipos de testes existentes na literatura de QA e Tecnologia da Informação**. Após pesquisa exaustiva em fontes acadêmicas (IEEE, ACM, SEI/CMU), glossários oficiais (ISTQB), e literatura especializada, foram catalogados **mais de 600 tipos de testes** organizados em 35 categorias principais.

A taxonomia do SEI/CMU identifica aproximadamente **200 tipos de testes abstratos** organizados em 7 categorias baseadas nas perguntas 5W+2H (What, When, Why, Who, Where, How, How Well).[481][560][557]

---

## PARTE 1: TESTES FUNCIONAIS

### 1.1. Testes de Unidade e Componente

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 1 | **Unit Testing** | Testa unidades individuais de código (funções, métodos, classes) em isolamento.[473][474][477][493][497] |
| 2 | **Component Testing** | Sinônimo de Unit Testing; testa componentes individuais.[474][477][493] |
| 3 | **Module Testing** | Sinônimo de Unit Testing; testa módulos individuais.[493][497] |
| 4 | **Function Testing** | Testa funções específicas do código.[493] |
| 5 | **Class Testing** | Testa classes individuais em POO.[477] |
| 6 | **Method Testing** | Testa métodos individuais.[477] |

### 1.2. Testes de Integração

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 7 | **Integration Testing** | Testa a integração entre módulos/componentes.[473][474][477][493][497] |
| 8 | **Big Bang Integration Testing** | Integra todos os módulos de uma vez e testa.[473][477][493][497] |
| 9 | **Top-Down Integration Testing** | Integra de cima para baixo usando Stubs.[473][477][493][497][592] |
| 10 | **Bottom-Up Integration Testing** | Integra de baixo para cima usando Drivers.[473][477][493][497][592] |
| 11 | **Hybrid/Sandwich Integration Testing** | Combina Top-Down e Bottom-Up.[473][477][493][497][592] |
| 12 | **Incremental Integration Testing** | Integra módulos incrementalmente.[477][493][592] |
| 13 | **Component Integration Testing** | Testa integração entre componentes específicos.[474] |
| 14 | **System Integration Testing** | Testa integração de subsistemas.[474][560] |
| 15 | **Hardware-Software Integration Testing** | Integração entre hardware e software.[546][560] |
| 16 | **Outside-In Integration Testing** | Começa das interfaces externas para o centro.[560] |
| 17 | **Inside-Out Integration Testing** | Começa do centro para as interfaces.[560] |
| 18 | **Backbone Integration Testing** | Integra usando estrutura backbone.[560] |
| 19 | **Layer Integration Testing** | Integra por camadas arquiteturais.[560] |
| 20 | **Thread Integration Testing** | Integra funcionalidades thread-based.[560] |

### 1.3. Testes de Sistema

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 21 | **System Testing** | Testa o sistema completo e integrado.[473][474][477][493][497] |
| 22 | **End-to-End Testing (E2E)** | Testa fluxos completos do início ao fim.[473][474][493][497][547] |
| 23 | **Subsystem Testing** | Testa subsistemas individuais.[560] |
| 24 | **System of Systems (SoS) Testing** | Testa sistemas de sistemas.[560] |
| 25 | **Full System Testing** | Teste completo do sistema.[560] |

### 1.4. Testes de Aceitação

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 26 | **Acceptance Testing** | Valida se o software atende requisitos de negócio.[473][474][477][493][497] |
| 27 | **User Acceptance Testing (UAT)** | Testes feitos por usuários finais.[473][474][477][493][497][594] |
| 28 | **Alpha Testing** | Testes internos antes do release.[473][474][477][493][497] |
| 29 | **Beta Testing** | Testes por usuários externos limitados.[473][474][477][493][497] |
| 30 | **Gamma Testing** | Testes finais antes do lançamento.[493][497] |
| 31 | **Operational Acceptance Testing (OAT)** | Valida requisitos operacionais.[474][478][546] |
| 32 | **Contract Acceptance Testing** | Valida conformidade contratual.[546] |
| 33 | **Regulatory Acceptance Testing** | Valida conformidade regulatória.[478] |
| 34 | **Factory Acceptance Testing (FAT)** | Testes de aceitação em fábrica.[546] |
| 35 | **Site Acceptance Testing (SAT)** | Testes de aceitação no local de instalação.[546] |
| 36 | **Business Acceptance Testing (BAT)** | Valida requisitos de negócio.[480] |

### 1.5. Testes de Verificação de Build

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 37 | **Smoke Testing** | Verifica funcionalidades básicas após build.[473][474][477][493][497][592][595] |
| 38 | **Build Verification Testing (BVT)** | Sinônimo de Smoke Testing.[493][592] |
| 39 | **Sanity Testing** | Verifica funcionalidades específicas após correções.[473][474][477][493][497][592][595][597] |
| 40 | **Health Check Testing** | Verifica saúde básica do sistema.[497] |
| 41 | **Confidence Testing** | Garante confiança básica no build.[497] |

### 1.6. Testes de Regressão

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 42 | **Regression Testing** | Verifica que mudanças não quebraram funcionalidades existentes.[473][474][477][493][497][592] |
| 43 | **Re-testing** | Re-testa defeitos corrigidos.[493][497] |
| 44 | **Confirmation Testing** | Confirma correção de defeitos.[493][497] |
| 45 | **Full Regression Testing** | Regressão completa do sistema.[497] |
| 46 | **Partial Regression Testing** | Regressão em áreas afetadas.[497] |
| 47 | **Unit Regression Testing** | Regressão em nível de unidade.[497] |
| 48 | **Progressive Regression Testing** | Regressão com novas funcionalidades.[497] |
| 49 | **Selective Regression Testing** | Regressão seletiva baseada em risco.[497] |
| 50 | **Complete Regression Testing** | Regressão total do sistema.[497] |

---

## PARTE 2: TESTES NÃO-FUNCIONAIS

### 2.1. Testes de Performance

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 51 | **Performance Testing** | Avalia velocidade, escalabilidade, estabilidade.[473][474][477][493][497] |
| 52 | **Load Testing** | Testa comportamento sob carga esperada.[473][474][477][493][497][559] |
| 53 | **Stress Testing** | Testa comportamento além da capacidade.[473][474][477][493][497][559] |
| 54 | **Spike Testing** | Testa picos súbitos de carga.[493][497][547] |
| 55 | **Volume Testing** | Testa com grandes volumes de dados.[473][474][477][493][497][601] |
| 56 | **Soak Testing** | Testa carga sustentada por longo período.[493][497][601] |
| 57 | **Endurance Testing** | Sinônimo de Soak Testing.[473][493][497][601] |
| 58 | **Stability Testing** | Testa estabilidade sob carga contínua.[493][497][559] |
| 59 | **Scalability Testing** | Testa capacidade de escalar.[473][493][497][559] |
| 60 | **Capacity Testing** | Determina capacidade máxima do sistema.[497][547] |
| 61 | **Benchmark Testing** | Compara com padrões estabelecidos.[474][493] |
| 62 | **Baseline Testing** | Estabelece linha de base de performance.[493] |
| 63 | **Ramp Testing** | Aumenta carga gradualmente até falha.[496] |
| 64 | **Peak Testing** | Testa em picos de carga.[520] |
| 65 | **Concurrency Testing** | Testa acessos simultâneos.[493][497] |
| 66 | **Throughput Testing** | Mede taxa de processamento.[497] |
| 67 | **Latency Testing** | Mede tempo de resposta.[497] |
| 68 | **Response Time Testing** | Testa tempos de resposta.[497] |

### 2.2. Testes de Confiabilidade

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 69 | **Reliability Testing** | Avalia confiabilidade do sistema.[473][474][493][497][547] |
| 70 | **Availability Testing** | Testa disponibilidade do sistema.[497][547] |
| 71 | **Failover Testing** | Testa transferência para backup.[493][497] |
| 72 | **Recovery Testing** | Testa recuperação após falhas.[473][474][493][497] |
| 73 | **Disaster Recovery Testing** | Testa recuperação de desastres.[497][547] |
| 74 | **Backup and Restore Testing** | Testa backup e restauração.[497] |
| 75 | **Fault Tolerance Testing** | Testa tolerância a falhas.[497] |

### 2.3. Testes de Compatibilidade

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 76 | **Compatibility Testing** | Testa compatibilidade com ambientes.[473][474][493][497][600] |
| 77 | **Cross-Browser Testing** | Testa em diferentes navegadores.[474][493][497] |
| 78 | **Cross-Platform Testing** | Testa em diferentes plataformas.[497] |
| 79 | **Browser Compatibility Testing** | Sinônimo de Cross-Browser Testing.[493] |
| 80 | **Device Compatibility Testing** | Testa em diferentes dispositivos.[497] |
| 81 | **Forward Compatibility Testing** | Testa com versões futuras.[493][497] |
| 82 | **Backward Compatibility Testing** | Testa com versões anteriores.[493][497] |
| 83 | **Downward Compatibility Testing** | Sinônimo de Backward.[493] |
| 84 | **Hardware Compatibility Testing** | Testa com diferentes hardwares.[497] |
| 85 | **Software Compatibility Testing** | Testa com diferentes softwares.[497] |
| 86 | **Network Compatibility Testing** | Testa em diferentes redes.[497] |
| 87 | **OS Compatibility Testing** | Testa em diferentes sistemas operacionais.[497] |
| 88 | **Version Compatibility Testing** | Testa entre versões.[497] |

### 2.4. Testes de Configuração e Instalação

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 89 | **Configuration Testing** | Testa diferentes configurações.[473][493][497] |
| 90 | **Installation Testing** | Testa processo de instalação.[473][493][497] |
| 91 | **Uninstallation Testing** | Testa processo de desinstalação.[497] |
| 92 | **Upgrade Testing** | Testa processo de atualização.[497] |
| 93 | **Migration Testing** | Testa migração de dados/sistemas.[497] |
| 94 | **Conversion Testing** | Testa conversão de dados.[497][547] |
| 95 | **Portability Testing** | Testa portabilidade entre ambientes.[497][547] |

### 2.5. Testes de Internacionalização

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 96 | **Internationalization Testing (i18n)** | Testa adaptação a diferentes idiomas.[493][497][547] |
| 97 | **Globalization Testing** | Testa funcionamento global.[493][497] |
| 98 | **Localization Testing (L10n)** | Testa adaptação a locais específicos.[473][493][497] |
| 99 | **Locale Testing** | Testa configurações regionais.[497] |
| 100 | **Translation Testing** | Testa traduções.[497] |
| 101 | **Currency Testing** | Testa formatos de moeda.[497] |
| 102 | **Date/Time Format Testing** | Testa formatos de data/hora.[497] |

### 2.6. Testes de Usabilidade

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 103 | **Usability Testing** | Avalia facilidade de uso.[473][474][493][497][525][534] |
| 104 | **User Experience (UX) Testing** | Avalia experiência do usuário.[497][525] |
| 105 | **Accessibility Testing (a11y)** | Testa acessibilidade para pessoas com deficiência.[493][495][497][528] |
| 106 | **Heuristic Evaluation** | Avaliação baseada em heurísticas.[537][546] |
| 107 | **Cognitive Walkthrough** | Avaliação cognitiva passo a passo.[525] |
| 108 | **Card Sorting Testing** | Teste de organização de informação.[534] |
| 109 | **5 Second Test** | Teste de primeira impressão.[534] |
| 110 | **A/B Testing (UX)** | Compara duas versões de UX.[493][497] |
| 111 | **Eye Tracking Testing** | Rastreia movimento dos olhos.[525] |
| 112 | **Think Aloud Testing** | Usuário verbaliza pensamentos.[525] |
| 113 | **Remote Usability Testing** | Usabilidade remota.[525][534] |
| 114 | **Moderated Usability Testing** | Com moderador presente.[534][537] |
| 115 | **Unmoderated Usability Testing** | Sem moderador.[534][537] |

### 2.7. Testes de Conformidade

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 116 | **Compliance Testing** | Verifica conformidade com padrões.[473][493][497][499] |
| 117 | **Conformance Testing** | Sinônimo de Compliance.[493][497] |
| 118 | **Regulatory Testing** | Testa requisitos regulatórios.[497] |
| 119 | **Standards Testing** | Testa conformidade com normas.[497] |
| 120 | **Certification Testing** | Testa para certificação.[497] |
| 121 | **Audit Testing** | Testes para auditoria.[497] |

---

## PARTE 3: TESTES DE SEGURANÇA

### 3.1. Testes de Segurança Gerais

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 122 | **Security Testing** | Avalia segurança do sistema.[473][474][493][497][526][529] |
| 123 | **Penetration Testing (Pen Testing)** | Simula ataques para encontrar vulnerabilidades.[493][497][520][526][529] |
| 124 | **Vulnerability Testing** | Identifica vulnerabilidades.[493][497] |
| 125 | **Vulnerability Assessment** | Avaliação de vulnerabilidades.[497][526] |
| 126 | **Vulnerability Scanning** | Varredura automatizada de vulnerabilidades.[526][548] |
| 127 | **Ethical Hacking** | Hacking autorizado para encontrar falhas.[497] |
| 128 | **Red Team Testing** | Simulação de ataques por equipe ofensiva.[497] |
| 129 | **Blue Team Testing** | Defesa contra ataques simulados.[497] |
| 130 | **Purple Team Testing** | Combinação Red + Blue.[497] |
| 131 | **Bug Bounty Testing** | Programa de recompensas por bugs.[497] |

### 3.2. Testes de Segurança de Aplicação

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 132 | **Static Application Security Testing (SAST)** | Análise estática de código fonte.[526][529][532][535][538][548] |
| 133 | **Dynamic Application Security Testing (DAST)** | Teste dinâmico em runtime.[526][529][532][535][538][548] |
| 134 | **Interactive Application Security Testing (IAST)** | Combina SAST e DAST.[526][529][532][535][538] |
| 135 | **Runtime Application Self-Protection (RASP)** | Proteção em tempo de execução.[526][529][538] |
| 136 | **Software Composition Analysis (SCA)** | Analisa dependências de terceiros.[535] |
| 137 | **Fuzz Testing (Fuzzing)** | Injeta dados aleatórios/malformados.[493][497][547] |
| 138 | **Input Validation Testing** | Testa validação de entradas.[497] |
| 139 | **SQL Injection Testing** | Testa vulnerabilidades de SQL injection.[526][529] |
| 140 | **XSS Testing** | Testa Cross-Site Scripting.[526][529] |
| 141 | **CSRF Testing** | Testa Cross-Site Request Forgery.[497] |
| 142 | **Authentication Testing** | Testa mecanismos de autenticação.[497][526] |
| 143 | **Authorization Testing** | Testa controles de autorização.[497] |
| 144 | **Session Management Testing** | Testa gerenciamento de sessões.[497][526] |
| 145 | **Encryption Testing** | Testa criptografia.[497] |
| 146 | **API Security Testing** | Testa segurança de APIs.[497][548] |
| 147 | **Mobile Security Testing** | Testa segurança mobile.[497][548] |
| 148 | **Web Application Security Testing** | Segurança de aplicações web.[526][529] |
| 149 | **Network Security Testing** | Testa segurança de rede.[497] |
| 150 | **Cloud Security Testing** | Testa segurança em cloud.[497] |
| 151 | **Container Security Testing** | Testa segurança de containers.[497] |

---

## PARTE 4: TÉCNICAS DE DESIGN DE TESTES

### 4.1. Técnicas Black Box

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 152 | **Black Box Testing** | Testa sem conhecimento do código interno.[473][474][476][477][493][497] |
| 153 | **Specification-Based Testing** | Baseado em especificações.[476][493][497] |
| 154 | **Equivalence Partitioning** | Divide entradas em classes equivalentes.[476][493][497][547] |
| 155 | **Equivalence Class Testing** | Sinônimo de Equivalence Partitioning.[493] |
| 156 | **Boundary Value Analysis (BVA)** | Testa valores de fronteira.[476][493][497][545][547][601] |
| 157 | **Boundary Value Testing** | Sinônimo de BVA.[493][601] |
| 158 | **Decision Table Testing** | Usa tabelas de decisão.[476][493][497][545][547] |
| 159 | **Cause-Effect Graph Testing** | Usa grafos causa-efeito.[476][493][497][547] |
| 160 | **State Transition Testing** | Testa transições de estado.[476][493][497][545][547] |
| 161 | **State Machine Testing** | Baseado em máquinas de estado.[497] |
| 162 | **Classification Tree Method** | Usa árvores de classificação.[476][480][547] |
| 163 | **Pairwise Testing** | Testa pares de variáveis.[493][497][547] |
| 164 | **All-Pairs Testing** | Sinônimo de Pairwise.[493][497] |
| 165 | **Orthogonal Array Testing** | Usa arrays ortogonais.[497] |
| 166 | **Combinatorial Testing** | Testa combinações de variáveis.[497][545][547] |
| 167 | **Domain Testing** | Testa domínios de entrada.[480][497] |
| 168 | **Syntax Testing** | Testa sintaxe de entradas.[476][547] |
| 169 | **Random Testing** | Testes aleatórios.[480][547] |
| 170 | **Error Guessing** | Baseado em intuição/experiência.[476][493][497] |
| 171 | **Checklist-Based Testing** | Baseado em checklists.[493][497][546][558] |
| 172 | **Use Case Testing** | Baseado em casos de uso.[476][493][497] |
| 173 | **Scenario Testing** | Baseado em cenários.[473][493][497] |
| 174 | **Requirements-Based Testing** | Baseado em requisitos.[497][547] |

### 4.2. Técnicas White Box

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 175 | **White Box Testing** | Testa com conhecimento do código.[473][476][477][493][497][509] |
| 176 | **Glass Box Testing** | Sinônimo de White Box.[476][493][497] |
| 177 | **Clear Box Testing** | Sinônimo de White Box.[493] |
| 178 | **Structural Testing** | Baseado na estrutura do código.[476][493][497] |
| 179 | **Code-Based Testing** | Baseado no código fonte.[497] |
| 180 | **Statement Testing** | Testa cada declaração.[493][497][547] |
| 181 | **Statement Coverage Testing** | Cobertura de declarações.[493][546][547] |
| 182 | **Decision Testing** | Testa cada decisão.[493][497][547] |
| 183 | **Decision Coverage Testing** | Cobertura de decisões.[493][547] |
| 184 | **Branch Testing** | Testa cada branch.[474][493][496][547] |
| 185 | **Branch Coverage Testing** | Cobertura de branches.[493][546][547] |
| 186 | **Condition Testing** | Testa cada condição.[493][497][547] |
| 187 | **Condition Coverage Testing** | Cobertura de condições.[493][546][547] |
| 188 | **Multiple Condition Testing** | Testa múltiplas condições.[493][497][546][547] |
| 189 | **Modified Condition/Decision Coverage (MC/DC)** | Cobertura MC/DC.[497][546][547] |
| 190 | **Path Testing** | Testa cada caminho.[493][497][545][547] |
| 191 | **Path Coverage Testing** | Cobertura de caminhos.[493] |
| 192 | **Control Flow Testing** | Testa fluxo de controle.[497][547][561][563] |
| 193 | **Data Flow Testing** | Testa fluxo de dados.[476][497][547][593] |
| 194 | **All-Defs Testing** | Testa todas as definições.[497][547] |
| 195 | **All-Uses Testing** | Testa todos os usos.[497][547] |
| 196 | **All-DU-Paths Testing** | Testa todos os caminhos DU.[497][547] |
| 197 | **Loop Testing** | Testa loops.[493][497] |
| 198 | **Mutation Testing** | Introduz mutações no código.[493][497][510][547][593] |
| 199 | **Basis Path Testing** | Testa caminhos básicos.[480][497] |
| 200 | **Code Review Testing** | Revisão de código.[497] |
| 201 | **Code Inspection** | Inspeção de código.[497] |

### 4.3. Técnicas Gray Box

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 202 | **Gray Box Testing** | Combinação de Black e White Box.[473][476][477][493][497][509] |
| 203 | **Matrix Testing** | Testa usando matrizes.[497] |
| 204 | **Pattern Testing** | Testa padrões de comportamento.[497] |

---

## PARTE 5: TESTES BASEADOS EM EXPERIÊNCIA

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 205 | **Experience-Based Testing** | Baseado na experiência do testador.[476][493][497][546][558] |
| 206 | **Exploratory Testing** | Explora o sistema sem scripts.[473][474][477][493][497][547] |
| 207 | **Ad Hoc Testing** | Testes improvisados sem planejamento.[473][493][497][520] |
| 208 | **Intuitive Testing** | Baseado na intuição.[493] |
| 209 | **Monkey Testing** | Ações aleatórias no sistema.[493][497][507][547] |
| 210 | **Gorilla Testing** | Testa intensivamente uma funcionalidade.[473][493][497] |
| 211 | **Smart Monkey Testing** | Monkey testing com inteligência.[507] |
| 212 | **Dumb Monkey Testing** | Monkey testing puramente aleatório.[507] |
| 213 | **Session-Based Testing** | Exploratory testing em sessões.[497][547] |
| 214 | **Tour-Based Testing** | Exploratory testing por "tours".[497] |
| 215 | **Charter-Based Testing** | Exploratory com objetivos definidos.[497] |

---

## PARTE 6: TESTES DE CHAOS E RESILIÊNCIA

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 216 | **Chaos Testing** | Introduz caos controlado no sistema.[507][513][515][518][521][524][547] |
| 217 | **Chaos Engineering** | Disciplina de testes de caos.[507][513][515][518][521] |
| 218 | **Chaos Monkey Testing** | Desliga instâncias aleatoriamente (Netflix).[507][513][515][518][524] |
| 219 | **Latency Monkey Testing** | Introduz latência artificial.[515] |
| 220 | **Conformity Monkey Testing** | Verifica conformidade de instâncias.[515] |
| 221 | **Security Monkey Testing** | Verifica segurança de instâncias.[515] |
| 222 | **Janitor Monkey Testing** | Remove recursos não utilizados.[515] |
| 223 | **Chaos Gorilla Testing** | Simula falha de zona inteira.[515] |
| 224 | **Simian Army Testing** | Conjunto de ferramentas de caos.[515] |
| 225 | **Fault Injection Testing** | Injeta falhas propositais.[493][497][507][511] |
| 226 | **Failure Mode Testing** | Testa modos de falha.[497] |
| 227 | **Disaster Injection Testing** | Injeta desastres simulados.[497] |
| 228 | **Game Day Testing** | Simulação de incidentes.[497] |
| 229 | **Resilience Testing** | Testa resiliência do sistema.[507][513][547] |
| 230 | **Robustness Testing** | Testa robustez do sistema.[473][493][497] |
| 231 | **Destructive Testing** | Testa até o sistema quebrar.[493][497] |
| 232 | **Breaking Point Testing** | Identifica ponto de quebra.[497] |
| 233 | **Stress Injection Testing** | Injeta stress no sistema.[507] |

---

## PARTE 7: TESTES MOBILE

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 234 | **Mobile Application Testing** | Testes de apps mobile.[482][497][525][531][537][543][599] |
| 235 | **Native App Testing** | Testes de apps nativos.[497] |
| 236 | **Hybrid App Testing** | Testes de apps híbridos.[497] |
| 237 | **Mobile Web Testing** | Testes de sites mobile.[497] |
| 238 | **Mobile Usability Testing** | Usabilidade em mobile.[525][534][537] |
| 239 | **Mobile Performance Testing** | Performance em mobile.[531][543] |
| 240 | **Mobile Security Testing** | Segurança em mobile.[531][543][548] |
| 241 | **Mobile Compatibility Testing** | Compatibilidade em mobile.[531][543] |
| 242 | **Device Farm Testing** | Testes em farm de dispositivos.[497] |
| 243 | **Gesture Testing** | Testes de gestos (swipe, tap, etc.).[531][543] |
| 244 | **Interrupt Testing** | Testes de interrupções (chamadas, SMS).[531][543] |
| 245 | **Battery Consumption Testing** | Testes de consumo de bateria.[531][543] |
| 246 | **Network Condition Testing** | Testes em diferentes condições de rede.[531][543] |
| 247 | **App Store Compliance Testing** | Conformidade com lojas de apps.[497] |
| 248 | **Mobile GUI Testing** | GUI em mobile.[531][543] |
| 249 | **Push Notification Testing** | Testes de notificações push.[543] |
| 250 | **Deep Link Testing** | Testes de deep links.[543] |

---

## PARTE 8: TESTES WEB

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 251 | **Web Application Testing** | Testes de aplicações web.[497][528] |
| 252 | **Cross-Browser Testing** | Testes em múltiplos navegadores.[474][493][497] |
| 253 | **Responsive Design Testing** | Testes de design responsivo.[497][528] |
| 254 | **Visual Regression Testing** | Testes de regressão visual.[497][528] |
| 255 | **Screenshot Testing** | Compara screenshots.[497] |
| 256 | **DOM Testing** | Testes do Document Object Model.[497] |
| 257 | **JavaScript Testing** | Testes de código JavaScript.[497] |
| 258 | **CSS Testing** | Testes de estilos CSS.[497] |
| 259 | **HTML Validation Testing** | Validação de HTML.[497] |
| 260 | **Web Accessibility Testing** | Acessibilidade web.[495][497][528] |
| 261 | **SEO Testing** | Testes de otimização para buscadores.[497] |
| 262 | **Cookie Testing** | Testes de cookies.[497] |
| 263 | **Session Testing** | Testes de sessões web.[497] |
| 264 | **Navigation Testing** | Testes de navegação.[497] |
| 265 | **Link Testing** | Testes de links quebrados.[497] |
| 266 | **Form Testing** | Testes de formulários.[497] |
| 267 | **Page Load Testing** | Testes de tempo de carregamento.[497] |

---

## PARTE 9: TESTES DE API

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 268 | **API Testing** | Testes de APIs.[473][474][493][497][527][530][533] |
| 269 | **REST API Testing** | Testes de APIs REST.[497][527][530][533][536] |
| 270 | **SOAP API Testing** | Testes de APIs SOAP.[497][527][530][533][536] |
| 271 | **GraphQL Testing** | Testes de APIs GraphQL.[527][530][533][536] |
| 272 | **gRPC Testing** | Testes de APIs gRPC.[533] |
| 273 | **Webhook Testing** | Testes de webhooks.[497] |
| 274 | **API Contract Testing** | Testes de contratos de API.[572][575][578][581][584] |
| 275 | **Consumer-Driven Contract Testing** | Contratos definidos pelo consumidor.[572][575][578][581] |
| 276 | **Provider-Driven Contract Testing** | Contratos definidos pelo provedor.[575] |
| 277 | **API Functional Testing** | Funcionalidade de APIs.[497] |
| 278 | **API Performance Testing** | Performance de APIs.[497] |
| 279 | **API Security Testing** | Segurança de APIs.[497][548] |
| 280 | **API Load Testing** | Carga em APIs.[497] |
| 281 | **API Versioning Testing** | Testes de versionamento.[497] |
| 282 | **API Documentation Testing** | Testes de documentação de API.[497] |

---

## PARTE 10: TESTES DE DATABASE

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 283 | **Database Testing** | Testes de banco de dados.[493][497][564] |
| 284 | **Data Integrity Testing** | Testes de integridade de dados.[497] |
| 285 | **Data Validation Testing** | Validação de dados.[497] |
| 286 | **Data Migration Testing** | Testes de migração de dados.[497][564] |
| 287 | **Data Conversion Testing** | Testes de conversão de dados.[497] |
| 288 | **Schema Testing** | Testes de esquema do banco.[497] |
| 289 | **Table Testing** | Testes de tabelas.[497] |
| 290 | **Stored Procedure Testing** | Testes de stored procedures.[497] |
| 291 | **Trigger Testing** | Testes de triggers.[497] |
| 292 | **View Testing** | Testes de views.[497] |
| 293 | **Index Testing** | Testes de índices.[497] |
| 294 | **Transaction Testing** | Testes de transações.[497] |
| 295 | **ACID Testing** | Testes de propriedades ACID.[497] |
| 296 | **Query Testing** | Testes de queries.[497] |
| 297 | **SQL Testing** | Testes de SQL.[497] |
| 298 | **NoSQL Testing** | Testes de bancos NoSQL.[497] |
| 299 | **ETL Testing** | Testes de Extract-Transform-Load.[493][497] |
| 300 | **Data Warehouse Testing** | Testes de data warehouse.[493][497] |
| 301 | **OLAP Testing** | Testes de OLAP.[497] |
| 302 | **Backup and Recovery Testing** | Testes de backup e recuperação.[497] |

---

## PARTE 11: TESTES DE DOMÍNIOS ESPECIALIZADOS

### 11.1. IoT Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 303 | **IoT Testing** | Testes de Internet das Coisas.[508][514] |
| 304 | **IoT Security Testing** | Segurança de dispositivos IoT.[508][514] |
| 305 | **IoT Performance Testing** | Performance de IoT.[508][514] |
| 306 | **IoT Interoperability Testing** | Interoperabilidade de IoT.[508][514] |
| 307 | **IoT Protocol Testing** | Protocolos de IoT (MQTT, CoAP).[508] |
| 308 | **Sensor Testing** | Testes de sensores.[508] |
| 309 | **Edge Computing Testing** | Testes de edge computing.[508] |
| 310 | **Digital Twin Testing** | Testes com gêmeos digitais.[508] |

### 11.2. Blockchain Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 311 | **Blockchain Testing** | Testes de blockchain.[511][519] |
| 312 | **Smart Contract Testing** | Testes de contratos inteligentes.[511][519] |
| 313 | **Consensus Testing** | Testes de mecanismos de consenso.[511] |
| 314 | **Node Testing** | Testes de nós blockchain.[511] |
| 315 | **Blockchain Security Testing** | Segurança de blockchain.[511][519] |
| 316 | **Blockchain Performance Testing** | Performance de blockchain.[511] |
| 317 | **DApp Testing** | Testes de aplicações descentralizadas.[519] |

### 11.3. Embedded Systems Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 318 | **Embedded Systems Testing** | Testes de sistemas embarcados.[546][576][582][590] |
| 319 | **Embedded Software Testing** | Testes de software embarcado.[576][582][589] |
| 320 | **Firmware Testing** | Testes de firmware.[574][577][583][586][589] |
| 321 | **Hardware Testing** | Testes de hardware.[560][577][580] |
| 322 | **Hardware-in-the-Loop (HIL) Testing** | Testes com hardware real.[576][579][583][585] |
| 323 | **Software-in-the-Loop (SIL) Testing** | Testes com software simulado.[560][576] |
| 324 | **Model-in-the-Loop (MIL) Testing** | Testes com modelos.[560][576] |
| 325 | **Processor-in-the-Loop (PIL) Testing** | Testes com processador real.[560][576] |
| 326 | **Human-in-the-Loop Testing** | Testes com humanos no loop.[560] |
| 327 | **JTAG Testing** | Testes via interface JTAG.[574] |
| 328 | **UART Testing** | Testes via interface UART.[574] |
| 329 | **Real-Time Testing** | Testes de sistemas real-time.[576] |
| 330 | **Timing Testing** | Testes de temporização.[576] |

### 11.4. Automotive Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 331 | **Automotive Testing** | Testes automotivos.[573][579][585][588] |
| 332 | **ADAS Testing** | Testes de sistemas avançados de assistência.[573][579] |
| 333 | **ECU Testing** | Testes de unidades de controle eletrônico.[573][576][579] |
| 334 | **CAN Bus Testing** | Testes de barramento CAN.[573] |
| 335 | **OBD-II Testing** | Testes de diagnóstico veicular.[573] |
| 336 | **ISO 26262 Testing** | Testes de segurança funcional.[573][579] |
| 337 | **Autonomous Vehicle Testing** | Testes de veículos autônomos.[579] |

### 11.5. AI/ML Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 338 | **AI/ML Model Testing** | Testes de modelos de IA/ML.[507][508] |
| 339 | **Model Validation Testing** | Validação de modelos.[508] |
| 340 | **Model Verification Testing** | Verificação de modelos.[508] |
| 341 | **Training Data Testing** | Testes de dados de treino.[508] |
| 342 | **Feature Engineering Testing** | Testes de engenharia de features.[508] |
| 343 | **Model Performance Testing** | Performance de modelos.[508] |
| 344 | **Model Accuracy Testing** | Acurácia de modelos.[508] |
| 345 | **Model Bias Testing** | Testes de viés em modelos.[508] |
| 346 | **Model Fairness Testing** | Testes de equidade de modelos.[508] |
| 347 | **Model Drift Testing** | Testes de deriva de modelo.[508] |
| 348 | **Concept Drift Testing** | Testes de deriva de conceito.[508] |
| 349 | **Data Drift Testing** | Testes de deriva de dados.[508] |
| 350 | **Model Inference Testing** | Testes de inferência.[508] |
| 351 | **A/B Testing (ML Models)** | Testes A/B de modelos.[508] |
| 352 | **Canary Testing (ML Models)** | Testes canary de modelos.[508] |
| 353 | **Shadow Testing (ML Models)** | Testes shadow de modelos.[508] |
| 354 | **Champion-Challenger Testing** | Testes champion-challenger.[508] |
| 355 | **Adversarial Testing** | Testes adversariais.[508][547] |
| 356 | **Metamorphic Testing** | Testes metamórficos.[547] |

### 11.6. Cloud Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 357 | **Cloud Testing** | Testes em ambiente cloud.[480][497] |
| 358 | **SaaS Testing** | Testes de Software as a Service.[497] |
| 359 | **PaaS Testing** | Testes de Platform as a Service.[497] |
| 360 | **IaaS Testing** | Testes de Infrastructure as a Service.[497] |
| 361 | **Multi-Cloud Testing** | Testes em múltiplas clouds.[497] |
| 362 | **Hybrid Cloud Testing** | Testes em cloud híbrida.[497] |
| 363 | **Cloud Migration Testing** | Testes de migração para cloud.[497] |
| 364 | **Serverless Testing** | Testes de arquiteturas serverless.[497] |
| 365 | **Container Testing** | Testes de containers.[497] |
| 366 | **Kubernetes Testing** | Testes em Kubernetes.[497] |

### 11.7. Microservices Testing

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 367 | **Microservices Testing** | Testes de microserviços.[572][578][581][584] |
| 368 | **Service Testing** | Testes de serviços individuais.[572] |
| 369 | **Service Integration Testing** | Integração de serviços.[572] |
| 370 | **Contract Testing (Microservices)** | Contratos entre serviços.[572][575][578][581][584] |
| 371 | **Consumer Contract Testing** | Contratos do lado consumidor.[572][575][578] |
| 372 | **Provider Contract Testing** | Contratos do lado provedor.[575] |
| 373 | **Pact Testing** | Testes com framework Pact.[572][578][581] |
| 374 | **Spring Cloud Contract Testing** | Testes com Spring Cloud Contract.[572][581] |

---

## PARTE 12: TESTES ESTÁTICOS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 375 | **Static Testing** | Testes sem execução de código.[493][497][547] |
| 376 | **Static Analysis** | Análise estática de código.[497] |
| 377 | **Static Code Analysis** | Análise estática de código fonte.[497][593] |
| 378 | **Code Review** | Revisão de código por pares.[497] |
| 379 | **Peer Review** | Revisão por colegas.[497] |
| 380 | **Walkthrough** | Apresentação guiada do código.[497][563] |
| 381 | **Inspection** | Inspeção formal de artefatos.[497][563] |
| 382 | **Technical Review** | Revisão técnica.[497][546] |
| 383 | **Formal Review** | Revisão formal documentada.[493][497] |
| 384 | **Informal Review** | Revisão informal.[497] |
| 385 | **Management Review** | Revisão gerencial.[497] |
| 386 | **Desk Checking** | Verificação manual no desk.[497][563] |
| 387 | **Requirements Review** | Revisão de requisitos.[497] |
| 388 | **Design Review** | Revisão de design.[497] |
| 389 | **Architecture Review** | Revisão de arquitetura.[497] |
| 390 | **Test Plan Review** | Revisão de plano de teste.[497] |
| 391 | **Test Case Review** | Revisão de casos de teste.[497] |

---

## PARTE 13: TESTES DINÂMICOS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 392 | **Dynamic Testing** | Testes com execução de código.[493][497][547] |
| 393 | **Dynamic Analysis** | Análise durante execução.[497] |
| 394 | **Runtime Testing** | Testes em tempo de execução.[497] |
| 395 | **Execution Testing** | Testes de execução.[497] |

---

## PARTE 14: TESTES POR ESCOPO E ABORDAGEM

### 14.1. Testes por Tipo de Entrada

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 396 | **Positive Testing** | Testa com entradas válidas.[493][497] |
| 397 | **Negative Testing** | Testa com entradas inválidas.[493][497] |
| 398 | **Boundary Testing** | Testa valores limites.[497][601] |
| 399 | **Edge Case Testing** | Testa casos de borda.[497] |
| 400 | **Corner Case Testing** | Testa casos de canto.[480][497] |
| 401 | **Happy Path Testing** | Testa fluxo feliz/principal.[497][547] |
| 402 | **Sad Path Testing** | Testa fluxos de erro.[497] |
| 403 | **Golden Path Testing** | Testa caminho ideal.[497] |
| 404 | **Exception Testing** | Testa tratamento de exceções.[497] |
| 405 | **Error Handling Testing** | Testa tratamento de erros.[497] |

### 14.2. Testes de Execução

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 406 | **Manual Testing** | Testes executados manualmente.[473][474][477][493][497][509][594] |
| 407 | **Automated Testing** | Testes automatizados.[473][474][477][493][497][509] |
| 408 | **Semi-Automated Testing** | Parcialmente automatizado.[497][550] |
| 409 | **Keyword-Driven Testing** | Automação por keywords.[497][501][547][563] |
| 410 | **Data-Driven Testing** | Automação orientada a dados.[474][480][497][547] |
| 411 | **Model-Based Testing** | Testes baseados em modelos.[480][497][547] |
| 412 | **Behavior-Driven Development (BDD) Testing** | Testes BDD.[497][547] |
| 413 | **Test-Driven Development (TDD) Testing** | Testes TDD.[497] |
| 414 | **Acceptance Test-Driven Development (ATDD)** | Testes ATDD.[480][497] |
| 415 | **Continuous Testing** | Testes contínuos em CI/CD.[478][497][547] |
| 416 | **Shift-Left Testing** | Testes antecipados no ciclo.[497] |
| 417 | **Shift-Right Testing** | Testes em produção.[497] |

---

## PARTE 15: TESTES DE DEPLOY E RELEASE

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 418 | **A/B Testing** | Compara duas versões.[493][497] |
| 419 | **Split Testing** | Sinônimo de A/B Testing.[493][497] |
| 420 | **Bucket Testing** | Sinônimo de A/B Testing.[493] |
| 421 | **Multivariate Testing** | Testa múltiplas variáveis.[497] |
| 422 | **Canary Testing** | Deploy gradual para % pequeno.[497][547] |
| 423 | **Canary Deployment Testing** | Testes de deploy canary.[497] |
| 424 | **Blue-Green Deployment Testing** | Testes de deploy blue-green.[497][547] |
| 425 | **Rolling Deployment Testing** | Testes de deploy rolling.[497] |
| 426 | **Feature Flag Testing** | Testes com feature flags.[497][547] |
| 427 | **Dark Launch Testing** | Launch oculto para testes.[497] |
| 428 | **Shadow Traffic Testing** | Tráfego shadow para testes.[497] |
| 429 | **Production Testing** | Testes em produção.[497] |
| 430 | **Testing in Production (TiP)** | Testes em ambiente de produção.[497] |
| 431 | **Pilot Testing** | Testes piloto com usuários limitados.[493][497] |
| 432 | **Parallel Testing** | Testes em paralelo.[497] |
| 433 | **Comparison Testing** | Compara versões.[497] |
| 434 | **Side-by-Side Testing** | Testes lado a lado.[497] |

---

## PARTE 16: TESTES CROWDSOURCED E EXTERNOS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 435 | **Crowdsourced Testing** | Testes por comunidade externa.[493][497] |
| 436 | **Crowdtesting** | Sinônimo de Crowdsourced Testing.[497] |
| 437 | **Beta Testing** | Testes por usuários beta.[473][474][477][493][497] |
| 438 | **Early Access Testing** | Testes de acesso antecipado.[497] |
| 439 | **Public Preview Testing** | Testes de preview público.[497] |
| 440 | **Bug Bounty Testing** | Programa de recompensas por bugs.[497] |
| 441 | **Outsourced Testing** | Testes terceirizados.[497][546] |
| 442 | **Third-Party Testing** | Testes por terceiros.[497] |
| 443 | **Independent Testing** | Testes independentes.[497][563] |
| 444 | **Certification Testing** | Testes para certificação.[497] |

---

## PARTE 17: TESTES AGILE E DEVOPS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 445 | **Agile Testing** | Testes em metodologia ágil.[480][493][497][547] |
| 446 | **Sprint Testing** | Testes dentro de sprints.[497] |
| 447 | **Iteration Testing** | Testes por iteração.[497] |
| 448 | **Continuous Integration Testing** | Testes em CI.[497] |
| 449 | **Continuous Delivery Testing** | Testes em CD.[497] |
| 450 | **Continuous Deployment Testing** | Testes em deploy contínuo.[497] |
| 451 | **Pipeline Testing** | Testes em pipeline CI/CD.[497] |
| 452 | **DevOps Testing** | Testes em cultura DevOps.[497] |
| 453 | **DevSecOps Testing** | Testes com segurança integrada.[497] |
| 454 | **Infrastructure as Code Testing** | Testes de IaC.[497] |
| 455 | **GitOps Testing** | Testes em GitOps.[497] |

---

## PARTE 18: TESTES DE REDE E PROTOCOLO

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 456 | **Network Testing** | Testes de rede.[497] |
| 457 | **Protocol Testing** | Testes de protocolos.[497][572] |
| 458 | **Protocol Conformance Testing** | Conformidade de protocolos.[497] |
| 459 | **Protocol Interoperability Testing** | Interoperabilidade de protocolos.[497] |
| 460 | **Network Performance Testing** | Performance de rede.[497] |
| 461 | **Network Security Testing** | Segurança de rede.[497] |
| 462 | **Packet Analysis Testing** | Análise de pacotes.[497] |
| 463 | **Latency Testing** | Testes de latência.[497] |
| 464 | **Bandwidth Testing** | Testes de largura de banda.[497] |
| 465 | **Throughput Testing** | Testes de throughput.[497] |
| 466 | **TCP/IP Testing** | Testes de pilha TCP/IP.[497] |
| 467 | **HTTP/HTTPS Testing** | Testes de HTTP(S).[497] |
| 468 | **WebSocket Testing** | Testes de WebSocket.[497] |
| 469 | **MQTT Testing** | Testes de MQTT.[508] |
| 470 | **DNS Testing** | Testes de DNS.[497] |
| 471 | **SSL/TLS Testing** | Testes de SSL/TLS.[497] |
| 472 | **VPN Testing** | Testes de VPN.[497] |
| 473 | **Firewall Testing** | Testes de firewall.[497] |
| 474 | **Load Balancer Testing** | Testes de load balancer.[497] |

---

## PARTE 19: TESTES GUI E INTERFACE

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 475 | **GUI Testing** | Testes de interface gráfica.[473][493][497][528][564] |
| 476 | **UI Testing** | Testes de interface de usuário.[493][497] |
| 477 | **Visual Testing** | Testes visuais.[497][528] |
| 478 | **Layout Testing** | Testes de layout.[528] |
| 479 | **Look and Feel Testing** | Testes de aparência.[528] |
| 480 | **Pixel-Perfect Testing** | Testes pixel a pixel.[497] |
| 481 | **Visual Regression Testing** | Regressão visual.[497][528] |
| 482 | **Screenshot Comparison Testing** | Comparação de screenshots.[497] |
| 483 | **Color Testing** | Testes de cores.[528] |
| 484 | **Font Testing** | Testes de fontes.[528] |
| 485 | **Icon Testing** | Testes de ícones.[528] |
| 486 | **Button Testing** | Testes de botões.[528] |
| 487 | **Menu Testing** | Testes de menus.[528] |
| 488 | **Dialog Testing** | Testes de diálogos.[528] |
| 489 | **Window Testing** | Testes de janelas.[528] |
| 490 | **Toolbar Testing** | Testes de barras de ferramentas.[528] |
| 491 | **Scrolling Testing** | Testes de scroll.[528] |
| 492 | **Drag and Drop Testing** | Testes de arrastar e soltar.[497] |
| 493 | **Keyboard Navigation Testing** | Testes de navegação por teclado.[497] |
| 494 | **Touch Testing** | Testes de toque (mobile).[531] |

---

## PARTE 20: TESTES DE DOCUMENTAÇÃO

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 495 | **Documentation Testing** | Testes de documentação.[493][497][547] |
| 496 | **User Manual Testing** | Testes de manuais de usuário.[497] |
| 497 | **Help File Testing** | Testes de arquivos de ajuda.[497] |
| 498 | **Online Help Testing** | Testes de ajuda online.[497] |
| 499 | **Release Notes Testing** | Testes de notas de versão.[497] |
| 500 | **Installation Guide Testing** | Testes de guias de instalação.[497] |
| 501 | **API Documentation Testing** | Testes de documentação de API.[497] |
| 502 | **Traceability Testing** | Testes de rastreabilidade.[497] |
| 503 | **Process Testing** | Testes de processos.[497] |
| 504 | **Workflow Testing** | Testes de workflows.[497] |
| 505 | **Business Rule Testing** | Testes de regras de negócio.[497] |

---

## PARTE 21: TESTES DIVERSOS E ESPECIAIS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 506 | **Dependency Testing** | Testes de dependências.[493][497] |
| 507 | **Compatibility Matrix Testing** | Testes de matriz de compatibilidade.[497] |
| 508 | **Baseline Testing** | Estabelece linha de base.[497] |
| 509 | **Confirmation Testing** | Confirma correção de bugs.[497] |
| 510 | **Context-Driven Testing** | Testes orientados por contexto.[497] |
| 511 | **Thread Testing** | Testes de threads.[497][563] |
| 512 | **Operational Testing** | Testes operacionais.[497] |
| 513 | **Certification Testing** | Testes para certificação.[497] |
| 514 | **Qualification Testing** | Testes de qualificação.[497][560] |
| 515 | **Acceptance Criteria Testing** | Testes de critérios de aceite.[497] |
| 516 | **Dry Run Testing** | Simulação sem execução real.[480][497] |
| 517 | **Pre-Flight Testing** | Verificações pré-deploy.[497] |
| 518 | **Health Check Testing** | Verificações de saúde.[497] |
| 519 | **Synthetic Monitoring** | Monitoramento sintético.[497] |
| 520 | **Real User Monitoring (RUM)** | Monitoramento de usuários reais.[497] |
| 521 | **Observability Testing** | Testes de observabilidade.[497] |
| 522 | **Distributed Tracing Testing** | Testes de tracing distribuído.[497] |
| 523 | **Logging Testing** | Testes de logging.[497] |
| 524 | **Metrics Testing** | Testes de métricas.[497] |
| 525 | **SLA Testing** | Testes de SLA.[497] |
| 526 | **SLO Testing** | Testes de SLO.[497] |
| 527 | **Golden Signal Testing** | Testes de sinais golden.[497] |
| 528 | **Spike Testing** | Testes de picos súbitos.[493][497][547] |
| 529 | **Breakpoint Testing** | Identifica ponto de quebra.[497] |
| 530 | **Capacity Planning Testing** | Testes de planejamento de capacidade.[497] |
| 531 | **Resource Utilization Testing** | Testes de utilização de recursos.[497] |
| 532 | **Memory Leak Testing** | Testes de vazamento de memória.[497] |
| 533 | **CPU Usage Testing** | Testes de uso de CPU.[497] |
| 534 | **Disk I/O Testing** | Testes de I/O de disco.[497] |
| 535 | **Network I/O Testing** | Testes de I/O de rede.[497] |
| 536 | **Garbage Collection Testing** | Testes de coleta de lixo.[497] |
| 537 | **Thread Pool Testing** | Testes de pool de threads.[497] |
| 538 | **Connection Pool Testing** | Testes de pool de conexões.[497] |
| 539 | **Cache Testing** | Testes de cache.[497] |
| 540 | **Queue Testing** | Testes de filas.[497] |
| 541 | **Message Bus Testing** | Testes de barramento de mensagens.[497] |
| 542 | **Event-Driven Testing** | Testes orientados a eventos.[497] |
| 543 | **Async Testing** | Testes assíncronos.[497] |
| 544 | **Callback Testing** | Testes de callbacks.[497] |
| 545 | **Promise Testing** | Testes de promises.[497] |
| 546 | **WebHook Testing** | Testes de webhooks.[497] |
| 547 | **Pub/Sub Testing** | Testes de publish/subscribe.[497] |
| 548 | **Event Sourcing Testing** | Testes de event sourcing.[497] |
| 549 | **CQRS Testing** | Testes de CQRS.[497] |
| 550 | **Saga Testing** | Testes de sagas.[497] |

---

## PARTE 22: TESTES POR TAXONOMIA SEI/CMU

Baseado na taxonomia do SEI/CMU que organiza ~200 tipos de testes nas categorias 5W+2H:[481][560][557]

### 22.1. What-Based Testing (O QUE está sendo testado)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 551 | **Object-Under-Test (OUT) Testing** | Testes baseados no objeto sob teste.[560] |
| 552 | **Domain-Based Testing** | Testes baseados em domínio.[560] |
| 553 | **Hardware Testing** | Testes de hardware.[560] |
| 554 | **Software Testing** | Testes de software.[560] |
| 555 | **System Testing** | Testes de sistema.[560] |
| 556 | **Data Testing** | Testes de dados.[560] |
| 557 | **Documentation Testing** | Testes de documentação.[560] |
| 558 | **Personnel Testing** | Testes de pessoal/treinamento.[560] |
| 559 | **Procedures Testing** | Testes de procedimentos.[560] |

### 22.2. When-Based Testing (QUANDO o teste é realizado)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 560 | **Order-Based Testing** | Testes baseados em ordem.[560] |
| 561 | **Lifecycle-Based Testing** | Testes baseados em ciclo de vida.[560] |
| 562 | **Phase-Based Testing** | Testes baseados em fase.[560] |
| 563 | **Built-In-Test (BIT) Testing** | Testes embutidos.[560] |
| 564 | **Sequential Testing** | Testes sequenciais.[560] |
| 565 | **Waterfall Testing** | Testes em modelo cascata.[560] |
| 566 | **V-Model Testing** | Testes em modelo V.[560] |
| 567 | **Agile Testing** | Testes em metodologia ágil.[560] |
| 568 | **Iterative Testing** | Testes iterativos.[560] |
| 569 | **Incremental Testing** | Testes incrementais.[560] |

### 22.3. Why-Based Testing (POR QUE o teste é realizado)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 570 | **Defect-Based Testing** | Testes para encontrar defeitos.[560][563] |
| 571 | **Risk-Based Testing** | Testes baseados em risco.[493][497][558] |
| 572 | **Requirements-Based Testing** | Testes baseados em requisitos.[560] |
| 573 | **Compliance-Based Testing** | Testes para conformidade.[560] |
| 574 | **Verification Testing** | Testes de verificação.[560] |
| 575 | **Validation Testing** | Testes de validação.[560] |

### 22.4. Who-Based Testing (QUEM realiza o teste)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 576 | **Developer Testing** | Testes por desenvolvedores.[560] |
| 577 | **Tester Testing** | Testes por testadores.[560] |
| 578 | **User Testing** | Testes por usuários.[560] |
| 579 | **Independent Testing** | Testes independentes.[560] |
| 580 | **Third-Party Testing** | Testes por terceiros.[560] |

### 22.5. Where-Based Testing (ONDE o teste é realizado)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 581 | **Development Environment Testing** | Testes em ambiente de dev.[560] |
| 582 | **Test Environment Testing** | Testes em ambiente de teste.[560] |
| 583 | **Staging Environment Testing** | Testes em staging.[560] |
| 584 | **Production Environment Testing** | Testes em produção.[560] |
| 585 | **Lab Testing** | Testes em laboratório.[560] |
| 586 | **Field Testing** | Testes em campo.[560] |

### 22.6. How-Based Testing (COMO o teste é realizado)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 587 | **Manual Testing** | Testes manuais.[560] |
| 588 | **Automated Testing** | Testes automatizados.[560] |
| 589 | **Static Testing** | Testes estáticos.[560] |
| 590 | **Dynamic Testing** | Testes dinâmicos.[560] |
| 591 | **Black-Box Testing** | Testes caixa-preta.[560] |
| 592 | **White-Box Testing** | Testes caixa-branca.[560] |
| 593 | **Gray-Box Testing** | Testes caixa-cinza.[560] |

### 22.7. How Well-Based Testing (QUÃO BEM o teste cobre)

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 594 | **Coverage-Based Testing** | Testes baseados em cobertura.[560] |
| 595 | **Statement Coverage Testing** | Cobertura de declarações.[560] |
| 596 | **Branch Coverage Testing** | Cobertura de branches.[560] |
| 597 | **Condition Coverage Testing** | Cobertura de condições.[560] |
| 598 | **Path Coverage Testing** | Cobertura de caminhos.[560] |
| 599 | **MC/DC Coverage Testing** | Cobertura MC/DC.[560] |
| 600 | **Requirements Coverage Testing** | Cobertura de requisitos.[560] |

---

## PARTE 23: TESTES ADICIONAIS

| # | Tipo de Teste | Descrição |
|---|---------------|-----------|
| 601 | **Exhaustive Testing** | Testa todas as combinações possíveis (teórico).[545][549][551][568] |
| 602 | **Complete Testing** | Sinônimo de Exhaustive Testing.[546][551][568] |
| 603 | **Audio Testing** | Testes de áudio.[570] |
| 604 | **Video Testing** | Testes de vídeo.[497] |
| 605 | **Animation Testing** | Testes de animações.[525][537] |
| 606 | **Transition Testing** | Testes de transições.[525] |
| 607 | **Input Method Testing** | Testes de métodos de entrada.[497] |
| 608 | **Clipboard Testing** | Testes de clipboard.[497] |
| 609 | **Print Testing** | Testes de impressão.[497] |
| 610 | **Email Testing** | Testes de emails.[497] |
| 611 | **Notification Testing** | Testes de notificações.[497] |
| 612 | **Localization Testing** | Testes de localização.[493][497] |
| 613 | **Biometric Testing** | Testes de biometria.[497] |
| 614 | **Voice Testing** | Testes de voz.[497] |
| 615 | **Speech Recognition Testing** | Testes de reconhecimento de fala.[497] |
| 616 | **NLP Testing** | Testes de processamento de linguagem natural.[497] |
| 617 | **Chatbot Testing** | Testes de chatbots.[497] |
| 618 | **Virtual Assistant Testing** | Testes de assistentes virtuais.[497] |
| 619 | **Wearable Testing** | Testes de dispositivos vestíveis.[497] |
| 620 | **Smart Home Testing** | Testes de casa inteligente.[508] |
| 621 | **Connected Car Testing** | Testes de carros conectados.[573] |
| 622 | **Medical Device Testing** | Testes de dispositivos médicos.[576][582] |
| 623 | **Aerospace Testing** | Testes aeroespaciais.[560][576] |
| 624 | **Defense Testing** | Testes de defesa.[560] |
| 625 | **Nuclear Testing** | Testes para sistemas nucleares.[560] |
| 626 | **Gaming Testing** | Testes de jogos.[497] |
| 627 | **VR Testing** | Testes de realidade virtual.[497] |
| 628 | **AR Testing** | Testes de realidade aumentada.[480][497] |
| 629 | **XR Testing** | Testes de realidade estendida.[497] |
| 630 | **3D Printing Testing** | Testes de impressão 3D.[497] |
| 631 | **Robotics Testing** | Testes de robótica.[497] |
| 632 | **Drone Testing** | Testes de drones.[497] |

---

## Resumo Estatístico

| Categoria | Quantidade |
|-----------|------------|
| Testes Funcionais | 50 |
| Testes Não-Funcionais | 95 |
| Testes de Segurança | 30 |
| Técnicas Black Box | 23 |
| Técnicas White Box | 27 |
| Técnicas Gray Box | 3 |
| Testes Baseados em Experiência | 11 |
| Testes de Chaos/Resiliência | 18 |
| Testes Mobile | 17 |
| Testes Web | 17 |
| Testes de API | 15 |
| Testes de Database | 20 |
| Testes de Domínios Especializados | 65 |
| Testes Estáticos | 17 |
| Testes Dinâmicos | 4 |
| Testes por Escopo | 22 |
| Testes de Deploy/Release | 17 |
| Testes Crowdsourced | 10 |
| Testes Agile/DevOps | 11 |
| Testes de Rede/Protocolo | 19 |
| Testes GUI/Interface | 20 |
| Testes de Documentação | 11 |
| Testes Diversos | 45 |
| Taxonomia SEI/CMU | 50 |
| Testes Adicionais | 32 |
| **TOTAL** | **632** |

---

## Referências e Fontes

Esta compilação foi baseada em pesquisa exaustiva nas seguintes fontes:

1. **ISTQB Glossary** - Glossário oficial de termos de teste de software
2. **SEI/CMU Taxonomy of Testing** - Taxonomia de ~200 tipos de testes
3. **IEEE Standards** - Padrões IEEE de engenharia de software
4. **OWASP** - Open Web Application Security Project
5. **ISO/IEC 25010** - Modelo de qualidade de software
6. **ISO/IEC 29119** - Padrões internacionais de teste de software
7. **Literature review** de mais de 100 artigos e publicações especializadas

---

*Documento gerado em Dezembro de 2025*
*Compilação exaustiva de tipos de testes de QA para TI*
