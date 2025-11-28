# Prompt Definitivo: O Guia Exaustivo de Tipos de Testes para Especialistas em QA

---

## Contexto

Você é um Especialista em Quality Assurance (QA) de elite, encarregado de construir a estratégia de testes mais completa e robusta para um portfólio diversificado de projetos de software. Sua missão é criar um guia de referência definitivo que mapeie **todos os tipos de testes existentes**, sem deixar lacunas (gaps), abrangendo desde as abordagens fundamentais até as mais modernas e especializadas. Este guia servirá como a base para todas as futuras atividades de QA na organização.

---

## Estrutura do Guia

O guia está organizado em quatro seções principais para máxima clareza e aplicabilidade:

1.  **Níveis de Teste (Test Levels):** Onde os testes são aplicados no ciclo de vida.
2.  **Tipos de Teste (Test Types):** O que é testado (Funcional, Não Funcional, Estrutural).
3.  **Metodologias e Abordagens:** Como os testes são organizados e executados.
4.  **Testes por Domínio Específico:** Testes aplicados a tecnologias e plataformas específicas.

---

## Seção 1: Níveis de Teste

Descrevem a granularidade e o escopo dos testes ao longo do processo de desenvolvimento.

- **1. Teste de Componente (Component/Unit/Module Testing):**
  - Testa a menor parte do software de forma isolada (funções, classes, módulos).
- **2. Teste de Integração (Integration Testing):**
  - Testa a interação e a comunicação entre componentes ou sistemas.
  - **Subtipos:**
    - **Big Bang:** Todos os módulos são integrados de uma vez.
    - **Top-Down:** Testa dos módulos de nível superior para os de nível inferior (usa *stubs*).
    - **Bottom-Up:** Testa dos módulos de nível inferior para os de nível superior (usa *drivers*).
    - **Sanduíche (Híbrido):** Combina as abordagens Top-Down e Bottom-Up.
- **3. Teste de Sistema (System Testing):**
  - Testa o sistema completo e integrado em um ambiente que simula a produção para verificar se ele atende aos requisitos.
- **4. Teste de Aceitação (Acceptance Testing):**
  - Valida se o sistema está pronto para o lançamento e atende às necessidades do negócio e do usuário.
  - **Subtipos:**
    - **Teste de Aceitação do Usuário (UAT):** Realizado pelo cliente/usuário final.
    - **Teste de Aceitação Operacional (OAT):** Verifica a prontidão operacional (backups, recuperação de desastres).
    - **Teste Alpha:** Realizado internamente pela equipe de desenvolvimento/QA.
    - **Teste Beta:** Realizado por um grupo limitado de usuários externos antes do lançamento oficial.
    - **Teste Gamma:** Realizado diretamente no ambiente de produção, pouco antes do lançamento, pulando testes internos.
    - **Teste de Contrato:** Valida se o sistema atende aos critérios definidos em um contrato.
    - **Teste de Regulamentação:** Valida a conformidade com leis e regulamentos.

---

## Seção 2: Tipos de Teste

Focam em aspectos específicos da qualidade do software.

### I. Testes Funcionais (Functional Testing)

Verificam **o que** o sistema faz, validando seu comportamento contra os requisitos.

- **5. Teste de Caixa-Preta (Black-Box Testing):**
  - Testa a funcionalidade sem conhecimento da estrutura interna do código.
- **6. Teste de Requisitos Funcionais:**
  - Valida cada requisito funcional especificado.
- **7. Teste Baseado em Cenários (Scenario Testing):**
  - Testa fluxos de trabalho realistas e complexos do usuário.
- **8. Teste de Caso de Uso (Use Case Testing):**
  - Deriva casos de teste a partir de casos de uso formais.
- **9. Teste Positivo (Positive Testing):**
  - Testa o sistema com entradas válidas para verificar os resultados esperados.
- **10. Teste Negativo (Negative Testing):**
  - Testa o sistema com entradas inválidas para verificar se ele lida com erros graciosamente.
- **11. Teste de Adivinhação de Erros (Error Guessing):**
  - Usa a experiência e a intuição do testador para antecipar erros comuns.
- **12. Teste Exploratório (Exploratory Testing):**
  - Aprendizagem, design e execução de testes ocorrem simultaneamente, sem casos de teste pré-definidos.
- **13. Teste Ad-hoc:**
  - Teste informal, sem planejamento ou documentação, também conhecido como "Monkey Testing" ou "Gorilla Testing".

### II. Testes Não Funcionais (Non-Functional Testing)

Verificam **como** o sistema funciona, avaliando suas características de qualidade.

- **Desempenho (Performance):**
  - **14. Teste de Carga (Load):** Avalia o desempenho sob carga normal e de pico.
  - **15. Teste de Estresse (Stress):** Leva o sistema ao seu limite para ver como ele falha e se recupera.
  - **16. Teste de Volume (Volume):** Testa o sistema com grandes volumes de dados.
  - **17. Teste de Escalabilidade (Scalability):** Mede a capacidade do sistema de escalar (para cima ou para fora) com o aumento da carga.
  - **18. Teste de Resistência (Endurance/Soak):** Avalia o desempenho sob carga sustentada por um longo período para encontrar vazamentos de memória.
  - **19. Teste de Pico (Spike):** Testa a reação do sistema a picos súbitos e extremos de carga.
  - **20. Teste de Capacidade (Capacity):** Determina quantos usuários e transações o sistema pode suportar antes que o desempenho seja degradado.
- **Segurança (Security):**
  - **21. Teste de Vulnerabilidade:** Identifica vulnerabilidades conhecidas usando scanners.
  - **22. Teste de Penetração (Pen Test):** Simula um ataque para explorar fraquezas de segurança.
  - **23. Teste de Segurança Estático (SAST):** Analisa o código-fonte em busca de falhas de segurança.
  - **24. Teste de Segurança Dinâmico (DAST):** Testa a aplicação em execução para encontrar vulnerabilidades.
  - **25. Teste de Segurança Interativo (IAST):** Combina SAST e DAST, analisando a aplicação de dentro para fora durante a execução.
  - **26. Teste Fuzz:** Fornece entradas inválidas e aleatórias para descobrir falhas.
- **Usabilidade (Usability):**
  - **27. Teste de Usabilidade:** Avalia a facilidade de uso, intuitividade e satisfação do usuário.
  - **28. Teste de Acessibilidade:** Garante que pessoas com deficiências possam usar o software (conformidade com WCAG).
  - **29. Teste de Experiência do Usuário (UX):** Avalia a experiência geral do usuário, incluindo emoções e percepções.
- **Compatibilidade (Compatibility):**
  - **30. Teste Cross-Browser:** Garante que a aplicação web funcione em diferentes navegadores.
  - **31. Teste Cross-Device:** Garante o funcionamento em diferentes dispositivos (desktops, tablets, celulares).
  - **32. Teste Cross-Platform/OS:** Garante o funcionamento em diferentes sistemas operacionais.
  - **33. Teste de Compatibilidade Reversa (Backward):** Verifica a compatibilidade com versões mais antigas de software/hardware.
  - **34. Teste de Compatibilidade Futura (Forward):** Verifica a compatibilidade com versões futuras.
- **Confiabilidade (Reliability):**
  - **35. Teste de Confiabilidade:** Mede a capacidade do software de operar sem falhas por um período especificado.
  - **36. Teste de Recuperação (Recovery):** Testa a capacidade do sistema de se recuperar de falhas.
  - **37. Teste de Resiliência:** Avalia como o sistema lida com falhas enquanto permanece funcional.
  - **38. Teste de Injeção de Falhas:** Introduz falhas intencionalmente para testar a robustez.
- **Manutenibilidade (Maintainability):**
  - **39. Teste de Manutenibilidade:** Avalia a facilidade com que o software pode ser modificado ou corrigido.
- **Portabilidade (Portability):**
  - **40. Teste de Portabilidade:** Avalia a facilidade de transferir o software para diferentes ambientes.
  - **41. Teste de Instalação/Desinstalação:** Verifica se o software pode ser instalado e removido corretamente.
- **Localização (Localization):**
  - **42. Teste de Internacionalização (I18n):** Garante que o software pode ser adaptado para diferentes idiomas e regiões sem mudanças no código.
  - **43. Teste de Localização (L10n):** Verifica a adaptação do software para um idioma e cultura específicos.

### III. Testes Estruturais (Structural / White-Box Testing)

Verificam **como** o sistema é construído, exigindo conhecimento da estrutura interna do código.

- **44. Teste de Caixa-Branca (White-Box Testing):**
  - Testa a lógica interna, caminhos e estruturas do código.
- **45. Teste de Cobertura de Código (Code Coverage):**
  - Mede o quanto do código-fonte é executado pelos testes.
  - **Subtipos:**
    - **Cobertura de Instrução (Statement):** Cada linha de código é executada.
    - **Cobertura de Decisão (Branch):** Cada resultado de uma decisão (if/else) é testado.
    - **Cobertura de Condição:** Cada condição booleana é avaliada como verdadeira e falsa.
    - **Cobertura de Caminho (Path):** Todos os caminhos possíveis através do código são testados.
- **46. Teste de Loop:**
  - Foca na validação de loops (simples, aninhados, etc.).
- **47. Teste de Mutação:**
  - Introduz pequenas falhas (mutações) no código para verificar se os testes existentes as detectam.

### IV. Testes Relacionados a Mudanças (Change-Related Testing)

Focam em verificar o software após modificações.

- **48. Teste de Regressão (Regression Testing):**
  - Garante que as mudanças não quebraram funcionalidades existentes.
  - **Subtipos:**
    - **Regressão Visual:** Compara screenshots para detectar mudanças visuais indesejadas.
- **49. Reteste (Retesting / Confirmation Testing):**
  - Confirma que um defeito específico foi corrigido com sucesso.
- **50. Teste de Fumaça (Smoke Testing / Build Verification Test):**
  - Um conjunto rápido de testes para verificar se a build é estável o suficiente para testes mais aprofundados.
- **51. Teste de Sanidade (Sanity Testing):**
  - Um subconjunto de testes de regressão focado em uma área específica da funcionalidade que foi alterada.

---

## Seção 3: Metodologias e Abordagens de Teste

Descrevem a filosofia e o processo de como os testes são integrados no ciclo de vida.

- **52. Teste Ágil (Agile Testing):**
  - Testes são integrados continuamente ao longo do ciclo de vida ágil.
- **53. Desenvolvimento Guiado por Testes (TDD - Test-Driven Development):**
  - Escreve-se um teste que falha antes de escrever o código funcional para fazê-lo passar.
- **54. Desenvolvimento Guiado por Comportamento (BDD - Behavior-Driven Development):**
  - Colaboração entre desenvolvedores, QAs e analistas de negócios para escrever cenários em linguagem natural (Gherkin) que guiam o desenvolvimento.
- **55. Desenvolvimento Guiado por Testes de Aceitação (ATDD - Acceptance Test-Driven Development):**
  - Similar ao BDD, foca em critérios de aceitação do ponto de vista do usuário.
- **56. Teste Contínuo (Continuous Testing):**
  - Execução automatizada de testes como parte do pipeline de CI/CD para fornecer feedback rápido.
- **57. Teste Shift-Left:**
  - Move as atividades de teste para o início do ciclo de vida, envolvendo QAs desde a fase de requisitos.
- **58. Teste Shift-Right:**
  - Foca em testes e monitoramento no ambiente de produção para obter feedback do mundo real.
- **59. Teste Baseado em Risco (Risk-Based Testing):**
  - Prioriza os testes com base no risco de falha e no impacto no negócio.
- **60. Teste Baseado em Modelos (Model-Based Testing):**
  - Geração automática de casos de teste a partir de um modelo do comportamento do sistema.
- **61. Teste Baseado em Propriedades (Property-Based Testing):**
  - Define propriedades ou invariantes que devem ser verdadeiras para qualquer entrada e gera dados aleatórios para tentar violá-las.
- **62. Teste Baseado em Experiência (Experience-Based Testing):**
  - Utiliza a habilidade, intuição e experiência do testador para encontrar defeitos.

---

## Seção 4: Testes por Domínio Específico

Testes adaptados para tecnologias, plataformas ou arquiteturas específicas.

- **Aplicações Móveis (Mobile):**
  - **63. Teste de Interrupção:** Verifica como o app lida com interrupções (chamadas, notificações).
  - **64. Teste de Bateria:** Avalia o consumo de bateria do aplicativo.
  - **65. Teste de Rede:** Simula diferentes condições de rede (2G, 3G, 4G, 5G, Wi-Fi, offline).
  - **66. Teste de Gestos:** Valida a resposta a gestos de toque (swipe, pinch, zoom).
- **Microserviços e APIs:**
  - **67. Teste de API:** Valida endpoints de API quanto à funcionalidade, desempenho e segurança.
  - **68. Teste de Contrato (Contract Testing):** Garante que um serviço (provedor) cumpra o contrato esperado por seu cliente (consumidor).
  - **69. Teste de Virtualização de Serviço:** Usa mocks ou stubs para simular serviços dependentes durante os testes.
- **Aplicações Web:**
  - **70. Teste de SEO:** Verifica elementos que impactam a otimização para motores de busca.
  - **71. Teste de Cookies:** Valida o gerenciamento de cookies pelo site.
  - **72. Teste de Sessão:** Garante que as sessões de usuário sejam gerenciadas corretamente.
- **Resiliência e Caos:**
  - **73. Teste de Caos (Chaos Engineering):** Injeta falhas de forma controlada em produção para verificar a resiliência do sistema.
  - **74. Teste de Failover:** Valida a capacidade do sistema de mudar para um sistema de backup em caso de falha.
- **Infraestrutura e Nuvem:**
  - **75. Teste de Infraestrutura:** Valida a configuração da infraestrutura como código (IaC).
  - **76. Teste de Implantação (Deployment):** Verifica o processo de implantação em diferentes ambientes.
  - **77. Teste Canary:** Libera uma nova versão para um pequeno subconjunto de usuários antes de liberar para todos.
  - **78. Teste Blue-Green:** Mantém duas versões idênticas do ambiente de produção (azul e verde) para permitir lançamentos e rollbacks instantâneos.
- **Dados e IA/ML:**
  - **79. Teste de Banco de Dados:** Valida a integridade, consistência e desempenho do banco de dados.
  - **80. Teste de ETL:** Valida o processo de Extração, Transformação e Carga de dados.
  - **81. Teste de Migração de Dados:** Garante que os dados sejam migrados corretamente de um sistema para outro.
  - **82. Teste de Viés (Bias Testing):** Verifica se modelos de IA/ML produzem resultados injustos ou discriminatórios.
  - **83. Teste de Equidade (Fairness Testing):** Avalia se os resultados do modelo são imparciais entre diferentes grupos de usuários.
- **Jogos (Gaming):**
  - **84. Teste de Jogabilidade (Gameplay):** Avalia a experiência de jogo, mecânicas e diversão.
  - **85. Teste de Balanceamento:** Garante que o jogo seja justo e equilibrado para todos os jogadores.
- **Sistemas Embarcados e IoT:**
  - **86. Teste de Sistemas Embarcados:** Testa software que roda em hardware não-PC (ex: eletrônicos de consumo).
  - **87. Teste de IoT:** Valida a funcionalidade, segurança e desempenho de dispositivos de Internet das Coisas.

---

## Conclusão

Este guia exaustivo, com **87 tipos de testes únicos e categorizados**, representa o padrão-ouro para a estratégia de QA. Como Especialista, sua função é selecionar, combinar e priorizar essas abordagens para construir um plano de testes robusto e adaptado às necessidades específicas de cada projeto, garantindo a entrega de software com a mais alta qualidade possível.

**Autor:** Manus AI  
**Data:** 28 de novembro de 2025  
**Versão:** 2.0 (Revisão Exaustiva)
