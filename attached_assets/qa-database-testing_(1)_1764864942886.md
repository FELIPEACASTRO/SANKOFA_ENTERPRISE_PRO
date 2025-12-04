# Guia Supremo de Testes de QA para Bancos de Dados

## Visão Geral

Este guia consolida **todos os tipos de testes** que um Especialista em QA deve executar para garantir a qualidade de soluções baseadas em **bancos de dados relacionais** (SQL), **não relacionais** (NoSQL: documentos, colunas, grafos, chave‑valor) e **em memória** (Redis, Memcached). O objetivo é fornecer um **checklist exaustivo** para validar integridade, performance, segurança, disponibilidade e conformidade em qualquer cenário de dados.[120][121][122][123][126][129][131][135][138]

---

## 1. Fundamentos de Testes de Banco de Dados

1. **Por que testar bancos de dados?**
   - Garantir integridade e consistência dos dados ao longo do ciclo de vida da aplicação.[120][123][126][129][138]
   - Prevenir perdas de dados, corrupção, violações de segurança e degradação de performance.[120][123][126][135][138]
   - Validar conformidade com regras de negócio, SLAs de disponibilidade e requisitos regulatórios.[120][123][126][135][138]

2. **Tipos principais de testes de banco de dados**
   - Estrutural (schema, tabelas, índices, constraints, views, triggers, procedures).[120][123][126][129][135][138][140][143]
   - Funcional (CRUD, regras de negócio, stored procedures, triggers).[120][123][126][129][135][138][140][143]
   - Integridade de dados (chaves primárias/estrangeiras, unicidade, referencial).[120][123][126][129][132][135][138]
   - Performance e carga (latência, throughput, escalabilidade, stress).[120][123][126][135][138][141][144][147][150][153][158]
   - Segurança (SQL injection, controles de acesso, criptografia).[120][123][126][135][138][142][145][148][151][154][156]
   - Migração e ETL.[159][161][162][164][165][167][168][170][173][174][176][177]
   - Backup, recuperação e alta disponibilidade.[160][163][166][169][172][175][178][200][203][206][209][212][215]

---

## 2. Testes Estruturais (Schema e Objetos)

### 2.1. Validação de schema

1. **Tabelas e colunas**
   - Existência de todas as tabelas previstas no modelo de dados.[120][123][126][129][135][138]
   - Tipos de dados corretos para cada coluna (INT, VARCHAR, DATE, etc.).[120][123][126][129][135]
   - Constraints NOT NULL, DEFAULT, CHECK aplicadas corretamente.[120][123][126][129][135]

2. **Chaves primárias e estrangeiras**
   - Toda tabela possui PK definida e única.[120][123][126][129][132][135]
   - FKs referenciam PKs válidas; integridade referencial garantida.[120][123][126][129][132][135]

3. **Índices**
   - Índices necessários criados para colunas de busca frequente.[120][123][126][135][140][143]
   - Índices únicos onde aplicável (e-mails, CPFs, etc.).[120][123][126][135]

4. **Views, stored procedures, triggers e functions**
   - Existência e sintaxe correta de todos os objetos programáveis.[120][123][126][135][140][143][146][149][152][157]
   - Dependências entre objetos mapeadas e válidas.[140][143]

### 2.2. Testes de schema em bancos NoSQL

1. **MongoDB: validação de schema (JSON Schema)**
   - Regras de validação definidas com `$jsonSchema`; `validationLevel` e `validationAction` configurados.[181][184][190][193]
   - Documentos inválidos são rejeitados ou logados conforme política.[181][184][190]

2. **Cassandra / wide‑column stores**
   - Keyspaces, column families/tables e tipos de dados conferidos.[124][127][130][133][136][139]

3. **Graph DBs (Neo4j)**
   - Labels, relationship types e properties conferidos via Cypher.[199][202][205][208][211][214][217]

---

## 3. Testes Funcionais (CRUD e Regras de Negócio)

### 3.1. Operações CRUD

1. **CREATE**
   - Inserção de registros válidos funciona; contagem de registros incrementa corretamente.[120][121][123][126][129][135][138]
   - Inserção de dados inválidos (violação de constraint) é rejeitada com erro adequado.[120][123][126][129][135]

2. **READ**
   - Consultas retornam dados corretos, completos e no formato esperado.[120][121][123][126][129][135][138]
   - Filtros, ordenações, paginações funcionam conforme especificado.[120][123][126][135]

3. **UPDATE**
   - Atualizações alteram apenas os registros e campos esperados.[120][121][123][126][129][135][138]
   - Histórico/auditoria de alterações (se aplicável) é registrado.[120][123][126][135]

4. **DELETE**
   - Exclusões removem apenas os registros esperados.[120][121][123][126][129][135][138]
   - Soft delete (se implementado) marca registros sem remoção física.[120][123][126][135]

### 3.2. Stored procedures, triggers e functions

1. **Execução correta**
   - Procedures retornam resultados esperados para inputs válidos.[120][123][126][135][140][143][146][157]
   - Triggers disparam nos eventos corretos (INSERT, UPDATE, DELETE) e executam lógica prevista.[120][123][126][135][140][143]

2. **Tratamento de erros**
   - Exceptions são capturadas e tratadas; transações são revertidas quando necessário.[140][143][146]

3. **Performance de procedures**
   - Procedures pesadas são otimizadas e monitoradas.[140][143]

### 3.3. Testes funcionais em NoSQL

1. **MongoDB**
   - Operações `insertOne`, `insertMany`, `find`, `updateOne`, `deleteMany` validadas.[121][181][184][187][190][193][196]
   - Aggregation pipelines retornam resultados corretos.[121][181][187]

2. **Redis / Memcached**
   - Operações SET, GET, DEL, EXPIRE funcionam conforme esperado.[122][125][128][131][134][137][179][182][185][188][191][194][197]
   - Estruturas de dados (hashes, lists, sets, sorted sets) manipuladas corretamente.[122][131][134][185][191]

3. **Grafos (Neo4j)**
   - Queries Cypher retornam nós e relacionamentos esperados.[199][202][205][208][211][214][217]

---

## 4. Testes de Integridade de Dados

### 4.1. Integridade de entidade

1. **Chaves primárias**
   - Cada registro é unicamente identificado; não há duplicatas de PK.[120][123][126][129][132][135][138]

2. **Constraints de unicidade**
   - Campos únicos (e-mail, documento, código) não permitem duplicação.[120][123][126][129][135]

### 4.2. Integridade referencial

1. **Foreign keys**
   - Não existem registros órfãos (FK apontando para PK inexistente).[120][123][126][129][132][135][138]
   - Ações de CASCADE, SET NULL, RESTRICT funcionam conforme definido.[120][123][126][129][135]

2. **Consistência entre tabelas relacionadas**
   - Joins retornam dados consistentes; não há inconsistências de estado.[120][123][126][129][132][135]

### 4.3. Integridade de domínio

1. **Validação de valores**
   - Valores estão dentro de ranges permitidos (CHECK constraints, ENUM, etc.).[120][123][126][129][135]
   - Formatos de dados (datas, moedas, percentuais) são válidos.[120][123][126][129][135]

### 4.4. Integridade em NoSQL

1. **MongoDB**
   - Schema validation rules aplicadas; documentos fora do schema são tratados.[181][184][190][193]

2. **Grafos**
   - Relacionamentos apontam para nós existentes; não há edges órfãos.[199][202][205][208]

---

## 5. Testes de Transações e Concorrência

### 5.1. ACID e níveis de isolamento

1. **Atomicidade**
   - Transações são all‑or‑nothing; em caso de erro, rollback completo.[198][201][204][207][210][213][216]

2. **Consistência**
   - Após cada transação, o banco permanece em estado válido.[198][201][204][207][210]

3. **Isolamento**
   - Testar comportamento em diferentes níveis: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE.[198][201][204][207][210][213][216]
   - Verificar dirty reads, non‑repeatable reads, phantom reads conforme nível.[198][201][204][207][210][216]

4. **Durabilidade**
   - Dados commitados persistem mesmo após falha de energia/crash.[198][201][204][207][210]

### 5.2. Testes de concorrência e deadlocks

1. **Cenários de concorrência**
   - Múltiplas transações simultâneas lendo/escrevendo mesmos registros.[180][183][186][189][192][195]
   - Validar que locks são adquiridos e liberados corretamente.[180][183][186][189][192]

2. **Detecção e resolução de deadlocks**
   - Simular cenários de deadlock; verificar que o DBMS detecta e resolve (rollback de vítima).[180][183][186][189][192]
   - Validar que aplicação trata deadlock e faz retry.[180][183][186][189]

3. **Estratégias de prevenção**
   - Ordem consistente de locks, timeouts, uso de OCC vs PCC.[180][183][186][189][192]

### 5.3. Transações em NoSQL

1. **MongoDB**
   - Multi‑document transactions (ACID) funcionam conforme esperado.[127][130][133][181][184][193][196][210]

2. **Bancos eventualmente consistentes (Cassandra)**
   - Testar níveis de consistência (ONE, QUORUM, ALL) e comportamento de leitura/escrita.[124][127][130][133][136][139]

---

## 6. Testes de Performance e Carga

### 6.1. Métricas‑chave

1. **Latência de queries**
   - Tempo de resposta de consultas críticas (p50, p95, p99).[120][123][126][135][138][141][144][147][150][153][158]

2. **Throughput**
   - Número de transações/queries por segundo suportadas.[120][123][126][135][138][141][144][147][150][153][158]

3. **Utilização de recursos**
   - CPU, memória, I/O de disco, rede do servidor de banco.[120][123][126][135][138][141][144][147][150][153]

### 6.2. Tipos de testes de performance

1. **Load testing**
   - Simular carga esperada de usuários/transações; medir métricas sob condições normais.[141][144][147][150][153][158]

2. **Stress testing**
   - Aumentar carga além do esperado para encontrar breaking point.[141][144][147][150][153][155][158]

3. **Volume testing**
   - Testar com grandes volumes de dados (milhões/bilhões de registros).[141][144][147][150][153][158]

4. **Scalability testing**
   - Avaliar como o banco escala com aumento de dados/usuários; testar sharding, particionamento, read replicas.[141][144][147][150][153][155][158]

5. **Soak/endurance testing**
   - Carga sustentada por período prolongado para detectar memory leaks, degradação.[141][144][150][153]

### 6.3. Testes de performance para cache (Redis/Memcached)

1. **Cache hit ratio**
   - Monitorar proporção de hits vs misses.[122][125][131][134][179][182][185][188][191][194][197]

2. **Latência de cache**
   - Tempo de GET/SET deve ser sub‑milissegundo em condições normais.[122][125][131][134][179][182][185][191][194][197]

3. **Eviction policies**
   - Testar comportamento quando memória cheia (LRU, LFU, FIFO, TTL).[122][125][131][134][179][185][188][191]

4. **Benchmark tools**
   - `redis-benchmark`, `memtier_benchmark`, JMeter, k6 etc.[125][131][179]

---

## 7. Testes de Segurança

### 7.1. SQL Injection

1. **Testes de injeção**
   - Tentar payloads clássicos (`' OR 1=1 --`, `UNION SELECT`, etc.) em todos os inputs que geram queries.[142][145][148][151][154][156]
   - Validar que queries parametrizadas/prepared statements são usadas.[142][145][148][151][154][156]

2. **Stored procedure injection**
   - Testar procedures que constroem SQL dinâmico.[142][145][148]

### 7.2. NoSQL Injection

1. **MongoDB injection**
   - Testar payloads como `{$gt: ""}` em inputs JSON.[121][181][187]

### 7.3. Controles de acesso

1. **Autenticação**
   - Conexões não autenticadas são rejeitadas.[120][123][126][135][138][142][145][148][154]

2. **Autorização e privilégios**
   - Usuários têm apenas permissões necessárias (least privilege).[120][123][126][135][138][142][145][148][154]
   - Roles e grants estão corretos; testar escalação de privilégios.[120][123][126][135][138][145][148]

3. **Auditoria**
   - Acessos e operações sensíveis são logados.[120][123][126][135][138]

### 7.4. Criptografia e proteção de dados

1. **Em trânsito**
   - Conexões usam TLS/SSL; certificados válidos.[120][123][126][135][138][203][209]

2. **Em repouso**
   - Dados sensíveis são criptografados no storage.[120][123][126][135][138]

3. **Masking e pseudonimização**
   - Dados pessoais são mascarados em ambientes não‑produtivos.[120][123][126][135]

---

## 8. Testes de Migração de Dados e ETL

### 8.1. Pré‑migração

1. **Análise de dados fonte**
   - Qualidade, volume, tipos de dados, anomalias identificadas.[159][162][165][168][171][174][177]

2. **Mapeamento fonte‑destino**
   - Documentação de correspondência de campos, transformações, regras de negócio.[159][162][165][168][171][174][177]

### 8.2. Execução de migração

1. **Piloto com subset de dados**
   - Migrar amostra representativa; validar antes de migração completa.[159][162][165][168][171][174][177]

2. **Monitoramento de erros**
   - Capturar e logar falhas de transformação, rejeições, exceções.[159][162][165][168][171][174][177]

### 8.3. Pós‑migração

1. **Reconciliação de contagens**
   - Comparar row counts entre fonte e destino.[159][162][165][168][171][174][177]

2. **Validação de integridade**
   - Checksums, comparação de samples, validação de FKs.[159][162][165][168][171][174][177]

3. **Testes funcionais**
   - Aplicações funcionam corretamente com dados migrados.[159][162][165][168][171][174][177]

4. **User Acceptance Testing (UAT)**
   - Usuários de negócio validam dados críticos.[159][162][165][168][171][174][177]

### 8.4. Testes de ETL e Data Warehouse

1. **Extração**
   - Dados são extraídos completamente e no formato esperado.[161][164][167][170][173][176]

2. **Transformação**
   - Regras de negócio aplicadas corretamente; cálculos, agregações, joins validados.[161][164][167][170][173][176]

3. **Carga**
   - Dados carregados no destino sem perda, duplicação ou corrupção.[161][164][167][170][173][176]

4. **Testes de regressão ETL**
   - Após mudanças em pipelines, dados continuam corretos.[161][164][167][170][173][176]

---

## 9. Testes de Backup, Recuperação e Alta Disponibilidade

### 9.1. Backup

1. **Tipos de backup**
   - Full, incremental, diferencial; todos funcionam e são restauráveis.[160][163][166][169][172][175][178]

2. **Verificação de integridade**
   - Backups não estão corrompidos; checksums validados.[160][163][166][169][175][178]

3. **Armazenamento**
   - Backups armazenados em local seguro (offsite, nuvem, imutáveis).[160][163][166][169][175][178]

### 9.2. Recuperação (Restore)

1. **Testes de restore**
   - Restaurar backup completo e incremental; validar dados recuperados.[160][163][166][169][172][175][178]

2. **Recovery Point Objective (RPO)**
   - Perda de dados aceitável; backups atendem RPO definido.[160][163][166][169][175]

3. **Recovery Time Objective (RTO)**
   - Tempo de recuperação dentro do SLA.[160][163][166][169][175]

### 9.3. Alta disponibilidade e replicação

1. **Replicação**
   - Replicas estão sincronizadas (sync ou async conforme arquitetura).[200][203][206][209][212][215]
   - Lag de replicação monitorado e dentro de limites aceitáveis.[200][203][206][209][212][215]

2. **Failover**
   - Em caso de falha do primário, replica assume automaticamente.[200][203][206][209][212][215]
   - Testar failover manual e automático; medir tempo de recuperação.[200][203][206][209][212][215]

3. **Failback**
   - Após recuperação do primário, operações retornam sem perda de dados.[200][203][206][209][212]

### 9.4. Disaster Recovery

1. **Simulação de desastres**
   - Testar cenários: falha de hardware, datacenter, ransomware, corrupção de dados.[160][163][166][169][172][175][178]

2. **Documentação e runbooks**
   - Procedimentos de DR documentados e testados periodicamente.[160][163][166][169][175][178]

---

## 10. Testes Específicos para Bancos em Memória (Redis, Memcached)

### 10.1. Funcionalidade de cache

1. **Operações básicas**
   - SET, GET, DEL, EXPIRE, TTL funcionam conforme esperado.[122][125][128][131][134][137][179][182][185][188][191][194][197]

2. **Estruturas de dados (Redis)**
   - Hashes, Lists, Sets, Sorted Sets, Streams manipulados corretamente.[122][131][134][185][191]

3. **Pub/Sub (Redis)**
   - Mensagens publicadas e recebidas por subscribers.[131][185][191]

### 10.2. Consistência e invalidação de cache

1. **Cache invalidation**
   - Após atualização no banco principal, cache é invalidado/atualizado.[122][125][179][185][188][191][197]

2. **TTL e expiração**
   - Dados expiram no tempo correto; não há dados stale além do aceitável.[122][125][179][185][188][191][197]

3. **Cache stampede / thundering herd**
   - Mecanismos de proteção (mutex, probabilistic early expiration) funcionam.[179][188]

### 10.3. Persistência (Redis)

1. **RDB e AOF**
   - Snapshots e logs de append funcionam; dados persistem após restart.[131][134][185][191]

2. **Restore de dados persistidos**
   - Após crash, Redis recupera dados do RDB/AOF.[131][134][185][191]

### 10.4. Clustering e replicação (Redis)

1. **Redis Cluster**
   - Sharding funciona; keys são distribuídas corretamente.[131][134][185][191]

2. **Replicação master‑replica**
   - Replicas estão sincronizadas; failover funciona.[131][134][185][191]

---

## 11. Testes Específicos para Bancos NoSQL Documentais (MongoDB)

### 11.1. Schema validation

1. **JSON Schema rules**
   - Regras de validação aplicadas; documentos inválidos rejeitados ou logados.[181][184][190][193]

2. **validationLevel e validationAction**
   - Configurações strict/moderate e error/warn funcionam conforme esperado.[181][184][190]

### 11.2. Índices e performance

1. **Índices criados e usados**
   - Queries utilizam índices; explain plans validados.[121][181][184][187][193][196]

2. **Índices compostos e geoespaciais**
   - Funcionam para queries complexas e de localização.[121][181][184][193]

### 11.3. Agregações

1. **Aggregation pipelines**
   - Estágios ($match, $group, $project, $lookup etc.) retornam resultados corretos.[121][181][184][187]

### 11.4. Transações multi‑documento

1. **ACID transactions**
   - Transações multi‑documento funcionam com atomicidade.[121][127][181][184][193][196][210]

---

## 12. Testes Específicos para Bancos NoSQL Colunares (Cassandra)

### 12.1. Modelo de dados

1. **Keyspaces e tables**
   - Existem e têm schema correto.[124][127][130][133][136][139]

2. **Partition keys e clustering keys**
   - Dados são distribuídos e ordenados conforme design.[124][127][130][133][136][139]

### 12.2. Consistência

1. **Níveis de consistência**
   - Testar ONE, QUORUM, ALL em reads e writes.[124][127][130][133][136][139]

2. **Hinted handoff e read repair**
   - Mecanismos de consistência eventual funcionam.[124][127][130][133][139]

### 12.3. Performance e escalabilidade

1. **Distribuição de dados**
   - Dados balanceados entre nós; não há hot spots.[124][127][130][133][136][139]

2. **Compaction e tombstones**
   - Processos de compaction não degradam performance excessivamente.[124][127][130][133][139]

---

## 13. Testes Específicos para Bancos de Grafos (Neo4j)

### 13.1. Modelo de dados

1. **Labels e relationship types**
   - Nós e relacionamentos criados com labels/types corretos.[199][202][205][208][211][214][217]

2. **Properties**
   - Propriedades de nós e edges estão corretas e tipadas.[199][202][205][208][211]

### 13.2. Queries Cypher

1. **Traversals e pattern matching**
   - Queries retornam caminhos e padrões esperados.[199][202][205][208][211][214][217]

2. **Aggregations e projections**
   - Funções de agregação e retorno de dados funcionam.[199][202][205][208][211]

### 13.3. Procedures e plugins

1. **Stored procedures**
   - Procedures customizadas funcionam e retornam resultados corretos.[199][205][214]

2. **APOC e outros plugins**
   - Funções de plugins instalados funcionam.[199][202][211]

### 13.4. Testes com neo4j‑harness

1. **Testes unitários de queries**
   - Usar neo4j‑harness para testes em banco embarcado.[199][205][208]

---

## 14. Checklist Sintético de QA para Bancos de Dados

Use esta seção como check‑list diretamente em repositórios e planos de teste.

### 14.1. Estrutura e schema

- [ ] Todas as tabelas/collections/keyspaces existem conforme modelo.
- [ ] Tipos de dados e constraints corretos.
- [ ] PKs, FKs, índices e objetos programáveis validados.
- [ ] Schema validation (NoSQL) configurada e testada.

### 14.2. Funcionalidade

- [ ] Operações CRUD testadas com dados válidos e inválidos.
- [ ] Stored procedures, triggers e functions validados.
- [ ] Aggregations e queries complexas retornam resultados corretos.

### 14.3. Integridade de dados

- [ ] Integridade de entidade, referencial e de domínio validadas.
- [ ] Não há registros órfãos ou duplicatas indevidas.

### 14.4. Transações e concorrência

- [ ] Propriedades ACID validadas.
- [ ] Níveis de isolamento testados conforme requisitos.
- [ ] Cenários de deadlock simulados e tratados.

### 14.5. Performance

- [ ] Load, stress, volume e scalability tests executados.
- [ ] Latência e throughput dentro de SLOs.
- [ ] Cache hit ratio e performance de cache validados.

### 14.6. Segurança

- [ ] Testes de SQL/NoSQL injection executados.
- [ ] Controles de autenticação, autorização e auditoria validados.
- [ ] Criptografia em trânsito e em repouso verificada.

### 14.7. Migração e ETL

- [ ] Migração testada com piloto antes de produção.
- [ ] Reconciliação de contagens e integridade pós‑migração.
- [ ] Pipelines ETL validados end‑to‑end.

### 14.8. Backup, recuperação e HA

- [ ] Backups são criados e restauráveis.
- [ ] RPO e RTO atendem SLAs.
- [ ] Failover e failback testados com sucesso.
- [ ] Simulações de DR executadas periodicamente.

### 14.9. Bancos em memória (Redis/Memcached)

- [ ] Operações de cache funcionam corretamente.
- [ ] TTL, invalidação e eviction policies validados.
- [ ] Persistência e replicação (se aplicável) testadas.

### 14.10. NoSQL documentais (MongoDB)

- [ ] Schema validation rules aplicadas.
- [ ] Índices utilizados e queries performáticas.
- [ ] Transações multi‑documento funcionam.

### 14.11. NoSQL colunares (Cassandra)

- [ ] Modelo de partição e clustering correto.
- [ ] Níveis de consistência testados.
- [ ] Distribuição de dados balanceada.

### 14.12. Grafos (Neo4j)

- [ ] Labels, relationships e properties corretos.
- [ ] Queries Cypher retornam resultados esperados.
- [ ] Procedures e plugins funcionam.

---

Este guia pode ser expandido por domínio (fintech, healthtech, e‑commerce, IoT, etc.) e por tecnologia específica, adicionando casos de teste detalhados e scripts de validação para cada contexto.[120][121][122][123][124][125][126][127][128][129][131][135][138][140][141][142][143][144][145][148][159][160][161][163][164][165][179][180][181][184][185][198][199][200][203]
