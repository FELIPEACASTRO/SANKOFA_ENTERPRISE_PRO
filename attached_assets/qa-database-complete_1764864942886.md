# Guia Supremo de Testes de QA para Bancos de Dados (Versão Completa e Revisada)

## Visão Geral

Este guia consolida **todos os tipos de testes** que um Especialista em QA deve executar para garantir a qualidade de soluções baseadas em **bancos de dados relacionais** (SQL), **não relacionais** (NoSQL: documentos, colunas, grafos, chave‑valor, time‑series) e **em memória** (Redis, Memcached). O objetivo é fornecer um **checklist exaustivo** para validar integridade, performance, segurança, disponibilidade, conformidade e governança em qualquer cenário de dados.[120][123][126][129][135][138][219][220][222]

---

## 1. Fundamentos de Testes de Banco de Dados

### 1.1. Por que testar bancos de dados?

- Garantir **integridade e consistência** dos dados ao longo do ciclo de vida da aplicação.[120][123][126][129][138][219][220]
- Prevenir **perdas de dados, corrupção**, violações de segurança e degradação de performance.[120][123][126][135][138][219][222]
- Validar conformidade com **regras de negócio, SLAs** de disponibilidade e requisitos regulatórios (GDPR, LGPD, HIPAA etc.).[120][123][126][135][138][257][260][263][266][269]
- Assegurar **qualidade dos dados**: acurácia, completude, consistência, unicidade, validade e atualidade.[221][224][226][229][232][234]

### 1.2. Tipos principais de testes de banco de dados

| Categoria | Exemplos |
|-----------|----------|
| **Estrutural** | Schema, tabelas, colunas, índices, constraints, views, triggers, procedures, functions [120][123][126][129][135][138][219][220][222] |
| **Funcional** | CRUD, regras de negócio, stored procedures, triggers, functions [120][123][126][129][135][138][219][220][222] |
| **Integridade de dados** | PKs, FKs, unicidade, referencial, constraints de domínio [120][123][126][129][132][135][138][220][221] |
| **Qualidade de dados** | Completude, acurácia, consistência, unicidade, validade, atualidade [221][224][226][229][232][234] |
| **Performance e carga** | Latência, throughput, escalabilidade, stress, volume, endurance [120][123][126][135][138][141][144][147][150][153][158][239] |
| **Segurança** | SQL injection, NoSQL injection, controles de acesso, criptografia, auditoria [120][123][126][135][138][142][145][148][151][154][156][227] |
| **Migração e ETL** | Validação de migração, reconciliação, transformações ETL [159][161][162][164][165][167][168][170][173][174][176][177] |
| **Backup, recuperação e HA** | Backup, restore, RPO/RTO, replicação, failover, DR [160][163][166][169][172][175][178][200][203][206][209][212][215] |
| **Compliance e auditoria** | GDPR, audit logs, rastreabilidade, data lineage [257][260][263][266][269][272][275][277][280][283][286][289] |

---

## 2. Testes Estruturais (Schema e Objetos)

### 2.1. Validação de schema

1. **Tabelas e colunas**
   - Existência de todas as tabelas previstas no modelo de dados.[120][123][126][129][135][138][219][220][222]
   - Tipos de dados corretos para cada coluna (INT, VARCHAR, DATE, DECIMAL, BLOB, JSON etc.).[120][123][126][129][135][219][220][222][225][228][231]
   - Constraints NOT NULL, DEFAULT, CHECK aplicadas corretamente.[120][123][126][129][135][220][222]
   - Tamanho e precisão de campos (VARCHAR(255), DECIMAL(10,2) etc.).[219][220][222][225]

2. **Chaves primárias e estrangeiras**
   - Toda tabela possui PK definida e única.[120][123][126][129][132][135][219][220][222]
   - FKs referenciam PKs válidas; integridade referencial garantida.[120][123][126][129][132][135][220][221]
   - Ações de CASCADE, SET NULL, RESTRICT funcionam conforme definido.[120][123][126][129][135][220][258][261][264][267][273][276]

3. **Índices**
   - Índices necessários criados para colunas de busca frequente.[120][123][126][135][219][220][222]
   - Índices únicos onde aplicável (e-mails, CPFs, códigos únicos etc.).[120][123][126][135][220]
   - Índices compostos para queries com múltiplas colunas.[219][220][238][241][244][247][256]
   - Índices full-text, geoespaciais ou especializados conforme necessidade.[181][184][193]

4. **Views**
   - Views existem, retornam dados corretos e refletem estrutura esperada.[120][123][126][135][140][143][219][222]
   - Performance de views complexas é aceitável.[140][143][238][244]

5. **Stored procedures e functions**
   - Existência e sintaxe correta de todos os objetos programáveis.[120][123][126][135][140][143][146][149][152][157][219][220][222][278][281][284][287][292][295]
   - Dependências entre objetos mapeadas e válidas.[140][143]
   - Parâmetros de entrada e saída validados.[140][143][278][281][284]

6. **Triggers**
   - Triggers existem e disparam nos eventos corretos (BEFORE/AFTER INSERT/UPDATE/DELETE).[120][123][126][135][140][143][219][220][222][258][261][264][270][273][276]
   - Lógica de trigger executa conforme esperado.[140][143][258][261]
   - Triggers em cascata funcionam corretamente sem loops infinitos.[258][261][264][267][273][276]
   - Performance de triggers não degrada operações significativamente.[140][143][220][261]

7. **Sequences e auto-increment**
   - Sequences existem e geram valores únicos corretamente.[219][220][222]
   - Auto-increment funciona após inserts e deletes.[219][220]

### 2.2. Testes de mapeamento front-end/back-end

1. **Mapeamento de objetos (ORM)**
   - Entidades do ORM (Hibernate, JPA, Sequelize, TypeORM etc.) mapeiam corretamente para tabelas/colunas.[219][222][279][282][285][288][290][293]
   - Queries geradas pelo ORM são eficientes (evitar N+1 queries).[279][282][285][288]
   - Validações de bean/entity são executadas corretamente.[279][282][285][288][290][293]

2. **Schema mapping testing**
   - Objetos da aplicação front-end correspondem ao schema do banco.[219][222]

### 2.3. Testes de schema em bancos NoSQL

1. **MongoDB: validação de schema (JSON Schema)**
   - Regras de validação definidas com `$jsonSchema`; `validationLevel` e `validationAction` configurados.[181][184][190][193]
   - Documentos inválidos são rejeitados ou logados conforme política.[181][184][190]

2. **Cassandra / wide‑column stores**
   - Keyspaces, column families/tables e tipos de dados conferidos.[124][127][130][133][136][139]

3. **Graph DBs (Neo4j)**
   - Labels, relationship types e properties conferidos via Cypher.[199][202][205][208][211][214][217]

4. **Time-series DBs (InfluxDB, TimescaleDB, QuestDB)**
   - Measurements, tags, fields e retention policies configurados corretamente.[259][262][265][268][271][274]

---

## 3. Testes de Qualidade de Dados (Data Quality)

### 3.1. Dimensões de qualidade de dados

1. **Completude (Completeness)**
   - Todos os campos obrigatórios estão preenchidos.[221][224][226][229][232][234]
   - Não há valores NULL em campos que não devem ser nulos.[221][224][226][229][232]
   - Datasets estão completos sem lacunas.[221][224][226][229][232][234]

2. **Acurácia (Accuracy)**
   - Dados refletem corretamente a realidade que representam.[221][224][226][229][232][234]
   - Valores estão dentro de ranges válidos e esperados.[221][224][226][229][232]
   - Cross-reference com fontes autorizadas quando possível.[221][226][229]

3. **Consistência (Consistency)**
   - Dados aparecem uniformemente em diferentes datasets e sistemas.[221][224][226][229][232][234]
   - Mesma entidade tem mesma representação em diferentes tabelas/bases.[221][224][226][234]
   - Formatos consistentes (datas, moedas, unidades de medida).[221][224][226][232][234]

4. **Unicidade (Uniqueness)**
   - Não há registros duplicados onde não deveria haver.[221][224][226][229][232][234]
   - Campos únicos (IDs, e-mails, documentos) não têm duplicatas.[221][224][226][232]

5. **Validade (Validity/Conformity)**
   - Dados estão no formato, tipo e tamanho esperados.[221][224][226][229][232][234]
   - Valores seguem regras de negócio (ex.: idade > 0, status em lista válida).[221][226][229][232]
   - Formatos de e-mail, telefone, CEP etc. são válidos.[221][226][229][232]

6. **Atualidade/Frescor (Timeliness/Freshness)**
   - Dados estão atualizados para o uso pretendido.[221][224][226][229][232][234]
   - Timestamps de última atualização são recentes o suficiente.[221][224][226][229]

7. **Integridade relacional (Referential Integrity)**
   - FKs apontam para PKs existentes.[120][123][126][129][132][135][138][220][221]
   - Não há registros órfãos.[120][123][126][129][132][135][220][221]

### 3.2. Técnicas de teste de qualidade de dados

1. **Null set testing** - Verificar campos nulos ou vazios.[221][226]
2. **Boundary value testing** - Testar limites de valores (min/max).[221][226]
3. **Completeness testing** - Verificar preenchimento de campos obrigatórios.[221][224][226][232]
4. **Uniqueness testing** - Identificar duplicatas.[221][224][226][232]
5. **Referential integrity testing** - Validar relacionamentos entre tabelas.[221][226]
6. **Format/pattern testing** - Validar formatos de dados (regex, masks).[221][226][232]
7. **Cross-system consistency testing** - Comparar dados entre sistemas.[221][226][234]

### 3.3. Ferramentas para qualidade de dados

- Great Expectations, dbt tests, Soda, Monte Carlo, Atlan, Datafold.[221][223][226][232]

---

## 4. Testes Funcionais (CRUD e Regras de Negócio)

### 4.1. Operações CRUD

1. **CREATE**
   - Inserção de registros válidos funciona; contagem de registros incrementa corretamente.[120][121][123][126][129][135][138][219][220][222][230]
   - Inserção de dados inválidos (violação de constraint) é rejeitada com erro adequado.[120][123][126][129][135][219][220][222]
   - Auto-generated IDs são criados corretamente.[219][220]

2. **READ**
   - Consultas retornam dados corretos, completos e no formato esperado.[120][121][123][126][129][135][138][219][220][222]
   - Filtros, ordenações, paginações funcionam conforme especificado.[120][123][126][135][220][222]
   - JOINs retornam dados consistentes.[120][123][126][135][220]

3. **UPDATE**
   - Atualizações alteram apenas os registros e campos esperados.[120][121][123][126][129][135][138][219][220][222]
   - Histórico/auditoria de alterações (se aplicável) é registrado.[120][123][126][135][257][260][263][269][275]
   - Timestamps de updated_at são atualizados.[220]

4. **DELETE**
   - Exclusões removem apenas os registros esperados.[120][121][123][126][129][135][138][219][220][222]
   - Soft delete (se implementado) marca registros sem remoção física.[120][123][126][135][220]
   - Cascade delete funciona conforme configurado.[120][123][126][135][220][258][261][267][273]

### 4.2. Stored procedures, triggers e functions

1. **Testes unitários de stored procedures**
   - Procedures retornam resultados esperados para inputs válidos.[120][123][126][135][140][143][146][157][278][281][284][287][292][295]
   - Parâmetros de entrada são validados.[140][143][278][281][284]
   - Parâmetros de saída retornam valores corretos.[140][143][278][281][284]
   - Edge cases e boundary values são tratados.[140][143][220][278][281]

2. **Frameworks de teste para procedures**
   - tSQLt (SQL Server), utPLSQL (Oracle), pgTAP (PostgreSQL), DBFit.[278][281][284][287][292][295]
   - Integração com CI/CD pipelines.[278][281][284]

3. **Tratamento de erros**
   - Exceptions são capturadas e tratadas; transações são revertidas quando necessário.[140][143][146][220][278]

4. **Triggers**
   - Disparam nos eventos corretos.[120][123][126][135][140][143][219][220][222][258][261]
   - Validação de dados antes de insert funciona.[258][261]
   - Logging/auditoria via triggers funciona.[258][261][275]
   - Cascading triggers funcionam sem loops infinitos.[258][261][264][267][273][276]

5. **Functions**
   - Retornam valores corretos para inputs variados.[140][143][146][220][278]
   - Performance é aceitável.[140][143][220]

### 4.3. Regras de negócio no banco

1. **Check constraints**
   - Validações de domínio funcionam (ex.: idade >= 0, status IN ('A','I')).[120][123][126][135][220][261]

2. **Computed columns**
   - Colunas calculadas retornam valores corretos.[220]

3. **Default values**
   - Valores default são aplicados quando campos não são informados.[120][123][126][135][220]

### 4.4. Testes funcionais em NoSQL

1. **MongoDB**
   - Operações `insertOne`, `insertMany`, `find`, `updateOne`, `deleteMany` validadas.[121][181][184][187][190][193][196]
   - Aggregation pipelines retornam resultados corretos.[121][181][187]
   - Índices são utilizados nas queries (explain plan).[121][181][184][187]

2. **Redis / Memcached**
   - Operações SET, GET, DEL, EXPIRE funcionam conforme esperado.[122][125][128][131][134][137][179][182][185][188][191][194][197]
   - Estruturas de dados (hashes, lists, sets, sorted sets) manipuladas corretamente.[122][131][134][185][191]
   - Pub/Sub funciona (Redis).[131][185][191]

3. **Grafos (Neo4j)**
   - Queries Cypher retornam nós e relacionamentos esperados.[199][202][205][208][211][214][217]
   - Aggregations e projections funcionam.[199][202][205][208][211]

4. **Time-series (InfluxDB, TimescaleDB)**
   - Inserts de métricas funcionam.[259][262][265][268][271][274]
   - Queries de agregação temporal (rollups, downsampling) retornam resultados corretos.[259][262][265][268]
   - Retention policies funcionam.[259][262][265]

---

## 5. Testes de Transações e Concorrência

### 5.1. ACID e níveis de isolamento

1. **Atomicidade**
   - Transações são all‑or‑nothing; em caso de erro, rollback completo.[198][201][204][207][210][213][216][220]

2. **Consistência**
   - Após cada transação, o banco permanece em estado válido.[198][201][204][207][210][220]

3. **Isolamento**
   - Testar comportamento em diferentes níveis: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE.[198][201][204][207][210][213][216]
   - Verificar dirty reads, non‑repeatable reads, phantom reads conforme nível.[198][201][204][207][210][216]

4. **Durabilidade**
   - Dados commitados persistem mesmo após falha de energia/crash.[198][201][204][207][210]

### 5.2. Testes de concorrência e deadlocks

1. **Cenários de concorrência**
   - Múltiplas transações simultâneas lendo/escrevendo mesmos registros.[180][183][186][189][192][195]
   - Validar que locks são adquiridos e liberados corretamente.[180][183][186][189][192]
   - Testar race conditions em operações críticas.[180][183][186][189]

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
   - Tempo de resposta de consultas críticas (p50, p95, p99).[120][123][126][135][138][141][144][147][150][153][158][238][241][244][247]

2. **Throughput**
   - Número de transações/queries por segundo suportadas.[120][123][126][135][138][141][144][147][150][153][158]

3. **Utilização de recursos**
   - CPU, memória, I/O de disco, rede do servidor de banco.[120][123][126][135][138][141][144][147][150][153]

### 6.2. Tipos de testes de performance

1. **Load testing**
   - Simular carga esperada de usuários/transações; medir métricas sob condições normais.[141][144][147][150][153][158][220][227][239]

2. **Stress testing**
   - Aumentar carga além do esperado para encontrar breaking point.[141][144][147][150][153][155][158][219][220][222]

3. **Volume testing**
   - Testar com grandes volumes de dados (milhões/bilhões de registros).[141][144][147][150][153][158][219][220]

4. **Scalability testing**
   - Avaliar como o banco escala com aumento de dados/usuários; testar sharding, particionamento, read replicas.[141][144][147][150][153][155][158][239][242][245][248][251][254]

5. **Soak/endurance testing**
   - Carga sustentada por período prolongado para detectar memory leaks, degradação.[141][144][150][153]

### 6.3. Análise de planos de execução (Query Execution Plans)

1. **Verificar planos de execução**
   - Usar EXPLAIN/EXPLAIN ANALYZE para verificar como queries são executadas.[238][241][244][247][250][253][256]
   - Identificar table scans, index scans, index seeks.[238][241][244][247][256]
   - Detectar operações custosas (sorts, nested loops em grandes datasets).[238][241][244][247][256]

2. **Otimização baseada em planos**
   - Validar que índices são utilizados.[238][241][244][247][256]
   - Identificar queries que precisam de reescrita.[238][241][244][256]
   - Comparar planos estimados vs reais.[238][244][247]

### 6.4. Connection pooling

1. **Testes de pool de conexões**
   - Validar comportamento sob esgotamento de pool.[237][240][243][246][249][252][255]
   - Testar timeouts de conexão.[237][240][243]
   - Monitorar conexões ativas vs disponíveis.[237][240][243][246][252][255]
   - Detectar connection leaks.[237][240][243][252]

### 6.5. Particionamento e sharding

1. **Testes de particionamento**
   - Partition pruning funciona (queries acessam apenas partições relevantes).[239][242][245][248][251][254]
   - Adição/remoção de partições funciona sem downtime.[239][245][248]
   - Performance melhora com particionamento adequado.[239][242][245][248]

2. **Testes de sharding**
   - Dados são distribuídos corretamente entre shards.[239][242][245][248][251][254]
   - Queries são roteadas para shards corretos.[239][242][245][248][251][254]
   - Rebalanceamento de shards funciona.[239][242][245][248]
   - Failover dentro de shards funciona.[239][242][245]

### 6.6. Testes de performance para cache (Redis/Memcached)

1. **Cache hit ratio**
   - Monitorar proporção de hits vs misses.[122][125][131][134][179][182][185][188][191][194][197]

2. **Latência de cache**
   - Tempo de GET/SET deve ser sub‑milissegundo em condições normais.[122][125][131][134][179][182][185][191][194][197]

3. **Eviction policies**
   - Testar comportamento quando memória cheia (LRU, LFU, FIFO, TTL).[122][125][131][134][179][185][188][191]

4. **Benchmark tools**
   - `redis-benchmark`, `memtier_benchmark`, JMeter, k6 etc.[125][131][179]

### 6.7. Testes de performance para time-series DBs

1. **Insert performance**
   - Throughput de inserção sob diferentes cardinalidades.[259][262][265][268][271][274]

2. **Query performance**
   - Latência de queries de agregação temporal.[259][262][265][268][271][274]
   - Performance de queries complexas (JOINs, window functions).[259][262][265]

---

## 7. Testes de Segurança

### 7.1. SQL Injection

1. **Testes de injeção**
   - Tentar payloads clássicos (`' OR 1=1 --`, `UNION SELECT`, `'; DROP TABLE--` etc.) em todos os inputs que geram queries.[142][145][148][151][154][156][227][230]
   - Validar que queries parametrizadas/prepared statements são usadas.[142][145][148][151][154][156][227]
   - Testar second-order SQL injection.[142][145][148]

2. **Stored procedure injection**
   - Testar procedures que constroem SQL dinâmico.[142][145][148]

3. **Blind SQL injection**
   - Testar vulnerabilidades de inferência (time-based, boolean-based).[142][145][148][151]

### 7.2. NoSQL Injection

1. **MongoDB injection**
   - Testar payloads como `{$gt: ""}`, `{$ne: null}` em inputs JSON.[121][181][187]

2. **Redis injection**
   - Testar comandos injetados em inputs.[122][131]

### 7.3. Controles de acesso

1. **Autenticação**
   - Conexões não autenticadas são rejeitadas.[120][123][126][135][138][142][145][148][154][227]
   - Senhas fortes são exigidas.[227]
   - MFA/2FA quando aplicável.[227]

2. **Autorização e privilégios**
   - Usuários têm apenas permissões necessárias (least privilege).[120][123][126][135][138][142][145][148][154][227]
   - Roles e grants estão corretos; testar escalação de privilégios.[120][123][126][135][138][145][148][227]
   - Separation of duties entre usuários de aplicação, DBA, auditoria.[227]

3. **Row-level security (RLS)**
   - Usuários veem apenas dados que deveriam ver.[227]

### 7.4. Criptografia e proteção de dados

1. **Em trânsito**
   - Conexões usam TLS/SSL; certificados válidos.[120][123][126][135][138][203][209][227]

2. **Em repouso**
   - Dados sensíveis são criptografados no storage.[120][123][126][135][138][227]

3. **Transparent Data Encryption (TDE)**
   - TDE está habilitado quando requerido.[227]

4. **Masking e pseudonimização**
   - Dados pessoais são mascarados em ambientes não‑produtivos.[120][123][126][135][227][263]
   - Dynamic data masking funciona conforme configurado.[227]

### 7.5. Auditoria e logging

1. **Audit trails**
   - Acessos e operações sensíveis são logados.[120][123][126][135][138][257][260][263][269][275]
   - Logs incluem quem, quando, o quê, de onde.[257][260][263][269][275]

2. **Proteção de logs**
   - Logs são protegidos contra tampering.[257][260][263][269]
   - Acesso a logs é controlado e logado.[257][260][263][269]

3. **Retenção de logs**
   - Políticas de retenção estão definidas e são cumpridas.[257][260][263][269]

---

## 8. Testes de Compliance, Governança e Data Lineage

### 8.1. Compliance regulatório

1. **GDPR/LGPD**
   - Direito de acesso, retificação e exclusão (right to erasure) podem ser exercidos.[257][260][263][266][269][272]
   - Consentimento é registrado e rastreável.[257][260][263][266]
   - Data minimization é praticada.[257][260][263][266]
   - Logs de processamento de dados pessoais são mantidos.[257][260][263][266][269]

2. **Auditorias periódicas**
   - Processos de auditoria de compliance estão definidos e são executados.[257][260][263][266][269][272]
   - Documentação está atualizada (RoPA, DPIAs).[257][260][266][272]

3. **Retenção e exclusão de dados**
   - Políticas de retenção são cumpridas.[257][260][263][269]
   - Dados são excluídos após período de retenção.[257][260][263][269]

### 8.2. Data lineage e rastreabilidade

1. **Rastreamento de origem**
   - É possível rastrear a origem de cada dado.[277][280][283][286][289][291][294]
   - Transformações são documentadas e rastreáveis.[277][280][283][286][289]

2. **Audit trails de modificações**
   - Modificações de dados são logadas com quem, quando, o quê.[277][280][283][286][289]

3. **Impacto de mudanças**
   - É possível avaliar impacto de mudanças em dados upstream/downstream.[277][280][283][286][289]

4. **Ferramentas de lineage**
   - Apache Atlas, DataHub, Collibra, Atlan, Monte Carlo, OpenLineage.[277][280][283][286][289]

---

## 9. Testes de Migração de Dados e ETL

### 9.1. Pré‑migração

1. **Análise de dados fonte**
   - Qualidade, volume, tipos de dados, anomalias identificadas.[159][162][165][168][171][174][177]

2. **Mapeamento fonte‑destino**
   - Documentação de correspondência de campos, transformações, regras de negócio.[159][162][165][168][171][174][177]

3. **Planejamento de rollback**
   - Estratégia de rollback definida caso migração falhe.[159][162][165][168][171]

### 9.2. Execução de migração

1. **Piloto com subset de dados**
   - Migrar amostra representativa; validar antes de migração completa.[159][162][165][168][171][174][177]

2. **Monitoramento de erros**
   - Capturar e logar falhas de transformação, rejeições, exceções.[159][162][165][168][171][174][177]

### 9.3. Pós‑migração

1. **Reconciliação de contagens**
   - Comparar row counts entre fonte e destino.[159][162][165][168][171][174][177]

2. **Validação de integridade**
   - Checksums, comparação de samples, validação de FKs.[159][162][165][168][171][174][177]

3. **Testes funcionais**
   - Aplicações funcionam corretamente com dados migrados.[159][162][165][168][171][174][177]

4. **User Acceptance Testing (UAT)**
   - Usuários de negócio validam dados críticos.[159][162][165][168][171][174][177]

### 9.4. Testes de ETL e Data Warehouse

1. **Extração**
   - Dados são extraídos completamente e no formato esperado.[161][164][167][170][173][176]

2. **Transformação**
   - Regras de negócio aplicadas corretamente; cálculos, agregações, joins validados.[161][164][167][170][173][176]
   - Transformações são idempotentes quando esperado.[161][164][167]

3. **Carga**
   - Dados carregados no destino sem perda, duplicação ou corrupção.[161][164][167][170][173][176]
   - Upserts funcionam corretamente.[161][164][167]

4. **Testes de regressão ETL**
   - Após mudanças em pipelines, dados continuam corretos.[161][164][167][170][173][176]

5. **SLA de ETL**
   - Jobs completam dentro do tempo esperado.[161][164][167]

---

## 10. Testes de Backup, Recuperação e Alta Disponibilidade

### 10.1. Backup

1. **Tipos de backup**
   - Full, incremental, diferencial; todos funcionam e são restauráveis.[160][163][166][169][172][175][178]

2. **Verificação de integridade**
   - Backups não estão corrompidos; checksums validados.[160][163][166][169][175][178]

3. **Armazenamento**
   - Backups armazenados em local seguro (offsite, nuvem, imutáveis).[160][163][166][169][175][178]

4. **Automação e agendamento**
   - Backups são executados automaticamente conforme schedule.[160][163][166][169][175]

### 10.2. Recuperação (Restore)

1. **Testes de restore**
   - Restaurar backup completo e incremental; validar dados recuperados.[160][163][166][169][172][175][178]

2. **Recovery Point Objective (RPO)**
   - Perda de dados aceitável; backups atendem RPO definido.[160][163][166][169][175]

3. **Recovery Time Objective (RTO)**
   - Tempo de recuperação dentro do SLA.[160][163][166][169][175]

4. **Point-in-time recovery (PITR)**
   - Recuperação para momento específico funciona.[160][163][166][169]

### 10.3. Alta disponibilidade e replicação

1. **Replicação**
   - Replicas estão sincronizadas (sync ou async conforme arquitetura).[200][203][206][209][212][215]
   - Lag de replicação monitorado e dentro de limites aceitáveis.[200][203][206][209][212][215]

2. **Failover**
   - Em caso de falha do primário, replica assume automaticamente.[200][203][206][209][212][215]
   - Testar failover manual e automático; medir tempo de recuperação.[200][203][206][209][212][215]

3. **Failback**
   - Após recuperação do primário, operações retornam sem perda de dados.[200][203][206][209][212]

4. **Split-brain prevention**
   - Mecanismos de prevenção de split-brain funcionam.[200][203][206][209]

### 10.4. Disaster Recovery

1. **Simulação de desastres**
   - Testar cenários: falha de hardware, datacenter, ransomware, corrupção de dados.[160][163][166][169][172][175][178]

2. **Documentação e runbooks**
   - Procedimentos de DR documentados e testados periodicamente.[160][163][166][169][175][178]

3. **Geo-replicação**
   - Replicas em outras regiões funcionam e podem assumir operação.[200][203][206][209]

---

## 11. Testes Específicos para Bancos em Memória (Redis, Memcached)

### 11.1. Funcionalidade de cache

1. **Operações básicas**
   - SET, GET, DEL, EXPIRE, TTL funcionam conforme esperado.[122][125][128][131][134][137][179][182][185][188][191][194][197]

2. **Estruturas de dados (Redis)**
   - Hashes, Lists, Sets, Sorted Sets, Streams, HyperLogLog, Bitmaps manipulados corretamente.[122][131][134][185][191]

3. **Pub/Sub (Redis)**
   - Mensagens publicadas e recebidas por subscribers.[131][185][191]

4. **Transactions (Redis MULTI/EXEC)**
   - Transações atômicas funcionam.[131][185][191]

5. **Lua scripting (Redis)**
   - Scripts Lua executam corretamente.[131][185][191]

### 11.2. Consistência e invalidação de cache

1. **Cache invalidation**
   - Após atualização no banco principal, cache é invalidado/atualizado.[122][125][179][185][188][191][197]

2. **TTL e expiração**
   - Dados expiram no tempo correto; não há dados stale além do aceitável.[122][125][179][185][188][191][197]

3. **Cache stampede / thundering herd**
   - Mecanismos de proteção (mutex, probabilistic early expiration) funcionam.[179][188]

4. **Cache-aside, write-through, write-behind patterns**
   - Padrões de cache implementados funcionam corretamente.[179][188]

### 11.3. Persistência (Redis)

1. **RDB e AOF**
   - Snapshots e logs de append funcionam; dados persistem após restart.[131][134][185][191]

2. **Restore de dados persistidos**
   - Após crash, Redis recupera dados do RDB/AOF.[131][134][185][191]

### 11.4. Clustering e replicação (Redis)

1. **Redis Cluster**
   - Sharding funciona; keys são distribuídas corretamente.[131][134][185][191]

2. **Replicação master‑replica**
   - Replicas estão sincronizadas; failover funciona.[131][134][185][191]

3. **Redis Sentinel**
   - Monitoramento e failover automático funcionam.[131][134][185][191]

---

## 12. Testes Específicos para Bancos NoSQL Documentais (MongoDB)

### 12.1. Schema validation

1. **JSON Schema rules**
   - Regras de validação aplicadas; documentos inválidos rejeitados ou logados.[181][184][190][193]

2. **validationLevel e validationAction**
   - Configurações strict/moderate e error/warn funcionam conforme esperado.[181][184][190]

### 12.2. Índices e performance

1. **Índices criados e usados**
   - Queries utilizam índices; explain plans validados.[121][181][184][187][193][196]

2. **Índices compostos, text e geoespaciais**
   - Funcionam para queries complexas e de localização.[121][181][184][193]

3. **Index intersection**
   - Múltiplos índices são combinados quando apropriado.[181][184]

### 12.3. Agregações

1. **Aggregation pipelines**
   - Estágios ($match, $group, $project, $lookup, $unwind, $facet etc.) retornam resultados corretos.[121][181][184][187]

### 12.4. Transações multi‑documento

1. **ACID transactions**
   - Transações multi‑documento funcionam com atomicidade.[121][127][181][184][193][196][210]

### 12.5. Replicação e sharding

1. **Replica sets**
   - Replicas estão sincronizadas; failover funciona.[181][184][193]

2. **Sharded clusters**
   - Dados distribuídos corretamente; balancer funciona.[181][184][193]

---

## 13. Testes Específicos para Bancos NoSQL Colunares (Cassandra)

### 13.1. Modelo de dados

1. **Keyspaces e tables**
   - Existem e têm schema correto.[124][127][130][133][136][139]

2. **Partition keys e clustering keys**
   - Dados são distribuídos e ordenados conforme design.[124][127][130][133][136][139]

### 13.2. Consistência

1. **Níveis de consistência**
   - Testar ONE, QUORUM, LOCAL_QUORUM, ALL em reads e writes.[124][127][130][133][136][139]

2. **Hinted handoff e read repair**
   - Mecanismos de consistência eventual funcionam.[124][127][130][133][139]

### 13.3. Performance e escalabilidade

1. **Distribuição de dados**
   - Dados balanceados entre nós; não há hot spots.[124][127][130][133][136][139]

2. **Compaction e tombstones**
   - Processos de compaction não degradam performance excessivamente.[124][127][130][133][139]
   - Tombstones são limpados adequadamente.[124][127][130][133]

---

## 14. Testes Específicos para Bancos de Grafos (Neo4j)

### 14.1. Modelo de dados

1. **Labels e relationship types**
   - Nós e relacionamentos criados com labels/types corretos.[199][202][205][208][211][214][217]

2. **Properties**
   - Propriedades de nós e edges estão corretas e tipadas.[199][202][205][208][211]

### 14.2. Queries Cypher

1. **Traversals e pattern matching**
   - Queries retornam caminhos e padrões esperados.[199][202][205][208][211][214][217]

2. **Aggregations e projections**
   - Funções de agregação e retorno de dados funcionam.[199][202][205][208][211]

3. **Performance de queries**
   - Queries complexas executam em tempo aceitável.[199][202][205][208]

### 14.3. Procedures e plugins

1. **Stored procedures**
   - Procedures customizadas funcionam e retornam resultados corretos.[199][205][214]

2. **APOC e outros plugins**
   - Funções de plugins instalados funcionam.[199][202][211]

### 14.4. Testes com neo4j‑harness

1. **Testes unitários de queries**
   - Usar neo4j‑harness para testes em banco embarcado.[199][205][208]

---

## 15. Testes Específicos para Bancos Time-Series

### 15.1. InfluxDB

1. **Measurements, tags e fields**
   - Estrutura de dados está correta.[259][262][265][268][271][274]

2. **Retention policies**
   - Dados são retidos e expirados conforme políticas.[259][265][271]

3. **Continuous queries**
   - Downsampling automático funciona.[259][265][271]

### 15.2. TimescaleDB

1. **Hypertables**
   - Tabelas são particionadas automaticamente por tempo.[259][262][265][268]

2. **Compression**
   - Compressão funciona e economiza espaço sem perda.[259][262][265]

3. **Continuous aggregates**
   - Agregações materializadas estão corretas e atualizadas.[259][262][265]

### 15.3. Performance comparativa

1. **Insert throughput**
   - Performance de inserção atende requisitos.[259][262][265][268][271][274]

2. **Query latency**
   - Latência de queries dentro do aceitável.[259][262][265][268][271][274]

---

## 16. Checklist Sintético de QA para Bancos de Dados

### 16.1. Estrutura e schema

- [ ] Todas as tabelas/collections/keyspaces existem conforme modelo.
- [ ] Tipos de dados, tamanhos e constraints corretos.
- [ ] PKs, FKs, índices e objetos programáveis validados.
- [ ] Schema validation (NoSQL) configurada e testada.
- [ ] Mapeamento ORM validado (se aplicável).

### 16.2. Qualidade de dados

- [ ] Completude: campos obrigatórios preenchidos, sem NULLs indevidos.
- [ ] Acurácia: dados refletem realidade, valores em ranges válidos.
- [ ] Consistência: dados uniformes entre sistemas e tabelas.
- [ ] Unicidade: sem duplicatas indevidas.
- [ ] Validade: formatos e regras de negócio respeitados.
- [ ] Atualidade: dados suficientemente recentes.

### 16.3. Funcionalidade

- [ ] Operações CRUD testadas com dados válidos e inválidos.
- [ ] Stored procedures, triggers e functions validados com testes unitários.
- [ ] Aggregations e queries complexas retornam resultados corretos.
- [ ] Regras de negócio no banco funcionam (check constraints, defaults).

### 16.4. Integridade de dados

- [ ] Integridade de entidade, referencial e de domínio validadas.
- [ ] Não há registros órfãos ou duplicatas indevidas.
- [ ] Cascading actions funcionam corretamente.

### 16.5. Transações e concorrência

- [ ] Propriedades ACID validadas.
- [ ] Níveis de isolamento testados conforme requisitos.
- [ ] Cenários de deadlock simulados e tratados.
- [ ] Race conditions testadas em operações críticas.

### 16.6. Performance

- [ ] Load, stress, volume e scalability tests executados.
- [ ] Latência e throughput dentro de SLOs.
- [ ] Planos de execução analisados e otimizados.
- [ ] Connection pooling configurado e testado.
- [ ] Particionamento/sharding validado (se aplicável).
- [ ] Cache hit ratio e performance de cache validados (se aplicável).

### 16.7. Segurança

- [ ] Testes de SQL/NoSQL injection executados.
- [ ] Controles de autenticação, autorização e auditoria validados.
- [ ] Criptografia em trânsito e em repouso verificada.
- [ ] Masking de dados em ambientes não-prod configurado.
- [ ] Audit logging habilitado e funcionando.

### 16.8. Compliance e governança

- [ ] Requisitos de GDPR/LGPD atendidos (direitos de titular, consentimento, retenção).
- [ ] Audit trails completos e protegidos.
- [ ] Data lineage rastreável.
- [ ] Documentação de compliance atualizada.

### 16.9. Migração e ETL

- [ ] Migração testada com piloto antes de produção.
- [ ] Reconciliação de contagens e integridade pós‑migração.
- [ ] Pipelines ETL validados end‑to‑end.
- [ ] Regressão ETL após mudanças em pipelines.

### 16.10. Backup, recuperação e HA

- [ ] Backups são criados automaticamente e restauráveis.
- [ ] RPO e RTO atendem SLAs.
- [ ] Failover e failback testados com sucesso.
- [ ] Simulações de DR executadas periodicamente.
- [ ] Replicação sincronizada e lag monitorado.

### 16.11. Bancos em memória (Redis/Memcached)

- [ ] Operações de cache funcionam corretamente.
- [ ] TTL, invalidação e eviction policies validados.
- [ ] Persistência e replicação (se aplicável) testadas.
- [ ] Clustering funciona (se aplicável).

### 16.12. NoSQL documentais (MongoDB)

- [ ] Schema validation rules aplicadas.
- [ ] Índices utilizados e queries performáticas.
- [ ] Transações multi‑documento funcionam.
- [ ] Replicação e sharding validados.

### 16.13. NoSQL colunares (Cassandra)

- [ ] Modelo de partição e clustering correto.
- [ ] Níveis de consistência testados.
- [ ] Distribuição de dados balanceada.
- [ ] Compaction não degrada performance.

### 16.14. Grafos (Neo4j)

- [ ] Labels, relationships e properties corretos.
- [ ] Queries Cypher retornam resultados esperados.
- [ ] Procedures e plugins funcionam.

### 16.15. Time-series (InfluxDB, TimescaleDB)

- [ ] Estrutura de dados (measurements, tags, fields, hypertables) correta.
- [ ] Retention policies e compression funcionam.
- [ ] Performance de insert e query dentro do esperado.

---

Este guia pode ser expandido por domínio (fintech, healthtech, e‑commerce, IoT, telecom etc.) e por tecnologia específica, adicionando casos de teste detalhados e scripts de validação para cada contexto.
