# Sankofa Enterprise Pro - Database

## Estrutura do Banco de Dados

Este diretório contém todos os scripts e documentação relacionados ao banco de dados PostgreSQL do sistema Sankofa Enterprise Pro.

## Diretórios

```
DB/
├── migrations/          # Scripts de migração versionados
├── seeds/              # Dados iniciais e de teste
├── scripts/            # Scripts de manutenção e utilitários
├── backup/             # Scripts de backup e restore
├── docs/               # Documentação do banco de dados
├── schema.sql          # Schema completo (DDL)
├── init.sql            # Script de inicialização
└── README.md           # Este arquivo
```

## Tabelas Principais

| Tabela | Descrição | Registros Esperados |
|--------|-----------|---------------------|
| `transactions` | Transações financeiras | 300M+/dia |
| `fraud_detections` | Resultados de detecção de fraude | 3-5% das transações |
| `audit_trail` | Log de auditoria (LGPD/BACEN) | Todos os eventos |
| `users` | Usuários do sistema | ~100-1000 |
| `customers` | Clientes do banco | Milhões |
| `model_versions` | Versões dos modelos ML | ~10-50 |
| `api_keys` | Chaves de API | ~50-100 |
| `events` | Event sourcing | Milhões |

## Configuração

### Variáveis de Ambiente

```bash
DATABASE_URL=postgresql://user:password@host:5432/database
PGHOST=localhost
PGPORT=5432
PGUSER=sankofa
PGPASSWORD=***
PGDATABASE=sankofa_fraud
```

### Conexão

```python
import psycopg2

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
```

## Comandos Rápidos

### Inicializar Banco

```bash
psql -f DB/init.sql
psql -f DB/schema.sql
psql -f DB/seeds/initial_data.sql
```

### Executar Migrações

```bash
python DB/scripts/migrate.py
```

### Backup

```bash
./DB/backup/backup.sh
```

### Restore

```bash
./DB/backup/restore.sh backup_20251127.sql
```

## Performance

### Índices Principais

- `idx_transactions_timestamp` - Consultas por período
- `idx_transactions_cliente_cpf` - Consultas por cliente
- `idx_transactions_is_fraud` - Filtro de fraudes
- `idx_audit_trail_timestamp` - Auditoria por período

### Particionamento

Para volumes de 300M req/dia, recomenda-se particionar por:
- `transactions`: Por mês
- `audit_trail`: Por mês (retenção 7 anos)

## Compliance

### LGPD
- CPF mascarado (XXX.XXX.XXX-XX)
- Audit trail completo
- Right to be forgotten implementado

### BACEN Resolução 6/2023
- Retenção 5 anos
- Rastreabilidade completa
- Relatórios regulatórios

### PCI DSS
- Dados sensíveis criptografados
- Sem armazenamento de CVV
- Logs sem dados sensíveis

## Contato

Para questões sobre o banco de dados, consulte a documentação em `DB/docs/` ou o time de DBA.
