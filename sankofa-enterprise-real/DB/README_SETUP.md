# Sankofa Enterprise Pro - Setup do Banco de Dados

## Arquivos Disponíveis

| Arquivo | Descrição |
|---------|-----------|
| `complete_setup.sql` | Schema completo de todas as tabelas |
| `seed_data.sql` | Dados iniciais (usuários, configs, exemplos) |
| `hard_rules_insert.sql` | 216 regras duras de detecção de fraude |
| `REDIS_CONFIG.md` | Configuração do cache Redis |

## Setup Rápido (Replit)

O banco PostgreSQL já está configurado automaticamente no Replit.
A variável `DATABASE_URL` está disponível para conexão.

### 1. Criar as tabelas:
```bash
psql "$DATABASE_URL" -f DB/complete_setup.sql
```

### 2. Inserir dados de seed:
```bash
psql "$DATABASE_URL" -f DB/seed_data.sql
```

### 3. Inserir regras duras (216 regras):
```bash
psql "$DATABASE_URL" -f DB/hard_rules_insert.sql
```

## Setup Manual (Ambiente Local)

### 1. Criar banco de dados:
```bash
createdb sankofa_fraud
```

### 2. Configurar variável de ambiente:
```bash
export DATABASE_URL="postgresql://usuario:senha@localhost:5432/sankofa_fraud"
```

### 3. Executar scripts:
```bash
psql "$DATABASE_URL" -f DB/complete_setup.sql
psql "$DATABASE_URL" -f DB/seed_data.sql
psql "$DATABASE_URL" -f DB/hard_rules_insert.sql
```

## Tabelas Criadas

### Tabelas Principais
| Tabela | Descrição | Registros Iniciais |
|--------|-----------|-------------------|
| `users` | Usuários do sistema | 3 |
| `transactions` | Transações processadas | 5 |
| `hard_rules` | Regras de detecção | 216 |
| `alerts` | Alertas de fraude | 2 |
| `vip_list` | Lista branca | 3 |
| `hot_list` | Lista negra | 3 |

### Tabelas de Suporte
| Tabela | Descrição |
|--------|-----------|
| `audit_logs` | Log de auditoria (LGPD) |
| `feedback` | Feedback de analistas |
| `model_metrics` | Métricas do ML |
| `system_configs` | Configurações |
| `rbac_roles` | Roles de acesso |
| `rbac_user_roles` | Associação usuário-role |
| `rbac_sessions` | Sessões ativas |
| `cpf_tokens` | Tokenização CPF |
| `cpf_access_log` | Log de acesso CPF |

## Usuários Padrão

| Username | Senha | Role | Permissões |
|----------|-------|------|------------|
| admin | admin123 | admin | Acesso total |
| analista | admin123 | analyst | Transações, alertas, regras |
| viewer | admin123 | viewer | Somente leitura |

**IMPORTANTE:** Altere as senhas em produção!

## 216 Regras Duras

### Distribuição por Ação
- **block** (63 regras): Bloqueio imediato
- **review** (106 regras): Análise manual
- **alert** (28 regras): Monitoramento
- **step_up** (19 regras): Verificação extra

### Categorias de Regras
- PIX (horários, valores, chaves)
- Cartão de Crédito (CNP, velocity)
- Cartão de Débito (ATM, saques)
- BACEN (limites regulatórios)
- Engenharia Social (golpes)
- Malware (Mão Fantasma, BrasDex)
- Velocity (ataques automatizados)
- Combinações multi-fator

## Verificar Instalação

```sql
-- Contar tabelas
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public';

-- Verificar regras duras
SELECT COUNT(*) FROM hard_rules;

-- Verificar usuários
SELECT username, role FROM users;
```

## Backup e Restore

### Backup
```bash
pg_dump "$DATABASE_URL" > backup_sankofa.sql
```

### Restore
```bash
psql "$DATABASE_URL" < backup_sankofa.sql
```

## Suporte

Para problemas com o banco de dados:
1. Verifique logs: `psql "$DATABASE_URL" -c "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 10;"`
2. Verifique conexão: `psql "$DATABASE_URL" -c "SELECT 1;"`
3. Verifique regras: `psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM hard_rules WHERE enabled = true;"`
