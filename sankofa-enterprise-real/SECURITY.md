# 🔒 GUIA DE SEGURANÇA - SANKOFA ENTERPRISE PRO

## ⚠️ CONFIGURAÇÕES CRÍTICAS DE SEGURANÇA

### 1. Variáveis de Ambiente Sensíveis

**NUNCA** commitar os seguintes valores em arquivos .env:

```bash
# ❌ NUNCA FAZER ISSO
SECRET_KEY=minha-chave-secreta
JWT_SECRET_KEY=jwt-123
DB_PASSWORD=senha123
```

**✅ FAZER ASSIM:**
- Use o sistema de secrets do Devin
- Configure via variáveis de ambiente do sistema
- Use gerenciadores de secrets (AWS Secrets Manager, Azure Key Vault, etc.)

### 2. Configuração de Produção

#### Variáveis Obrigatórias para Produção:
- `SECRET_KEY`: Mínimo 32 caracteres aleatórios
- `JWT_SECRET_KEY`: Mínimo 32 caracteres aleatórios  
- `DATABASE_URL`: String de conexão segura
- `REDIS_URL`: URL de conexão Redis com autenticação

#### Geração de Chaves Seguras:
```python
import secrets
# Gerar chave segura de 32 bytes
secret_key = secrets.token_urlsafe(32)
print(f"SECRET_KEY={secret_key}")
```

### 3. Configurações de Banco de Dados

#### ✅ Boas Práticas:
- Use conexões SSL/TLS
- Configure usuários com privilégios mínimos
- Use connection pooling
- Configure timeouts apropriados

#### ❌ Evitar:
- Usuários com privilégios de admin
- Conexões sem SSL
- Senhas fracas ou padrão

### 4. Configurações de API

#### Headers de Segurança Obrigatórios:
```python
# Configurar no Flask
from flask_talisman import Talisman

Talisman(app, {
    'force_https': True,
    'strict_transport_security': True,
    'content_security_policy': {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline'"
    }
})
```

### 5. Monitoramento e Logs

#### ⚠️ NUNCA logar informações sensíveis:
- Senhas
- Tokens JWT
- Chaves de API
- Dados pessoais (PII)

#### ✅ Logar para auditoria:
- Tentativas de login
- Alterações de dados críticos
- Erros de autenticação
- Acessos a recursos sensíveis

### 6. Dependências e Vulnerabilidades

#### Comandos para verificar vulnerabilidades:
```bash
# Python
pip audit

# Node.js
npm audit
npm audit fix

# Verificar dependências desatualizadas
pip list --outdated
npm outdated
```

### 7. Configuração de CORS

#### ✅ Produção:
```python
CORS(app, origins=['https://your-domain.com'])
```

#### ❌ NUNCA em produção:
```python
CORS(app, origins=['*'])  # Permite qualquer origem
```

### 8. Rate Limiting

Configure rate limiting para APIs:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

### 9. Validação de Input

#### ✅ Sempre validar:
- Dados de entrada da API
- Parâmetros de query
- Headers HTTP
- Uploads de arquivo

#### Use bibliotecas como:
- `marshmallow` para serialização
- `cerberus` para validação
- `pydantic` para modelos de dados

### 10. Backup e Recuperação

#### Configurar backups automatizados:
- Banco de dados
- Arquivos de configuração
- Modelos de ML treinados
- Logs de auditoria

## 📞 Reportar Vulnerabilidades

Se encontrar vulnerabilidades de segurança:
1. **NÃO** abra issues públicas
2. Envie email para: security@sankofa-enterprise.com
3. Inclua detalhes técnicos e steps para reproduzir
4. Aguarde resposta em até 48h

## 🔄 Atualizações de Segurança

- Revisar este documento mensalmente
- Atualizar dependências regularmente
- Monitorar CVEs relacionadas
- Realizar auditorias de segurança trimestrais