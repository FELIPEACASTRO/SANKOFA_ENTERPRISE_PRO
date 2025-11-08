# 🛡️ SOLUÇÕES PARA VULNERABILIDADES DE SEGURANÇA - SANKOFA ENTERPRISE PRO

**Data**: 08 de Novembro de 2025  
**Status**: Plano de Remediação  
**Prioridade**: CRÍTICA  

---

## 📋 SUMÁRIO EXECUTIVO

Este documento apresenta soluções concretas e implementáveis para as vulnerabilidades críticas identificadas no projeto SANKOFA_ENTERPRISE_PRO. Todas as soluções foram projetadas para serem aplicadas de forma incremental, sem quebrar a funcionalidade existente.

---

## 🔴 VULNERABILIDADE 1: FLASK DEBUG MODE HABILITADO

### Problema Identificado

**Severidade**: CRÍTICA  
**Arquivos Afetados**: 3  
- `backend/simple_api.py:116`
- `backend/api/compliance_api.py:48`
- `backend/api/main_integrated_api.py:363`

**Risco**: Exposição do debugger Werkzeug permite execução remota de código (RCE).

### Solução Proposta

#### Opção 1: Usar Variável de Ambiente (RECOMENDADO)

```python
import os
from flask import Flask

app = Flask(__name__)

# Solução: Usar variável de ambiente
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8445,
        debug=DEBUG_MODE,  # Controlado por variável de ambiente
        threaded=True
    )
```

#### Opção 2: Usar Arquivo de Configuração

```python
# config/settings.py
import os

class Config:
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# Selecionar configuração baseada no ambiente
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}

def get_config():
    env = os.getenv('FLASK_ENV', 'production')
    return config.get(env, config['default'])
```

```python
# main_integrated_api.py
from config.settings import get_config

app = Flask(__name__)
app.config.from_object(get_config())

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8445,
        debug=app.config['DEBUG']
    )
```

### Implementação

**Passo 1**: Criar arquivo `.env` na raiz do projeto:
```bash
FLASK_ENV=production
FLASK_DEBUG=False
```

**Passo 2**: Instalar python-dotenv:
```bash
pip install python-dotenv
```

**Passo 3**: Carregar variáveis de ambiente no início de cada arquivo:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Passo 4**: Atualizar todos os 3 arquivos afetados.

---

## 🔴 VULNERABILIDADE 2: SSL CERTIFICATE VALIDATION DESABILITADA

### Problema Identificado

**Severidade**: CRÍTICA  
**Arquivo Afetado**: `backend/infrastructure/disaster_recovery_system.py:212`

**Risco**: Vulnerável a ataques Man-in-the-Middle (MITM).

### Solução Proposta

#### Opção 1: Usar Certificados Válidos (RECOMENDADO)

```python
import requests
import os

# Solução: Usar certificado CA válido
ca_bundle_path = os.getenv('CA_BUNDLE_PATH', '/etc/ssl/certs/ca-certificates.crt')

response = requests.post(
    backup_url,
    json=backup_data,
    verify=ca_bundle_path,  # Usar certificado CA
    timeout=30
)
```

#### Opção 2: Usar Certificado Auto-Assinado com Validação

```python
import requests
import os

# Solução: Usar certificado auto-assinado específico
cert_path = os.getenv('CUSTOM_CERT_PATH', '/path/to/self-signed-cert.pem')

response = requests.post(
    backup_url,
    json=backup_data,
    verify=cert_path if os.path.exists(cert_path) else True,
    timeout=30
)
```

#### Opção 3: Permitir Desabilitação APENAS em Desenvolvimento

```python
import requests
import os
import warnings

# Solução: Permitir verify=False APENAS em desenvolvimento
ALLOW_INSECURE_SSL = os.getenv('ALLOW_INSECURE_SSL', 'False').lower() == 'true'

if ALLOW_INSECURE_SSL:
    warnings.warn("SSL verification is disabled. This is INSECURE and should only be used in development!")
    verify_ssl = False
else:
    verify_ssl = True

response = requests.post(
    backup_url,
    json=backup_data,
    verify=verify_ssl,
    timeout=30
)
```

### Implementação

**Passo 1**: Obter certificados SSL válidos (Let's Encrypt, DigiCert, etc.).

**Passo 2**: Configurar variável de ambiente:
```bash
CA_BUNDLE_PATH=/etc/ssl/certs/ca-certificates.crt
```

**Passo 3**: Atualizar o arquivo `disaster_recovery_system.py`.

---

## 🔴 VULNERABILIDADE 3: USO DE HASH MD5 FRACO

### Problema Identificado

**Severidade**: ALTA  
**Arquivos Afetados**: 14 ocorrências em múltiplos arquivos

**Risco**: MD5 é criptograficamente quebrado, possível colisão de hash.

### Solução Proposta

#### Substituir MD5 por SHA-256

```python
import hashlib

# ANTES (INSEGURO)
cpf_hash = hashlib.md5(cpf.encode()).hexdigest()

# DEPOIS (SEGURO)
cpf_hash = hashlib.sha256(cpf.encode()).hexdigest()
```

#### Para Casos de Cache (onde performance é crítica)

```python
import hashlib

# Usar BLAKE2 (mais rápido que SHA-256, mais seguro que MD5)
cache_key = hashlib.blake2b(key_data.encode(), digest_size=16).hexdigest()
```

### Implementação

**Passo 1**: Criar função utilitária para hash seguro:

```python
# backend/utils/security.py
import hashlib

def secure_hash(data: str, algorithm: str = 'sha256') -> str:
    """
    Gera hash seguro de uma string.
    
    Args:
        data: String a ser hasheada
        algorithm: Algoritmo de hash ('sha256', 'sha512', 'blake2b')
    
    Returns:
        Hash hexadecimal da string
    """
    if algorithm == 'sha256':
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(data.encode()).hexdigest()
    elif algorithm == 'blake2b':
        return hashlib.blake2b(data.encode(), digest_size=16).hexdigest()
    else:
        raise ValueError(f"Algoritmo não suportado: {algorithm}")
```

**Passo 2**: Substituir todas as 14 ocorrências de `hashlib.md5()` por `secure_hash()`.

**Passo 3**: Atualizar testes para refletir os novos hashes.

---

## 🔴 VULNERABILIDADE 4: TARFILE EXTRACTION SEM VALIDAÇÃO

### Problema Identificado

**Severidade**: ALTA  
**Arquivos Afetados**: 3  
- `backend/data/external_dataset_integration.py:13`
- `backend/infrastructure/backup_recovery_system.py:327`
- `backend/infrastructure/disaster_recovery_system.py:539`

**Risco**: Path traversal, possível sobrescrita de arquivos do sistema.

### Solução Proposta

#### Validar Membros do Arquivo Antes de Extrair

```python
import tarfile
import os

def safe_extract(tar_path: str, extract_to: str):
    """
    Extrai arquivo tar de forma segura, validando path traversal.
    
    Args:
        tar_path: Caminho do arquivo tar
        extract_to: Diretório de destino
    """
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            # Validar path traversal
            member_path = os.path.join(extract_to, member.name)
            if not member_path.startswith(os.path.abspath(extract_to)):
                raise ValueError(f"Path traversal detectado: {member.name}")
            
            # Validar links simbólicos
            if member.issym() or member.islnk():
                link_target = member.linkname
                if os.path.isabs(link_target):
                    raise ValueError(f"Link absoluto detectado: {member.name} -> {link_target}")
        
        # Se todas as validações passaram, extrair
        tar.extractall(path=extract_to)
```

### Implementação

**Passo 1**: Criar função utilitária para extração segura:

```python
# backend/utils/file_operations.py
import tarfile
import zipfile
import os
from pathlib import Path

def safe_extract_tar(tar_path: str, extract_to: str) -> None:
    """Extrai arquivo tar de forma segura."""
    extract_to = os.path.abspath(extract_to)
    
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extract_to, member.name))
            
            # Validar path traversal
            if not member_path.startswith(extract_to):
                raise ValueError(f"Path traversal detectado: {member.name}")
            
            # Validar links simbólicos
            if member.issym() or member.islnk():
                if os.path.isabs(member.linkname):
                    raise ValueError(f"Link absoluto detectado: {member.name}")
        
        tar.extractall(path=extract_to)

def safe_extract_zip(zip_path: str, extract_to: str) -> None:
    """Extrai arquivo zip de forma segura."""
    extract_to = os.path.abspath(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            member_path = os.path.abspath(os.path.join(extract_to, member))
            
            # Validar path traversal
            if not member_path.startswith(extract_to):
                raise ValueError(f"Path traversal detectado: {member}")
        
        zip_ref.extractall(extract_to)
```

**Passo 2**: Substituir todas as 3 ocorrências de `extractall()` direto por `safe_extract_tar()` ou `safe_extract_zip()`.

---

## 🔴 VULNERABILIDADE 5: HARDCODED SECRETS

### Problema Identificado

**Severidade**: MÉDIA  
**Arquivos Afetados**: 2  
- `backend/api/secure_main_api.py:429` (password)
- `backend/config/settings.py:185` (secret)

**Risco**: Exposição de credenciais no código-fonte.

### Solução Proposta

#### Usar Variáveis de Ambiente

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ANTES (INSEGURO)
password = "minha_senha_123"
secret_key = "chave_secreta_abc"

# DEPOIS (SEGURO)
password = os.getenv('DB_PASSWORD')
secret_key = os.getenv('SECRET_KEY')

if not password or not secret_key:
    raise ValueError("Variáveis de ambiente DB_PASSWORD e SECRET_KEY são obrigatórias")
```

### Implementação

**Passo 1**: Criar arquivo `.env.example`:
```bash
# Database
DB_PASSWORD=your_secure_password_here

# Flask
SECRET_KEY=your_secret_key_here

# API Keys
API_KEY=your_api_key_here
```

**Passo 2**: Adicionar `.env` ao `.gitignore`:
```bash
echo ".env" >> .gitignore
```

**Passo 3**: Atualizar todos os arquivos com secrets hardcoded.

---

## 📊 ROADMAP DE IMPLEMENTAÇÃO

### Fase 1: Correções Críticas (Semana 1)
- [ ] Desabilitar Flask debug mode em produção
- [ ] Habilitar validação SSL
- [ ] Mover secrets para variáveis de ambiente

### Fase 2: Correções de Alta Prioridade (Semana 2)
- [ ] Substituir MD5 por SHA-256
- [ ] Implementar extração segura de arquivos
- [ ] Adicionar testes de segurança

### Fase 3: Validação e Testes (Semana 3)
- [ ] Executar testes de penetração
- [ ] Validar conformidade com PCI DSS
- [ ] Auditoria de segurança externa

---

## ✅ CHECKLIST DE VALIDAÇÃO

- [ ] Todas as ocorrências de `debug=True` foram removidas ou controladas por variável de ambiente
- [ ] Todas as ocorrências de `verify=False` foram removidas ou justificadas
- [ ] Todos os usos de MD5 foram substituídos por SHA-256 ou superior
- [ ] Todas as extrações de arquivos foram validadas contra path traversal
- [ ] Todos os secrets foram movidos para variáveis de ambiente
- [ ] Arquivo `.env.example` foi criado
- [ ] `.env` foi adicionado ao `.gitignore`
- [ ] Testes de segurança foram executados
- [ ] Documentação foi atualizada

---

**Documento preparado por**: Análise Automatizada  
**Data**: 08 de Novembro de 2025  
**Versão**: 1.0  
