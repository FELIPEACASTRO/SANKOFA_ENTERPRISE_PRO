#!/usr/bin/env python3
"""
Script para corrigir vulnerabilidades de segurança no Sankofa Enterprise Pro
- Substitui SHA256 por SHA256
- Corrige debug=True em produção
- Corrige verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados em requests
"""

import os
import re
import glob
from pathlib import Path

def fix_sha256_to_sha256(file_path):
    """Substitui sha256() por sha256() em um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Substitui hashlib.sha256() por hashlib.sha256()
        content = re.sub(r'hashlib\.sha256\(', 'hashlib.sha256(', content)
        
        # Substitui hash_sha256 por hash_sha256 em nomes de variáveis
        content = re.sub(r'\bhash_sha256\b', 'hash_sha256', content)
        
        # Substitui comentários que mencionam SHA256
        content = re.sub(r'SHA256', 'SHA256', content)
        content = re.sub(r'sha256', 'sha256', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido SHA256 → SHA256: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Erro ao processar {file_path}: {e}")
        return False

def fix_debug_mode(file_path):
    """Corrige debug=True para usar configuração segura"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Substitui debug=True por configuração segura
        if 'app.run(' in content and 'debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"' in content:
            # Adiciona imports necessários se não existirem
            if 'import os' not in content:
                content = 'import os\n' + content
            
            # Substitui debug=os.getenv("FLASK_DEBUG", "False").lower() == "true" por configuração condicional
            content = re.sub(
                r'app\.run\([^)]*debug=True[^)]*\)',
                lambda m: m.group(0).replace(
                    'debug=True', 
                    'debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"'
                ),
                content
            )
            
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido debug mode: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Erro ao processar {file_path}: {e}")
        return False

def fix_ssl_verification(file_path):
    """Corrige verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados para usar configuração segura"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Substitui verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados por configuração condicional
        if 'verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados' in content:
            # Adiciona imports necessários se não existirem
            if 'import os' not in content:
                content = 'import os\n' + content
            
            # Substitui verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados por configuração condicional
            content = re.sub(
                r'verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados',
                'verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados',
                content
            )
            
            # Adiciona comentário explicativo
            content = re.sub(
                r'verify=os\.getenv\("VERIFY_SSL_CERTS", "true"\)\.lower\(\) == "true"',
                'verify=os.getenv("VERIFY_SSL_CERTS", "true").lower() == "true"  # Use False apenas em dev com certs auto-assinados  # Use False apenas em dev com certs auto-assinados',
                content
            )
            
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corrigido SSL verification: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Erro ao processar {file_path}: {e}")
        return False

def main():
    """Executa todas as correções de segurança"""
    print("🔒 Iniciando correções de segurança...")
    
    # Encontra todos os arquivos Python
    base_path = "/home/ubuntu/repos/SANKOFA_ENTERPRISE_PRO"
    python_files = []
    
    for root, dirs, files in os.walk(base_path):
        # Ignora diretórios desnecessários
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Encontrados {len(python_files)} arquivos Python")
    
    # Contadores
    sha256_fixes = 0
    debug_fixes = 0
    ssl_fixes = 0
    
    # Aplica correções
    for file_path in python_files:
        if fix_sha256_to_sha256(file_path):
            sha256_fixes += 1
        
        if fix_debug_mode(file_path):
            debug_fixes += 1
            
        if fix_ssl_verification(file_path):
            ssl_fixes += 1
    
    print("\n📊 Resumo das correções:")
    print(f"   🔐 SHA256 → SHA256: {sha256_fixes} arquivos")
    print(f"   🐛 Debug mode: {debug_fixes} arquivos")
    print(f"   🔒 SSL verification: {ssl_fixes} arquivos")
    
    print("\n✅ Correções de segurança concluídas!")
    
    # Cria arquivo de relatório
    report_path = os.path.join(base_path, "security_fixes_report.md")
    with open(report_path, 'w') as f:
        f.write("# Relatório de Correções de Segurança\n\n")
        f.write(f"**Data**: {os.popen('date').read().strip()}\n\n")
        f.write("## Correções Aplicadas\n\n")
        f.write(f"- **SHA256 → SHA256**: {sha256_fixes} arquivos corrigidos\n")
        f.write(f"- **Debug Mode**: {debug_fixes} arquivos corrigidos\n")
        f.write(f"- **SSL Verification**: {ssl_fixes} arquivos corrigidos\n\n")
        f.write("## Próximos Passos\n\n")
        f.write("1. Testar a aplicação após as correções\n")
        f.write("2. Atualizar variáveis de ambiente conforme necessário\n")
        f.write("3. Executar testes de segurança\n")
        f.write("4. Fazer commit das alterações\n")
    
    print(f"📋 Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()