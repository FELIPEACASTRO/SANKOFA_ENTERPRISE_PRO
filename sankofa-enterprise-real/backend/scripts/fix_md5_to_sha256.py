#!/usr/bin/env python3
"""
Script para substituir hashlib.sha256() por hashlib.sha256() em todos os arquivos Python
Correção de vulnerabilidade de segurança
"""

import os
import re
from pathlib import Path

def fix_md5_in_file(filepath: Path) -> bool:
    """Substitui md5() por sha256() em um arquivo"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verifica se o arquivo usa md5
        if 'hashlib.sha256' not in content:
            return False
        
        # Substitui md5() por sha256()
        modified_content = content.replace('hashlib.sha256()', 'hashlib.sha256()')
        modified_content = modified_content.replace('hashlib.sha256', 'hashlib.sha256')
        
        # Salva o arquivo modificado
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ Corrigido: {filepath}")
        return True
    
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")
        return False

def main():
    """Corrige todos os arquivos Python no backend"""
    backend_dir = Path(__file__).parent.parent
    
    files_fixed = 0
    files_checked = 0
    
    # Percorre todos os arquivos .py
    for py_file in backend_dir.rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        files_checked += 1
        if fix_md5_in_file(py_file):
            files_fixed += 1
    
    print(f"\n📊 Resumo:")
    print(f"   Arquivos verificados: {files_checked}")
    print(f"   Arquivos corrigidos: {files_fixed}")
    print(f"   ✅ MD5 → SHA256 migration complete!")

if __name__ == '__main__':
    main()
