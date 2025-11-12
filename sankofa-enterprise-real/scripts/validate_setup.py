#!/usr/bin/env python3
"""
🔍 VALIDADOR DE SETUP PROFISSIONAL - SANKOFA ENTERPRISE PRO
Verifica se o ambiente está configurado corretamente para produção.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple

class SetupValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def check_python_version(self) -> bool:
        """Verifica se está usando Python 3.12+"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 12:
            self.passed.append("✅ Python 3.12+ detectado")
            return True
        else:
            self.errors.append(f"❌ Python {version.major}.{version.minor} detectado. Requer Python 3.12+")
            return False
    
    def check_environment_files(self) -> bool:
        """Verifica configuração de arquivos de ambiente"""
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        env_prod_example = self.project_root / ".env.production.example"
        
        issues = []
        
        if not env_example.exists():
            issues.append("❌ Arquivo .env.example não encontrado")
        
        if not env_prod_example.exists():
            issues.append("❌ Arquivo .env.production.example não encontrado")
        
        if env_file.exists():
            # Verificar se não há secrets hardcoded
            content = env_file.read_text()
            dangerous_patterns = [
                "SECRET_KEY=dev-secret",
                "JWT_SECRET_KEY=jwt-secret",
                "password=postgres",
                "password=123",
                "FLASK_DEBUG=True"
            ]
            
            for pattern in dangerous_patterns:
                if pattern.lower() in content.lower():
                    issues.append(f"⚠️  Possível configuração insegura encontrada: {pattern}")
        
        if issues:
            self.warnings.extend(issues)
            return False
        else:
            self.passed.append("✅ Arquivos de ambiente configurados corretamente")
            return True
    
    def check_dependencies(self) -> bool:
        """Verifica se as dependências estão atualizadas"""
        requirements_file = self.project_root / "backend" / "requirements.txt"
        
        if not requirements_file.exists():
            self.errors.append("❌ Arquivo requirements.txt não encontrado")
            return False
        
        # Verificar se há versões específicas para dependências críticas
        content = requirements_file.read_text()
        critical_deps = ["Flask", "SQLAlchemy", "cryptography", "requests"]
        
        for dep in critical_deps:
            if dep.lower() not in content.lower():
                self.warnings.append(f"⚠️  Dependência crítica {dep} não encontrada")
        
        self.passed.append("✅ Arquivo de dependências encontrado")
        return True
    
    def check_security_files(self) -> bool:
        """Verifica se arquivos de segurança existem"""
        security_files = [
            "SECURITY.md",
            ".gitignore"
        ]
        
        missing = []
        for file in security_files:
            if not (self.project_root / file).exists():
                missing.append(file)
        
        if missing:
            self.errors.append(f"❌ Arquivos de segurança ausentes: {', '.join(missing)}")
            return False
        else:
            self.passed.append("✅ Arquivos de segurança presentes")
            return True
    
    def check_git_status(self) -> bool:
        """Verifica se o git está limpo"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=self.project_root,
                capture_output=True, 
                text=True
            )
            
            if result.stdout.strip():
                self.warnings.append("⚠️  Git working directory não está limpo")
                return False
            else:
                self.passed.append("✅ Git working directory limpo")
                return True
        except Exception as e:
            self.warnings.append(f"⚠️  Erro ao verificar git status: {e}")
            return False
    
    def check_package_json(self) -> bool:
        """Verifica configuração do package.json"""
        package_file = self.project_root / "frontend" / "package.json"
        
        if not package_file.exists():
            self.errors.append("❌ package.json não encontrado")
            return False
        
        try:
            with open(package_file) as f:
                package_data = json.load(f)
            
            # Verificar nome do projeto
            if "booster" in package_data.get("name", "").lower():
                self.warnings.append("⚠️  Nome do projeto ainda contém 'booster'")
            
            # Verificar versão
            if package_data.get("version") == "0.0.0":
                self.warnings.append("⚠️  Versão do projeto ainda é 0.0.0")
            
            self.passed.append("✅ package.json configurado")
            return True
            
        except json.JSONDecodeError:
            self.errors.append("❌ package.json inválido")
            return False
    
    def check_build_process(self) -> bool:
        """Verifica se o build funciona"""
        try:
            # Verificar build do frontend
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.project_root / "frontend",
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                self.passed.append("✅ Build do frontend funcionando")
                return True
            else:
                self.errors.append(f"❌ Build do frontend falhou: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.errors.append("❌ Build do frontend timeout (>2min)")
            return False
        except Exception as e:
            self.errors.append(f"❌ Erro no build: {e}")
            return False
    
    def run_all_checks(self) -> Dict[str, any]:
        """Executa todas as verificações"""
        print("🔍 INICIANDO VALIDAÇÃO PROFISSIONAL DO SETUP...")
        print("=" * 60)
        
        checks = [
            ("Versão do Python", self.check_python_version),
            ("Arquivos de Ambiente", self.check_environment_files),
            ("Dependências", self.check_dependencies),
            ("Arquivos de Segurança", self.check_security_files),
            ("Status do Git", self.check_git_status),
            ("Configuração Frontend", self.check_package_json),
            ("Processo de Build", self.check_build_process),
        ]
        
        results = {}
        for check_name, check_func in checks:
            print(f"\n🔍 Verificando: {check_name}")
            try:
                results[check_name] = check_func()
            except Exception as e:
                self.errors.append(f"❌ Erro em {check_name}: {e}")
                results[check_name] = False
        
        return results
    
    def print_summary(self):
        """Imprime resumo dos resultados"""
        print("\n" + "=" * 60)
        print("📊 RESUMO DA VALIDAÇÃO")
        print("=" * 60)
        
        if self.passed:
            print(f"\n✅ SUCESSOS ({len(self.passed)}):")
            for item in self.passed:
                print(f"   {item}")
        
        if self.warnings:
            print(f"\n⚠️  AVISOS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"   {item}")
        
        if self.errors:
            print(f"\n❌ ERROS CRÍTICOS ({len(self.errors)}):")
            for item in self.errors:
                print(f"   {item}")
        
        print("\n" + "=" * 60)
        
        if self.errors:
            print("🚨 SETUP NÃO ESTÁ PRONTO PARA PRODUÇÃO")
            print("   Corrija os erros críticos antes de prosseguir.")
            return False
        elif self.warnings:
            print("⚠️  SETUP FUNCIONAL MAS COM MELHORIAS NECESSÁRIAS")
            print("   Considere resolver os avisos para melhor qualidade.")
            return True
        else:
            print("🎉 SETUP PROFISSIONAL VALIDADO COM SUCESSO!")
            print("   Pronto para uso em produção.")
            return True

def main():
    validator = SetupValidator()
    validator.run_all_checks()
    success = validator.print_summary()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()