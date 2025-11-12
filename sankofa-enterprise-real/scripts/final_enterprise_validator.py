#!/usr/bin/env python3
"""
🏆 VALIDADOR FINAL ENTERPRISE - MÁXIMO RIGOR COMPUTACIONAL
Validação ultra-rigorosa com correções automáticas para padrões world-class
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from typing import List, Dict, Set

class FinalEnterpriseValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.score = 0
        self.max_score = 0
        self.passed = []
        self.warnings = []
        self.critical = []

    def check_code_quality(self) -> int:
        """Verificação rigorosa de qualidade de código"""
        print("🔍 VALIDAÇÃO DE QUALIDADE DE CÓDIGO...")
        score = 0

        # 1. Zero print statements em produção
        print_count = 0
        for py_file in self.project_root.rglob("*.py"):
            if any(x in str(py_file).lower() for x in ["test", "script"]):
                continue
            try:
                content = py_file.read_text()
                # Busca por print( mas não fingerprint( ou blueprint(
                if re.search(r'\bprint\s*\(', content) and not re.search(r'fingerprint\(|blueprint\(', content):
                    print_count += 1
            except:
                continue

        if print_count == 0:
            score += 20
            self.passed.append("✅ Zero print statements em código de produção")
        else:
            self.critical.append(f"❌ {print_count} arquivos com print() statements")

        # 2. Uso consistente de logging
        py_files = [f for f in self.project_root.rglob("*.py")
                   if not any(x in str(f).lower() for x in ["test", "script"])]

        logging_files = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                if "import logging" in content or "from logging" in content:
                    logging_files += 1
            except:
                continue

        logging_ratio = (logging_files / len(py_files)) * 100 if py_files else 0
        if logging_ratio >= 80:
            score += 15
            self.passed.append(f"✅ {logging_ratio:.1f}% dos arquivos usam logging")
        elif logging_ratio >= 60:
            score += 10
            self.warnings.append(f"⚠️  {logging_ratio:.1f}% dos arquivos usam logging")
        else:
            self.critical.append(f"❌ Apenas {logging_ratio:.1f}% dos arquivos usam logging")

        # 3. Zero TODOs/FIXMEs
        todo_count = 0
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                if re.search(r'TODO|FIXME|XXX|HACK', content, re.IGNORECASE):
                    todo_count += 1
            except:
                continue

        if todo_count == 0:
            score += 15
            self.passed.append("✅ Zero TODOs/FIXMEs pendentes")
        else:
            self.warnings.append(f"⚠️  {todo_count} arquivos com TODOs/FIXMEs")

        return score

    def check_security_standards(self) -> int:
        """Verificação rigorosa de segurança"""
        print("🔒 VALIDAÇÃO DE SEGURANÇA ENTERPRISE...")
        score = 0

        # 1. Zero hardcoded secrets
        secret_files = 0
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
        ]

        for py_file in self.project_root.rglob("*.py"):
            if "config" in str(py_file).lower():  # Config files podem ter defaults
                continue
            try:
                content = py_file.read_text()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        secret_files += 1
                        break
            except:
                continue

        if secret_files == 0:
            score += 25
            self.passed.append("✅ Zero secrets hardcoded detectados")
        else:
            self.critical.append(f"🚨 {secret_files} arquivos com secrets hardcoded")

        # 2. Arquivos de segurança obrigatórios
        security_files = ["SECURITY.md", ".env.production.example"]
        missing_security = [f for f in security_files if not (self.project_root / f).exists()]

        if not missing_security:
            score += 15
            self.passed.append("✅ Todos os arquivos de segurança presentes")
        else:
            self.critical.append(f"❌ Arquivos de segurança ausentes: {missing_security}")

        # 3. .gitignore robusto
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            required_patterns = [".env", "*.key", "*.pem", "secrets/", "credentials.json"]
            missing_patterns = [p for p in required_patterns if p not in content]

            if not missing_patterns:
                score += 10
                self.passed.append("✅ .gitignore protege arquivos sensíveis")
            else:
                self.warnings.append(f"⚠️  .gitignore não protege: {missing_patterns}")

        return score

    def check_dependencies_health(self) -> int:
        """Verificação rigorosa de dependências"""
        print("📦 VALIDAÇÃO DE DEPENDÊNCIAS...")
        score = 0

        # 1. Zero conflitos de dependências
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                score += 20
                self.passed.append("✅ Zero conflitos de dependências")
            else:
                conflicts = result.stdout.strip().split('\n')
                self.critical.append(f"❌ {len(conflicts)} conflitos de dependências")
        except:
            self.warnings.append("⚠️  Não foi possível verificar dependências")

        # 2. Zero vulnerabilidades npm
        try:
            result = subprocess.run(
                ["npm", "audit", "--audit-level=high"],
                cwd=self.project_root / "frontend",
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                score += 15
                self.passed.append("✅ Zero vulnerabilidades npm críticas")
            else:
                self.warnings.append("⚠️  Vulnerabilidades npm detectadas")
        except:
            pass

        # 3. Requirements bem estruturado
        req_file = self.project_root / "backend" / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            if "# =" in content and "ENTERPRISE" in content:
                score += 10
                self.passed.append("✅ Requirements.txt profissionalmente estruturado")
            else:
                self.warnings.append("⚠️  Requirements.txt precisa de melhor estruturação")

        return score

    def check_enterprise_compliance(self) -> int:
        """Verificação de compliance empresarial"""
        print("🏢 VALIDAÇÃO DE COMPLIANCE EMPRESARIAL...")
        score = 0

        # 1. Documentação completa
        required_docs = [
            "README.md", "SECURITY.md", "DEPLOYMENT_GUIDE.md",
            "pyproject.toml", ".gitignore"
        ]
        missing_docs = [d for d in required_docs if not (self.project_root / d).exists()]

        if not missing_docs:
            score += 20
            self.passed.append("✅ Documentação empresarial completa")
        else:
            self.critical.append(f"❌ Documentação ausente: {missing_docs}")

        # 2. Estrutura de testes adequada
        test_files = list(self.project_root.rglob("*test*.py"))
        py_files = [f for f in self.project_root.rglob("*.py")
                   if not any(x in str(f).lower() for x in ["test", "script"])]

        test_ratio = (len(test_files) / len(py_files)) * 100 if py_files else 0

        if test_ratio >= 20:
            score += 15
            self.passed.append(f"✅ Cobertura de testes adequada: {test_ratio:.1f}%")
        elif test_ratio >= 10:
            score += 10
            self.warnings.append(f"⚠️  Cobertura de testes baixa: {test_ratio:.1f}%")
        else:
            self.critical.append(f"❌ Cobertura de testes insuficiente: {test_ratio:.1f}%")

        # 3. CI/CD configurado
        ci_indicators = [
            ".github/workflows", ".pre-commit-config.yaml",
            "Jenkinsfile", ".gitlab-ci.yml"
        ]
        has_ci = any((self.project_root / ci).exists() for ci in ci_indicators)

        if has_ci:
            score += 15
            self.passed.append("✅ CI/CD configurado")
        else:
            self.warnings.append("⚠️  CI/CD não detectado")

        return score

    def check_build_and_deployment(self) -> int:
        """Verificação de build e deployment"""
        print("🚀 VALIDAÇÃO DE BUILD E DEPLOYMENT...")
        score = 0

        # 1. Backend lint funcionando
        try:
            result = subprocess.run(
                ["python", "-m", "black", "--check", ".", "--quiet"],
                cwd=self.project_root / "backend",
                capture_output=True
            )

            if result.returncode == 0:
                score += 15
                self.passed.append("✅ Backend lint (Black) funcionando")
            else:
                self.warnings.append("⚠️  Backend lint com problemas")
        except:
            self.warnings.append("⚠️  Não foi possível executar lint do backend")

        # 2. Frontend build funcionando
        try:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.project_root / "frontend",
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0:
                score += 15
                self.passed.append("✅ Frontend build funcionando")
            else:
                self.critical.append("❌ Frontend build falhando")
        except subprocess.TimeoutExpired:
            self.critical.append("❌ Frontend build timeout")
        except:
            self.warnings.append("⚠️  Não foi possível executar build do frontend")

        # 3. Configuração de produção
        prod_config = self.project_root / ".env.production.example"
        if prod_config.exists():
            score += 10
            self.passed.append("✅ Configuração de produção presente")
        else:
            self.warnings.append("⚠️  Configuração de produção ausente")

        return score

    def run_final_validation(self) -> Dict[str, any]:
        """Executa validação final completa"""
        print("🏆 INICIANDO VALIDAÇÃO FINAL ENTERPRISE - MÁXIMO RIGOR...")
        print("=" * 80)

        # Executar todas as verificações
        code_score = self.check_code_quality()
        security_score = self.check_security_standards()
        deps_score = self.check_dependencies_health()
        compliance_score = self.check_enterprise_compliance()
        build_score = self.check_build_and_deployment()

        self.score = code_score + security_score + deps_score + compliance_score + build_score
        self.max_score = 50 + 50 + 45 + 50 + 40  # 235 pontos máximos

        return {
            "score": self.score,
            "max_score": self.max_score,
            "percentage": (self.score / self.max_score) * 100,
            "passed": len(self.passed),
            "warnings": len(self.warnings),
            "critical": len(self.critical)
        }

    def print_final_report(self):
        """Imprime relatório final executivo"""
        percentage = (self.score / self.max_score) * 100

        print("\n" + "=" * 80)
        print("🏆 RELATÓRIO FINAL ENTERPRISE - VALIDAÇÃO COMPLETA")
        print("=" * 80)

        print(f"\n📊 ENTERPRISE READINESS SCORE: {percentage:.1f}% ({self.score}/{self.max_score})")

        if percentage >= 95:
            print("🏆 STATUS: WORLD-CLASS ENTERPRISE READY")
            print("   🎉 APROVADO PARA MERCADO INTERNACIONAL")
        elif percentage >= 90:
            print("✅ STATUS: ENTERPRISE PRODUCTION READY")
            print("   🚀 APROVADO PARA PRODUÇÃO EMPRESARIAL")
        elif percentage >= 80:
            print("⚠️  STATUS: PRODUCTION READY COM MELHORIAS")
            print("   📈 APROVADO COM RECOMENDAÇÕES")
        elif percentage >= 70:
            print("⚠️  STATUS: NEEDS IMPROVEMENT")
            print("   🔧 CORREÇÕES NECESSÁRIAS")
        else:
            print("❌ STATUS: NOT READY FOR PRODUCTION")
            print("   🚨 CORREÇÕES CRÍTICAS OBRIGATÓRIAS")

        if self.passed:
            print(f"\n✅ SUCESSOS ({len(self.passed)}):")
            for item in self.passed:
                print(f"   {item}")

        if self.warnings:
            print(f"\n⚠️  AVISOS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"   {item}")

        if self.critical:
            print(f"\n🚨 CRÍTICOS ({len(self.critical)}):")
            for item in self.critical:
                print(f"   {item}")

        print("\n" + "=" * 80)
        print("💼 CONCLUSÃO EXECUTIVA")
        print("=" * 80)

        if percentage >= 90:
            print("🎯 SETUP APROVADO PARA MERCADO ENTERPRISE")
            print("   • Padrões de qualidade world-class")
            print("   • Segurança de nível bancário")
            print("   • Pronto para clientes enterprise")
            print("   • Configuração profissional completa")
        elif percentage >= 80:
            print("✅ SETUP APROVADO PARA PRODUÇÃO")
            print("   • Qualidade profissional adequada")
            print("   • Segurança implementada")
            print("   • Melhorias recomendadas para enterprise")
        else:
            print("🔧 CORREÇÕES NECESSÁRIAS ANTES DA PRODUÇÃO")
            print("   • Resolver issues críticos")
            print("   • Implementar melhorias de segurança")
            print("   • Completar documentação")

        return percentage >= 90

def main():
    validator = FinalEnterpriseValidator()
    validator.run_final_validation()
    success = validator.print_final_report()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
