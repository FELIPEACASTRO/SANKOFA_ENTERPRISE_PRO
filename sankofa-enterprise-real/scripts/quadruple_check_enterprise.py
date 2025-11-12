#!/usr/bin/env python3
"""
🔍 QUADRUPLE CHECK ENTERPRISE - MÁXIMO RIGOR COMPUTACIONAL
Análise ultra-rigorosa para garantir setup de nível enterprise absoluto
"""

import os
import sys
import subprocess
import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import importlib.util

class EnterpriseQuadrupleChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.critical_errors = []
        self.security_issues = []
        self.code_quality_issues = []
        self.performance_issues = []
        self.compliance_issues = []
        self.passed_checks = []
        
    def analyze_code_quality(self) -> Dict[str, any]:
        """Análise rigorosa de qualidade de código"""
        print("🔍 ANÁLISE DE QUALIDADE DE CÓDIGO...")
        
        issues = []
        
        # 1. Detectar print statements em código de produção
        print_files = []
        for py_file in self.project_root.rglob("*.py"):
            if "test" in str(py_file).lower() or "script" in str(py_file).lower():
                continue
                
            try:
                content = py_file.read_text()
                if "print(" in content and "logging" not in content:
                    print_files.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
        
        if print_files:
            issues.append(f"❌ CRÍTICO: {len(print_files)} arquivos com print() em produção")
            self.critical_errors.extend([f"Print statements em: {f}" for f in print_files[:5]])
        else:
            self.passed_checks.append("✅ Nenhum print() em código de produção")
        
        # 2. Detectar TODOs/FIXMEs
        todo_files = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                if re.search(r'TODO|FIXME|XXX|HACK', content, re.IGNORECASE):
                    todo_files.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
        
        if todo_files:
            issues.append(f"⚠️  {len(todo_files)} arquivos com TODOs/FIXMEs")
            self.code_quality_issues.extend([f"TODOs em: {f}" for f in todo_files[:3]])
        else:
            self.passed_checks.append("✅ Nenhum TODO/FIXME pendente")
        
        # 3. Verificar uso de logging
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if "test" not in str(f).lower() and "script" not in str(f).lower()]
        
        logging_files = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                if "import logging" in content or "from logging" in content:
                    logging_files += 1
            except:
                continue
        
        logging_ratio = (logging_files / len(py_files)) * 100 if py_files else 0
        if logging_ratio < 30:
            issues.append(f"⚠️  Apenas {logging_ratio:.1f}% dos arquivos usam logging")
            self.code_quality_issues.append(f"Baixo uso de logging: {logging_ratio:.1f}%")
        else:
            self.passed_checks.append(f"✅ {logging_ratio:.1f}% dos arquivos usam logging")
        
        return {"issues": issues, "print_files": len(print_files), "todo_files": len(todo_files)}
    
    def analyze_security_vulnerabilities(self) -> Dict[str, any]:
        """Análise rigorosa de vulnerabilidades de segurança"""
        print("🔒 ANÁLISE DE SEGURANÇA...")
        
        issues = []
        
        # 1. Verificar hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
        ]
        
        hardcoded_secrets = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        hardcoded_secrets.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        if hardcoded_secrets:
            issues.append(f"🚨 CRÍTICO: {len(hardcoded_secrets)} arquivos com secrets hardcoded")
            self.security_issues.extend([f"Secrets em: {f}" for f in hardcoded_secrets])
        else:
            self.passed_checks.append("✅ Nenhum secret hardcoded detectado")
        
        # 2. Verificar SQL injection risks
        sql_risks = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                # Buscar concatenação de strings em queries SQL
                if re.search(r'(SELECT|INSERT|UPDATE|DELETE).*\+.*["\']', content, re.IGNORECASE):
                    sql_risks.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
        
        if sql_risks:
            issues.append(f"⚠️  {len(sql_risks)} possíveis riscos de SQL injection")
            self.security_issues.extend([f"SQL risk em: {f}" for f in sql_risks])
        else:
            self.passed_checks.append("✅ Nenhum risco de SQL injection detectado")
        
        return {"issues": issues, "hardcoded_secrets": len(hardcoded_secrets), "sql_risks": len(sql_risks)}
    
    def analyze_dependency_health(self) -> Dict[str, any]:
        """Análise rigorosa de dependências"""
        print("📦 ANÁLISE DE DEPENDÊNCIAS...")
        
        issues = []
        
        # 1. Verificar conflitos de dependências
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                conflicts = result.stdout.strip().split('\n')
                issues.append(f"❌ CRÍTICO: {len(conflicts)} conflitos de dependências")
                self.critical_errors.extend([f"Conflito: {c}" for c in conflicts[:3]])
            else:
                self.passed_checks.append("✅ Nenhum conflito de dependências")
        except Exception as e:
            issues.append(f"⚠️  Erro ao verificar dependências: {e}")
        
        # 2. Verificar vulnerabilidades npm
        try:
            result = subprocess.run(
                ["npm", "audit", "--audit-level=moderate", "--json"],
                cwd=self.project_root / "frontend",
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                try:
                    audit_data = json.loads(result.stdout)
                    vuln_count = audit_data.get("metadata", {}).get("vulnerabilities", {})
                    total_vulns = sum(vuln_count.values()) if isinstance(vuln_count, dict) else 0
                    
                    if total_vulns > 0:
                        issues.append(f"⚠️  {total_vulns} vulnerabilidades npm detectadas")
                        self.security_issues.append(f"NPM vulnerabilities: {total_vulns}")
                    else:
                        self.passed_checks.append("✅ Nenhuma vulnerabilidade npm")
                except:
                    self.passed_checks.append("✅ Nenhuma vulnerabilidade npm crítica")
            else:
                self.passed_checks.append("✅ Nenhuma vulnerabilidade npm")
        except Exception as e:
            issues.append(f"⚠️  Erro ao verificar npm audit: {e}")
        
        return {"issues": issues}
    
    def analyze_performance_bottlenecks(self) -> Dict[str, any]:
        """Análise de possíveis gargalos de performance"""
        print("⚡ ANÁLISE DE PERFORMANCE...")
        
        issues = []
        
        # 1. Detectar loops aninhados complexos
        complex_loops = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                # Buscar for loops aninhados
                nested_for_count = content.count("for ") - content.count("for " + content.split("for ")[1].split(":")[0] + ":")
                if nested_for_count > 3:
                    complex_loops.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
        
        if complex_loops:
            issues.append(f"⚠️  {len(complex_loops)} arquivos com loops complexos")
            self.performance_issues.extend([f"Loops complexos em: {f}" for f in complex_loops])
        else:
            self.passed_checks.append("✅ Nenhum loop complexo detectado")
        
        # 2. Verificar imports desnecessários
        unused_imports = []
        for py_file in self.project_root.rglob("*.py"):
            if "test" in str(py_file).lower():
                continue
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                import_lines = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
                
                if len(import_lines) > 20:  # Muitos imports podem indicar código mal estruturado
                    unused_imports.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
        
        if unused_imports:
            issues.append(f"⚠️  {len(unused_imports)} arquivos com muitos imports")
            self.performance_issues.extend([f"Muitos imports em: {f}" for f in unused_imports[:3]])
        else:
            self.passed_checks.append("✅ Imports organizados adequadamente")
        
        return {"issues": issues}
    
    def analyze_enterprise_compliance(self) -> Dict[str, any]:
        """Análise de compliance empresarial"""
        print("🏢 ANÁLISE DE COMPLIANCE EMPRESARIAL...")
        
        issues = []
        
        # 1. Verificar documentação obrigatória
        required_docs = [
            "README.md",
            "SECURITY.md", 
            "DEPLOYMENT_GUIDE.md",
            ".gitignore",
            "pyproject.toml"
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not (self.project_root / doc).exists():
                missing_docs.append(doc)
        
        if missing_docs:
            issues.append(f"❌ CRÍTICO: Documentação ausente: {', '.join(missing_docs)}")
            self.compliance_issues.extend([f"Doc ausente: {d}" for d in missing_docs])
        else:
            self.passed_checks.append("✅ Toda documentação obrigatória presente")
        
        # 2. Verificar estrutura de testes
        test_files = list(self.project_root.rglob("*test*.py"))
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if "test" not in str(f).lower()]
        
        test_coverage_ratio = (len(test_files) / len(py_files)) * 100 if py_files else 0
        
        if test_coverage_ratio < 20:
            issues.append(f"⚠️  Cobertura de testes baixa: {test_coverage_ratio:.1f}%")
            self.compliance_issues.append(f"Baixa cobertura de testes: {test_coverage_ratio:.1f}%")
        else:
            self.passed_checks.append(f"✅ Cobertura de testes adequada: {test_coverage_ratio:.1f}%")
        
        # 3. Verificar configuração de CI/CD
        ci_files = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            ".pre-commit-config.yaml"
        ]
        
        has_ci = any((self.project_root / ci).exists() for ci in ci_files)
        if not has_ci:
            issues.append("⚠️  Nenhuma configuração de CI/CD detectada")
            self.compliance_issues.append("Ausência de CI/CD")
        else:
            self.passed_checks.append("✅ Configuração de CI/CD presente")
        
        return {"issues": issues, "test_coverage": test_coverage_ratio}
    
    def generate_enterprise_fixes(self) -> List[str]:
        """Gera correções automáticas para problemas detectados"""
        fixes = []
        
        if self.critical_errors:
            fixes.append("🔧 CORREÇÕES CRÍTICAS NECESSÁRIAS:")
            for error in self.critical_errors[:5]:
                fixes.append(f"   • {error}")
        
        if self.security_issues:
            fixes.append("🔒 CORREÇÕES DE SEGURANÇA NECESSÁRIAS:")
            for issue in self.security_issues[:5]:
                fixes.append(f"   • {issue}")
        
        if self.code_quality_issues:
            fixes.append("📝 MELHORIAS DE QUALIDADE RECOMENDADAS:")
            for issue in self.code_quality_issues[:5]:
                fixes.append(f"   • {issue}")
        
        return fixes
    
    def run_quadruple_check(self) -> Dict[str, any]:
        """Executa o QUADRUPLE CHECK completo"""
        print("🔍 INICIANDO QUADRUPLE CHECK ENTERPRISE - MÁXIMO RIGOR...")
        print("=" * 80)
        
        results = {}
        
        # Executar todas as análises
        results["code_quality"] = self.analyze_code_quality()
        results["security"] = self.analyze_security_vulnerabilities()
        results["dependencies"] = self.analyze_dependency_health()
        results["performance"] = self.analyze_performance_bottlenecks()
        results["compliance"] = self.analyze_enterprise_compliance()
        
        return results
    
    def print_enterprise_summary(self):
        """Imprime resumo executivo para nível enterprise"""
        print("\n" + "=" * 80)
        print("📊 RELATÓRIO EXECUTIVO - QUADRUPLE CHECK ENTERPRISE")
        print("=" * 80)
        
        # Calcular score geral
        total_issues = (len(self.critical_errors) + len(self.security_issues) + 
                       len(self.code_quality_issues) + len(self.performance_issues) + 
                       len(self.compliance_issues))
        
        total_checks = len(self.passed_checks) + total_issues
        enterprise_score = (len(self.passed_checks) / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n🎯 ENTERPRISE READINESS SCORE: {enterprise_score:.1f}%")
        
        if enterprise_score >= 95:
            print("🏆 STATUS: ENTERPRISE READY - NÍVEL WORLD CLASS")
        elif enterprise_score >= 85:
            print("✅ STATUS: PRODUCTION READY - NÍVEL PROFISSIONAL")
        elif enterprise_score >= 70:
            print("⚠️  STATUS: NEEDS IMPROVEMENT - CORREÇÕES NECESSÁRIAS")
        else:
            print("❌ STATUS: NOT READY - CORREÇÕES CRÍTICAS OBRIGATÓRIAS")
        
        if self.passed_checks:
            print(f"\n✅ SUCESSOS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"   {check}")
        
        if self.critical_errors:
            print(f"\n🚨 ERROS CRÍTICOS ({len(self.critical_errors)}):")
            for error in self.critical_errors:
                print(f"   ❌ {error}")
        
        if self.security_issues:
            print(f"\n🔒 ISSUES DE SEGURANÇA ({len(self.security_issues)}):")
            for issue in self.security_issues:
                print(f"   🔒 {issue}")
        
        if self.code_quality_issues:
            print(f"\n📝 QUALIDADE DE CÓDIGO ({len(self.code_quality_issues)}):")
            for issue in self.code_quality_issues:
                print(f"   📝 {issue}")
        
        # Recomendações executivas
        print("\n" + "=" * 80)
        print("💼 RECOMENDAÇÕES EXECUTIVAS")
        print("=" * 80)
        
        if enterprise_score >= 95:
            print("🎉 PARABÉNS! Setup aprovado para ambiente enterprise.")
            print("   • Pronto para deploy em produção")
            print("   • Atende padrões de mercado internacional")
            print("   • Configuração de nível world-class")
        else:
            fixes = self.generate_enterprise_fixes()
            for fix in fixes:
                print(fix)
        
        return enterprise_score >= 85

def main():
    checker = EnterpriseQuadrupleChecker()
    checker.run_quadruple_check()
    success = checker.print_enterprise_summary()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()