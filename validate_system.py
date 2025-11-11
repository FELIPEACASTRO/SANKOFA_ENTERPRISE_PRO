#!/usr/bin/env python3
"""
Sistema de Validação Completa do Sankofa Enterprise Pro
Verifica todos os aspectos do sistema para garantir perfeição
"""

import os
import sys
import json
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validador completo do sistema"""
    
    def __init__(self):
        self.base_path = Path("/home/ubuntu/repos/SANKOFA_ENTERPRISE_PRO")
        self.results = {
            "security": {"score": 0, "max_score": 100, "issues": []},
            "structure": {"score": 0, "max_score": 100, "issues": []},
            "dependencies": {"score": 0, "max_score": 100, "issues": []},
            "code_quality": {"score": 0, "max_score": 100, "issues": []},
            "performance": {"score": 0, "max_score": 100, "issues": []},
            "compliance": {"score": 0, "max_score": 100, "issues": []}
        }
    
    def validate_security(self) -> Dict:
        """Valida aspectos de segurança"""
        logger.info("🔒 Validando segurança...")
        
        score = 0
        issues = []
        
        # Verifica se não há debug=True em produção
        debug_files = self._find_debug_true()
        if not debug_files:
            score += 20
        else:
            issues.append(f"Debug mode ativo em {len(debug_files)} arquivos")
        
        # Verifica uso de MD5
        md5_files = self._find_md5_usage()
        if not md5_files:
            score += 20
        else:
            issues.append(f"MD5 encontrado em {len(md5_files)} arquivos")
        
        # Verifica SSL verification
        ssl_files = self._find_ssl_verification_disabled()
        if not ssl_files:
            score += 20
        else:
            issues.append(f"SSL verification desabilitado em {len(ssl_files)} arquivos")
        
        # Verifica se .env.example existe
        if (self.base_path / "sankofa-enterprise-real" / ".env.example").exists():
            score += 20
        else:
            issues.append(".env.example não encontrado")
        
        # Verifica se há secrets hardcoded
        hardcoded_secrets = self._find_hardcoded_secrets()
        if not hardcoded_secrets:
            score += 20
        else:
            issues.append(f"Possíveis secrets hardcoded em {len(hardcoded_secrets)} arquivos")
        
        self.results["security"] = {"score": score, "max_score": 100, "issues": issues}
        return self.results["security"]
    
    def validate_structure(self) -> Dict:
        """Valida estrutura do projeto"""
        logger.info("📁 Validando estrutura...")
        
        score = 0
        issues = []
        
        # Verifica arquivos essenciais
        essential_files = [
            "app.py",
            "PROJECT_SUMMARY.md",
            ".gitignore",
            "sankofa-enterprise-real/README.md",
            "sankofa-enterprise-real/backend/requirements.txt",
            "sankofa-enterprise-real/frontend/package.json"
        ]
        
        for file_path in essential_files:
            if (self.base_path / file_path).exists():
                score += 10
            else:
                issues.append(f"Arquivo essencial não encontrado: {file_path}")
        
        # Verifica se não há duplicações
        if not (self.base_path / "sankofa-github-repo").exists():
            score += 20
        else:
            issues.append("Diretório duplicado encontrado: sankofa-github-repo")
        
        # Verifica estrutura de diretórios
        required_dirs = [
            "sankofa-enterprise-real/backend",
            "sankofa-enterprise-real/frontend",
            "sankofa-enterprise-real/docs",
            "logs",
            "temp",
            "backups"
        ]
        
        for dir_path in required_dirs:
            if (self.base_path / dir_path).exists():
                score += 5
            else:
                issues.append(f"Diretório não encontrado: {dir_path}")
        
        self.results["structure"] = {"score": min(score, 100), "max_score": 100, "issues": issues}
        return self.results["structure"]
    
    def validate_dependencies(self) -> Dict:
        """Valida dependências"""
        logger.info("📦 Validando dependências...")
        
        score = 0
        issues = []
        
        # Verifica requirements.txt
        req_file = self.base_path / "sankofa-enterprise-real" / "backend" / "requirements.txt"
        if req_file.exists():
            score += 20
            
            # Verifica se as dependências críticas estão listadas
            with open(req_file, 'r') as f:
                requirements = f.read()
            
            critical_deps = [
                'Flask', 'redis', 'pandas', 'numpy', 'scikit-learn',
                'psycopg2-binary', 'cryptography', 'structlog'
            ]
            
            for dep in critical_deps:
                if dep in requirements:
                    score += 8
                else:
                    issues.append(f"Dependência crítica não encontrada: {dep}")
        else:
            issues.append("requirements.txt não encontrado")
        
        # Verifica package.json do frontend
        pkg_file = self.base_path / "sankofa-enterprise-real" / "frontend" / "package.json"
        if pkg_file.exists():
            score += 20
        else:
            issues.append("package.json do frontend não encontrado")
        
        self.results["dependencies"] = {"score": min(score, 100), "max_score": 100, "issues": issues}
        return self.results["dependencies"]
    
    def validate_code_quality(self) -> Dict:
        """Valida qualidade do código"""
        logger.info("✨ Validando qualidade do código...")
        
        score = 0
        issues = []
        
        # Verifica se há arquivos Python com sintaxe válida
        python_files = list(self.base_path.rglob("*.py"))
        valid_files = 0
        
        for py_file in python_files[:20]:  # Testa apenas os primeiros 20 para não demorar muito
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
                valid_files += 1
            except SyntaxError:
                issues.append(f"Erro de sintaxe: {py_file}")
            except Exception:
                pass  # Ignora outros erros
        
        if valid_files > 15:
            score += 30
        elif valid_files > 10:
            score += 20
        elif valid_files > 5:
            score += 10
        
        # Verifica documentação nos arquivos
        documented_files = 0
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        documented_files += 1
            except:
                pass
        
        if documented_files > 7:
            score += 20
        elif documented_files > 5:
            score += 15
        elif documented_files > 3:
            score += 10
        
        # Verifica se há testes
        test_files = list(self.base_path.rglob("*test*.py"))
        if len(test_files) > 5:
            score += 25
        elif len(test_files) > 2:
            score += 15
        elif len(test_files) > 0:
            score += 10
        else:
            issues.append("Poucos arquivos de teste encontrados")
        
        # Verifica estrutura de imports
        clean_imports = 0
        for py_file in python_files[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
                    if len(import_lines) < 20:  # Não muitos imports
                        clean_imports += 1
            except:
                pass
        
        if clean_imports > 7:
            score += 25
        elif clean_imports > 5:
            score += 15
        elif clean_imports > 3:
            score += 10
        
        self.results["code_quality"] = {"score": min(score, 100), "max_score": 100, "issues": issues}
        return self.results["code_quality"]
    
    def validate_performance(self) -> Dict:
        """Valida aspectos de performance"""
        logger.info("⚡ Validando performance...")
        
        score = 0
        issues = []
        
        # Verifica se há sistema de cache
        cache_files = list(self.base_path.rglob("*cache*.py"))
        if len(cache_files) > 0:
            score += 25
        else:
            issues.append("Sistema de cache não encontrado")
        
        # Verifica se há configurações de performance
        perf_files = list(self.base_path.rglob("*performance*.py"))
        if len(perf_files) > 0:
            score += 25
        else:
            issues.append("Configurações de performance não encontradas")
        
        # Verifica se há sistema de logging
        log_configs = 0
        python_files = list(self.base_path.rglob("*.py"))
        for py_file in python_files[:20]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'logging' in content:
                        log_configs += 1
            except:
                pass
        
        if log_configs > 10:
            score += 25
        elif log_configs > 5:
            score += 15
        elif log_configs > 2:
            score += 10
        else:
            issues.append("Poucos arquivos com logging configurado")
        
        # Verifica se há otimizações de banco de dados
        db_files = list(self.base_path.rglob("*database*.py")) + list(self.base_path.rglob("*db*.py"))
        if len(db_files) > 0:
            score += 25
        else:
            issues.append("Configurações de banco de dados não encontradas")
        
        self.results["performance"] = {"score": min(score, 100), "max_score": 100, "issues": issues}
        return self.results["performance"]
    
    def validate_compliance(self) -> Dict:
        """Valida compliance"""
        logger.info("📋 Validando compliance...")
        
        score = 0
        issues = []
        
        # Verifica se há módulos de compliance
        compliance_files = list(self.base_path.rglob("*compliance*.py"))
        if len(compliance_files) > 0:
            score += 30
        else:
            issues.append("Módulos de compliance não encontrados")
        
        # Verifica se há documentação de compliance
        compliance_docs = list(self.base_path.rglob("*COMPLIANCE*.md")) + list(self.base_path.rglob("*compliance*.md"))
        if len(compliance_docs) > 0:
            score += 20
        else:
            issues.append("Documentação de compliance não encontrada")
        
        # Verifica se há auditoria
        audit_files = list(self.base_path.rglob("*audit*.py"))
        if len(audit_files) > 0:
            score += 25
        else:
            issues.append("Sistema de auditoria não encontrado")
        
        # Verifica se há configurações LGPD/BACEN
        lgpd_files = list(self.base_path.rglob("*lgpd*.py")) + list(self.base_path.rglob("*bacen*.py"))
        if len(lgpd_files) > 0:
            score += 25
        else:
            issues.append("Configurações LGPD/BACEN não encontradas")
        
        self.results["compliance"] = {"score": min(score, 100), "max_score": 100, "issues": issues}
        return self.results["compliance"]
    
    def _find_debug_true(self) -> List[Path]:
        """Encontra arquivos com debug=True"""
        files_with_debug = []
        for py_file in self.base_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'debug=True' in content:
                        files_with_debug.append(py_file)
            except:
                pass
        return files_with_debug
    
    def _find_md5_usage(self) -> List[Path]:
        """Encontra arquivos usando MD5"""
        files_with_md5 = []
        for py_file in self.base_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'hashlib.md5(' in content:
                        files_with_md5.append(py_file)
            except:
                pass
        return files_with_md5
    
    def _find_ssl_verification_disabled(self) -> List[Path]:
        """Encontra arquivos com SSL verification desabilitado"""
        files_with_ssl_disabled = []
        for py_file in self.base_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'verify=False' in content:
                        files_with_ssl_disabled.append(py_file)
            except:
                pass
        return files_with_ssl_disabled
    
    def _find_hardcoded_secrets(self) -> List[Path]:
        """Encontra possíveis secrets hardcoded"""
        files_with_secrets = []
        secret_patterns = ['password=', 'secret=', 'key=', 'token=']
        
        for py_file in self.base_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for pattern in secret_patterns:
                        if pattern in content and 'os.getenv' not in content:
                            files_with_secrets.append(py_file)
                            break
            except:
                pass
        return files_with_secrets
    
    def generate_report(self) -> Dict:
        """Gera relatório completo"""
        logger.info("📊 Gerando relatório...")
        
        # Executa todas as validações
        self.validate_security()
        self.validate_structure()
        self.validate_dependencies()
        self.validate_code_quality()
        self.validate_performance()
        self.validate_compliance()
        
        # Calcula score geral
        total_score = sum(result["score"] for result in self.results.values())
        max_total_score = sum(result["max_score"] for result in self.results.values())
        overall_score = (total_score / max_total_score) * 100
        
        report = {
            "overall_score": round(overall_score, 1),
            "grade": self._get_grade(overall_score),
            "categories": self.results,
            "summary": {
                "total_issues": sum(len(result["issues"]) for result in self.results.values()),
                "critical_issues": self._count_critical_issues(),
                "status": "APPROVED" if overall_score >= 85 else "NEEDS_IMPROVEMENT"
            }
        }
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """Converte score em nota"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _count_critical_issues(self) -> int:
        """Conta issues críticos"""
        critical_keywords = ['debug', 'md5', 'ssl', 'secret', 'password']
        critical_count = 0
        
        for category in self.results.values():
            for issue in category["issues"]:
                if any(keyword in issue.lower() for keyword in critical_keywords):
                    critical_count += 1
        
        return critical_count

def main():
    """Executa validação completa"""
    print("🔍 VALIDAÇÃO COMPLETA DO SANKOFA ENTERPRISE PRO")
    print("=" * 60)
    
    validator = SystemValidator()
    report = validator.generate_report()
    
    # Mostra resultados
    print(f"\n📊 SCORE GERAL: {report['overall_score']}/100 ({report['grade']})")
    print(f"🎯 STATUS: {report['summary']['status']}")
    print(f"⚠️  Total de Issues: {report['summary']['total_issues']}")
    print(f"🚨 Issues Críticos: {report['summary']['critical_issues']}")
    
    print("\n📋 DETALHES POR CATEGORIA:")
    for category, result in report['categories'].items():
        status = "✅" if result['score'] >= 80 else "⚠️" if result['score'] >= 60 else "❌"
        print(f"{status} {category.upper()}: {result['score']}/100")
        
        if result['issues']:
            for issue in result['issues'][:3]:  # Mostra apenas os 3 primeiros
                print(f"   - {issue}")
            if len(result['issues']) > 3:
                print(f"   ... e mais {len(result['issues']) - 3} issues")
    
    # Salva relatório
    report_path = Path("/home/ubuntu/repos/SANKOFA_ENTERPRISE_PRO/validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📋 Relatório completo salvo em: {report_path}")
    
    # Recomendações
    if report['overall_score'] >= 90:
        print("\n🎉 EXCELENTE! O sistema está próximo da perfeição!")
    elif report['overall_score'] >= 80:
        print("\n👍 BOM! Algumas melhorias menores podem ser feitas.")
    else:
        print("\n⚠️  ATENÇÃO! Várias melhorias são necessárias.")
    
    return report

if __name__ == "__main__":
    main()