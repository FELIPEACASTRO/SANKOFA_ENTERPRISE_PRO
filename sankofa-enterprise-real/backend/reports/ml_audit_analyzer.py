"""
ML Engine Audit Analyzer
Analisa todos os arquivos .py do ml_engine e gera relatório detalhado
"""

import os
import ast
import json
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

class MLCodeAnalyzer:
    """Analisador de código Python para auditoria de ML Engine"""

    # Métodos principais que indicam funcionalidade ML
    ML_METHODS = {
        'train', 'fit', 'predict', 'predict_proba', 'transform',
        'fit_transform', 'analyze', 'detect', 'score', 'evaluate'
    }

    # Imports opcionais conhecidos
    OPTIONAL_IMPORTS = {
        'tensorflow', 'torch', 'pytorch', 'h2o', 'networkx',
        'scipy', 'statsmodels', 'xgboost', 'lightgbm', 'catboost',
        'torch_geometric', 'transformers', 'sentence_transformers'
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.tree = None
        self.analysis_result = {
            'pode_importar': False,
            'erro_import': None,
            'classes': {},
            'aliases': [],
            'imports_opcionais': [],
            'problemas': []
        }

    def analyze(self) -> Dict[str, Any]:
        """Executa análise completa do arquivo"""

        # 1. Tentar parsear o AST
        if not self._parse_ast():
            return self.analysis_result

        # 2. Verificar se pode ser importado
        self._check_importability()

        # 3. Analisar imports
        self._analyze_imports()

        # 4. Analisar classes
        self._analyze_classes()

        # 5. Buscar aliases (funções factory no final)
        self._find_aliases()

        # 6. Identificar problemas
        self._identify_issues()

        return self.analysis_result

    def _parse_ast(self) -> bool:
        """Tenta parsear o arquivo como AST"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            self.tree = ast.parse(code)
            return True
        except SyntaxError as e:
            self.analysis_result['erro_import'] = f"SyntaxError: {str(e)}"
            self.analysis_result['problemas'].append(f"Erro de sintaxe: {str(e)}")
            return False
        except Exception as e:
            self.analysis_result['erro_import'] = f"Parse error: {str(e)}"
            self.analysis_result['problemas'].append(f"Erro ao parsear: {str(e)}")
            return False

    def _check_importability(self):
        """Verifica se o módulo pode ser importado"""
        try:
            spec = importlib.util.spec_from_file_location("test_module", self.file_path)
            if spec and spec.loader:
                # Não executa o import de fato para evitar side effects
                # Apenas verifica se o spec pode ser criado
                self.analysis_result['pode_importar'] = True
        except Exception as e:
            self.analysis_result['pode_importar'] = False
            self.analysis_result['erro_import'] = str(e)
            self.analysis_result['problemas'].append(f"Não pode ser importado: {str(e)}")

    def _analyze_imports(self):
        """Analisa imports e identifica bibliotecas opcionais"""
        if not self.tree:
            return

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.OPTIONAL_IMPORTS:
                        self.analysis_result['imports_opcionais'].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.OPTIONAL_IMPORTS:
                        self.analysis_result['imports_opcionais'].append(node.module)

            # Detectar try/except ImportError
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type and isinstance(handler.type, ast.Name):
                        if handler.type.id == 'ImportError':
                            # Imports dentro de try/except são opcionais
                            for stmt in node.body:
                                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                                    if isinstance(stmt, ast.Import):
                                        for alias in stmt.names:
                                            self.analysis_result['imports_opcionais'].append(alias.name)
                                    elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                                        self.analysis_result['imports_opcionais'].append(stmt.module)

        # Remove duplicatas
        self.analysis_result['imports_opcionais'] = list(set(self.analysis_result['imports_opcionais']))

    def _analyze_classes(self):
        """Analisa todas as classes definidas no arquivo"""
        if not self.tree:
            return

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                self.analysis_result['classes'][node.name] = class_info

    def _analyze_class(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Analisa uma classe específica"""
        class_info = {
            'metodos': [],
            'metodos_incompletos': [],
            'tem_train': False,
            'tem_predict': False,
            'tem_fit': False,
            'tem_transform': False,
            'tem_analyze': False,
            'tem_detect': False,
            'status': 'completo'
        }

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                class_info['metodos'].append(method_name)

                # Verificar se é um método ML principal
                if method_name in self.ML_METHODS:
                    class_info[f'tem_{method_name}'] = True

                # Verificar se está implementado
                if self._is_method_incomplete(item):
                    class_info['metodos_incompletos'].append(method_name)

        # Determinar status
        if class_info['metodos_incompletos']:
            class_info['status'] = 'incompleto'

        return class_info

    def _is_method_incomplete(self, func_node: ast.FunctionDef) -> bool:
        """Verifica se um método está incompleto (stub)"""
        if not func_node.body:
            return True

        # Ignorar docstrings
        body_without_docstring = []
        for stmt in func_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue  # É docstring
            body_without_docstring.append(stmt)

        if not body_without_docstring:
            return True

        # Verificar se só tem 'pass'
        if len(body_without_docstring) == 1:
            stmt = body_without_docstring[0]
            if isinstance(stmt, ast.Pass):
                return True

            # Verificar 'raise NotImplementedError'
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if stmt.exc.func.id == 'NotImplementedError':
                            return True
                elif isinstance(stmt.exc, ast.Name):
                    if stmt.exc.id == 'NotImplementedError':
                        return True

        return False

    def _find_aliases(self):
        """Busca funções factory/aliases no final do arquivo"""
        if not self.tree:
            return

        # Funções que parecem ser factory/singleton
        factory_patterns = ['get_', 'create_', 'make_', 'build_']

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                if any(func_name.startswith(pattern) for pattern in factory_patterns):
                    self.analysis_result['aliases'].append(func_name)

    def _identify_issues(self):
        """Identifica problemas potenciais no código"""

        # Já adicionamos problemas durante a análise
        # Aqui podemos adicionar verificações extras

        # Verificar se tem TODOs ou FIXMEs
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if 'TODO' in content or 'FIXME' in content:
                # Contar ocorrências
                todo_count = content.count('TODO')
                fixme_count = content.count('FIXME')
                if todo_count > 0:
                    self.analysis_result['problemas'].append(f"Contém {todo_count} TODOs")
                if fixme_count > 0:
                    self.analysis_result['problemas'].append(f"Contém {fixme_count} FIXMEs")
        except:
            pass

def analyze_ml_engine_directory(ml_engine_dir: str) -> Dict[str, Any]:
    """Analisa todos os arquivos .py do diretório ml_engine"""

    ml_engine_path = Path(ml_engine_dir)

    if not ml_engine_path.exists():
        return {
            'error': f'Diretório não encontrado: {ml_engine_dir}',
            'files': {}
        }

    # Encontrar todos os arquivos .py
    py_files = list(ml_engine_path.rglob('*.py'))

    audit_report = {
        'metadata': {
            'audit_date': datetime.now().isoformat(),
            'directory': str(ml_engine_path),
            'total_files_analyzed': len(py_files),
            'analysis_type': 'comprehensive_code_audit'
        },
        'summary': {
            'total_classes': 0,
            'total_methods': 0,
            'incomplete_methods': 0,
            'files_with_issues': 0,
            'optional_imports_unique': set()
        },
        'files': {}
    }

    # Analisar cada arquivo
    for py_file in py_files:
        try:
            analyzer = MLCodeAnalyzer(str(py_file))
            file_analysis = analyzer.analyze()

            # Adicionar ao relatório
            relative_name = py_file.name
            audit_report['files'][relative_name] = file_analysis

            # Atualizar sumário
            for class_name, class_info in file_analysis['classes'].items():
                audit_report['summary']['total_classes'] += 1
                audit_report['summary']['total_methods'] += len(class_info['metodos'])
                audit_report['summary']['incomplete_methods'] += len(class_info['metodos_incompletos'])

            if file_analysis['problemas']:
                audit_report['summary']['files_with_issues'] += 1

            for imp in file_analysis['imports_opcionais']:
                audit_report['summary']['optional_imports_unique'].add(imp)

        except Exception as e:
            print(f"Erro ao analisar {py_file.name}: {e}")
            audit_report['files'][py_file.name] = {
                'pode_importar': False,
                'erro_import': str(e),
                'classes': {},
                'aliases': [],
                'imports_opcionais': [],
                'problemas': [f'Erro na análise: {str(e)}']
            }

    # Converter set para list no sumário
    audit_report['summary']['optional_imports_unique'] = list(
        audit_report['summary']['optional_imports_unique']
    )

    return audit_report

if __name__ == '__main__':
    # Diretório do ml_engine
    ml_engine_dir = r'c:\Users\davis\Workspace\SANKOFA_ENTERPRISE_PRO\sankofa-enterprise-real\backend\ml_engine'

    print(f"Analisando diretório: {ml_engine_dir}")
    print("Isso pode levar alguns minutos...")

    # Executar análise
    audit_report = analyze_ml_engine_directory(ml_engine_dir)

    # Salvar relatório
    output_path = r'c:\Users\davis\Workspace\SANKOFA_ENTERPRISE_PRO\sankofa-enterprise-real\backend\reports\ml_audit_detailed.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    # Imprimir sumário
    print(f"\n{'='*60}")
    print("RELATÓRIO DE AUDITORIA - ML ENGINE")
    print(f"{'='*60}")
    print(f"Total de arquivos analisados: {audit_report['metadata']['total_files_analyzed']}")
    print(f"Total de classes: {audit_report['summary']['total_classes']}")
    print(f"Total de métodos: {audit_report['summary']['total_methods']}")
    print(f"Métodos incompletos: {audit_report['summary']['incomplete_methods']}")
    print(f"Arquivos com problemas: {audit_report['summary']['files_with_issues']}")
    print(f"\nImports opcionais encontrados ({len(audit_report['summary']['optional_imports_unique'])}):")
    for imp in sorted(audit_report['summary']['optional_imports_unique']):
        print(f"  - {imp}")
    print(f"\nRelatório detalhado salvo em:")
    print(f"  {output_path}")
    print(f"{'='*60}\n")
