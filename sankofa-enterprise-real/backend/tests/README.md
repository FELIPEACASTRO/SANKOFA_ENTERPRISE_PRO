# 🧪 Sankofa Enterprise Pro - Tests

**Checklist 5.2**: Pirâmide de testes equilibrada

---

## 📁 Estrutura

```
tests/
├── unit/               # Testes unitários (70-80%)
│   ├── test_production_fraud_engine.py
│   ├── test_error_handling.py
│   └── ...
├── integration/        # Testes de integração (15-20%)
│   └── (a implementar)
└── e2e/               # Testes end-to-end (<5%)
    └── (a implementar)
```

---

## 🚀 Executar Testes

### Todos os testes
```bash
cd backend
pytest
```

### Apenas unitários
```bash
pytest tests/unit/ -v
```

### Com cobertura
```bash
pytest --cov=. --cov-report=html
```

### Específico
```bash
pytest tests/unit/test_production_fraud_engine.py::TestProductionFraudEngine::test_fit_creates_model
```

---

## ✅ Status Atual

| Categoria | Cobertura | Status |
|-----------|-----------|--------|
| **Unit Tests** | 2 arquivos | 🟡 Iniciado |
| **Integration Tests** | 0 arquivos | 🔴 Pendente |
| **E2E Tests** | 0 arquivos | 🔴 Pendente |

---

## 📋 Testes Implementados

### ✅ test_production_fraud_engine.py (11 testes)

**Cobertura**:
- ✅ Inicialização
- ✅ Fit e criação de modelo
- ✅ Cálculo de métricas
- ✅ Predict sem fit (erro esperado)
- ✅ Predict retorna binário
- ✅ Predict_proba retorna probabilidades
- ✅ Thresholds diferentes
- ✅ F1-Score razoável
- ✅ Tratamento de NaN

**Checklist atendido**:
- ✅ 5.1 Testabilidade desde o design
- ✅ 5.3 Cobertura de caminhos críticos

---

### ✅ test_error_handling.py (9 testes)

**Cobertura**:
- ✅ Hierarquia de erros
- ✅ Cada tipo de erro customizado
- ✅ Decorator de tratamento
- ✅ Captura de exceções

**Checklist atendido**:
- ✅ 4.3 Tratamento e exposição de erros

---

## 🎯 Próximos Testes

### Alta Prioridade
- [ ] test_kaggle_dataset_downloader.py
- [ ] test_real_data_trainer.py
- [ ] test_feature_engineering.py
- [ ] test_config_management.py

### Média Prioridade
- [ ] test_api_endpoints.py (integration)
- [ ] test_database_operations.py (integration)
- [ ] test_cache_system.py (integration)

### Baixa Prioridade
- [ ] test_frontend_integration.py (e2e)
- [ ] test_full_workflow.py (e2e)

---

## 📊 Meta de Cobertura

| Componente | Meta | Atual |
|------------|------|-------|
| ML Engine | 80% | 15% |
| API | 70% | 0% |
| Utils | 70% | 10% |
| **Total** | **70%** | **~8%** |

---

## 🔧 Configuração

Arquivo: `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
]
```

---

**Última Atualização**: 08 de Novembro de 2025  
**Status**: Testes básicos implementados (20 testes)
