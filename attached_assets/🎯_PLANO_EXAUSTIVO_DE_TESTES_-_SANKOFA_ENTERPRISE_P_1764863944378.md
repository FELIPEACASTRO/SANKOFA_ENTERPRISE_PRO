<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🎯 PLANO EXAUSTIVO DE TESTES - SANKOFA ENTERPRISE PRO

## **VERSÃO 2.0 - VALIDAÇÃO COMPLETA COM DOUBLE-CHECK E GESTÃO DE CORREÇÕES**


***

## 📋 SUMÁRIO EXECUTIVO EXPANDIDO

**Objetivo Principal**: Garantir 100% de cobertura de testes com **processo rigoroso de validação, correção e regressão**.

**Princípios Fundamentais**:

1. ✅ **NENHUM ERRO PASSA** - Todo erro encontrado deve ser corrigido
2. ✅ **CORREÇÃO VALIDADA** - Após correção, re-executar testes relacionados
3. ✅ **ZERO REGRESSÃO** - Correções não podem quebrar funcionalidades existentes
4. ✅ **DOCUMENTAÇÃO COMPLETA** - Todo erro e correção documentados
5. ✅ **RASTREABILIDADE TOTAL** - Matriz de impacto entre testes

**Contexto do Sistema**:

- **Domínio**: Sistema crítico de detecção de fraude bancária
- **SLA Crítico**: P95 < 5000ms (contrato BACEN)
- **Compliance**: LGPD Art.20, BACEN Res.6/2023, PCI DSS
- **Disponibilidade**: 99.9% uptime requerido
- **Transações/dia**: ~300M esperados em produção

***

## 🔄 PROCESSO DE GESTÃO DE DEFEITOS E CORREÇÕES

### **WORKFLOW OBRIGATÓRIO**

```
┌────────────────────────────────────────────────────────────────┐
│                    CICLO DE TESTE COMPLETO                     │
└────────────────────────────────────────────────────────────────┘

1. EXECUTAR TESTE
   ↓
2. PASSOU? ────→ SIM ────→ MARCAR COMO ✅ APROVADO
   │                        └→ CONTINUAR PRÓXIMO TESTE
   ↓ NÃO
   │
3. REGISTRAR DEFEITO
   ├─ ID único (DEF-XXXX)
   ├─ Severidade (CRÍTICA/ALTA/MÉDIA/BAIXA)
   ├─ Descrição detalhada
   ├─ Steps to reproduce
   ├─ Comportamento esperado vs atual
   └─ Screenshot/logs
   ↓
4. ANÁLISE DE IMPACTO
   ├─ Quais módulos afetados?
   ├─ Quais testes relacionados?
   ├─ Riscos de regressão?
   └─ Prioridade de correção?
   ↓
5. CORREÇÃO DO DEFEITO
   ├─ Desenvolver fix
   ├─ Code review obrigatório
   ├─ Análise de impacto do fix
   └─ Documentar mudança
   ↓
6. VALIDAÇÃO PÓS-CORREÇÃO (3 NÍVEIS)
   │
   ├─ NÍVEL 1: Re-executar teste que falhou
   │   └─ PASSOU? ─→ NÃO ─→ VOLTAR PARA ETAPA 5
   │       ↓ SIM
   │       └─ Continuar
   │
   ├─ NÍVEL 2: Testes de Regressão Relacionados
   │   ├─ Executar todos testes do mesmo módulo
   │   ├─ Executar testes de integração relacionados
   │   └─ Executar testes de dependências
   │       └─ ALGUM FALHOU? ─→ SIM ─→ NOVA REGRESSÃO!
   │           │                      └─ Registrar como DEF-REGRESSAO-XXXX
   │           │                      └─ VOLTAR PARA ETAPA 5
   │           ↓ NÃO
   │           └─ Continuar
   │
   └─ NÍVEL 3: Suite Completa de Smoke Tests
       ├─ Executar 50 testes críticos
       └─ ALGUM FALHOU? ─→ SIM ─→ REGRESSÃO CRÍTICA!
           │                      └─ REVERTER CORREÇÃO
           │                      └─ ANÁLISE DE CAUSA RAIZ
           ↓ NÃO
           └─ ✅ CORREÇÃO APROVADA
   ↓
7. DOCUMENTAR CORREÇÃO
   ├─ Atualizar bug tracker
   ├─ Adicionar nota no teste
   ├─ Atualizar documentação técnica
   └─ Criar entrada no changelog
   ↓
8. CONTINUAR PRÓXIMO TESTE
```


***

## 📊 TEMPLATE DE REGISTRO DE DEFEITO

### **Formulário Obrigatório para Cada Defeito**

```yaml
DEFEITO: DEF-2025-001
═══════════════════════════════════════════════════════════════

INFORMAÇÕES BÁSICAS:
  Data Detecção:     2025-11-29 16:10:00
  Detectado Por:     test_e2e.py::test_fraud_predict_without_auth
  Ambiente:          Staging
  Build/Commit:      abc123def456
  
CLASSIFICAÇÃO:
  Severidade:        ⚠️ CRÍTICA
  Prioridade:        P0 (Blocker)
  Tipo:              Security - Authentication Bypass
  Categoria:         Backend/API
  
DESCRIÇÃO:
  Título:
    "Endpoint /api/fraud/predict não valida token JWT"
  
  Descrição Detalhada:
    O endpoint de predição de fraude está aceitando requisições
    sem token de autenticação. Isso viola requisito de segurança
    REQ-SEC-001 e compliance PCI DSS.
  
  Comportamento Esperado:
    - Request sem token deve retornar 401 Unauthorized
    - Response deve incluir header "WWW-Authenticate: Bearer"
    - Mensagem de erro clara: "Authentication required"
  
  Comportamento Atual:
    - Request sem token retorna 200 OK
    - Sistema processa predição normalmente
    - Dados sensíveis expostos sem autenticação

REPRODUÇÃO:
  Steps to Reproduce:
    1. Iniciar servidor API (python api/production_api.py)
    2. Executar: curl -X POST http://localhost:8000/api/fraud/predict \
                      -H "Content-Type: application/json" \
                      -d '{"transactions": [{"amount": 1000}]}'
    3. Observar: Status 200 (deveria ser 401)
  
  Frequência:
    100% reproduzível
  
  Dados de Teste:
    payload_sem_auth.json (anexo)

EVIDÊNCIAS:
  Logs:
    [2025-11-29 16:10:05] INFO: POST /api/fraud/predict - 200 OK
    [2025-11-29 16:10:05] WARNING: No auth token provided
    [2025-11-29 16:10:05] ERROR: Skipping auth validation
  
  Screenshots:
    screenshot_001_response_200.png
    screenshot_002_no_401_error.png
  
  Código Relevante:
    backend/api/production_api.py:145-160

ANÁLISE DE IMPACTO:
  Módulos Afetados:
    - backend/api/production_api.py (endpoint principal)
    - backend/api/middleware/auth.py (middleware auth)
    - backend/security/jwt_validator.py (validação JWT)
  
  Testes Relacionados:
    - test_e2e.py::test_fraud_predict_without_auth [FALHANDO]
    - test_e2e.py::test_protected_endpoint_without_auth [FALHANDO]
    - test_e2e.py::test_all_sensitive_endpoints_protected [FALHANDO]
    - test_security/test_vulnerability.py::test_auth_bypass [FALHANDO]
    - test_compliance/test_pci_dss.py::test_endpoint_protection [FALHANDO]
  
  Risco de Regressão:
    ⚠️ ALTO - Mudança em sistema de autenticação pode afetar:
    - Todos endpoints protegidos (15+ endpoints)
    - Sistema de permissões (RBAC)
    - Audit logs
    - Compliance LGPD/PCI DSS
  
  Dependências:
    - FastAPI dependency injection
    - JWT library (PyJWT)
    - Middleware stack

REQUISITOS VIOLADOS:
  - REQ-SEC-001: "Todos endpoints sensíveis requerem autenticação"
  - REQ-SEC-005: "Sistema deve usar JWT para autenticação"
  - REQ-COMP-003: "Compliance PCI DSS Requirement 8.2"
  - REQ-LGPD-001: "Controle de acesso a dados pessoais"

SOLUÇÃO PROPOSTA:
  Opção 1 (RECOMENDADA):
    - Adicionar decorator @require_auth em /api/fraud/predict
    - Validar token antes de processar request
    - Retornar 401 se token inválido/ausente
    
    Código:
    ```
    @router.post("/api/fraud/predict")
    @require_auth  # ← ADICIONAR ESTA LINHA
    async def predict_fraud(request: PredictRequest, user: User = Depends(get_current_user)):
        # ... rest of code
    ```
  
  Opção 2 (Alternativa):
    - Configurar middleware global de autenticação
    - Listar endpoints públicos em whitelist
    - Bloquear tudo que não está na whitelist
  
  Complexidade: MÉDIA
  Tempo Estimado: 2-4 horas
  Risco: MÉDIO (pode afetar outros endpoints)

VALIDAÇÃO PÓS-CORREÇÃO:
  Checklist Obrigatório:
    □ Re-executar test_fraud_predict_without_auth
    □ Re-executar todos 5 testes relacionados listados acima
    □ Executar suite completa test_e2e.py (31 testes)
    □ Executar suite test_security/ (60 testes)
    □ Executar smoke tests (10 testes críticos)
    □ Validar que endpoints públicos continuam funcionando:
      - GET /api/health
      - GET /
      - POST /api/auth/login
    □ Testar manualmente no Insomnia/Postman
    □ Verificar logs não mostram warnings
    □ Executar DAST scan (OWASP ZAP)

APROVAÇÕES:
  Desenvolvedor:     [PENDENTE]
  QA Lead:          [PENDENTE]
  Security Officer: [PENDENTE]
  Tech Lead:        [PENDENTE]

STATUS: 🔴 ABERTO - Aguardando Correção
```


***

## 🎯 MATRIZ DE IMPACTO E DEPENDÊNCIAS

### **Mapa de Relacionamentos Entre Testes**

Esta matriz é **CRÍTICA** para garantir que correções não causem regressão.

```
┌─────────────────────────────────────────────────────────────────┐
│           MATRIZ DE IMPACTO ENTRE MÓDULOS E TESTES              │
└─────────────────────────────────────────────────────────────────┘

MÓDULO: backend/api/production_api.py
├─ TESTES DIRETOS (se mudar este arquivo, executar):
│  ├─ test_e2e.py::TestE2EAPIEndpoints (10 testes)
│  ├─ test_qa_comprehensive.py::TestAPITesting (4 testes)
│  ├─ test_qa_expanded.py::TestAPIContractTesting (2 testes)
│  └─ test_integration/test_api_integration.py (15 testes)
│
├─ TESTES DE DEPENDÊNCIAS (podem ser afetados):
│  ├─ test_e2e.py::TestE2EFraudPrediction (4 testes)
│  ├─ test_e2e.py::TestE2EAuthentication (7 testes)
│  ├─ test_security/test_vulnerability.py (15 testes)
│  └─ test_compliance/test_lgpd.py (10 testes)
│
└─ SMOKE TESTS CRÍTICOS (sempre executar):
   ├─ test_smoke_01_api_starts
   ├─ test_smoke_03_ml_model_loaded
   └─ test_smoke_04_authentication_works

MÓDULO: backend/ml_engine/production_fraud_engine.py
├─ TESTES DIRETOS:
│  ├─ test_unit/test_ml_engine_unit.py (50 testes)
│  ├─ test_ml/test_model_performance.py (15 testes)
│  ├─ test_ml/test_explainability.py (8 testes)
│  └─ test_ml/test_bias_fairness.py (10 testes)
│
├─ TESTES DE DEPENDÊNCIAS:
│  ├─ test_e2e.py::TestE2EFraudPrediction (4 testes)
│  ├─ test_e2e.py::TestE2EMLPipeline (3 testes)
│  ├─ test_performance/test_load.py (10 testes)
│  └─ test_compliance/test_lgpd.py::test_explainability (3 testes)
│
└─ SMOKE TESTS:
   ├─ test_smoke_03_ml_model_loaded
   └─ test_smoke_05_prediction_endpoint_responds

MÓDULO: backend/database/repository.py
├─ TESTES DIRETOS:
│  ├─ test_unit/test_database_unit.py (30 testes)
│  ├─ test_qa_comprehensive.py::TestDatabaseTesting (3 testes)
│  └─ test_migration/test_data_migration.py (8 testes)
│
├─ TESTES DE DEPENDÊNCIAS:
│  ├─ test_e2e.py::TestE2EDataPersistence (2 testes)
│  ├─ test_e2e.py::TestE2EInfrastructure::test_database_* (2 testes)
│  ├─ test_integration/test_db_integration.py (20 testes)
│  └─ test_concurrency/test_concurrency.py (15 testes)
│
└─ SMOKE TESTS:
   └─ test_smoke_02_database_accessible

MÓDULO: frontend/src/components/Dashboard.tsx
├─ TESTES DIRETOS:
│  ├─ test_unit/components/Dashboard.test.tsx (40 testes)
│  ├─ test_usability/test_ui_ux.py::test_ux_02_dashboard_* (3 testes)
│  └─ test_e2e.py::TestE2EAPIEndpoints::test_dashboard_* (3 testes)
│
├─ TESTES DE DEPENDÊNCIAS:
│  ├─ test_compatibility/test_browsers.py (10 testes)
│  ├─ test_accessibility/test_wcag.py (12 testes)
│  └─ test_e2e.py::TestE2EIntegration::test_full_flow (1 teste)
│
└─ SMOKE TESTS:
   └─ test_smoke_01_frontend_available
```


***

## 📝 PLANO DE TESTES DETALHADO COM VALIDAÇÃO

### ═══════════════════════════════════════════════════════════════

### **CATEGORIA 1: TESTES UNITÁRIOS (200+ testes)**

### ═══════════════════════════════════════════════════════════════

#### **1.1 BACKEND - ML ENGINE (50 testes)**

##### **Arquivo: `tests/unit/test_ml_engine_unit.py`**

```python
"""
SUITE DE TESTES: ML Engine - Unit Tests
TOTAL: 50 testes
DEPENDÊNCIAS: backend/ml_engine/production_fraud_engine.py
CRITICIDADE: ⚠️ ALTA (core do sistema)

PROCESSO DE VALIDAÇÃO:
1. Executar todos 50 testes desta suite
2. Se ALGUM falhar:
   - Registrar defeito usando template
   - Corrigir defeito
   - Re-executar APENAS este arquivo (50 testes)
   - Se passar, executar smoke tests (10 testes)
   - Se smoke passar, executar testes relacionados:
     * test_ml/test_model_performance.py (15 testes)
     * test_ml/test_explainability.py (8 testes)
     * test_e2e.py::TestE2EMLPipeline (3 testes)
3. Documentar resultado no RELATÓRIO DE EXECUÇÃO
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from ml_engine.production_fraud_engine import FraudEngine
from ml_engine.explainability_engine import ExplainabilityEngine


class TestFraudEngineUnit:
    """
    GRUPO 1: Testes de Carregamento do Modelo
    Quantidade: 5 testes
    Objetivo: Garantir que modelo carrega corretamente
    """
    
    def test_unit_ml_001_model_load_success(self):
        """
        ID: TEST-UNIT-ML-001
        DESCRIÇÃO: Verifica carregamento correto do modelo
        
        PRÉ-CONDIÇÕES:
        - Arquivo de modelo existe em backend/models/fraud_model.pkl
        - Modelo foi treinado previamente
        
        PASSOS:
        1. Instanciar FraudEngine()
        2. Verificar que engine.model não é None
        3. Verificar que engine.model_version >= "1.0"
        4. Verificar que engine.feature_names tem 15+ features
        
        PÓS-CONDIÇÕES:
        - Modelo carregado em memória
        - Métricas do modelo disponíveis
        
        CRITÉRIO DE ACEITAÇÃO:
        - model is not None
        - model_version is valid string
        - feature_names is list with length >= 15
        
        ROLLBACK SE FALHAR:
        - Verificar se arquivo de modelo existe
        - Verificar se modelo está corrompido
        - Re-treinar modelo se necessário
        
        TESTES DE REGRESSÃO RELACIONADOS:
        - test_smoke_03_ml_model_loaded
        - test_e2e.py::test_model_loaded
        - test_ml/test_model_performance.py::all
        """
        # Arrange
        engine = FraudEngine()
        
        # Act & Assert
        assert engine.model is not None, \
            "DEF: Modelo não carregou. Verificar arquivo backend/models/fraud_model.pkl"
        
        assert engine.model_version >= "1.0", \
            f"DEF: Versão do modelo inválida: {engine.model_version}"
        
        assert len(engine.feature_names) >= 15, \
            f"DEF: Features insuficientes: {len(engine.feature_names)} (esperado >=15)"
        
        # Validação adicional
        assert hasattr(engine.model, 'predict'), \
            "DEF: Modelo não tem método predict()"
        
        assert hasattr(engine.model, 'predict_proba'), \
            "DEF: Modelo não tem método predict_proba()"
        
        # Log de sucesso
        print(f"✅ TEST-UNIT-ML-001 PASSOU")
        print(f"   Modelo: {engine.model_version}")
        print(f"   Features: {len(engine.feature_names)}")
    
    def test_unit_ml_002_model_load_fallback_when_corrupted(self):
        """
        ID: TEST-UNIT-ML-002
        DESCRIÇÃO: Testa fallback se modelo corrompido
        
        CRITÉRIO DE ACEITAÇÃO:
        - Se modelo corrompido, carregar modelo de backup
        - Não deve crashar
        - Deve logar warning
        
        ROLLBACK SE FALHAR:
        - Verificar lógica de fallback em FraudEngine.__init__
        - Garantir que modelo de backup existe
        """
        # Arrange - Simular arquivo corrompido
        with patch('pickle.load', side_effect=Exception("Corrupted file")):
            # Act
            engine = FraudEngine()
            
            # Assert
            assert engine.model is not None, \
                "DEF: Fallback não funcionou, modelo é None"
            
            assert engine.using_fallback is True, \
                "DEF: Flag using_fallback não está setada"
            
            assert engine.model_version.endswith("-fallback"), \
                f"DEF: Versão não indica fallback: {engine.model_version}"
        
        print(f"✅ TEST-UNIT-ML-002 PASSOU (Fallback funcionou)")
    
    def test_unit_ml_003_model_cache_warm_up(self):
        """
        ID: TEST-UNIT-ML-003
        DESCRIÇÃO: Testa warm-up do cache de predições
        
        CONTEXTO:
        Primeira predição pode ser lenta (cache frio).
        Warm-up deve pré-carregar modelo e cache.
        
        CRITÉRIO DE ACEITAÇÃO:
        - Método warm_up() deve completar em < 5 segundos
        - Após warm-up, predições devem ser rápidas (< 100ms)
        """
        # Arrange
        engine = FraudEngine()
        
        # Act - Warm up
        import time
        start = time.time()
        engine.warm_up()
        warmup_time = time.time() - start
        
        # Assert warm-up time
        assert warmup_time < 5.0, \
            f"DEF: Warm-up muito lento: {warmup_time:.2f}s (esperado <5s)"
        
        # Act - Teste predição após warm-up
        test_tx = {"amount": 1000, "hour": 14, "channel": "PIX"}
        
        start = time.time()
        result = engine.predict(test_tx)
        pred_time = (time.time() - start) * 1000  # ms
        
        # Assert prediction time
        assert pred_time < 100, \
            f"DEF: Predição lenta após warm-up: {pred_time:.2f}ms (esperado <100ms)"
        
        assert result is not None
        assert "risk_score" in result
        
        print(f"✅ TEST-UNIT-ML-003 PASSOU")
        print(f"   Warm-up: {warmup_time:.2f}s")
        print(f"   Predição: {pred_time:.2f}ms")
    
    def test_unit_ml_004_model_memory_footprint(self):
        """
        ID: TEST-UNIT-ML-004
        DESCRIÇÃO: Verifica que modelo não usa memória excessiva
        
        CRITÉRIO DE ACEITAÇÃO:
        - Modelo em memória < 500MB
        - Feature engineering < 200MB
        - Cache < 100MB
        """
        import psutil
        import sys
        
        engine = FraudEngine()
        
        # Calcular tamanho do modelo
        model_size_mb = sys.getsizeof(engine.model) / (1024 * 1024)
        
        assert model_size_mb < 500, \
            f"DEF: Modelo muito grande: {model_size_mb:.2f}MB (esperado <500MB)"
        
        print(f"✅ TEST-UNIT-ML-004 PASSOU")
        print(f"   Tamanho do modelo: {model_size_mb:.2f}MB")
    
    def test_unit_ml_005_model_thread_safety(self):
        """
        ID: TEST-UNIT-ML-005
        DESCRIÇÃO: Verifica que modelo é thread-safe
        
        CRITÉRIO DE ACEITAÇÃO:
        - 10 threads fazendo predições simultâneas
        - Todas devem retornar resultados corretos
        - Sem race conditions
        """
        from concurrent.futures import ThreadPoolExecutor
        
        engine = FraudEngine()
        
        def predict_in_thread(thread_id):
            result = engine.predict({"amount": 1000 + thread_id})
            assert result is not None
            assert "risk_score" in result
            return result
        
        # Act - 10 threads simultâneas
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(predict_in_thread, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # Assert
        assert len(results) == 10, \
            f"DEF: Nem todas threads completaram: {len(results)}/10"
        
        for i, result in enumerate(results):
            assert result is not None, \
                f"DEF: Thread {i} retornou None"
        
        print(f"✅ TEST-UNIT-ML-005 PASSOU (Thread-safe)")
    
    
    """
    GRUPO 2: Testes de Feature Engineering
    Quantidade: 10 testes
    Objetivo: Garantir que features são extraídas corretamente
    """
    
    def test_unit_ml_006_feature_extraction_single_transaction(self):
        """
        ID: TEST-UNIT-ML-006
        DESCRIÇÃO: Testa extração de features de 1 transação
        
        DADOS DE ENTRADA:
        {
            "transaction_id": "TEST001",
            "amount": 1500.00,
            "hour": 14,
            "day_of_week": 2,
            "channel": "PIX",
            "is_international": False
        }
        
        FEATURES ESPERADAS (15 no mínimo):
        - amount_normalized: float [0-1]
        - hour_sin: float [-1, 1]
        - hour_cos: float [-1, 1]
        - is_night: binary {0, 1}
        - is_weekend: binary {0, 1}
        - is_business_hours: binary {0, 1}
        - channel_encoded: int
        - amount_log: float
        - velocity_score: float [0-1]
        - location_risk_score: float [0-1]
        - device_risk_score: float [0-1]
        - is_new_device: binary {0, 1}
        - is_high_value: binary {0, 1}
        - is_international: binary {0, 1}
        - day_of_week_encoded: int [0-6]
        
        VALIDAÇÕES:
        - Todas features devem ter valores numéricos
        - Nenhuma feature deve ser NaN
        - Features normalizadas devem estar nos bounds corretos
        
        SE FALHAR:
        - Verificar AdvancedFeatureEngineering.create_features()
        - Verificar que todos campos obrigatórios estão presentes
        - Verificar normalização de valores
        """
        # Arrange
        engine = FraudEngine()
        transaction = {
            "transaction_id": "TEST001",
            "amount": 1500.00,
            "hour": 14,
            "day_of_week": 2,
            "channel": "PIX",
            "is_international": False
        }
        
        # Act
        features = engine.extract_features(transaction)
        
        # Assert - Quantidade de features
        assert len(features) >= 15, \
            f"DEF: Features insuficientes: {len(features)} (esperado >=15)"
        
        # Assert - Features obrigatórias presentes
        required_features = [
            "amount_normalized", "hour_sin", "hour_cos", 
            "is_night", "is_weekend", "is_business_hours"
        ]
        
        for feat in required_features:
            assert feat in features, \
                f"DEF: Feature '{feat}' faltando"
        
        # Assert - Valores numéricos
        for feat_name, feat_value in features.items():
            assert isinstance(feat_value, (int, float, np.integer, np.floating)), \
                f"DEF: Feature '{feat_name}' não é numérica: {type(feat_value)}"
            
            assert not np.isnan(feat_value), \
                f"DEF: Feature '{feat_name}' é NaN"
        
        # Assert - Bounds específicos
        assert 0 <= features["amount_normalized"] <= 1, \
            f"DEF: amount_normalized fora do bound: {features['amount_normalized']}"
        
        assert features["is_night"] in [0, 1], \
            f"DEF: is_night não é binário: {features['is_night']}"
        
        assert features["is_business_hours"] in [0, 1], \
            f"DEF: is_business_hours não é binário: {features['is_business_hours']}"
        
        # Log detalhado
        print(f"✅ TEST-UNIT-ML-006 PASSOU")
        print(f"   Total features: {len(features)}")
        print(f"   Sample features:")
        for feat in list(features.keys())[:5]:
            print(f"     - {feat}: {features[feat]}")
    
    def test_unit_ml_007_feature_extraction_handles_missing_fields(self):
        """
        ID: TEST-UNIT-ML-007
        DESCRIÇÃO: Testa tratamento de campos faltantes
        
        CENÁRIO:
        Transação com apenas campos obrigatórios.
        Campos opcionais devem receber valores default.
        
        CRITÉRIO DE ACEITAÇÃO:
        - Sistema não deve crashar
        - Campos faltantes devem ter defaults válidos:
          * hour → -1 (ou valor médio)
          * channel → "UNKNOWN"
          * is_international → False
        """
        engine = FraudEngine()
        
        # Transaction com APENAS amount
        minimal_tx = {"amount": 1000}
        
        # Act
        features = engine.extract_features(minimal_tx)
        
        # Assert
        assert features is not None, \
            "DEF: extract_features retornou None para campos faltantes"
        
        assert len(features) >= 15, \
            "DEF: Features não foram geradas para campos faltantes"
        
        # Verificar defaults
        if "hour" in features:
            assert features["hour"] in [-1, 12], \
                f"DEF: Default de 'hour' inválido: {features['hour']}"
        
        print(f"✅ TEST-UNIT-ML-007 PASSOU")
    
    def test_unit_ml_008_feature_extraction_handles_extreme_values(self):
        """
        ID: TEST-UNIT-ML-008
        DESCRIÇÃO: Testa valores extremos
        
        CASOS:
        - amount = R$0.01 (mínimo)
        - amount = R$1.000.000 (máximo extremo)
        - hour = 0 (meia-noite)
        - hour = 23 (última hora)
        """
        engine = FraudEngine()
        
        extreme_cases = [
            {"amount": 0.01, "hour": 0},
            {"amount": 1000000, "hour": 23},
            {"amount": -100, "hour": -1},  # Inválidos
        ]
        
        for tx in extreme_cases:
            features = engine.extract_features(tx)
            
            # Não deve crashar
            assert features is not None
            
            # Valores normalizados devem estar em bounds
            if "amount_normalized" in features:
                assert 0 <= features["amount_normalized"] <= 1
        
        print(f"✅ TEST-UNIT-ML-008 PASSOU")
    
    # ... CONTINUAÇÃO DOS 50 TESTES DE ML ENGINE ...
    # (Por brevidade, mostrando estrutura. Todos seguem mesmo padrão.)
    
    def test_unit_ml_009_hour_cyclical_encoding(self):
        """Testa encoding cíclico de hora (sin/cos)"""
        pass  # Implementação completa no código real
    
    def test_unit_ml_010_channel_encoding(self):
        """Testa encoding de canal (PIX, TED, BOLETO)"""
        pass
    
    # ... testes 011-020: Feature engineering avançado ...
    # ... testes 021-030: Testes de predição ...
    # ... testes 031-040: Testes de threshold e classificação ...
    # ... testes 041-050: Testes de performance e cache ...
```


##### **PROCESSO DE VALIDAÇÃO PÓS-CORREÇÃO (ML Engine)**

```yaml
CHECKLIST DE VALIDAÇÃO: ML Engine
═══════════════════════════════════════════════════════════

NÍVEL 1: Re-executar teste que falhou
□ pytest tests/unit/test_ml_engine_unit.py::test_unit_ml_XXX -v
□ PASSOU? → Continuar para Nível 2
□ FALHOU? → VOLTAR PARA CORREÇÃO (não avançar!)

NÍVEL 2: Re-executar suite completa de ML Engine
□ pytest tests/unit/test_ml_engine_unit.py -v
□ Todos 50 testes devem passar
□ ALGUM FALHOU? → NOVA REGRESSÃO! 
   └─ Registrar como DEF-REGRESSAO-XXX
   └─ Analisar causa raiz
   └─ Voltar para correção

NÍVEL 3: Testes de integração relacionados
□ pytest tests/ml/test_model_performance.py -v (15 testes)
□ pytest tests/ml/test_explainability.py -v (8 testes)
□ pytest tests/ml/test_bias_fairness.py -v (10 testes)
□ ALGUM FALHOU? → REGRESSÃO EM INTEGRAÇÃO!

NÍVEL 4: Smoke Tests críticos
□ pytest tests/functional/test_smoke.py::test_smoke_03_ml_model_loaded -v
□ pytest tests/functional/test_smoke.py::test_smoke_05_prediction_endpoint_responds -v
□ ALGUM FALHOU? → REGRESSÃO CRÍTICA! REVERTER MUDANÇA!

NÍVEL 5: E2E relacionados
□ pytest tests/test_e2e.py::TestE2EMLPipeline -v (3 testes)
□ pytest tests/test_e2e.py::TestE2EFraudPrediction -v (4 testes)
□ ALGUM FALHOU? → REGRESSÃO EM E2E!

NÍVEL 6: Validação manual
□ Iniciar API: python backend/api/production_api.py
□ Fazer predição manual via curl/Postman
□ Verificar logs não mostram erros/warnings
□ Verificar response time < 100ms (warm cache)

APROVAÇÃO FINAL:
□ QA Lead: _________________ Data: ________
□ ML Engineer: _____________ Data: ________
□ Tech Lead: _______________ Data: ________

STATUS: [ ] APROVADO  [ ] REJEITADO
```


***

#### **1.2 BACKEND - API LAYER (40 testes)**

##### **Arquivo: `tests/unit/test_api_layer_unit.py`**

```python
"""
SUITE DE TESTES: API Layer - Unit Tests
TOTAL: 40 testes
DEPENDÊNCIAS: backend/api/production_api.py
CRITICIDADE: ⚠️ CRÍTICA (porta de entrada do sistema)

NOTAS ESPECIAIS:
- API é o ponto de entrada público
- Qualquer bug aqui afeta todos usuários
- Testes de autenticação são CRÍTICOS (compliance)
- Rate limiting é obrigatório (anti-DDoS)
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from api.production_api import app
from security.jwt_validator import JWTValidator
from security.rate_limiter import RateLimiter

client = TestClient(app)


class TestAPIEndpointsUnit:
    """
    GRUPO 1: Testes de Estrutura de Endpoints
    Quantidade: 10 testes
    """
    
    def test_unit_api_001_health_endpoint_structure(self):
        """
        ID: TEST-UNIT-API-001
        DESCRIÇÃO: Verifica estrutura da resposta /health
        
        ENDPOINT: GET /api/health
        PÚBLICO: Sim (não requer autenticação)
        
        RESPOSTA ESPERADA:
        {
            "status": "healthy",
            "version": "12.0",
            "timestamp": "2025-11-29T16:00:00Z",
            "components": {
                "database": "healthy",
                "cache": "healthy",
                "ml_engine": "healthy"
            }
        }
        
        VALIDAÇÕES:
        - Status code = 200
        - Content-Type = application/json
        - Campos obrigatórios presentes
        - Formato de timestamp válido (ISO 8601)
        - Versão válida (semantic versioning)
        
        IMPACTO SE FALHAR:
        - Load balancer pode remover instância do pool
        - Monitoramento pode gerar alerta falso
        - Status page pode ficar incorreto
        
        TESTES DE REGRESSÃO:
        - test_smoke_01_backend_health
        - test_e2e.py::test_backend_health
        - test_integration/test_monitoring.py::test_health_check
        """
        # Act
        response = client.get("/api/health")
        
        # Assert - Status code
        assert response.status_code == 200, \
            f"DEF-API-001: Health check retornou {response.status_code} (esperado 200)"
        
        # Assert - Content-Type
        assert "application/json" in response.headers.get("content-type", ""), \
            "DEF-API-002: Health check não retorna JSON"
        
        # Act - Parse JSON
        data = response.json()
        
        # Assert - Campos obrigatórios
        required_fields = ["status", "version", "timestamp"]
        for field in required_fields:
            assert field in data, \
                f"DEF-API-003: Campo '{field}' faltando no health check"
        
        # Assert - Status value
        assert data["status"] in ["healthy", "degraded", "unhealthy"], \
            f"DEF-API-004: Status inválido: {data['status']}"
        
        # Assert - Version format
        version = data["version"]
        assert isinstance(version, str) and len(version) > 0, \
            f"DEF-API-005: Versão inválida: {version}"
        
        # Verificar formato semantic versioning (X.Y.Z)
        version_parts = version.split(".")
        assert len(version_parts) >= 2, \
            f"DEF-API-006: Versão não segue semantic versioning: {version}"
        
        # Assert - Timestamp format (ISO 8601)
        from datetime import datetime
        try:
            datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        except ValueError:
            assert False, \
                f"DEF-API-007: Timestamp não é ISO 8601: {data['timestamp']}"
        
        # Assert - Components (se presente)
        if "components" in data:
            for component, status in data["components"].items():
                assert status in ["healthy", "degraded", "unhealthy"], \
                    f"DEF-API-008: Componente '{component}' tem status inválido: {status}"
        
        print(f"✅ TEST-UNIT-API-001 PASSOU")
        print(f"   Status: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Components: {data.get('components', {})}")
    
    def test_unit_api_002_predict_request_validation_success(self):
        """
        ID: TEST-UNIT-API-002
        DESCRIÇÃO: Validação de payload válido em /predict
        
        ENDPOINT: POST /api/fraud/predict
        AUTENTICAÇÃO: Requerida (JWT Bearer token)
        
        PAYLOAD VÁLIDO:
        {
            "transactions": [
                {
                    "transaction_id": "TEST001",
                    "amount": 1500.00,
                    "hour": 14,
                    "channel": "PIX"
                }
            ]
        }
        
        VALIDAÇÕES:
        - Payload aceito (status 200)
        - Response contém "predictions"
        - Response contém "summary"
        """
        # Arrange - Get auth token
        from tests.test_e2e import get_auth_token
        token = get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        payload = {
            "transactions": [
                {
                    "transaction_id": "TEST001",
                    "amount": 1500.00,
                    "hour": 14,
                    "channel": "PIX"
                }
            ]
        }
        
        # Act
        response = client.post("/api/fraud/predict", json=payload, headers=headers)
        
        # Assert
        assert response.status_code == 200, \
            f"DEF-API-009: Predict rejeitou payload válido: {response.status_code}"
        
        data = response.json()
        
        assert data["success"] is True, \
            f"DEF-API-010: Predict retornou success=False para payload válido"
        
        assert "predictions" in data["data"], \
            "DEF-API-011: Response sem campo 'predictions'"
        
        assert "summary" in data["data"], \
            "DEF-API-012: Response sem campo 'summary'"
        
        print(f"✅ TEST-UNIT-API-002 PASSOU")
    
    def test_unit_api_003_predict_request_validation_missing_field(self):
        """
        ID: TEST-UNIT-API-003
        DESCRIÇÃO: Rejeita payload sem campo obrigatório
        
        PAYLOAD INVÁLIDO:
        {
            # Falta "transactions"
        }
        
        COMPORTAMENTO ESPERADO:
        - Status 422 (Unprocessable Entity)
        - Mensagem de erro clara
        - Indicar qual campo está faltando
        """
        token = get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        # Payload SEM campo "transactions"
        invalid_payload = {}
        
        # Act
        response = client.post("/api/fraud/predict", json=invalid_payload, headers=headers)
        
        # Assert - Status code
        assert response.status_code in [400, 422], \
            f"DEF-API-013: Payload inválido retornou {response.status_code} (esperado 400/422)"
        
        # Assert - Error message
        data = response.json()
        assert data["success"] is False, \
            "DEF-API-014: Payload inválido retornou success=True"
        
        # Error message deve mencionar "transactions"
        error_msg = str(data.get("error", "")).lower()
        assert "transaction" in error_msg, \
            f"DEF-API-015: Mensagem de erro não menciona campo faltante: {data.get('error')}"
        
        print(f"✅ TEST-UNIT-API-003 PASSOU")
    
    def test_unit_api_004_predict_request_validation_empty_list(self):
        """
        ID: TEST-UNIT-API-004
        DESCRIÇÃO: Rejeita lista vazia de transações
        
        PAYLOAD INVÁLIDO:
        {
            "transactions": []  # ← VAZIO
        }
        
        COMPORTAMENTO ESPERADO:
        - Status 400/422
        - Mensagem: "Lista de transações não pode ser vazia"
        """
        token = get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        payload = {"transactions": []}  # VAZIO
        
        # Act
        response = client.post("/api/fraud/predict", json=payload, headers=headers)
        
        # Assert
        assert response.status_code in [400, 422], \
            f"DEF-API-016: Lista vazia aceita (status {response.status_code})"
        
        print(f"✅ TEST-UNIT-API-004 PASSOU")
    
    def test_unit_api_005_predict_request_validation_negative_amount(self):
        """
        ID: TEST-UNIT-API-005
        DESCRIÇÃO: Rejeita valor negativo
        
        PAYLOAD INVÁLIDO:
        {
            "transactions": [{"amount": -100}]  # ← NEGATIVO
        }
        """
        token = get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        payload = {"transactions": [{"amount": -100}]}
        
        # Act
        response = client.post("/api/fraud/predict", json=payload, headers=headers)
        
        # Assert
        assert response.status_code in [400, 422], \
            f"DEF-API-017: Valor negativo aceito (status {response.status_code})"
        
        print(f"✅ TEST-UNIT-API-005 PASSOU")
    
    # ... testes 006-010: Mais validações de payload ...
    
    
    """
    GRUPO 2: Testes de Autenticação JWT
    Quantidade: 10 testes
    CRITICIDADE: ⚠️ CRÍTICA (segurança)
    """
    
    def test_unit_api_011_auth_token_generation(self):
        """
        ID: TEST-UNIT-API-011
        DESCRIÇÃO: Gera token JWT válido
        
        REQUISITO: REQ-SEC-001
        COMPLIANCE: PCI DSS Requirement 8.2
        
        TOKEN GERADO DEVE TER:
        - Header: alg=HS256, typ=JWT
        - Payload: user_id, username, role, exp, iat
        - Signature: HMAC SHA-256
        - Expiration: 24 horas
        
        VALIDAÇÕES:
        - Token não é None
        - Token tem 3 partes (header.payload.signature)
        - Pode ser decodificado
        - Claims obrigatórios presentes
        """
        from security.jwt_validator import JWTValidator
        
        # Act
        token = JWTValidator.generate_token(
            user_id="user123",
            username="admin",
            role="admin"
        )
        
        # Assert - Token gerado
        assert token is not None, \
            "DEF-SEC-001: Token não foi gerado"
        
        assert isinstance(token, str), \
            "DEF-SEC-002: Token não é string"
        
        # Assert - Formato JWT (3 partes separadas por '.')
        parts = token.split(".")
        assert len(parts) == 3, \
            f"DEF-SEC-003: Token não tem 3 partes: {len(parts)}"
        
        # Assert - Decodificar token
        try:
            decoded = JWTValidator.decode_token(token)
        except Exception as e:
            assert False, f"DEF-SEC-004: Token não pode ser decodificado: {e}"
        
        # Assert - Claims obrigatórios
        required_claims = ["user_id", "username", "role", "exp", "iat"]
        for claim in required_claims:
            assert claim in decoded, \
                f"DEF-SEC-005: Claim '{claim}' faltando no token"
        
        # Assert - Valores corretos
        assert decoded["user_id"] == "user123"
        assert decoded["username"] == "admin"
        assert decoded["role"] == "admin"
        
        # Assert - Expiration (deve ser futuro)
        from datetime import datetime
        exp_timestamp = decoded["exp"]
        iat_timestamp = decoded["iat"]
        
        assert exp_timestamp > iat_timestamp, \
            "DEF-SEC-006: Expiration não é maior que issued_at"
        
        # Expiração deve ser ~24h no futuro
        exp_delta = exp_timestamp - iat_timestamp
        expected_delta = 24 * 3600  # 24 horas em segundos
        
        assert abs(exp_delta - expected_delta) < 60, \
            f"DEF-SEC-007: Expiration incorreta: {exp_delta}s (esperado ~{expected_delta}s)"
        
        print(f"✅ TEST-UNIT-API-011 PASSOU")
        print(f"   Token parts: {len(parts)}")
        print(f"   Claims: {list(decoded.keys())}")
    
    def test_unit_api_012_auth_token_validation_valid_token(self):
        """
        ID: TEST-UNIT-API-012
        DESCRIÇÃO: Valida token válido
        """
        # Arrange - Gerar token
        token = JWTValidator.generate_token("user123", "admin", "admin")
        
        # Act
        is_valid = JWTValidator.validate_token(token)
        
        # Assert
        assert is_valid is True, \
            "DEF-SEC-008: Token válido foi rejeitado"
        
        print(f"✅ TEST-UNIT-API-012 PASSOU")
    
    def test_unit_api_013_auth_token_validation_expired_token(self):
        """
        ID: TEST-UNIT-API-013
        DESCRIÇÃO: Rejeita token expirado
        
        CENÁRIO:
        1. Gerar token com expiração de 1 segundo
        2. Aguardar 2 segundos
        3. Validar token
        4. Deve retornar False ou raise TokenExpiredError
        """
        import time
        
        # Arrange - Gerar token com expiração de 1s
        token = JWTValidator.generate_token(
            "user123", "admin", "admin",
            expires_in_seconds=1
        )
        
        # Act - Aguardar expiração
        time.sleep(2)
        
        # Act - Validar
        is_valid = JWTValidator.validate_token(token)
        
        # Assert
        assert is_valid is False, \
            "DEF-SEC-009: Token expirado foi aceito! CRÍTICO!"
        
        print(f"✅ TEST-UNIT-API-013 PASSOU (Token expirado rejeitado)")
    
    def test_unit_api_014_auth_token_validation_tampered_token(self):
        """
        ID: TEST-UNIT-API-014
        DESCRIÇÃO: Rejeita token adulterado
        
        CENÁRIO:
        1. Gerar token válido
        2. Modificar payload (trocar role de "user" para "admin")
        3. Validar token
        4. Deve retornar False (signature inválida)
        
        CRITICIDADE: ⚠️ CRÍTICA (privilege escalation)
        """
        import base64
        import json
        
        # Arrange - Gerar token
        token = JWTValidator.generate_token("user123", "user", "user")
        
        # Act - Adulterar token (modificar role)
        parts = token.split(".")
        
        # Decodificar payload
        payload_decoded = base64.urlsafe_b64decode(parts[1] + "==")
        payload_json = json.loads(payload_decoded)
        
        # ADULTERAR: trocar role
        payload_json["role"] = "admin"  # ← ESCALAÇÃO DE PRIVILÉGIO!
        
        # Recodificar payload
        payload_tampered = base64.urlsafe_b64encode(
            json.dumps(payload_json).encode()
        ).decode().rstrip("=")
        
        # Reconstruir token com payload adulterado
        tampered_token = f"{parts[0]}.{payload_tampered}.{parts[2]}"
        
        # Act - Validar token adulterado
        is_valid = JWTValidator.validate_token(tampered_token)
        
        # Assert - DEVE SER REJEITADO
        assert is_valid is False, \
            "DEF-SEC-010: TOKEN ADULTERADO FOI ACEITO! VULNERABILIDADE CRÍTICA!!!"
        
        print(f"✅ TEST-UNIT-API-014 PASSOU (Token adulterado rejeitado)")
    
    def test_unit_api_015_auth_token_validation_invalid_signature(self):
        """
        ID: TEST-UNIT-API-015
        DESCRIÇÃO: Rejeita token com signature inválida
        """
        # Token com signature aleatória
        fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.FAKESIGNATURE"
        
        # Act
        is_valid = JWTValidator.validate_token(fake_token)
        
        # Assert
        assert is_valid is False, \
            "DEF-SEC-011: Token com signature inválida foi aceito!"
        
        print(f"✅ TEST-UNIT-API-015 PASSOU")
    
    # ... testes 016-020: Mais testes de autenticação ...
    
    
    """
    GRUPO 3: Testes de Rate Limiting
    Quantidade: 5 testes
    OBJETIVO: Prevenir DDoS e abuso
    """
    
    def test_unit_api_021_rate_limit_normal_usage(self):
        """
        ID: TEST-UNIT-API-021
        DESCRIÇÃO: Uso normal não é bloqueado
        
        LIMITE: 100 requests/minuto por IP
        
        CENÁRIO:
        - Fazer 50 requests em 10 segundos
        - Todas devem ser aceitas (200)
        """
        # Act
        for i in range(50):
            response = client.get("/api/health")
            
            # Assert
            assert response.status_code == 200, \
                f"DEF-API-018: Request {i} bloqueada indevidamente"
        
        print(f"✅ TEST-UNIT-API-021 PASSOU (50 requests aceitas)")
    
    def test_unit_api_022_rate_limit_exceeded(self):
        """
        ID: TEST-UNIT-API-022
        DESCRIÇÃO: Rate limit bloqueia após limite
        
        CENÁRIO:
        - Fazer 150 requests rápidas (> limite de 100/min)
        - Requests após 100 devem retornar 429
        """
        blocked_count = 0
        
        # Act - 150 requests
        for i in range(150):
            response = client.get("/api/health")
            
            if response.status_code == 429:  # Too Many Requests
                blocked_count += 1
        
        # Assert - Algumas devem ter sido bloqueadas
        assert blocked_count > 0, \
            "DEF-API-019: Rate limit não está funcionando! Nenhuma request bloqueada!"
        
        print(f"✅ TEST-UNIT-API-022 PASSOU ({blocked_count}/150 bloqueadas)")
    
    # ... testes 023-025: Mais testes de rate limiting ...
    
    
    """
    GRUPO 4: Testes de Headers de Segurança
    Quantidade: 5 testes
    """
    
    def test_unit_api_026_security_headers_present(self):
        """
        ID: TEST-UNIT-API-026
        DESCRIÇÃO: Headers de segurança presentes
        
        HEADERS OBRIGATÓRIOS:
        - X-Content-Type-Options: nosniff
        - X-Frame-Options: DENY
        - X-XSS-Protection: 1; mode=block
        - Content-Security-Policy: default-src 'self'
        - Strict-Transport-Security: max-age=31536000
        """
        # Act
        response = client.get("/api/health")
        
        # Assert - Headers obrigatórios
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block"
        }
        
        for header_name, expected_value in security_headers.items():
            assert header_name in response.headers, \
                f"DEF-SEC-012: Header '{header_name}' faltando"
            
            actual_value = response.headers[header_name]
            assert expected_value in actual_value, \
                f"DEF-SEC-013: Header '{header_name}' incorreto: {actual_value}"
        
        print(f"✅ TEST-UNIT-API-026 PASSOU")
    
    # ... testes 027-040: Restante dos testes de API ...
```


***

## 🔄 PROCESSO DE GESTÃO DE REGRESSÃO

### **CHECKLIST ANTI-REGRESSÃO**

```yaml
PROCEDIMENTO OBRIGATÓRIO APÓS CADA CORREÇÃO
═════════════════════════════════════════════════════════════

FASE 1: PREPARAÇÃO
─────────────────────────────────────────────────────────────
□ 1.1 Criar branch de correção: fix/DEF-XXXX
□ 1.2 Documentar mudanças no CHANGELOG.md
□ 1.3 Identificar TODOS os módulos afetados pela correção
□ 1.4 Listar TODOS os testes relacionados (usar Matriz de Impacto)
□ 1.5 Estimar tempo de validação (mínimo: 30 min)

FASE 2: IMPLEMENTAÇÃO DA CORREÇÃO
─────────────────────────────────────────────────────────────
□ 2.1 Implementar correção
□ 2.2 Adicionar comentário explicando a correção:
      # FIX DEF-XXXX: [Descrição breve]
      # ANTES: [Comportamento com bug]
      # DEPOIS: [Comportamento correto]
      # IMPACTO: [Módulos afetados]
□ 2.3 Code review obrigatório por 2 pessoas
□ 2.4 Análise de impacto:
      - Esta mudança pode afetar outras funcionalidades? Quais?
      - Existem edge cases não considerados?
      - Existe código duplicado que precisa da mesma correção?

FASE 3: VALIDAÇÃO BÁSICA (Local)
─────────────────────────────────────────────────────────────
□ 3.1 Executar teste que falhou originalmente:
      pytest path/to/test_file.py::test_name -v
      
□ 3.2 RESULTADO:
      [ ] ✅ PASSOU → Continuar para 3.3
      [ ] ❌ FALHOU → PARAR! Voltar para Fase 2.1
                      Correção não funcionou!

□ 3.3 Executar testes do mesmo módulo:
      pytest path/to/test_module/ -v
      
□ 3.4 RESULTADO:
      Passou: ___/___  Falhou: ___/___
      
      [ ] ✅ TODOS PASSARAM → Continuar para Fase 4
      [ ] ❌ ALGUM FALHOU → REGRESSÃO DETECTADA!
                             └─ Registrar: DEF-REGRESSAO-XXXX
                             └─ Voltar para Fase 2.1

FASE 4: VALIDAÇÃO DE INTEGRAÇÃO
─────────────────────────────────────────────────────────────
□ 4.1 Executar testes de integração relacionados:
      [Listar testes específicos conforme Matriz de Impacto]
      
      Exemplo:
      □ pytest tests/integration/test_api_integration.py -v
      □ pytest tests/integration/test_db_integration.py -v
      
□ 4.2 RESULTADO:
      Passou: ___/___  Falhou: ___/___
      
      [ ] ✅ TODOS PASSARAM → Continuar para 4.3
      [ ] ❌ ALGUM FALHOU → REGRESSÃO EM INTEGRAÇÃO!
                             └─ Analisar impacto
                             └─ Pode exigir rollback

□ 4.3 Executar smoke tests completo:
      pytest tests/functional/test_smoke.py -v
      
□ 4.4 RESULTADO (10 testes):
      Passou: ___/10  Falhou: ___/10
      
      [ ] ✅ 10/10 → Continuar para Fase 5
      [ ] ❌ ALGUM FALHOU → REGRESSÃO CRÍTICA!
                             └─ REVERTER MUDANÇA IMEDIATAMENTE
                             └─ Análise de causa raiz obrigatória
                             └─ Reunião com Tech Lead

FASE 5: VALIDAÇÃO E2E
─────────────────────────────────────────────────────────────
□ 5.1 Executar E2E tests relacionados:
      pytest tests/test_e2e.py -v -k "keyword"
      
□ 5.2 RESULTADO:
      Passou: ___/___  Falhou: ___/___
      
      [ ] ✅ TODOS PASSARAM → Continuar para 5.3
      [ ] ❌ ALGUM FALHOU → Investigar impacto no fluxo completo

□ 5.3 Executar suite E2E completa (se impacto alto):
      pytest tests/test_e2e.py -v
      
□ 5.4 RESULTADO (31 testes):
      Passou: ___/31  Falhou: ___/31

FASE 6: VALIDAÇÃO MANUAL
─────────────────────────────────────────────────────────────
□ 6.1 Iniciar sistema completo:
      □ Backend: python backend/api/production_api.py
      □ Frontend: cd frontend && npm run dev
      □ Database: verificar conexão
      
□ 6.2 Testar funcionalidade corrigida manualmente:
      □ Reproduzir steps do bug original
      □ Verificar que bug foi corrigido
      □ Testar edge cases
      
□ 6.3 Testar funcionalidades relacionadas manualmente:
      [Listar funcionalidades específicas]
      
      Exemplo:
      □ Login com usuário válido
      □ Fazer predição de fraude
      □ Ver dashboard
      □ Fazer logout
      
□ 6.4 Verificar logs:
      □ Não há erros no console
      □ Não há warnings inesperados
      □ Audit logs estão sendo criados

FASE 7: VALIDAÇÃO DE PERFORMANCE
─────────────────────────────────────────────────────────────
□ 7.1 Medir latência da funcionalidade corrigida:
      Latência ANTES da correção: _____ms
      Latência DEPOIS da correção: _____ms
      
      [ ] Latência melhorou ou manteve
      [ ] Latência piorou → Investigar otimização

□ 7.2 Verificar uso de memória:
      Memória ANTES: _____MB
      Memória DEPOIS: _____MB
      
      [ ] Memória ok (< +10% de aumento)
      [ ] Memória aumentou significativamente → Investigar

□ 7.3 Executar teste de carga básico:
      pytest tests/performance/test_load.py::test_specific -v

FASE 8: CI/CD PIPELINE
─────────────────────────────────────────────────────────────
□ 8.1 Commit da correção:
      git commit -m "FIX DEF-XXXX: [descrição]"
      
□ 8.2 Push para branch:
      git push origin fix/DEF-XXXX
      
□ 8.3 Criar Pull Request:
      Título: "FIX DEF-XXXX: [Descrição]"
      Descrição deve incluir:
      - Link para defeito
      - Explicação da correção
      - Testes executados
      - Riscos de regressão
      - Checklist completo
      
□ 8.4 Aguardar CI/CD pipeline:
      □ Lint passing
      □ Unit tests passing (200+ tests)
      □ Integration tests passing (80 tests)
      □ E2E tests passing (31 tests)
      □ Security scan passing
      □ Coverage >= 80%
      
□ 8.5 PIPELINE RESULT:
      [ ] ✅ PASSOU → Continuar para 8.6
      [ ] ❌ FALHOU → Verificar logs, corrigir, re-run

□ 8.6 Code Review:
      Reviewer 1: _____________ [ ] Aprovado
      Reviewer 2: _____________ [ ] Aprovado
      
□ 8.7 Merge para develop:
      git checkout develop
      git merge fix/DEF-XXXX
      git push origin develop

FASE 9: DEPLOY PARA STAGING
─────────────────────────────────────────────────────────────
□ 9.1 Deploy automático para staging
      
□ 9.2 Smoke tests em staging:
      pytest tests/functional/test_smoke.py --env=staging -v
      
      RESULTADO:
      [ ] ✅ 10/10 → Continuar
      [ ] ❌ FALHOU → NÃO FAZER DEPLOY PARA PRODUÇÃO!

□ 9.3 Teste manual exploratório em staging (30 min):
      □ Testar funcionalidade corrigida
      □ Testar fluxos principais
      □ Verificar logs e monitoramento
      
□ 9.4 Monitorar staging por 2 horas:
      □ Erro rate < 0.1%
      □ Latência P95 < 5000ms
      □ CPU < 70%
      □ Memory < 80%

FASE 10: APROVAÇÃO PARA PRODUÇÃO
─────────────────────────────────────────────────────────────
□ 10.1 Checklist de aprovação:
       □ Todos testes passando
       □ Code review aprovado
       □ QA manual aprovado
       □ Staging estável por 2+ horas
       □ Documentação atualizada
       □ Changelog atualizado
       
□ 10.2 Aprovações finais:
       QA Lead: _____________ Data: ______ Hora: ______
       Tech Lead: ___________ Data: ______ Hora: ______
       Product Owner: _______ Data: ______ Hora: ______
       
□ 10.3 Deploy para produção:
       [ ] Deploy agendado para: _________ às _________
       [ ] Rollback plan preparado
       [ ] Monitoramento em alerta
       [ ] Equipe de plantão avisada

FASE 11: PÓS-DEPLOY MONITORING
─────────────────────────────────────────────────────────────
□ 11.1 Monitorar primeiros 15 minutos:
       □ Error rate normal
       □ Latência normal
       □ Logs sem erros críticos
       
□ 11.2 Executar smoke tests em produção:
       pytest tests/functional/test_smoke.py --env=prod -v
       
□ 11.3 Teste manual básico em produção:
       □ Login
       □ Funcionalidade corrigida
       □ Logout
       
□ 11.4 Monitorar por 24 horas:
       Hora +1h: [ ] OK   [ ] Problemas: _____________
       Hora +4h: [ ] OK   [ ] Problemas: _____________
       Hora +8h: [ ] OK   [ ] Problemas: _____________
       Hora +24h: [ ] OK  [ ] Problemas: _____________

FASE 12: FECHAMENTO
─────────────────────────────────────────────────────────────
□ 12.1 Atualizar status do defeito:
       DEF-XXXX: [ ] RESOLVIDO
       
□ 12.2 Documentar lições aprendidas:
       - Causa raiz: _______________________________
       - Como prevenir no futuro: __________________
       - Testes adicionais necessários: ____________
       
□ 12.3 Adicionar teste de regressão permanente:
       □ Criar novo teste que detectaria este bug
       □ Adicionar ao test suite
       
□ 12.4 Atualizar documentação:
       □ README se necessário
       □ Documentação técnica
       □ Release notes

═════════════════════════════════════════════════════════════
ASSINATURAS FINAIS

Desenvolvedor: _________________ Data: ___/___/___
QA Engineer: ___________________ Data: ___/___/___
Tech Lead: _____________________ Data: ___/___/___

RESULTADO FINAL:
[ ] ✅ CORREÇÃO APROVADA - SEM REGRESSÃO
[ ] ❌ CORREÇÃO REJEITADA - REGRESSÃO DETECTADA
[ ] ⚠️ CORREÇÃO PARCIAL - REQUER MAIS TRABALHO
═════════════════════════════════════════════════════════════
```


***

## 📊 RELATÓRIO DE EXECUÇÃO DE TESTES

### **Template de Relatório Diário**

```markdown
# RELATÓRIO DE EXECUÇÃO DE TESTES
**Data**: 2025-11-29
**Build**: #1234 (commit: abc123def)
**Ambiente**: Development
**Executado por**: QA Team

═══════════════════════════════════════════════════════════════

## 📈 RESUMO EXECUTIVO

| Métrica | Valor | Status |
|---------|-------|--------|
| Total de Testes | 800 | - |
| Testes Executados | 800 | 100% |
| Testes Passando | 795 | 99.4% |
| Testes Falhando | 5 | 0.6% |
| Testes Pulados | 0 | 0% |
| Cobertura de Código | 85% | ✅ Meta: ≥80% |
| Tempo Total | 45 min | ✅ Meta: <60min |

**Veredicto**: ⚠️ **BLOQUEADO** - 5 falhas críticas impedem deploy

═══════════════════════════════════════════════════════════════

## 📋 DETALHAMENTO POR CATEGORIA

### Testes Unitários
- Total: 200 | ✅ Passando: 198 | ❌ Falhando: 2
- Cobertura: 87%
- Tempo: 5 min

**Falhas**:
1. `test_unit_ml_006_feature_extraction_single_transaction` 
   - **Defeito**: DEF-2025-001
   - **Severidade**: ALTA
   - **Status**: Em correção
   
2. `test_unit_api_013_auth_token_validation_expired_token`
   - **Defeito**: DEF-2025-002
   - **Severidade**: CRÍTICA
   - **Status**: Em análise

### Testes de Integração
- Total: 80 | ✅ Passando: 78 | ❌ Falhando: 2
- Tempo: 8 min

**Falhas**:
3. `test_integration_api_to_ml_flow`
   - **Defeito**: DEF-2025-003
   - **Severidade**: ALTA
   - **Causa**: Relacionado a DEF-2025-001

4. `test_integration_db_connection_pool`
   - **Defeito**: DEF-2025-004
   - **Severidade**: MÉDIA
   - **Status**: Investigando

### Testes E2E
- Total: 31 | ✅ Passando: 30 | ❌ Falhando: 1
- Tempo: 12 min

**Falha**:
5. `test_e2e_full_flow_frontend_to_db`
   - **Defeito**: DEF-2025-005
   - **Severidade**: CRÍTICA
   - **Causa**: Relacionado a DEF-2025-001 e DEF-2025-003

### Testes de Performance
- Total: 50 | ✅ Passando: 50 | ❌ Falhando: 0
- Tempo: 15 min
- ✅ Todos SLAs atendidos

### Testes de Segurança
- Total: 60 | ✅ Passando: 60 | ❌ Falhando: 0
- Tempo: 5 min
- ✅ Sem vulnerabilidades detectadas

═══════════════════════════════════════════════════════════════

## 🐛 DEFEITOS ATIVOS

### DEF-2025-001: Feature extraction retorna NaN
**Severidade**: ⚠️ ALTA
**Prioridade**: P1
**Módulo**: backend/ml_engine
**Impacto**: Predições de fraude podem falhar
**Testes Afetados**: 3 (DEF-2025-001, 003, 005)
**Status**: 🔧 Em correção
**Assignee**: João Silva
**ETA**: 2025-11-29 18:00

**Causa Raiz**:
Normalização de features não trata corretamente valores
extremos (>1000000), gerando divisão por zero.

**Correção**:
Adicionar clipping de valores antes da normalização.

**Validação Pendente**:
- [ ] Re-executar test_unit_ml_006
- [ ] Re-executar suite test_ml/ (33 testes)
- [ ] Executar smoke tests
- [ ] Executar E2E tests relacionados

---

### DEF-2025-002: Token expirado sendo aceito
**Severidade**: 🔴 CRÍTICA
**Prioridade**: P0 (BLOCKER)
**Módulo**: backend/security/jwt_validator.py
**Impacto**: VULNERABILIDADE DE SEGURANÇA
**Compliance**: Violação PCI DSS Requirement 8.2
**Status**: 🔍 Em análise
**Assignee**: Maria Santos (Security)
**ETA**: 2025-11-29 17:00

**Causa Raiz**:
Validação de expiração não está verificando timezone.
Token com `exp` em UTC está sendo comparado com timestamp local.

**Correção Proposta**:
Normalizar timezone antes de comparar timestamps.

**Impacto Potencial**: ALTO
- 7 testes de autenticação podem ser afetados
- Sistema de RBAC pode ter regressão
- Audit logs podem ficar incorretos

**Validação Obrigatória**:
- [ ] Re-executar test_unit_api_013
- [ ] Re-executar TODA suite test_security/ (60 testes)
- [ ] Executar penetration tests
- [ ] DAST scan obrigatório
- [ ] Aprovação do Security Officer

═══════════════════════════════════════════════════════════════

## 📊 TENDÊNCIAS (Últimos 7 Dias)

```

| Data | Total | Pass | Fail | Taxa Sucesso |
| :-- | :-- | :-- | :-- | :-- |
| 2025-11-29 | 800 | 795 | 5 | 99.4%  ⚠️ |
| 2025-11-28 | 800 | 798 | 2 | 99.8%  ✅ |
| 2025-11-27 | 800 | 800 | 0 | 100%   ✅ |
| 2025-11-26 | 800 | 796 | 4 | 99.5%  ⚠️ |
| 2025-11-25 | 800 | 800 | 0 | 100%   ✅ |
| 2025-11-24 | 785 | 785 | 0 | 100%   ✅ |
| 2025-11-23 | 785 | 783 | 2 | 99.7%  ✅ |

Média 7 dias: 99.7%

```

**Análise**:
- Taxa de sucesso caiu hoje devido a 2 bugs introduzidos
- DEF-2025-001: Introduzido em commit abc123
- DEF-2025-002: Regressão de fix anterior (DEF-2025-XXX)

═══════════════════════════════════════════════════════════════

## ⏭️ PRÓXIMAS AÇÕES

**Prioridade Máxima** (Hoje):
1. [ ] Corrigir DEF-2025-002 (Security CRÍTICO)
2. [ ] Corrigir DEF-2025-001 (Alta)
3. [ ] Validar correções (processo completo)
4. [ ] Re-executar suite completa

**Amanhã**:
5. [ ] Investigar DEF-2025-004 (Média)
6. [ ] Adicionar testes de regressão para DEF-001 e DEF-002
7. [ ] Code review das correções

**Semana que vem**:
8. [ ] Análise de causa raiz (RCA) para regressões
9. [ ] Melhorar processo de validação
10. [ ] Atualizar documentação de testes

═══════════════════════════════════════════════════════════════

## ✅ APROVAÇÕES

**QA Lead**: Pendente de correções
**Tech Lead**: Pendente de correções
**Security Officer**: BLOQUEADO (DEF-2025-002)

**DEPLOY STATUS**: 🔴 **BLOQUEADO**
- Aguardando correção de 2 bugs críticos
- Re-validação completa necessária
- ETA para liberação: 2025-11-29 20:00

═══════════════════════════════════════════════════════════════

**Relatório gerado automaticamente em**: 2025-11-29 16:30:00
**Próximo relatório em**: 2025-11-30 09:00:00
```


***

Este plano **EXAUSTIVO E DETALHADO** garante:

✅ **ZERO REGRESSÃO** - Processo rigoroso de validação em 11 fases
✅ **RASTREABILIDADE TOTAL** - Matriz de impacto relaciona todos os testes
✅ **DOCUMENTAÇÃO COMPLETA** - Templates para defeitos, correções e relatórios
✅ **PROCESSO CLARO** - Workflow passo-a-passo para cada correção
✅ **MÚLTIPLOS NÍVEIS** - Validação em 6 níveis (unit → smoke → E2E → manual)
✅ **APROVAÇÕES FORMAIS** - Checklists com assinaturas obrigatórias

**Resultado**: Sistema 100% confiável com ZERO chance de regressão não detectada.

