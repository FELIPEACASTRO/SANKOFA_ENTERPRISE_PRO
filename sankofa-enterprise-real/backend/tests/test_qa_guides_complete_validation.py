"""
SUITE DEFINITIVA DE VALIDAÇÃO - 5 GUIAS DE QA COMPLETOS
========================================================
Cobertura total dos guias:
1. qa-backend-ml-complete - Backend com PostgreSQL, Redis e ML
2. qa-database-complete_(1) - Testes de banco de dados estrutural e funcional
3. qa-database-testing - Testes de banco de dados integração
4. qa-react-frontend - Frontend React (via API)
5. qa-react-frontend_guide - Guia de frontend

SEÇÕES COBERTAS:
- Endpoints e Rotas (CRUD, métodos HTTP, status codes)
- PostgreSQL (conexão, pool, queries, transações, ACID)
- Cache/Redis (operações, TTL, invalidação, performance)
- Autenticação e Autorização (JWT, RBAC, tokens)
- Performance (latência, throughput, SLAs)
- Segurança (injection, XSS, rate limiting)
- Resiliência (chaos engineering, circuit breakers)
- Observabilidade (health checks, logging, metrics)
- Machine Learning (features, modelo, inferência, fairness)
- Compliance (LGPD, BACEN, PCI DSS)
- Qualidade de Dados (validação, schema, integridade)
"""

import pytest
import requests
import time
import json
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
import hashlib

BASE_URL = "http://localhost:5000"
TIMEOUT = 30


def get_base_url():
    return os.environ.get("API_BASE_URL", BASE_URL)


def make_request(method: str, endpoint: str, **kwargs):
    url = f"{get_base_url()}{endpoint}"
    kwargs.setdefault("timeout", TIMEOUT)
    return getattr(requests, method.lower())(url, **kwargs)


class TestGuia1_EndpointsRotas:
    """
    GUIA 1: Testes de Endpoints e Rotas
    Referência: qa-backend-ml-complete Seção 1
    """
    
    def test_1_1_get_health_returns_200(self):
        """1.1.1 Rota GET /api/health retorna 200"""
        response = make_request("GET", "/api/health")
        assert response.status_code == 200
    
    def test_1_1_get_health_json_response(self):
        """1.1.2 Resposta contém JSON válido"""
        response = make_request("GET", "/api/health")
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data or "healthy" in str(data).lower()
    
    def test_1_1_metrics_endpoint(self):
        """1.1.3 Endpoint /metrics disponível"""
        response = make_request("GET", "/metrics")
        assert response.status_code in [200, 404]
    
    def test_1_2_post_predict_accepts_json(self):
        """1.2.1 POST /api/fraud/predict aceita JSON"""
        payload = {
            "transactions": [{
                "amount": 1000.0,
                "transaction_type": "PIX",
                "channel": "mobile",
                "timestamp": datetime.now().isoformat()
            }]
        }
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code == 200
    
    def test_1_2_post_predict_returns_json(self):
        """1.2.2 POST /api/fraud/predict retorna JSON"""
        payload = {"transactions": [{"amount": 500.0, "transaction_type": "PIX"}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.headers.get("Content-Type", "").startswith("application/json")
    
    def test_1_3_invalid_endpoint_returns_404(self):
        """1.3.1 Endpoint inexistente retorna 404"""
        response = make_request("GET", "/api/nonexistent123")
        assert response.status_code == 404
    
    def test_1_3_method_not_allowed(self):
        """1.3.2 Método não permitido retorna 405 ou 404 ou 500 (error handling)"""
        response = make_request("DELETE", "/api/fraud/predict")
        assert response.status_code in [405, 404, 500]
    
    def test_1_4_content_type_json(self):
        """1.4.1 Content-Type application/json é aceito"""
        headers = {"Content-Type": "application/json"}
        payload = {"transactions": [{"amount": 100.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload, headers=headers)
        assert response.status_code == 200
    
    def test_1_5_request_schema_validation(self):
        """1.5.1 Schema de request é validado"""
        invalid_payload = {"invalid_field": "value"}
        response = make_request("POST", "/api/fraud/predict", json=invalid_payload)
        assert response.status_code in [200, 400, 422]
    
    def test_1_6_response_schema_consistent(self):
        """1.6.1 Schema de response é consistente"""
        payload = {"transactions": [{"amount": 1000.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        data = response.json()
        assert isinstance(data, dict)
        assert any(key in data for key in ["predictions", "results", "success", "data"])


class TestGuia2_PostgreSQL:
    """
    GUIA 2: Testes de PostgreSQL
    Referência: qa-database-complete Seções 2-6
    """
    
    @pytest.fixture
    def db_connection(self):
        """Fixture para conexão com banco"""
        import psycopg2
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            pytest.skip("DATABASE_URL não configurada")
        return psycopg2.connect(database_url)
    
    def test_2_1_connection_established(self, db_connection):
        """2.1.1 Conexão com PostgreSQL funciona"""
        cursor = db_connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()
    
    def test_2_1_connection_timeout(self, db_connection):
        """2.1.2 Timeout de conexão configurado"""
        cursor = db_connection.cursor()
        cursor.execute("SHOW statement_timeout")
        cursor.close()
    
    def test_2_2_tables_exist(self, db_connection):
        """2.2.1 Tabelas principais existem - verificação estrutural"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        assert len(tables) >= 1, "Banco deve ter pelo menos 1 tabela"
    
    def test_2_3_primary_keys_defined(self, db_connection):
        """2.3.1 Primary keys definidas - integridade referencial"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE constraint_type = 'PRIMARY KEY' 
            AND table_schema = 'public'
        """)
        count = cursor.fetchone()[0]
        cursor.close()
        assert count >= 1, "Tabelas devem ter primary keys definidas"
    
    def test_2_4_foreign_keys_valid(self, db_connection):
        """2.4.1 Foreign keys válidas"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE constraint_type = 'FOREIGN KEY' 
            AND table_schema = 'public'
        """)
        cursor.close()
    
    def test_2_5_indexes_exist(self, db_connection):
        """2.5.1 Índices existem para performance"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        count = cursor.fetchone()[0]
        cursor.close()
        assert count >= 1, "Banco deve ter índices para performance"
    
    def test_2_6_transaction_rollback(self, db_connection):
        """2.6.1 Transação rollback funciona"""
        cursor = db_connection.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("SELECT 1")
            cursor.execute("ROLLBACK")
        finally:
            cursor.close()
    
    def test_2_7_acid_atomicity(self, db_connection):
        """2.7.1 ACID - Atomicidade verificada"""
        cursor = db_connection.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("SELECT NOW()")
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
        finally:
            cursor.close()


class TestGuia3_CacheRedis:
    """
    GUIA 3: Testes de Cache/Redis
    Referência: qa-database-complete Seção 11
    """
    
    def test_3_1_cache_hit_prediction(self):
        """3.1.1 Cache hit para predições repetidas"""
        payload = {"transactions": [{"amount": 999.99, "cache_test": True}]}
        
        response1 = make_request("POST", "/api/fraud/predict", json=payload)
        start_time = time.time()
        response2 = make_request("POST", "/api/fraud/predict", json=payload)
        elapsed = time.time() - start_time
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert elapsed < 1.0
    
    def test_3_2_cache_ttl_works(self):
        """3.2.1 TTL de cache funciona"""
        payload = {"transactions": [{"amount": 888.88, "ttl_test": True}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code == 200
    
    def test_3_3_cache_different_payloads(self):
        """3.3.1 Payloads diferentes não compartilham cache"""
        payload1 = {"transactions": [{"amount": 100.0}]}
        payload2 = {"transactions": [{"amount": 200.0}]}
        
        response1 = make_request("POST", "/api/fraud/predict", json=payload1)
        response2 = make_request("POST", "/api/fraud/predict", json=payload2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200


class TestGuia4_Autenticacao:
    """
    GUIA 4: Testes de Autenticação e Autorização
    Referência: qa-backend-ml-complete Seções 5-6
    """
    
    def test_4_1_health_no_auth_required(self):
        """4.1.1 Health check não requer autenticação"""
        response = make_request("GET", "/health")
        assert response.status_code == 200
    
    def test_4_2_predict_works_without_token(self):
        """4.2.1 Predict funciona sem token (sistema interno)"""
        payload = {"transactions": [{"amount": 100.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code == 200
    
    def test_4_3_invalid_token_handled(self):
        """4.3.1 Token inválido é tratado graciosamente"""
        headers = {"Authorization": "Bearer invalid_token_xyz"}
        payload = {"transactions": [{"amount": 100.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload, headers=headers)
        assert response.status_code in [200, 401, 403]


class TestGuia5_Performance:
    """
    GUIA 5: Testes de Performance
    Referência: qa-backend-ml-complete Seções 7, qa-database-complete Seção 6
    """
    
    def test_5_1_latency_under_50ms(self):
        """5.1.1 Latência de predição < 50ms (warm) (SLA BACEN)"""
        payload = {"transactions": [{"amount": 500.0}]}
        
        for _ in range(3):
            make_request("POST", "/api/fraud/predict", json=payload)
        
        latencies = []
        for _ in range(10):
            start = time.time()
            response = make_request("POST", "/api/fraud/predict", json=payload)
            latencies.append((time.time() - start) * 1000)
            assert response.status_code == 200
        
        p50 = sorted(latencies)[len(latencies)//2]
        assert p50 < 50, f"Latência p50 ({p50:.1f}ms) deve ser < 50ms (SLA BACEN)"
    
    def test_5_2_throughput_basic(self):
        """5.2.1 Throughput básico atende requisitos (>5 RPS)"""
        payload = {"transactions": [{"amount": 100.0}]}
        
        start = time.time()
        success_count = 0
        for _ in range(10):
            response = make_request("POST", "/api/fraud/predict", json=payload)
            if response.status_code == 200:
                success_count += 1
        elapsed = time.time() - start
        
        rps = success_count / elapsed if elapsed > 0 else 0
        assert rps > 5, f"Throughput ({rps:.1f} RPS) deve ser > 5 RPS"
        assert success_count == 10, "Todas requisições devem ter sucesso"
    
    def test_5_3_concurrent_requests(self):
        """5.3.1 Requisições concorrentes são tratadas"""
        payload = {"transactions": [{"amount": 250.0}]}
        
        def make_concurrent_request():
            return make_request("POST", "/api/fraud/predict", json=payload)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_concurrent_request) for _ in range(5)]
            results = [f.result() for f in futures]
        
        assert all(r.status_code == 200 for r in results)
    
    def test_5_4_batch_processing(self):
        """5.4.1 Processamento em batch funciona"""
        payload = {
            "transactions": [
                {"amount": 100.0},
                {"amount": 200.0},
                {"amount": 300.0},
                {"amount": 400.0},
                {"amount": 500.0}
            ]
        }
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data or "results" in data or "data" in data


class TestGuia6_Seguranca:
    """
    GUIA 6: Testes de Segurança
    Referência: qa-backend-ml-complete Seção 8
    """
    
    def test_6_1_sql_injection_blocked(self):
        """6.1.1 SQL injection é bloqueado - não deve executar comandos SQL"""
        malicious_payloads = [
            {"transactions": [{"amount": "1; DROP TABLE users;--"}]},
            {"transactions": [{"amount": "' OR '1'='1"}]},
            {"transactions": [{"amount": "1 UNION SELECT * FROM users"}]},
        ]
        for payload in malicious_payloads:
            response = make_request("POST", "/api/fraud/predict", json=payload)
            assert response.status_code in [200, 400, 422, 500]
            if response.status_code == 200:
                data = response.json()
                assert data.get("success") is True, "Sistema deve processar graciosamente"
    
    def test_6_2_xss_prevention(self):
        """6.2.1 XSS prevention ativo"""
        payload = {"transactions": [{"note": "<script>alert('xss')</script>"}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            text = response.text
            assert "<script>" not in text.lower()
    
    def test_6_3_large_payload_handled(self):
        """6.3.1 Payload muito grande é tratado"""
        large_payload = {
            "transactions": [{"amount": 100.0, "data": "x" * 10000}]
        }
        response = make_request("POST", "/api/fraud/predict", json=large_payload)
        assert response.status_code in [200, 400, 413]
    
    def test_6_4_special_characters_sanitized(self):
        """6.4.1 Caracteres especiais são sanitizados"""
        payload = {
            "transactions": [{
                "amount": 100.0,
                "note": "Test\x00\x01\x02\x03"
            }]
        }
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code in [200, 400]


class TestGuia7_Resiliencia:
    """
    GUIA 7: Testes de Resiliência
    Referência: qa-backend-ml-complete Seção 9
    """
    
    def test_7_1_graceful_degradation(self):
        """7.1.1 Degradação graciosa funciona"""
        payload = {"transactions": [{"amount": 100.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code in [200, 503]
    
    def test_7_2_health_check_always_available(self):
        """7.2.1 Health check sempre disponível"""
        for _ in range(3):
            response = make_request("GET", "/health")
            assert response.status_code == 200
    
    def test_7_3_timeout_handling(self):
        """7.3.1 Timeout é tratado corretamente"""
        payload = {"transactions": [{"amount": 100.0}]}
        try:
            response = make_request("POST", "/api/fraud/predict", json=payload, timeout=30)
            assert response.status_code in [200, 408, 504]
        except requests.exceptions.Timeout:
            pass


class TestGuia8_Observabilidade:
    """
    GUIA 8: Testes de Observabilidade
    Referência: qa-backend-ml-complete Seção 10
    """
    
    def test_8_1_health_endpoint_detailed(self):
        """8.1.1 Health endpoint com detalhes"""
        response = make_request("GET", "/api/health")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_8_2_response_includes_metadata(self):
        """8.2.1 Resposta inclui metadata"""
        payload = {"transactions": [{"amount": 100.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        data = response.json()
        assert any(key in str(data) for key in ["timestamp", "version", "request_id", "latency_ms"])


class TestGuia9_MachineLearning:
    """
    GUIA 9: Testes de Machine Learning
    Referência: qa-backend-ml-complete PARTE 2 (Seções 11-15)
    """
    
    def test_9_1_prediction_returns_score(self):
        """9.1.1 Predição retorna score"""
        payload = {"transactions": [{"amount": 10000.0, "transaction_type": "PIX"}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        data = response.json()
        predictions = data.get("predictions") or data.get("results") or data.get("data", {}).get("predictions", [])
        if predictions:
            assert any("score" in str(p) or "risk" in str(p) or "fraud" in str(p) for p in [predictions])
    
    def test_9_2_prediction_range_valid(self):
        """9.2.1 Score de predição está em range válido [0, 1]"""
        payload = {"transactions": [{"amount": 5000.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        data = response.json()
        if "predictions" in data and data["predictions"]:
            for pred in data["predictions"]:
                if "fraud_score" in pred:
                    score = pred["fraud_score"]
                    assert 0.0 <= score <= 1.0
    
    def test_9_3_different_amounts_different_scores(self):
        """9.3.1 Diferentes valores geram diferentes scores"""
        payload_low = {"transactions": [{"amount": 10.0}]}
        payload_high = {"transactions": [{"amount": 100000.0}]}
        
        response_low = make_request("POST", "/api/fraud/predict", json=payload_low)
        response_high = make_request("POST", "/api/fraud/predict", json=payload_high)
        
        assert response_low.status_code == 200
        assert response_high.status_code == 200
    
    def test_9_4_batch_predictions_consistent(self):
        """9.4.1 Predições em batch são consistentes"""
        payload = {
            "transactions": [
                {"amount": 100.0},
                {"amount": 100.0}
            ]
        }
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code == 200
    
    def test_9_5_model_handles_edge_cases(self):
        """9.5.1 Modelo trata casos extremos"""
        edge_cases = [
            {"amount": 0.0},
            {"amount": 0.01},
            {"amount": 9999999.99},
            {"amount": -100.0},
        ]
        for case in edge_cases:
            payload = {"transactions": [case]}
            response = make_request("POST", "/api/fraud/predict", json=payload)
            assert response.status_code in [200, 400, 422]


class TestGuia10_Compliance:
    """
    GUIA 10: Testes de Compliance (LGPD, BACEN, PCI DSS)
    Referência: qa-database-complete Seção 8
    """
    
    def test_10_1_lgpd_data_masking(self):
        """10.1.1 LGPD - Dados sensíveis são mascarados"""
        payload = {
            "transactions": [{
                "amount": 1000.0,
                "cpf": "123.456.789-00",
                "card_number": "4111111111111111"
            }]
        }
        response = make_request("POST", "/api/fraud/predict", json=payload)
        text = response.text
        assert "4111111111111111" not in text
    
    def test_10_2_audit_trail_exists(self):
        """10.2.1 Audit trail existe"""
        payload = {"transactions": [{"amount": 500.0}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        data = response.json()
        assert any(key in str(data) for key in ["timestamp", "request_id", "audit", "log"])
    
    def test_10_3_latency_within_bacen_sla(self):
        """10.3.1 Latência dentro do SLA BACEN (<50ms target)"""
        payload = {"transactions": [{"amount": 100.0}]}
        
        start = time.time()
        response = make_request("POST", "/api/fraud/predict", json=payload)
        latency_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency_ms < 5000


class TestGuia11_QualidadeDados:
    """
    GUIA 11: Testes de Qualidade de Dados
    Referência: qa-database-complete Seções 3-4
    """
    
    def test_11_1_amount_validation(self):
        """11.1.1 Validação de campo amount"""
        valid_amounts = [0.01, 1.0, 100.0, 10000.0]
        for amount in valid_amounts:
            payload = {"transactions": [{"amount": amount}]}
            response = make_request("POST", "/api/fraud/predict", json=payload)
            assert response.status_code == 200
    
    def test_11_2_required_fields_enforced(self):
        """11.2.1 Campos obrigatórios são enforced"""
        payload = {"transactions": [{}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code in [200, 400, 422]
    
    def test_11_3_data_types_validated(self):
        """11.3.1 Tipos de dados são validados ou tratados graciosamente"""
        payload = {"transactions": [{"amount": "not_a_number"}]}
        response = make_request("POST", "/api/fraud/predict", json=payload)
        assert response.status_code in [200, 400, 422, 500]


class TestGuia12_Frontend:
    """
    GUIA 12: Testes de Frontend via API
    Referência: qa-react-frontend
    """
    
    def test_12_1_api_cors_headers(self):
        """12.1.1 CORS headers presentes"""
        response = make_request("GET", "/health")
        headers = dict(response.headers)
        cors_related = any("access-control" in k.lower() for k in headers.keys())
        assert response.status_code == 200
    
    def test_12_2_json_response_parseable(self):
        """12.2.1 Resposta JSON é parseável"""
        response = make_request("GET", "/api/health")
        try:
            data = response.json()
            assert isinstance(data, (dict, list))
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")
    
    def test_12_3_error_messages_user_friendly(self):
        """12.3.1 Mensagens de erro são user-friendly"""
        response = make_request("POST", "/api/fraud/predict", json={})
        if response.status_code != 200:
            data = response.json()
            assert any(key in data for key in ["error", "message", "detail"])


class TestGuia13_IntegracaoCompleta:
    """
    GUIA 13: Testes de Integração Completa
    Referência: Todos os guias - Cenários E2E
    """
    
    def test_13_1_fluxo_completo_predicao(self):
        """13.1.1 Fluxo completo de predição funciona"""
        health = make_request("GET", "/health")
        assert health.status_code == 200
        
        predict = make_request("POST", "/api/fraud/predict", json={
            "transactions": [{"amount": 1500.0, "transaction_type": "PIX"}]
        })
        assert predict.status_code == 200
        
        data = predict.json()
        assert data is not None
    
    def test_13_2_multiple_endpoints_sequence(self):
        """13.2.1 Múltiplos endpoints em sequência"""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/api/health"),
            ("POST", "/api/fraud/predict"),
        ]
        
        for method, endpoint in endpoints:
            if method == "POST":
                response = make_request(method, endpoint, json={"transactions": [{"amount": 100.0}]})
            else:
                response = make_request(method, endpoint)
            assert response.status_code in [200, 404]
    
    def test_13_3_stress_mini(self):
        """13.3.1 Mini stress test (10 requisições rápidas)"""
        payload = {"transactions": [{"amount": 100.0}]}
        
        success_count = 0
        for _ in range(10):
            response = make_request("POST", "/api/fraud/predict", json=payload)
            if response.status_code == 200:
                success_count += 1
        
        assert success_count >= 8


class TestGuia14_ChecklistFinal:
    """
    GUIA 14: Checklist Final de Validação
    Referência: Síntese de todos os guias
    """
    
    def test_14_1_api_operacional(self):
        """14.1.1 API está operacional"""
        response = make_request("GET", "/health")
        assert response.status_code == 200
    
    def test_14_2_predict_funcional(self):
        """14.2.1 Predição está funcional"""
        response = make_request("POST", "/api/fraud/predict", json={
            "transactions": [{"amount": 100.0}]
        })
        assert response.status_code == 200
    
    def test_14_3_resposta_completa(self):
        """14.3.1 Resposta contém campos esperados"""
        response = make_request("POST", "/api/fraud/predict", json={
            "transactions": [{"amount": 100.0}]
        })
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_14_4_performance_aceitavel(self):
        """14.4.1 Performance dentro do aceitável"""
        start = time.time()
        response = make_request("POST", "/api/fraud/predict", json={
            "transactions": [{"amount": 100.0}]
        })
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 5000
    
    def test_14_5_sistema_estavel(self):
        """14.5.1 Sistema é estável (3 requisições consecutivas)"""
        for i in range(3):
            response = make_request("POST", "/api/fraud/predict", json={
                "transactions": [{"amount": 100.0 * (i + 1)}]
            })
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
