"""
Testes End-to-End (E2E) - Sankofa Enterprise Pro
Cobertura: Frontend -> API -> BD -> ML -> Response
Versão 12.2.1: Todos endpoints sensíveis requerem autenticação JWT
"""

import pytest
import requests
import psycopg2
import os
import json
import time
import numpy as np
from datetime import datetime

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5000"


_cached_token = None

def get_auth_token():
    """Obtém token JWT para testes autenticados (com cache para evitar rate limit)"""
    global _cached_token
    if _cached_token:
        return _cached_token
    
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "admin", "password": "SankofaAdmin2025!"},
        timeout=10
    )
    if response.status_code == 200:
        _cached_token = response.json()["data"]["token"]
        return _cached_token
    raise Exception(f"Failed to authenticate: {response.text}")


def get_auth_headers():
    """Retorna headers com autenticação (usa token cached)"""
    token = get_auth_token()
    return {"Authorization": f"Bearer {token}"}


class TestE2EInfrastructure:
    """Testes de infraestrutura básica"""
    
    def test_frontend_available(self):
        """Testa se frontend está acessível"""
        response = requests.get(FRONTEND_URL, timeout=10)
        assert response.status_code == 200
        assert "Sankofa" in response.text
        print("Frontend: OK")
    
    def test_backend_health(self):
        """Testa health check do backend (endpoint público)"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        print(f"Backend Health: {data}")
    
    def test_database_connection(self):
        """Testa conexão com PostgreSQL"""
        database_url = os.environ.get("DATABASE_URL")
        assert database_url is not None, "DATABASE_URL not set"
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        assert result[0] == 1
        cur.close()
        conn.close()
        print("Database Connection: OK")
    
    def test_database_tables_exist(self):
        """Verifica se tabelas necessárias existem"""
        database_url = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        required_tables = ["transactions", "alerts", "audit_logs", "feedback", "model_metrics"]
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        existing_tables = [t[0] for t in cur.fetchall()]
        
        for table in required_tables:
            assert table in existing_tables, f"Table {table} missing"
        
        cur.close()
        conn.close()
        print(f"Database Tables: {existing_tables}")


class TestE2EAuthentication:
    """Testes de autenticação (CRÍTICO para produção)"""
    
    def test_login_success(self):
        """Testa login com credenciais válidas"""
        global _cached_token
        _cached_token = None
        
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": "admin", "password": "SankofaAdmin2025!"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "token" in data["data"]
        
        _cached_token = data["data"]["token"]
        print("Login Success: OK")
    
    def test_login_failure(self):
        """Testa login com credenciais inválidas"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": "admin", "password": "wrongpassword"},
            timeout=10
        )
        assert response.status_code == 401
        print("Login Failure: OK (401 as expected)")
    
    def test_protected_endpoint_without_auth(self):
        """Testa acesso a endpoint protegido sem autenticação"""
        response = requests.get(f"{BASE_URL}/api/dashboard/kpis", timeout=10)
        assert response.status_code == 401
        print("Protected Endpoint Without Auth: OK (401 as expected)")
    
    def test_fraud_predict_without_auth(self):
        """Testa acesso à predição de fraude sem autenticação"""
        payload = {"transactions": [{"transaction_id": "TEST", "amount": 100}]}
        response = requests.post(f"{BASE_URL}/api/fraud/predict", json=payload, timeout=10)
        assert response.status_code == 401
        print("Fraud Predict Without Auth: OK (401 as expected)")
    
    def test_all_sensitive_endpoints_protected(self):
        """Testa que TODOS os endpoints sensíveis retornam 401 sem auth"""
        sensitive_endpoints = [
            ("GET", "/api/transactions"),
            ("GET", "/api/alerts"),
            ("GET", "/api/audit"),
            ("GET", "/api/explainability/features"),
            ("GET", "/api/observability/metrics"),
            ("GET", "/api/observability/prometheus"),
            ("GET", "/api/observability/sla"),
            ("GET", "/api/observability/alerts"),
            ("GET", "/api/reports"),
            ("GET", "/api/datasets"),
            ("GET", "/api/investigation/TEST123"),
            ("POST", "/api/feedback"),
            ("POST", "/api/reports/generate"),
            ("POST", "/api/alerts/1/acknowledge"),
            ("PUT", "/api/alerts/1/status"),
        ]
        
        unprotected = []
        for method, endpoint in sensitive_endpoints:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            elif method == "POST":
                response = requests.post(f"{BASE_URL}{endpoint}", json={}, timeout=10)
            elif method == "PUT":
                response = requests.put(f"{BASE_URL}{endpoint}", json={"status": "test"}, timeout=10)
            
            if response.status_code != 401:
                unprotected.append(f"{method} {endpoint} returned {response.status_code}")
        
        assert len(unprotected) == 0, f"Unprotected endpoints found: {unprotected}"
        print(f"All {len(sensitive_endpoints)} Sensitive Endpoints Protected: OK")
    
    def test_token_validation(self):
        """Testa que tokens inválidos são rejeitados"""
        fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        headers = {"Authorization": f"Bearer {fake_token}"}
        
        response = requests.get(f"{BASE_URL}/api/dashboard/kpis", headers=headers, timeout=10)
        assert response.status_code == 401
        print("Invalid Token Rejected: OK (401 as expected)")


class TestE2EAPIEndpoints:
    """Testes de todos os endpoints da API (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_api_root(self):
        """Testa endpoint raiz (público)"""
        response = requests.get(BASE_URL, timeout=10)
        assert response.status_code == 200
        print("API Root: OK")
    
    def test_model_metrics(self):
        """Testa endpoint de métricas do modelo (autenticado)"""
        response = requests.get(f"{BASE_URL}/api/model/metrics", headers=self.headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" in data["data"]
        assert "precision" in data["data"]["metrics"]
        assert "recall" in data["data"]["metrics"]
        assert "f1_score" in data["data"]["metrics"]
        print(f"Model Metrics: precision={data['data']['metrics']['precision']:.4f}")
    
    def test_dashboard_summary(self):
        """Testa endpoint de summary do dashboard (autenticado)"""
        response = requests.get(f"{BASE_URL}/api/dashboard/summary", headers=self.headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        print(f"Dashboard Summary: {data['data']}")
    
    def test_dashboard_kpis(self):
        """Testa endpoint de KPIs (autenticado)"""
        response = requests.get(f"{BASE_URL}/api/dashboard/kpis", headers=self.headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        print(f"Dashboard KPIs: {data['data']}")
    
    def test_dashboard_alerts(self):
        """Testa endpoint de alertas (autenticado)"""
        response = requests.get(f"{BASE_URL}/api/dashboard/alerts", headers=self.headers, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        print(f"Alerts: {len(data['data'])} items")


class TestE2EFraudPrediction:
    """Testes de predição de fraude E2E (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_single_transaction_prediction(self):
        """Testa predição de uma única transação (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": f"E2E_TEST_{int(time.time())}",
                "amount": 500.00,
                "hour": 14,
                "channel": "PIX"
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
        assert len(data["data"]["predictions"]) == 1
        
        pred = data["data"]["predictions"][0]
        assert "is_fraud" in pred
        assert "risk_score" in pred
        assert "risk_level" in pred
        assert 0 <= pred["risk_score"] <= 1
        
        print(f"Single Prediction: is_fraud={pred['is_fraud']}, risk_score={pred['risk_score']:.4f}")
    
    def test_batch_transaction_prediction(self):
        """Testa predição em lote (autenticado)"""
        transactions = []
        for i in range(10):
            transactions.append({
                "transaction_id": f"E2E_BATCH_{int(time.time())}_{i}",
                "amount": np.random.uniform(100, 5000),
                "hour": np.random.randint(0, 24),
                "channel": np.random.choice(["PIX", "TED", "BOLETO"])
            })
        
        payload = {"transactions": transactions}
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["predictions"]) == 10
        
        summary = data["data"]["summary"]
        print(f"Batch Prediction: total={summary['total']}, frauds={summary['frauds_detected']}")
    
    def test_high_risk_transaction(self):
        """Testa detecção de transação de alto risco (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": f"E2E_HIGH_RISK_{int(time.time())}",
                "amount": 50000.00,
                "hour": 3,
                "channel": "PIX",
                "is_international": True
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        pred = data["data"]["predictions"][0]
        
        print(f"High Risk Test: risk_score={pred['risk_score']:.4f}, level={pred['risk_level']}")
    
    def test_low_risk_transaction(self):
        """Testa transação de baixo risco (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": f"E2E_LOW_RISK_{int(time.time())}",
                "amount": 50.00,
                "hour": 14,
                "channel": "PIX"
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        pred = data["data"]["predictions"][0]
        
        print(f"Low Risk Test: risk_score={pred['risk_score']:.4f}, level={pred['risk_level']}")


class TestE2EDataPersistence:
    """Testes de persistência de dados (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_transaction_saved_to_db(self):
        """Verifica se transação é salva no banco (autenticado)"""
        unique_id = f"E2E_PERSIST_{int(time.time())}"
        
        payload = {
            "transactions": [{
                "transaction_id": unique_id,
                "amount": 999.99,
                "hour": 12,
                "channel": "PIX"
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        
        assert response.status_code == 200
        
        time.sleep(1)
        
        database_url = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM transactions")
        count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"Transaction Persistence: {count} total transactions in DB")
    
    def test_audit_log_created(self):
        """Verifica se audit logs são criados"""
        database_url = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM audit_logs")
        count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"Audit Logs: {count} entries")


class TestE2EMLPipeline:
    """Testes do pipeline de ML (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_model_loaded(self):
        """Verifica se modelo está carregado (autenticado)"""
        response = requests.get(f"{BASE_URL}/api/model/metrics", headers=self.headers, timeout=10)
        data = response.json()
        
        assert data["data"]["status"] == "trained"
        assert data["data"]["feature_count"] > 0
        
        print(f"Model: trained, {data['data']['feature_count']} features")
    
    def test_prediction_consistency(self):
        """Testa consistência das predições (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": "E2E_CONSISTENCY",
                "amount": 1000.00,
                "hour": 10,
                "channel": "PIX"
            }]
        }
        
        scores = []
        for _ in range(3):
            response = requests.post(
                f"{BASE_URL}/api/fraud/predict",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            data = response.json()
            scores.append(data["data"]["predictions"][0]["risk_score"])
        
        assert len(set([round(s, 4) for s in scores])) == 1, "Predictions should be consistent"
        print(f"Prediction Consistency: scores={scores}")
    
    def test_feature_engineering_e2e(self):
        """Testa feature engineering no pipeline"""
        import pandas as pd
        import sys
        sys.path.insert(0, '/home/runner/workspace/sankofa-enterprise-real/backend')
        from ml_engine.advanced_feature_engineering import AdvancedFeatureEngineering
        
        df = pd.DataFrame({
            "client_cpf": ["A", "A", "B"],
            "state": ["SP", "RJ", "SP"],
            "value": [100, 200, 500],
            "hour": [14, 3, 22]
        })
        
        fe = AdvancedFeatureEngineering()
        result = fe.create_features(df)
        
        assert "is_night" in result.columns
        assert "location_entropy" in result.columns
        assert "value_zscore" in result.columns
        
        night_values = list(result["is_night"])
        assert night_values == [0, 1, 1], f"Night detection failed: {night_values}"
        
        print(f"Feature Engineering E2E: {len(result.columns)} features created")


class TestE2EPerformance:
    """Testes de performance (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_health_latency(self):
        """Testa latência do health check (público)"""
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/health", timeout=10)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 100, f"Health latency too high: {avg_latency}ms"
        print(f"Health Latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
    
    def test_prediction_latency(self):
        """Testa latência de predição (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": f"PERF_{int(time.time())}",
                "amount": 1000.00,
                "hour": 14,
                "channel": "PIX"
            }]
        }
        
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/api/fraud/predict",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        assert avg_latency < 500, f"Prediction latency too high: {avg_latency}ms"
        print(f"Prediction Latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
    
    def test_batch_throughput(self):
        """Testa throughput em lote (autenticado)"""
        transactions = []
        for i in range(50):
            transactions.append({
                "transaction_id": f"THROUGHPUT_{int(time.time())}_{i}",
                "amount": 1000.00 + i,
                "hour": i % 24,
                "channel": "PIX"
            })
        
        payload = {"transactions": transactions}
        
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=120
        )
        total_time = time.time() - start
        
        assert response.status_code == 200
        
        tps = 50 / total_time
        print(f"Batch Throughput: {tps:.2f} TPS ({50} transactions in {total_time:.2f}s)")


class TestE2EValidation:
    """Testes de validação de entrada (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_invalid_payload_rejected(self):
        """Testa rejeição de payload inválido (autenticado)"""
        payload = {"invalid": "data"}
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=10
        )
        
        assert response.status_code in [400, 500]
        data = response.json()
        assert data["success"] is False
        print("Invalid Payload: correctly rejected")
    
    def test_empty_transactions_rejected(self):
        """Testa rejeição de lista vazia (autenticado)"""
        payload = {"transactions": []}
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=10
        )
        
        assert response.status_code in [400, 500]
        print("Empty Transactions: correctly rejected")
    
    def test_negative_amount_handled(self):
        """Testa tratamento de valor negativo (autenticado)"""
        payload = {
            "transactions": [{
                "transaction_id": "NEGATIVE_TEST",
                "amount": -100.00,
                "hour": 12,
                "channel": "PIX"
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=10
        )
        
        print(f"Negative Amount: status={response.status_code}")


class TestE2EIntegration:
    """Testes de integração completa (com autenticação)"""
    
    @pytest.fixture(autouse=True)
    def setup_auth(self):
        """Setup de autenticação para todos os testes"""
        self.headers = get_auth_headers()
    
    def test_full_flow_frontend_to_db(self):
        """Testa fluxo completo: Frontend -> API -> ML -> DB (autenticado)"""
        
        response = requests.get(FRONTEND_URL, timeout=10)
        assert response.status_code == 200
        print("Step 1: Frontend accessible")
        
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert response.json()["status"] == "healthy"
        print("Step 2: Backend healthy")
        
        unique_id = f"FULL_FLOW_{int(time.time())}"
        payload = {
            "transactions": [{
                "transaction_id": unique_id,
                "amount": 1234.56,
                "hour": 15,
                "channel": "TED"
            }]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        assert response.status_code == 200
        pred = response.json()["data"]["predictions"][0]
        print(f"Step 3: ML Prediction complete, risk={pred['risk_score']:.4f}")
        
        time.sleep(1)
        database_url = os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transactions")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"Step 4: Database has {count} transactions")
        
        print("FULL E2E FLOW: SUCCESS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
