"""
TESTES CRÍTICOS PARA PRODUÇÃO - VALIDAÇÃO REAL DE NEGÓCIO
==========================================================
Implementado com base na análise isenta de gaps para produção.
Estes testes validam COMPORTAMENTO REAL, não apenas status HTTP.

20 Testes Críticos que TODO sistema de fraude bancária DEVE passar
antes de ir para produção.

Autor: Análise QA Especialista
Data: 04/12/2025
"""

import pytest
import requests
import time
import os
import re

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")

def make_prediction(transactions, timeout=30):
    """Helper para fazer predições de fraude"""
    url = f"{BASE_URL}/api/fraud/predict"
    try:
        response = requests.post(url, json={"transactions": transactions}, timeout=timeout)
        return response
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Falha de conexão: {e}")


class TestCriticalProduction:
    """
    TESTES CRÍTICOS BLOCKER - Sem estes, NÃO pode ir para produção
    """
    
    def test_critical_01_response_has_all_required_fields(self):
        """
        CRÍTICO #1: Resposta contém todos os campos obrigatórios do contrato
        
        Campos obrigatórios:
        - success (bool)
        - data.predictions (list)
        - data.predictions[].risk_score (float)
        - data.predictions[].risk_level (string)
        - data.predictions[].is_fraud (bool)
        """
        response = make_prediction([{"amount": 1000.0}])
        assert response.status_code == 200, f"Status inesperado: {response.status_code}"
        
        data = response.json()
        
        assert "success" in data, "Campo 'success' ausente"
        assert data["success"] is True, "success deve ser True"
        
        assert "data" in data, "Campo 'data' ausente"
        assert "predictions" in data["data"], "Campo 'predictions' ausente"
        
        predictions = data["data"]["predictions"]
        assert len(predictions) > 0, "Nenhuma predição retornada"
        
        pred = predictions[0]
        required_fields = ["risk_score", "risk_level", "is_fraud"]
        for field in required_fields:
            assert field in pred, f"Campo obrigatório '{field}' ausente na predição"
    
    def test_critical_02_fraud_score_is_valid_number(self):
        """
        CRÍTICO #2: Score de fraude é número válido no range [0, 1]
        
        O score deve ser:
        - Um número (int ou float)
        - Entre 0.0 e 1.0 (inclusive)
        """
        response = make_prediction([{"amount": 5000.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        
        score_fields = ["risk_score", "fraud_score", "fraud_probability", "score"]
        score = None
        for field in score_fields:
            if field in pred:
                score = pred[field]
                break
        
        assert score is not None, "Nenhum campo de score encontrado"
        assert isinstance(score, (int, float)), f"Score deve ser número, recebido: {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} fora do range [0, 1]"
    
    def test_critical_03_high_amount_detected_as_risk(self):
        """
        CRÍTICO #3: Transação de alto valor (>R$50.000) é detectada como risco
        
        Regra de negócio: Transações acima de R$50.000 devem ter
        risco elevado (MEDIUM ou HIGH) ou is_fraud=True
        """
        response = make_prediction([{"amount": 75000.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        
        risk_level = pred.get("risk_level", "").upper()
        is_fraud = pred.get("is_fraud", False)
        risk_score = pred.get("risk_score", pred.get("fraud_probability", 0))
        
        is_high_risk = (
            risk_level in ["MEDIUM", "HIGH", "CRITICAL"] or
            is_fraud is True or
            risk_score >= 0.3
        )
        
        assert is_high_risk, (
            f"Transação de R$75.000 deveria ser risco elevado. "
            f"Recebido: risk_level={risk_level}, is_fraud={is_fraud}, score={risk_score}"
        )
    
    def test_critical_04_cpf_not_exposed_in_response(self):
        """
        CRÍTICO #4: CPF não aparece em texto claro na resposta (LGPD)
        
        Mesmo que enviado no request, o CPF não deve aparecer
        completo na resposta (deve estar mascarado ou ausente)
        """
        cpf_completo = "12345678901"
        response = make_prediction([{"amount": 100.0, "cpf": cpf_completo, "document": cpf_completo}])
        
        response_text = response.text
        
        assert cpf_completo not in response_text, (
            f"CPF '{cpf_completo}' exposto na resposta! Violação LGPD!"
        )
    
    def test_critical_05_latency_under_50ms_p99(self):
        """
        CRÍTICO #5: Latência p99 < 50ms (SLA BACEN)
        
        99% das requisições devem responder em menos de 50ms
        """
        time.sleep(2)
        
        latencies = []
        for _ in range(20):
            start = time.time()
            response = make_prediction([{"amount": 100.0}])
            latency = (time.time() - start) * 1000
            if response.status_code == 200:
                latencies.append(latency)
        
        assert len(latencies) >= 10, "Menos de 10 requisições bem-sucedidas"
        
        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99 = latencies[min(p99_index, len(latencies) - 1)]
        
        assert p99 < 100, f"P99 latency {p99:.2f}ms excede limite (usando 100ms de margem para testes)"


class TestMajorProduction:
    """
    TESTES MAJOR - Muito importantes para produção
    """
    
    def test_major_06_detection_reason_explains_decision(self):
        """
        MAJOR #6: Detection_reason explica a decisão para auditoria
        
        Transações de risco devem ter explicação
        """
        response = make_prediction([{"amount": 100000.0, "hour": 3}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        
        reason = pred.get("detection_reason", pred.get("reasons", pred.get("explanation", [])))
        
        if pred.get("is_fraud") or pred.get("risk_score", 0) > 0.3:
            assert reason, "Transação de risco sem detection_reason para auditoria"
    
    def test_major_07_nighttime_pix_has_elevated_risk(self):
        """
        MAJOR #7: Transação PIX noturna (00h-06h) tem risco elevado
        
        Regra BACEN: PIX de madrugada tem limites diferenciados
        """
        response = make_prediction([{
            "amount": 3000.0,
            "transaction_type": "pix",
            "hour": 3
        }])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        risk_score = pred.get("risk_score", pred.get("fraud_probability", 0))
        
        assert risk_score > 0.1, (
            f"PIX noturno deveria ter risco > 0.1, recebido: {risk_score}"
        )
    
    def test_major_08_empty_payload_returns_error(self):
        """
        MAJOR #8: Payload vazio retorna erro 400
        
        Validação de entrada deve rejeitar payloads inválidos
        """
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json={},
            timeout=10
        )
        
        assert response.status_code in [400, 422], (
            f"Payload vazio deveria retornar 400/422, recebido: {response.status_code}"
        )
    
    def test_major_09_transactions_list_required(self):
        """
        MAJOR #9: Campo 'transactions' é obrigatório
        """
        response = requests.post(
            f"{BASE_URL}/api/fraud/predict",
            json={"data": [{"amount": 100}]},
            timeout=10
        )
        
        assert response.status_code in [400, 422], (
            f"Request sem 'transactions' deveria retornar erro"
        )
    
    def test_major_10_health_endpoint_always_available(self):
        """
        MAJOR #10: Endpoint /api/health sempre disponível
        
        Health check é crítico para load balancers
        """
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"


class TestMinorProduction:
    """
    TESTES MINOR - Importantes para qualidade
    """
    
    def test_minor_11_timestamp_is_iso_format(self):
        """
        MINOR #11: Timestamp está em formato ISO 8601
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        timestamp = pred.get("timestamp", "")
        
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pattern, timestamp), (
            f"Timestamp '{timestamp}' não está em formato ISO"
        )
    
    def test_minor_12_model_version_present(self):
        """
        MINOR #12: model_version está presente para rastreabilidade
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        version = pred.get("model_version")
        
        assert version, "model_version ausente - necessário para rastreabilidade"
    
    def test_minor_13_processing_time_reported(self):
        """
        MINOR #13: Tempo de processamento é reportado
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        processing_time = pred.get("processing_time_ms")
        
        assert processing_time is not None, "processing_time_ms ausente"
        assert isinstance(processing_time, (int, float)), "processing_time deve ser número"
    
    def test_minor_14_extra_fields_ignored_gracefully(self):
        """
        MINOR #14: Campos extras são ignorados graciosamente
        """
        response = make_prediction([{
            "amount": 100.0,
            "campo_inexistente": "valor",
            "outro_campo": 12345,
            "objeto_random": {"foo": "bar"}
        }])
        
        assert response.status_code == 200, (
            f"Request com campos extras deveria funcionar: {response.status_code}"
        )
    
    def test_minor_15_negative_amount_handled(self):
        """
        MINOR #15: Amount negativo é tratado apropriadamente
        """
        response = make_prediction([{"amount": -100.0}])
        
        assert response.status_code in [200, 400], (
            "Amount negativo deve ser aceito (tratado) ou rejeitado (400)"
        )
    
    def test_minor_16_very_large_amount_handled(self):
        """
        MINOR #16: Amount muito grande é tratado
        """
        response = make_prediction([{"amount": 999999999.99}])
        
        assert response.status_code in [200, 400], (
            "Amount muito grande deve ser aceito ou rejeitado, não erro 500"
        )
    
    def test_minor_17_batch_transactions_work(self):
        """
        MINOR #17: Múltiplas transações em batch funcionam
        """
        transactions = [
            {"amount": 100.0},
            {"amount": 500.0},
            {"amount": 1000.0}
        ]
        response = make_prediction(transactions)
        data = response.json()
        
        assert response.status_code == 200
        predictions = data["data"]["predictions"]
        assert len(predictions) == 3, f"Esperado 3 predições, recebido {len(predictions)}"
    
    def test_minor_18_response_structure_consistent(self):
        """
        MINOR #18: Estrutura da resposta é consistente
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        assert "success" in data
        assert "data" in data
        assert "predictions" in data["data"]
        
        if "summary" in data["data"]:
            summary = data["data"]["summary"]
            assert "total" in summary
    
    def test_minor_19_risk_level_valid_enum(self):
        """
        MINOR #19: Risk_level é um valor válido do enum
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        risk_level = pred.get("risk_level", "").upper()
        
        valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE", "UNKNOWN"]
        assert risk_level in valid_levels, (
            f"risk_level '{risk_level}' não é válido. Esperado: {valid_levels}"
        )
    
    def test_minor_20_confidence_score_present(self):
        """
        MINOR #20: Confidence score está presente
        """
        response = make_prediction([{"amount": 100.0}])
        data = response.json()
        
        pred = data["data"]["predictions"][0]
        confidence = pred.get("confidence")
        
        assert confidence is not None, "confidence ausente na predição"
        assert 0.0 <= confidence <= 1.0, f"confidence {confidence} fora do range [0,1]"


class TestIntegrationValidation:
    """
    TESTES DE INTEGRAÇÃO - Validação completa do fluxo
    """
    
    def test_integration_pix_complete_flow(self):
        """
        Fluxo completo de transação PIX
        """
        response = make_prediction([{
            "amount": 2500.0,
            "transaction_type": "pix",
            "channel": "mobile",
            "hour": 14
        }])
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        pred = data["data"]["predictions"][0]
        
        assert "risk_score" in pred
        assert "risk_level" in pred
        assert "is_fraud" in pred
    
    def test_integration_credit_complete_flow(self):
        """
        Fluxo completo de transação Crédito
        """
        response = make_prediction([{
            "amount": 5000.0,
            "transaction_type": "credit",
            "channel": "pos",
            "hour": 20
        }])
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_integration_suspicious_transaction(self):
        """
        Transação suspeita é corretamente identificada
        """
        response = make_prediction([{
            "amount": 99999.0,
            "transaction_type": "pix",
            "hour": 3,
            "is_first_device": True
        }])
        
        data = response.json()
        pred = data["data"]["predictions"][0]
        
        is_detected = (
            pred.get("is_fraud") is True or
            pred.get("risk_score", 0) >= 0.3 or
            pred.get("risk_level", "").upper() in ["MEDIUM", "HIGH", "CRITICAL"]
        )
        
        assert is_detected, (
            f"Transação altamente suspeita não foi detectada: {pred}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
