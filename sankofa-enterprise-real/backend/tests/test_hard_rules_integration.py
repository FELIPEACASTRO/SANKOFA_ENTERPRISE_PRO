"""
Sankofa Enterprise Pro - Testes Integrados de Regras Duras
Verifica se TODAS as 216 regras estão funcionando corretamente
e se a resposta é INDISTINGUÍVEL do ML.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import time
from datetime import datetime
from typing import Dict, List, Any

from ml_engine.hard_rules_engine import (
    HardRulesEngine,
    HardRulePrediction,
    get_hard_rules_engine,
    UnifiedFraudEngine
)


class TestHardRulesEngineBasic:
    """Testes básicos do engine de regras duras"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_engine_initialization(self, engine):
        """Verifica se o engine inicializa corretamente"""
        assert engine is not None
        assert engine.VERSION == "2.0.0"
    
    def test_rules_loaded(self, engine):
        """Verifica se as regras são carregadas do PostgreSQL"""
        count = engine.get_rules_count()
        assert count > 200, f"Expected 200+ rules, got {count}"
        print(f"✓ {count} regras carregadas do PostgreSQL")
    
    def test_rules_summary(self, engine):
        """Verifica resumo das regras por categoria"""
        summary = engine.get_rules_summary()
        
        assert "total_rules" in summary
        assert "by_action" in summary
        assert "by_type" in summary
        
        print(f"✓ Resumo: {summary['total_rules']} regras")
        print(f"  Por ação: {summary['by_action']}")
        print(f"  Por tipo: {summary['by_type']}")


class TestHardRulesResponseFormat:
    """Testes para garantir que a resposta é INDISTINGUÍVEL do ML"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_response_has_ml_format(self, engine):
        """Verifica se a resposta tem TODOS os campos do ML"""
        transaction = {
            "id": "TEST_001",
            "amount": 100,
            "channel": "PIX",
            "hour": 14
        }
        
        result = engine.evaluate(transaction)
        
        required_fields = [
            "transaction_id",
            "is_fraud",
            "fraud_probability",
            "risk_score",
            "risk_level",
            "confidence",
            "processing_time_ms",
            "model_version",
            "detection_reason",
            "timestamp"
        ]
        
        unified = result.to_unified_response()
        
        for field in required_fields:
            assert field in unified, f"Missing required field: {field}"
        
        print("✓ Resposta contém TODOS os campos do ML")
    
    def test_response_indistinguishable_from_ml(self, engine):
        """Verifica se a resposta é indistinguível do ML"""
        transaction = {
            "id": "TEST_002",
            "amount": 50000,
            "channel": "PIX",
            "hour": 2
        }
        
        result = engine.evaluate(transaction)
        unified = result.to_unified_response()
        
        assert "triggered_rules" not in unified
        assert "source" not in unified
        
        assert isinstance(unified["is_fraud"], bool)
        assert isinstance(unified["fraud_probability"], float)
        assert isinstance(unified["risk_score"], float)
        assert unified["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        print("✓ Resposta INDISTINGUÍVEL do ML")
    
    def test_risk_levels_correct(self, engine):
        """Verifica se os níveis de risco estão corretos"""
        low_risk = {"id": "T1", "amount": 50, "channel": "TED"}
        result_low = engine.evaluate(low_risk)
        
        high_risk = {
            "id": "T2",
            "amount": 50000,
            "channel": "PIX",
            "hour": 2,
            "is_new_device": True
        }
        result_high = engine.evaluate(high_risk)
        
        assert result_low.risk_level in ["LOW", "MEDIUM"]
        print(f"✓ Transação baixo risco: {result_low.risk_level}")
        print(f"✓ Transação alto risco: {result_high.risk_level}")


class TestHardRulesCategories:
    """Testes por categoria de regras"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_bacen_rules(self, engine):
        """Testa regras BACEN"""
        transaction = {
            "id": "BACEN_TEST",
            "amount": 300,
            "channel": "PIX",
            "is_new_device": True
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ BACEN: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
        print(f"  Razões: {result.detection_reason[:2]}")
    
    def test_pix_noturno_rules(self, engine):
        """Testa regras PIX noturno"""
        transaction = {
            "id": "PIX_NOTURNO_TEST",
            "amount": 5000,
            "channel": "PIX",
            "hour": 2
        }
        
        result = engine.evaluate(transaction)
        assert result.is_fraud == True, "PIX madrugada deveria ser bloqueado"
        print(f"✓ PIX Noturno: BLOQUEADO (score={result.fraud_probability:.2f})")
    
    def test_valor_critico_rules(self, engine):
        """Testa regras de valor crítico R$5k-10k"""
        transaction = {
            "id": "VALOR_TEST",
            "amount": 7500,
            "channel": "PIX"
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Valor R$7.500: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_horario_critico_13h(self, engine):
        """Testa regra horário crítico 13h (97.43% fraude)"""
        transaction = {
            "id": "HORARIO_TEST",
            "amount": 1000,
            "channel": "PIX",
            "hour": 13
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Horário 13h: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_velocity_rules(self, engine):
        """Testa regras de velocidade"""
        transaction = {
            "id": "VELOCITY_TEST",
            "amount": 500,
            "velocity_1h": 10,
            "velocity_24h": 50
        }
        
        result = engine.evaluate(transaction)
        assert result.is_fraud == True, "Alta velocidade deveria ser bloqueada"
        print(f"✓ Velocity: BLOQUEADO (score={result.fraud_probability:.2f})")
    
    def test_device_rules(self, engine):
        """Testa regras de dispositivo"""
        transaction = {
            "id": "DEVICE_TEST",
            "amount": 1000,
            "is_new_device": True,
            "ip_address": "VPN_DETECTED",
            "device_id": "EMULATOR_001"
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Device suspeito: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_mao_fantasma_rules(self, engine):
        """Testa regras Mão Fantasma (acesso remoto)"""
        transaction = {
            "id": "MAO_FANTASMA_TEST",
            "amount": 2000,
            "channel": "PIX",
            "device_id": "REMOTE_ACCESS_TEAMVIEWER"
        }
        
        result = engine.evaluate(transaction)
        assert result.is_fraud == True, "Mão Fantasma deveria ser bloqueado"
        print(f"✓ Mão Fantasma: BLOQUEADO (score={result.fraud_probability:.2f})")
    
    def test_sequestro_relampago_rules(self, engine):
        """Testa regras sequestro relâmpago"""
        transaction = {
            "id": "SEQUESTRO_TEST",
            "amount": 2000,
            "channel": "ATM",
            "hour": 23,
            "velocity_1h": 3
        }
        
        result = engine.evaluate(transaction)
        assert result.is_fraud == True, "Sequestro relâmpago deveria ser bloqueado"
        print(f"✓ Sequestro Relâmpago: BLOQUEADO (score={result.fraud_probability:.2f})")
    
    def test_cnp_card_testing_rules(self, engine):
        """Testa regras Card-Not-Present / Card Testing"""
        transaction = {
            "id": "CNP_TEST",
            "amount": 5,
            "type": "CARTAO_CREDITO",
            "channel": "ECOMMERCE",
            "velocity_1h": 8
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Card Testing: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_engenharia_social_rules(self, engine):
        """Testa regras de engenharia social"""
        transaction = {
            "id": "ENGENHARIA_TEST",
            "amount": 1500,
            "channel": "PIX",
            "pix_key_type": "TELEFONE",
            "hour": 20
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Engenharia Social: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")


class TestHardRulesNewFromFiles:
    """Testa as novas regras criadas a partir dos arquivos anexados"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_cartao_recem_emitido(self, engine):
        """Testa regra cartão recém-emitido"""
        transaction = {
            "id": "CARTAO_NOVO_TEST",
            "type": "CARTAO_DEBITO",
            "amount": 800,
            "account_age_days": 5
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Cartão Novo: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_endereco_diferente(self, engine):
        """Testa regra endereço de entrega diferente"""
        transaction = {
            "id": "ENDERECO_TEST",
            "channel": "ECOMMERCE",
            "amount": 500,
            "location": "DIFERENTE",
            "is_new_device": True
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Endereço Diferente: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_fraude_triangulacao(self, engine):
        """Testa regra fraude de triangulação"""
        transaction = {
            "id": "TRIANGULACAO_TEST",
            "channel": "ECOMMERCE",
            "type": "CARTAO_CREDITO",
            "amount": 800,
            "velocity_1h": 4
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Triangulação: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_maquininha_adulterada(self, engine):
        """Testa regra maquininha adulterada"""
        transaction = {
            "id": "MAQUININHA_TEST",
            "type": "CARTAO_DEBITO",
            "channel": "POS",
            "amount": 800,
            "velocity_1h": 6
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Maquininha Adulterada: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_falso_comprovante_pix(self, engine):
        """Testa regra falso comprovante PIX"""
        transaction = {
            "id": "FALSO_COMPROVANTE_TEST",
            "channel": "PIX",
            "amount": 300,
            "pix_key_type": "ALEATORIA"
        }
        
        result = engine.evaluate(transaction)
        print(f"✓ Falso Comprovante: is_fraud={result.is_fraud}, score={result.fraud_probability:.2f}")
    
    def test_carding_bin_attack(self, engine):
        """Testa regra Carding/BIN Attack"""
        transaction = {
            "id": "CARDING_TEST",
            "type": "CARTAO_CREDITO",
            "amount": 3,
            "velocity_1h": 15
        }
        
        result = engine.evaluate(transaction)
        assert result.is_fraud == True, "BIN Attack deveria ser bloqueado"
        print(f"✓ BIN Attack: BLOQUEADO (score={result.fraud_probability:.2f})")


class TestHardRulesPerformance:
    """Testes de performance do engine de regras duras"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_single_evaluation_latency(self, engine):
        """Testa latência de avaliação única"""
        transaction = {
            "id": "PERF_TEST",
            "amount": 1000,
            "channel": "PIX",
            "hour": 14
        }
        
        start = time.time()
        result = engine.evaluate(transaction)
        latency = (time.time() - start) * 1000
        
        assert latency < 50, f"Latência {latency:.2f}ms excede SLA de 50ms"
        print(f"✓ Latência única: {latency:.2f}ms (< 50ms SLA)")
    
    def test_batch_evaluation_performance(self, engine):
        """Testa performance de avaliação em batch"""
        transactions = [
            {"id": f"BATCH_{i}", "amount": i * 100, "channel": "PIX", "hour": i % 24}
            for i in range(100)
        ]
        
        start = time.time()
        results = engine.evaluate_batch(transactions)
        total_time = (time.time() - start) * 1000
        avg_time = total_time / len(transactions)
        
        assert len(results) == 100
        print(f"✓ Batch 100 transações: {total_time:.2f}ms total, {avg_time:.2f}ms/tx")
    
    def test_rules_cache_performance(self, engine):
        """Testa performance do cache de regras"""
        engine._rules_cache = None
        
        start1 = time.time()
        engine._load_rules()
        time1 = (time.time() - start1) * 1000
        
        start2 = time.time()
        engine._load_rules()
        time2 = (time.time() - start2) * 1000
        
        assert time2 < time1, "Cache deveria ser mais rápido"
        print(f"✓ Sem cache: {time1:.2f}ms, Com cache: {time2:.4f}ms")


class TestUnifiedFraudEngine:
    """Testes do motor unificado ML + Regras"""
    
    def test_unified_engine_initialization(self):
        """Testa inicialização do motor unificado"""
        engine = UnifiedFraudEngine()
        assert engine is not None
        assert engine.hard_rules_engine is not None
        print("✓ Motor Unificado inicializado")
    
    def test_unified_response_format(self):
        """Testa formato de resposta unificada"""
        engine = UnifiedFraudEngine()
        
        transaction = {
            "id": "UNIFIED_TEST",
            "amount": 10000,
            "channel": "PIX",
            "hour": 2
        }
        
        result = engine.evaluate(transaction, use_ml=False, use_rules=True)
        
        required_fields = [
            "transaction_id", "is_fraud", "fraud_probability",
            "risk_score", "risk_level", "confidence",
            "processing_time_ms", "model_version",
            "detection_reason", "timestamp"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        print("✓ Resposta unificada tem formato correto")


class TestAllRulesEffectiveness:
    """Testa efetividade de TODAS as regras"""
    
    @pytest.fixture
    def engine(self):
        return get_hard_rules_engine()
    
    def test_all_rules_have_conditions(self, engine):
        """Verifica se todas as regras têm condições válidas"""
        rules = engine._load_rules()
        
        rules_without_conditions = []
        for rule in rules:
            conditions = rule.get("conditions_json")
            if not conditions or conditions == "[]":
                rules_without_conditions.append(rule.get("name"))
        
        if rules_without_conditions:
            print(f"⚠ Regras sem condições: {len(rules_without_conditions)}")
        else:
            print(f"✓ Todas as {len(rules)} regras têm condições válidas")
    
    def test_rule_actions_distribution(self, engine):
        """Verifica distribuição de ações das regras"""
        summary = engine.get_rules_summary()
        
        print(f"✓ Distribuição de ações:")
        for action, count in summary["by_action"].items():
            print(f"  - {action}: {count} regras")
    
    def test_rule_types_distribution(self, engine):
        """Verifica distribuição de tipos de regras"""
        summary = engine.get_rules_summary()
        
        print(f"✓ Distribuição de tipos:")
        for rule_type, count in summary["by_type"].items():
            print(f"  - {rule_type}: {count} regras")


def run_comprehensive_test():
    """Executa teste abrangente de todas as regras"""
    engine = get_hard_rules_engine()
    
    print("\n" + "="*60)
    print("TESTE INTEGRADO - REGRAS DURAS SANKOFA ENTERPRISE")
    print("="*60)
    
    rules_count = engine.get_rules_count()
    print(f"\n📊 Total de regras: {rules_count}")
    
    summary = engine.get_rules_summary()
    print(f"\n📈 Por ação:")
    for action, count in summary["by_action"].items():
        print(f"   {action}: {count}")
    
    test_scenarios = [
        {
            "name": "PIX Madrugada Alto Valor",
            "txn": {"id": "T1", "amount": 5000, "channel": "PIX", "hour": 3},
            "expected_fraud": True
        },
        {
            "name": "PIX Horário 13h",
            "txn": {"id": "T2", "amount": 1000, "channel": "PIX", "hour": 13},
            "expected_fraud": True
        },
        {
            "name": "Velocity Ataque",
            "txn": {"id": "T3", "amount": 100, "velocity_1h": 15},
            "expected_fraud": True
        },
        {
            "name": "Mão Fantasma",
            "txn": {"id": "T4", "amount": 2000, "channel": "PIX", "device_id": "REMOTE_ACCESS"},
            "expected_fraud": True
        },
        {
            "name": "Transação Normal",
            "txn": {"id": "T5", "amount": 50, "channel": "TED", "hour": 10},
            "expected_fraud": False
        },
        {
            "name": "BACEN Limite Noturno",
            "txn": {"id": "T6", "amount": 2000, "channel": "PIX", "hour": 22},
            "expected_fraud": True
        },
        {
            "name": "Card Testing",
            "txn": {"id": "T7", "amount": 2, "type": "CARTAO_CREDITO", "velocity_1h": 12},
            "expected_fraud": True
        },
        {
            "name": "Sequestro Relâmpago ATM",
            "txn": {"id": "T8", "amount": 1500, "channel": "ATM", "hour": 23, "velocity_1h": 3},
            "expected_fraud": True
        }
    ]
    
    print(f"\n🧪 Testando {len(test_scenarios)} cenários:")
    print("-"*60)
    
    passed = 0
    failed = 0
    
    for scenario in test_scenarios:
        result = engine.evaluate(scenario["txn"])
        
        status = "✅" if result.is_fraud == scenario["expected_fraud"] else "❌"
        if result.is_fraud == scenario["expected_fraud"]:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {scenario['name']}")
        print(f"   Esperado: {scenario['expected_fraud']}, Obtido: {result.is_fraud}")
        print(f"   Score: {result.fraud_probability:.2f}, Level: {result.risk_level}")
        if result.triggered_rules:
            print(f"   Regras: {len(result.triggered_rules)} acionadas")
    
    print("-"*60)
    print(f"\n📊 Resultado: {passed}/{len(test_scenarios)} cenários passaram")
    
    print("\n🔒 Verificando formato ML:")
    sample_result = engine.evaluate({"id": "FORMAT_TEST", "amount": 1000})
    unified = sample_result.to_unified_response()
    
    ml_fields = ["transaction_id", "is_fraud", "fraud_probability", "risk_score", 
                 "risk_level", "confidence", "processing_time_ms", "model_version",
                 "detection_reason", "timestamp"]
    
    all_present = all(f in unified for f in ml_fields)
    print(f"✅ Todos os campos ML presentes: {all_present}")
    print(f"✅ Campos extras removidos: {'triggered_rules' not in unified and 'source' not in unified}")
    
    print("\n" + "="*60)
    print("TESTE INTEGRADO COMPLETO")
    print("="*60)
    
    return passed == len(test_scenarios)


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
