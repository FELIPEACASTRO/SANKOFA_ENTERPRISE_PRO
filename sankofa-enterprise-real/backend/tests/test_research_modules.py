"""
Sankofa Enterprise Pro - Testes dos Módulos de Pesquisa
Testes unitários para os novos módulos baseados em pesquisa acadêmica

Módulos testados:
1. BahnsenFeatureEngineering (Bahnsen et al. 2016)
2. PIXFraudTaxonomy (arXiv:2511.20902)
3. NLPSocialEngineeringDetector (DIFrauD Dataset)
4. TransferLearningPipeline
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.bahnsen_feature_engineering import (
    BahnsenFeatureEngineering,
    UserBehaviorProfile,
    create_bahnsen_engineer
)
from ml_engine.pix_fraud_taxonomy import (
    PIXFraudTaxonomy,
    PIXFraudType,
    PIXFraudIndicator,
    create_pix_taxonomy
)
from ml_engine.nlp_social_engineering import (
    NLPSocialEngineeringDetector,
    create_nlp_detector
)
from ml_engine.transfer_learning_pipeline import (
    TransferLearningPipeline,
    DatasetConfig,
    create_transfer_pipeline
)


class TestBahnsenFeatureEngineering:
    """Testes para BahnsenFeatureEngineering"""
    
    @pytest.fixture
    def engineer(self):
        return BahnsenFeatureEngineering()
    
    @pytest.fixture
    def sample_timestamp(self):
        return datetime(2025, 12, 1, 14, 30, 0)
    
    def test_initialization(self, engineer):
        """Testa inicialização do módulo"""
        assert engineer is not None
        assert engineer.VERSION == "2.0.0"
        assert len(engineer.AGGREGATION_WINDOWS) == 5
    
    def test_temporal_aggregations_empty_history(self, engineer, sample_timestamp):
        """Testa agregações temporais sem histórico"""
        features = engineer.calculate_temporal_aggregations(
            user_id="user_001",
            current_timestamp=sample_timestamp
        )
        
        assert 'txn_count_last_1h' in features
        assert 'txn_count_last_24h' in features
        assert features['txn_count_last_1h'] == 0
    
    def test_temporal_aggregations_with_history(self, engineer, sample_timestamp):
        """Testa agregações temporais com histórico"""
        engineer.add_transaction_to_history(
            user_id="user_001",
            amount=100,
            timestamp=sample_timestamp - timedelta(minutes=30)
        )
        engineer.add_transaction_to_history(
            user_id="user_001",
            amount=200,
            timestamp=sample_timestamp - timedelta(hours=2)
        )
        
        features = engineer.calculate_temporal_aggregations(
            user_id="user_001",
            current_timestamp=sample_timestamp
        )
        
        assert features['txn_count_last_1h'] == 1
        assert features['txn_count_last_24h'] == 2
        assert features['txn_sum_last_24h'] == 300
    
    def test_periodic_features(self, engineer, sample_timestamp):
        """Testa features periódicas Von Mises"""
        features = engineer.calculate_periodic_features(sample_timestamp)
        
        assert 'hour_sin' in features
        assert 'hour_cos' in features
        assert 'day_of_week_sin' in features
        assert 'day_of_week_cos' in features
        
        assert -1 <= features['hour_sin'] <= 1
        assert -1 <= features['hour_cos'] <= 1
        
        assert features['is_night'] == 0
        assert features['is_business_hours'] == 1
    
    def test_periodic_features_night(self, engineer):
        """Testa features periódicas para horário noturno"""
        night_timestamp = datetime(2025, 12, 1, 23, 30, 0)
        features = engineer.calculate_periodic_features(night_timestamp)
        
        assert features['is_night'] == 1
        assert features['is_business_hours'] == 0
    
    def test_behavioral_deviation_new_user(self, engineer, sample_timestamp):
        """Testa desvio comportamental para usuário novo"""
        features = engineer.calculate_behavioral_deviation(
            user_id="new_user",
            amount=1000,
            timestamp=sample_timestamp
        )
        
        assert features['is_new_user'] == 1
        assert features['user_transaction_count'] == 0
    
    def test_behavioral_deviation_existing_user(self, engineer, sample_timestamp):
        """Testa desvio comportamental para usuário existente"""
        for i in range(10):
            engineer.add_transaction_to_history(
                user_id="existing_user",
                amount=100 + i * 10,
                timestamp=sample_timestamp - timedelta(days=i)
            )
        
        features = engineer.calculate_behavioral_deviation(
            user_id="existing_user",
            amount=5000,
            timestamp=sample_timestamp
        )
        
        assert features['is_new_user'] == 0
        assert features['amount_zscore'] > 2
        assert features['is_outlier'] == 1
    
    def test_velocity_features(self, engineer, sample_timestamp):
        """Testa features de velocidade"""
        for i in range(5):
            engineer.add_transaction_to_history(
                user_id="velocity_user",
                amount=100,
                timestamp=sample_timestamp - timedelta(minutes=i * 10)
            )
        
        features = engineer.calculate_velocity_features(
            user_id="velocity_user",
            current_timestamp=sample_timestamp
        )
        
        assert 'velocity_score' in features
        assert 'acceleration_score' in features
        assert 'burst_score' in features
        assert features['txn_frequency_1h'] >= 5
    
    def test_channel_risk(self, engineer):
        """Testa score de risco do canal"""
        ussd_risk = engineer.calculate_channel_risk('USSD')
        pix_risk = engineer.calculate_channel_risk('PIX')
        pos_risk = engineer.calculate_channel_risk('POS')
        
        assert ussd_risk['channel_risk_score'] > pix_risk['channel_risk_score']
        assert pix_risk['channel_risk_score'] > pos_risk['channel_risk_score']
    
    def test_generate_all_features(self, engineer, sample_timestamp):
        """Testa geração completa de features"""
        features = engineer.generate_all_features(
            user_id="test_user",
            amount=500,
            timestamp=sample_timestamp,
            channel='PIX',
            transaction_type='TRANSFER'
        )
        
        assert len(features) > 30
        assert 'amount_zscore' in features
        assert 'velocity_score' in features
        assert 'channel_risk_score' in features
        assert 'hour_sin' in features
    
    def test_transform_dataframe(self, engineer):
        """Testa transformação de DataFrame"""
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'amount': [100, 200, 150, 300],
            'timestamp': pd.date_range('2025-01-01', periods=4, freq='h'),
            'channel': ['PIX', 'TED', 'PIX', 'PIX'],
        })
        
        result = engineer.transform_dataframe(df)
        
        assert len(result) == 4
        assert len(result.columns) > len(df.columns)
        assert 'amount_zscore' in result.columns
    
    def test_factory_function(self):
        """Testa factory function"""
        engineer = create_bahnsen_engineer()
        assert isinstance(engineer, BahnsenFeatureEngineering)


class TestPIXFraudTaxonomy:
    """Testes para PIXFraudTaxonomy"""
    
    @pytest.fixture
    def analyzer(self):
        return PIXFraudTaxonomy()
    
    @pytest.fixture
    def sample_timestamp(self):
        return datetime(2025, 12, 1, 23, 30, 0)
    
    def test_initialization(self, analyzer):
        """Testa inicialização"""
        assert analyzer is not None
        assert len(analyzer.FRAUD_RULES) >= 10
    
    def test_analyze_legitimate_transaction(self, analyzer):
        """Testa análise de transação legítima"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN001",
            amount=100,
            timestamp=datetime(2025, 12, 1, 14, 0, 0),
            sender_id="USER001",
            receiver_id="KNOWN_RECEIVER",
            historical_data={'known_receivers': ['KNOWN_RECEIVER']}
        )
        
        assert result.fraud_probability < 0.5
        assert result.recommended_action in ['APPROVE', 'MONITOR']
    
    def test_analyze_night_high_value(self, analyzer, sample_timestamp):
        """Testa transação noturna de alto valor"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN002",
            amount=5000,
            timestamp=sample_timestamp,
            sender_id="USER001",
            receiver_id="NEW_RECEIVER",
            historical_data={'known_receivers': [], 'user_avg_amount': 500}
        )
        
        assert result.fraud_probability >= 0
        assert len(result.indicators_detected) > 0
        assert "BACEN_LIMITE_NOTURNO" in result.compliance_flags
    
    def test_analyze_ghost_hand(self, analyzer):
        """Testa detecção de Mão Fantasma"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN003",
            amount=2000,
            timestamp=datetime.now(),
            sender_id="USER001",
            receiver_id="SCAMMER",
            device_info={'remote_access_detected': True},
            context_indicators=['fear_inducing_context', 'bank_impersonation'],
            historical_data={'known_receivers': []}
        )
        
        assert result.fraud_probability > 0.5
        has_remote_access_indicator = any(
            'remote_access' in ind.indicator_type.lower()
            for ind in result.indicators_detected
        )
        assert has_remote_access_indicator
        assert result.recommended_action in ['REVIEW', 'BLOCK', 'MONITOR']
    
    def test_analyze_new_receiver(self, analyzer):
        """Testa detecção de destinatário novo"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN004",
            amount=1000,
            timestamp=datetime.now(),
            sender_id="USER001",
            receiver_id="NEVER_SEEN_BEFORE",
            historical_data={'known_receivers': ['receiver_a', 'receiver_b']}
        )
        
        has_new_receiver_indicator = any(
            'first_contact' in ind.indicator_type.lower() or 
            'recipient' in ind.indicator_type.lower()
            for ind in result.indicators_detected
        )
        assert has_new_receiver_indicator
    
    def test_compliance_flags(self, analyzer):
        """Testa geração de flags de compliance"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN005",
            amount=15000,
            timestamp=datetime(2025, 12, 1, 23, 0, 0),
            sender_id="USER001",
            receiver_id="RECEIVER",
            device_info={'remote_access_detected': True}
        )
        
        assert "BACEN_LIMITE_NOTURNO" in result.compliance_flags
        assert "BACEN_ALTO_VALOR" in result.compliance_flags
    
    def test_explanation_generation(self, analyzer):
        """Testa geração de explicação LGPD"""
        result = analyzer.analyze_transaction(
            transaction_id="TXN006",
            amount=3000,
            timestamp=datetime.now(),
            sender_id="USER001",
            receiver_id="RECEIVER",
            context_indicators=['bank_impersonation', 'urgency_pressure'],
            historical_data={'known_receivers': []}
        )
        
        assert result.explanation is not None
        assert len(result.explanation) > 10
    
    def test_fraud_types_summary(self, analyzer):
        """Testa resumo de tipos de fraude"""
        summary = analyzer.get_fraud_types_summary()
        
        assert len(summary) >= 10
        assert 'mao_fantasma' in summary
        assert 'clone_whatsapp' in summary
    
    def test_factory_function(self):
        """Testa factory function"""
        analyzer = create_pix_taxonomy()
        assert isinstance(analyzer, PIXFraudTaxonomy)


class TestNLPSocialEngineeringDetector:
    """Testes para NLPSocialEngineeringDetector"""
    
    @pytest.fixture
    def detector(self):
        return NLPSocialEngineeringDetector()
    
    def test_initialization(self, detector):
        """Testa inicialização"""
        assert detector is not None
        assert len(detector.compiled_urgency) > 0
        assert len(detector.compiled_phishing) > 0
    
    def test_legitimate_message(self, detector):
        """Testa mensagem legítima"""
        result = detector.analyze_text(
            "Bom dia! Seu pedido foi enviado. Acompanhe pelo site dos correios."
        )
        
        assert result.fraud_probability < 0.5
        assert result.recommendation in ['ALLOW', 'WARN_USER']
    
    def test_phishing_sms(self, detector):
        """Testa SMS de phishing"""
        result = detector.analyze_text(
            "URGENTE: Seu cartão foi bloqueado! Clique aqui para desbloquear: bit.ly/xyz123"
        )
        
        assert result.fraud_probability > 0.4
        assert 'URGENCY' in str(result.indicators) or 'PHISHING' in str(result.indicators)
        assert result.recommendation in ['WARN_USER', 'REVIEW', 'BLOCK']
    
    def test_whatsapp_clone(self, detector):
        """Testa golpe de clone de WhatsApp"""
        result = detector.analyze_text(
            "Oi mãe, troquei de número. Pode me fazer um PIX de R$500? Te pago amanhã."
        )
        
        assert result.fraud_probability > 0.25
        assert result.fraud_type in ['WHATSAPP_CLONE', 'EMOTIONAL_MANIPULATION', 'PIX_FRAUD']
    
    def test_pix_bug_scam(self, detector):
        """Testa golpe do bug do PIX"""
        result = detector.analyze_text(
            "Bug do PIX! Envie R$100 e receba R$200 de volta automaticamente!"
        )
        
        assert result.fraud_probability > 0.15
        assert 'PIX_FRAUD' in str(result.indicators) or result.fraud_type == 'PIX_FRAUD'
    
    def test_bank_impersonation(self, detector):
        """Testa impersonação de banco"""
        result = detector.analyze_text(
            "Central do Banco: Sua conta foi suspensa. Confirme seus dados: token e senha para reativar."
        )
        
        assert result.fraud_probability > 0.4
        assert 'BANK_IMPERSONATION' in str(result.indicators)
    
    def test_urgency_detection(self, detector):
        """Testa detecção de urgência"""
        result = detector.analyze_text(
            "ÚLTIMO AVISO: Ação necessária em 24 horas ou sua conta será encerrada!"
        )
        
        assert result.urgency_score > 0.3
    
    def test_emotional_manipulation(self, detector):
        """Testa manipulação emocional"""
        result = detector.analyze_text(
            "Parabéns! Você foi sorteado e ganhou R$10.000! Acesse agora para receber."
        )
        
        assert result.emotional_score > 0.3
    
    def test_batch_analyze(self, detector):
        """Testa análise em lote"""
        messages = [
            "Mensagem normal",
            "URGENTE: Clique no link!",
            "Você ganhou um prêmio!"
        ]
        
        results = detector.batch_analyze(messages)
        
        assert len(results) == 3
        assert all(r.text_id.startswith("BATCH_") for r in results)
    
    def test_pattern_summary(self, detector):
        """Testa resumo de padrões"""
        summary = detector.get_pattern_summary()
        
        assert summary['urgency_patterns'] > 0
        assert summary['phishing_patterns'] > 0
        assert summary['pix_patterns'] > 0
    
    def test_factory_function(self):
        """Testa factory function"""
        detector = create_nlp_detector()
        assert isinstance(detector, NLPSocialEngineeringDetector)


class TestTransferLearningPipeline:
    """Testes para TransferLearningPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return TransferLearningPipeline({'model_dir': '/tmp/test_models'})
    
    def test_initialization(self, pipeline):
        """Testa inicialização"""
        assert pipeline is not None
        assert len(pipeline.SUPPORTED_DATASETS) >= 4
    
    def test_list_supported_datasets(self, pipeline):
        """Testa listagem de datasets"""
        datasets = pipeline.list_supported_datasets()
        
        assert 'nigerian' in datasets
        assert 'paysim' in datasets
        assert 'feedzai_baf' in datasets
        
        assert datasets['nigerian']['compatible'] is True
    
    def test_dataset_compatibility(self, pipeline):
        """Testa compatibilidade de dataset"""
        nigerian = pipeline.get_dataset_compatibility('nigerian')
        ieee = pipeline.get_dataset_compatibility('ieee_cis')
        
        assert nigerian['compatible'] is True
        assert nigerian['size'] == 5_000_000
        
        assert ieee['compatible'] is False
    
    def test_prepare_features(self, pipeline):
        """Testa preparação de features"""
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'channel': ['PIX', 'TED', 'PIX'],
            'is_fraud': [0, 0, 1]
        })
        
        X, y = pipeline.prepare_features(df)
        
        assert len(X) == 3
        assert len(y) == 3
        assert 'is_fraud' not in X.columns
    
    def test_factory_function(self):
        """Testa factory function"""
        pipeline = create_transfer_pipeline()
        assert isinstance(pipeline, TransferLearningPipeline)


class TestIntegration:
    """Testes de integração entre módulos"""
    
    def test_bahnsen_with_taxonomy(self):
        """Testa integração Bahnsen + Taxonomia PIX"""
        engineer = BahnsenFeatureEngineering()
        analyzer = PIXFraudTaxonomy()
        
        timestamp = datetime.now()
        
        bahnsen_features = engineer.generate_all_features(
            user_id="test_user",
            amount=5000,
            timestamp=timestamp,
            channel='PIX'
        )
        
        pix_result = analyzer.analyze_transaction(
            transaction_id="TXN001",
            amount=5000,
            timestamp=timestamp,
            sender_id="test_user",
            receiver_id="receiver",
            historical_data={'user_avg_amount': bahnsen_features.get('user_avg_amount', 500)}
        )
        
        combined_score = (
            bahnsen_features.get('amount_zscore', 0) * 0.3 +
            pix_result.fraud_probability * 0.7
        )
        
        assert combined_score >= 0
    
    def test_nlp_with_taxonomy(self):
        """Testa integração NLP + Taxonomia PIX"""
        detector = NLPSocialEngineeringDetector()
        analyzer = PIXFraudTaxonomy()
        
        nlp_result = detector.analyze_text(
            "Central do banco: Confirme seus dados urgente!"
        )
        
        context_indicators = []
        if nlp_result.fraud_type == 'BANK_IMPERSONATION':
            context_indicators.append('bank_impersonation')
        if nlp_result.urgency_score > 0.3:
            context_indicators.append('urgency_pressure')
        
        pix_result = analyzer.analyze_transaction(
            transaction_id="TXN001",
            amount=1000,
            timestamp=datetime.now(),
            sender_id="user",
            receiver_id="receiver",
            context_indicators=context_indicators
        )
        
        assert pix_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
