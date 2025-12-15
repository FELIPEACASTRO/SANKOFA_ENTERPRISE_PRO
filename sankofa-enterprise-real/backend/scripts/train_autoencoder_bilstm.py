"""
Script para treinar e salvar modelos Autoencoder e BiLSTM
Utiliza dados sinteticos brasileiros para treinamento

Nota: Sem TensorFlow, usa implementacoes simplificadas baseadas em sklearn
"""
import sys
import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_engine.autoencoder_anomaly_detector import AutoencoderAnomalyDetector
from ml_engine.bilstm_sequence_analyzer import BiLSTMSequenceAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_transactions(n_samples: int = 10000, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Gera dados sinteticos de transacoes brasileiras"""

    np.random.seed(42)

    # Cidades brasileiras com coordenadas
    cities = {
        'Sao Paulo': (-23.5505, -46.6333),
        'Rio de Janeiro': (-22.9068, -43.1729),
        'Brasilia': (-15.7942, -47.8822),
        'Salvador': (-12.9714, -38.5014),
        'Belo Horizonte': (-19.9167, -43.9345),
        'Fortaleza': (-3.7172, -38.5433),
        'Curitiba': (-25.4290, -49.2671),
        'Recife': (-8.0476, -34.8770),
        'Porto Alegre': (-30.0346, -51.2177),
        'Manaus': (-3.1190, -60.0217),
    }

    city_names = list(cities.keys())

    data = []
    for i in range(n_samples):
        is_fraud = np.random.random() < fraud_rate

        city = np.random.choice(city_names)
        lat, lon = cities[city]

        # Padroes de fraude
        if is_fraud:
            # Fraudes tem padroes distintos
            hour = np.random.choice([1, 2, 3, 4, 5, 22, 23])  # Horarios suspeitos
            amount = np.random.exponential(5000) + 1000  # Valores maiores
            is_new_device = np.random.random() < 0.8
            is_new_recipient = np.random.random() < 0.9
            velocity_score = np.random.uniform(0.6, 1.0)
            transaction_count_24h = np.random.randint(5, 20)
            # Adicionar ruido nas coordenadas (dispositivos suspeitos)
            lat += np.random.normal(0, 0.5)
            lon += np.random.normal(0, 0.5)
        else:
            # Transacoes normais
            hour = np.random.choice(range(6, 23), p=np.array([0.02, 0.05, 0.08, 0.12, 0.12, 0.12, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])/sum([0.02, 0.05, 0.08, 0.12, 0.12, 0.12, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]))
            amount = np.random.exponential(500) + 50
            is_new_device = np.random.random() < 0.1
            is_new_recipient = np.random.random() < 0.3
            velocity_score = np.random.uniform(0.0, 0.4)
            transaction_count_24h = np.random.randint(0, 5)

        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour >= 22 else 0

        data.append({
            'transaction_id': f'TXN_{i:08d}',
            'customer_id': f'CUST_{np.random.randint(1, 1000):06d}',
            'amount': round(amount, 2),
            'log_amount': np.log1p(amount),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'latitude': lat,
            'longitude': lon,
            'is_new_device': int(is_new_device),
            'is_new_recipient': int(is_new_recipient),
            'velocity_score': velocity_score,
            'transaction_count_24h': transaction_count_24h,
            'avg_amount_30d': np.random.uniform(200, 2000),
            'std_amount_30d': np.random.uniform(50, 500),
            'is_fraud': int(is_fraud),
        })

    return pd.DataFrame(data)


def train_autoencoder(data: pd.DataFrame, model_path: Path) -> dict:
    """Treina e salva o Autoencoder"""
    logger.info("=" * 60)
    logger.info("TREINANDO AUTOENCODER ANOMALY DETECTOR")
    logger.info("=" * 60)

    # Usar apenas transacoes nao-fraudulentas para treinamento
    normal_data = data[data['is_fraud'] == 0].copy()
    logger.info(f"Dados de treinamento: {len(normal_data)} transacoes normais")

    # Criar e treinar modelo
    autoencoder = AutoencoderAnomalyDetector(
        encoding_dim=8,
        hidden_layers=[32, 16],
        anomaly_percentile=95.0,
        epochs=30,
        batch_size=64
    )

    # Treinar
    train_metrics = autoencoder.fit(normal_data)
    logger.info(f"Metricas de treinamento: {train_metrics}")

    # Testar em dados com fraude
    fraud_data = data[data['is_fraud'] == 1]
    detected_frauds = 0
    total_frauds = len(fraud_data)

    for _, row in fraud_data.iterrows():
        result = autoencoder.detect_anomaly(row.to_dict())
        if result.is_anomaly:
            detected_frauds += 1

    fraud_detection_rate = detected_frauds / total_frauds if total_frauds > 0 else 0
    logger.info(f"Taxa de deteccao de fraude: {fraud_detection_rate:.2%} ({detected_frauds}/{total_frauds})")

    # Salvar modelo
    model_file = model_path / "autoencoder_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(autoencoder, f)
    logger.info(f"Modelo salvo em: {model_file}")

    # Salvar metricas
    metrics = {
        **train_metrics,
        'fraud_detection_rate': fraud_detection_rate,
        'total_frauds_tested': total_frauds,
        'frauds_detected': detected_frauds,
        'trained_at': datetime.now().isoformat(),
    }

    metrics_file = model_path / "autoencoder_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_bilstm(data: pd.DataFrame, model_path: Path) -> dict:
    """Treina e salva o BiLSTM Sequence Analyzer"""
    logger.info("=" * 60)
    logger.info("TREINANDO BiLSTM SEQUENCE ANALYZER")
    logger.info("=" * 60)

    # Criar analisador
    bilstm = BiLSTMSequenceAnalyzer(
        sequence_length=10,
        lstm_units=32,
        attention_dim=16,
        risk_threshold=0.6
    )

    # Simular sequencias de transacoes por usuario
    customer_ids = data['customer_id'].unique()
    logger.info(f"Processando sequencias de {len(customer_ids)} clientes")

    # Adicionar transacoes ao buffer de sequencias
    for _, row in data.iterrows():
        bilstm.sequence_buffer.add_transaction(
            row['customer_id'],
            row.to_dict()
        )

    # Treinar modelo (se TF disponivel) ou usar heuristicas
    # Como nao temos TF, o BiLSTM usa analise baseada em regras

    # Testar deteccao em algumas transacoes
    test_results = []
    test_sample = data.sample(min(100, len(data)))

    suspicious_count = 0
    for _, row in test_sample.iterrows():
        customer_id = row['customer_id']
        result = bilstm.analyze_sequence(customer_id, row.to_dict())
        test_results.append(result)
        if result.is_suspicious_sequence:
            suspicious_count += 1

    suspicious_rate = suspicious_count / len(test_sample)
    logger.info(f"Taxa de sequencias suspeitas: {suspicious_rate:.2%}")

    # Salvar modelo
    model_file = model_path / "bilstm_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(bilstm, f)
    logger.info(f"Modelo salvo em: {model_file}")

    # Salvar metricas
    metrics = {
        'sequence_length': bilstm.sequence_length,
        'risk_threshold': bilstm.risk_threshold,
        'customers_tracked': len(customer_ids),
        'suspicious_rate': suspicious_rate,
        'trained_at': datetime.now().isoformat(),
    }

    metrics_file = model_path / "bilstm_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    """Executa treinamento completo"""
    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO DE MODELOS AUTOENCODER E BiLSTM")
    logger.info("=" * 60)

    # Diretorio de modelos
    model_path = Path(__file__).parent.parent / "models" / "production"
    model_path.mkdir(parents=True, exist_ok=True)

    # Gerar dados sinteticos
    logger.info("Gerando dados sinteticos...")
    data = generate_synthetic_transactions(n_samples=10000, fraud_rate=0.03)
    logger.info(f"Dataset: {len(data)} transacoes, {data['is_fraud'].sum()} fraudes ({data['is_fraud'].mean():.2%})")

    # Treinar Autoencoder
    ae_metrics = train_autoencoder(data, model_path)

    # Treinar BiLSTM
    bilstm_metrics = train_bilstm(data, model_path)

    # Resumo final
    logger.info("=" * 60)
    logger.info("RESUMO DO TREINAMENTO")
    logger.info("=" * 60)
    logger.info(f"Autoencoder:")
    logger.info(f"  - Threshold: {ae_metrics.get('threshold', 'N/A')}")
    logger.info(f"  - Taxa deteccao fraude: {ae_metrics.get('fraud_detection_rate', 0):.2%}")
    logger.info(f"BiLSTM:")
    logger.info(f"  - Clientes rastreados: {bilstm_metrics.get('customers_tracked', 0)}")
    logger.info(f"  - Taxa suspeitos: {bilstm_metrics.get('suspicious_rate', 0):.2%}")
    logger.info("=" * 60)
    logger.info("TREINAMENTO CONCLUIDO COM SUCESSO!")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
