#!/usr/bin/env python3
"""
Script para treinar modelo de produção com dataset real
Executa download Kaggle → feature engineering → treinamento → validação
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.kaggle_dataset_downloader import KaggleDatasetDownloader
from ml_engine.real_data_trainer import RealFraudDataTrainer
from utils.structured_logging import get_structured_logger

logger = get_structured_logger('train_production', 'INFO')


def main():
    """Treina modelo com Credit Card Fraud dataset (284K samples)"""
    
    logger.info("=== INICIANDO TREINAMENTO PRODUÇÃO ===")
    
    # 1. Download dataset
    logger.info("Etapa 1: Download dataset Kaggle")
    downloader = KaggleDatasetDownloader()
    
    dataset_path = downloader.download_dataset(
        'mlg-ulb/creditcardfraud',
        'Credit Card Fraud'
    )
    
    if not dataset_path:
        logger.error("Download falhou - verifique credenciais Kaggle")
        return False
    
    # 2. Treinar modelo
    logger.info("Etapa 2: Treinamento com feature engineering")
    trainer = RealFraudDataTrainer()
    
    success = trainer.train_from_file(
        str(dataset_path / 'creditcard.csv'),
        test_size=0.3,
        save_model=True
    )
    
    if success:
        logger.info(
            "✅ MODELO TREINADO COM SUCESSO",
            dataset="Credit Card Fraud (284K)",
            location="backend/models/fraud_model_production.pkl"
        )
        return True
    else:
        logger.error("❌ Treinamento falhou")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
