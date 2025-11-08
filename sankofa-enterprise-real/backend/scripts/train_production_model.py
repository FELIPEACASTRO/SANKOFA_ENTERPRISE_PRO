#!/usr/bin/env python3
"""
Script para treinar modelo de produção com dataset real
Executa download Kaggle → feature engineering → treinamento → validação
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.kaggle_dataset_downloader import KaggleDatasetDownloader
from ml_engine.real_data_trainer import RealDataTrainer
from utils.structured_logging import get_structured_logger

logger = get_structured_logger('train_production', 'INFO')


def main():
    """Treina modelo com Credit Card Fraud dataset (284K samples)"""
    
    logger.info("=== INICIANDO TREINAMENTO PRODUÇÃO ===")
    
    # 1. Download dataset
    logger.info("Etapa 1: Download dataset Kaggle")
    downloader = KaggleDatasetDownloader()
    
    # Usar chave correta do registro
    success = downloader.download_dataset(
        dataset_key='credit_card',
        force=False
    )
    
    if not success:
        logger.error("Download falhou - verifique credenciais Kaggle")
        logger.info("Execute: kaggle.com > Account > API > Create New Token")
        return False
    
    # 2. Treinar modelo
    logger.info("Etapa 2: Treinamento com feature engineering")
    trainer = RealDataTrainer()
    
    # Treinar com dataset Credit Card
    try:
        X, y = trainer._load_credit_card()
        
        logger.info(
            "Dataset carregado",
            num_samples=len(X),
            num_features=len(X.columns),
            fraud_rate=f"{y.mean() * 100:.2f}%"
        )
        
        # Treinar modelo
        from ml_engine.production_fraud_engine import ProductionFraudEngine
        
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        # Salvar modelo
        model_path = Path("./models/fraud_model_production.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(engine, model_path)
        
        logger.info(
            "✅ MODELO TREINADO COM SUCESSO",
            dataset="Credit Card Fraud (284K)",
            f1_score=f"{engine.metrics.f1_score:.3f}",
            precision=f"{engine.metrics.precision:.3f}",
            recall=f"{engine.metrics.recall:.3f}",
            location=str(model_path)
        )
        return True
        
    except FileNotFoundError as e:
        logger.error("Dataset não encontrado - execute download primeiro", error=str(e))
        return False
    except Exception as e:
        logger.error("Treinamento falhou", error=str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
