#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Real Data Trainer
Sistema de treinamento com dados reais (Kaggle datasets)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_engine.production_fraud_engine import ProductionFraudEngine, ModelMetrics
from utils.structured_logging import get_structured_logger

logger = get_structured_logger('real_data_trainer', 'INFO')

class RealDataTrainer:
    """
    Sistema de treinamento com datasets reais
    
    Features:
    - Suporta múltiplos datasets (IEEE-CIS, Credit Card, PaySim)
    - Preprocessamento automático
    - Feature engineering
    - Validação cruzada
    - Tracking de experimentos
    """
    
    def __init__(self, data_dir: str = "./data/datasets/raw"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path("./models/experiments")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_loaders = {
            'ieee_fraud': self._load_ieee_fraud,
            'credit_card': self._load_credit_card,
            'paysim': self._load_paysim,
            'bank_account': self._load_bank_account
        }
        
        logger.info("RealDataTrainer initialized", data_dir=str(self.data_dir))
    
    def _load_ieee_fraud(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Carrega IEEE-CIS Fraud Detection dataset"""
        logger.info("Loading IEEE-CIS Fraud Detection dataset...")
        
        dataset_path = self.data_dir / 'ieee_fraud'
        
        # Check if files exist
        train_transaction = dataset_path / 'train_transaction.csv'
        
        if not train_transaction.exists():
            raise FileNotFoundError(
                f"IEEE-CIS dataset not found at {dataset_path}. "
                "Run kaggle_dataset_downloader.py first."
            )
        
        # Load transaction data
        df = pd.read_csv(train_transaction)
        
        logger.info(
            "IEEE-CIS loaded",
            num_transactions=len(df),
            num_features=len(df.columns),
            fraud_rate=round(df['isFraud'].mean() * 100, 2)
        )
        
        # Separate features and target
        y = df['isFraud'].values
        X = df.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
        
        return X, y
    
    def _load_credit_card(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Carrega Credit Card Fraud dataset"""
        logger.info("Loading Credit Card Fraud dataset...")
        
        dataset_path = self.data_dir / 'credit_card' / 'creditcard.csv'
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Credit Card dataset not found at {dataset_path}. "
                "Run kaggle_dataset_downloader.py first."
            )
        
        df = pd.read_csv(dataset_path)
        
        logger.info(
            "Credit Card loaded",
            num_transactions=len(df),
            num_features=len(df.columns),
            fraud_rate=round(df['Class'].mean() * 100, 4)
        )
        
        y = df['Class'].values
        X = df.drop('Class', axis=1)
        
        return X, y
    
    def _load_paysim(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Carrega PaySim Mobile Money dataset"""
        logger.info("Loading PaySim dataset...")
        
        dataset_path = self.data_dir / 'paysim' / 'PS_20174392719_1491204439457_log.csv'
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"PaySim dataset not found at {dataset_path}. "
                "Run kaggle_dataset_downloader.py first."
            )
        
        # PaySim is large, sample if needed
        df = pd.read_csv(dataset_path)
        
        # Sample 10% for faster training (still 630K transactions!)
        df = df.sample(frac=0.1, random_state=42)
        
        logger.info(
            "PaySim loaded (10% sample)",
            num_transactions=len(df),
            num_features=len(df.columns),
            fraud_rate=round(df['isFraud'].mean() * 100, 4)
        )
        
        y = df['isFraud'].values
        X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], 
                     axis=1, errors='ignore')
        
        # Encode categorical
        if 'type' in X.columns:
            X = pd.get_dummies(X, columns=['type'], prefix='type')
        
        return X, y
    
    def _load_bank_account(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Carrega Bank Account Fraud dataset"""
        logger.info("Loading Bank Account Fraud dataset...")
        
        dataset_path = self.data_dir / 'bank_account'
        
        # Find CSV file
        csv_files = list(dataset_path.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(
                f"Bank Account dataset not found at {dataset_path}"
            )
        
        df = pd.read_csv(csv_files[0])
        
        logger.info(
            "Bank Account loaded",
            num_transactions=len(df),
            num_features=len(df.columns),
            fraud_rate=round(df['fraud_bool'].mean() * 100, 2)
        )
        
        y = df['fraud_bool'].values
        X = df.drop('fraud_bool', axis=1)
        
        return X, y
    
    def train_on_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        save_model: bool = True
    ) -> Dict[str, any]:
        """
        Treina modelo em um dataset específico
        
        Args:
            dataset_name: Nome do dataset ('ieee_fraud', 'credit_card', etc)
            sample_size: Número de samples (None = todos)
            save_model: Se True, salva o modelo treinado
        
        Returns:
            Dict com métricas e informações do treinamento
        """
        logger.info(f"Starting training on {dataset_name}...")
        
        # Validar dataset
        if dataset_name not in self.dataset_loaders:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.dataset_loaders.keys())}"
            )
        
        # Carregar dados
        start_time = datetime.now()
        X, y = self.dataset_loaders[dataset_name]()
        
        # Sample se necessário
        if sample_size and sample_size < len(X):
            logger.info(f"Sampling {sample_size} transactions...")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[indices]
            y = y[indices]
        
        # Criar e treinar engine
        logger.info("Training ProductionFraudEngine...")
        engine = ProductionFraudEngine()
        engine.fit(X, y)
        
        # Calcular tempo total
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Resultados
        results = {
            'dataset_name': dataset_name,
            'num_samples': len(X),
            'num_features': len(X.columns),
            'fraud_rate': round(y.mean() * 100, 4),
            'training_time_seconds': round(total_time, 2),
            'metrics': {
                'accuracy': round(engine.metrics.accuracy, 4),
                'precision': round(engine.metrics.precision, 4),
                'recall': round(engine.metrics.recall, 4),
                'f1_score': round(engine.metrics.f1_score, 4),
                'roc_auc': round(engine.metrics.roc_auc, 4),
                'threshold': round(engine.metrics.threshold, 4)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results
        logger.info(
            "Training completed",
            dataset=dataset_name,
            f1_score=results['metrics']['f1_score'],
            accuracy=results['metrics']['accuracy'],
            training_time=results['training_time_seconds']
        )
        
        # Salvar modelo
        if save_model:
            model_filename = f"{dataset_name}_f1_{results['metrics']['f1_score']:.3f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = self.models_dir / model_filename
            
            joblib.dump(engine, model_path)
            logger.info(f"Model saved to {model_path}")
            results['model_path'] = str(model_path)
        
        return results
    
    def train_all_datasets(self) -> Dict[str, Dict]:
        """Treina em todos os datasets disponíveis"""
        logger.info("Training on ALL datasets...")
        
        results = {}
        for dataset_name in self.dataset_loaders.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Dataset: {dataset_name}")
                logger.info(f"{'='*60}\n")
                
                result = self.train_on_dataset(dataset_name)
                results[dataset_name] = result
                
            except FileNotFoundError as e:
                logger.warning(f"Skipping {dataset_name}: {e}")
                results[dataset_name] = {'status': 'skipped', 'reason': str(e)}
            except Exception as e:
                logger.error(f"Error training on {dataset_name}: {e}")
                results[dataset_name] = {'status': 'failed', 'error': str(e)}
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        for dataset_name, result in results.items():
            if 'metrics' in result:
                logger.info(
                    f"{dataset_name}: F1={result['metrics']['f1_score']:.3f} | "
                    f"Accuracy={result['metrics']['accuracy']:.3f} | "
                    f"Time={result['training_time_seconds']:.1f}s"
                )
            else:
                logger.info(f"{dataset_name}: {result.get('status', 'unknown')}")
        
        return results

def main():
    """CLI para treinamento com dados reais"""
    print("🏦 Sankofa Enterprise Pro - Real Data Trainer")
    print("="*60)
    
    trainer = RealDataTrainer()
    
    print("\nAvailable datasets:")
    print("1. ieee_fraud - IEEE-CIS Fraud Detection (590K transactions)")
    print("2. credit_card - Credit Card Fraud (284K transactions)")
    print("3. paysim - PaySim Mobile Money (636K transactions - 10% sample)")
    print("4. bank_account - Bank Account Fraud (1M accounts)")
    print("5. ALL - Train on all datasets")
    
    choice = input("\nSelect dataset (1-5): ")
    
    dataset_map = {
        '1': 'ieee_fraud',
        '2': 'credit_card',
        '3': 'paysim',
        '4': 'bank_account'
    }
    
    if choice == '5':
        print("\n🚀 Training on ALL datasets...")
        results = trainer.train_all_datasets()
        print("\n✅ Training complete! Check logs for details.")
    elif choice in dataset_map:
        dataset_name = dataset_map[choice]
        print(f"\n🚀 Training on {dataset_name}...")
        result = trainer.train_on_dataset(dataset_name)
        
        print("\n📊 Results:")
        print(f"   F1-Score: {result['metrics']['f1_score']:.3f}")
        print(f"   Accuracy: {result['metrics']['accuracy']:.3f}")
        print(f"   Precision: {result['metrics']['precision']:.3f}")
        print(f"   Recall: {result['metrics']['recall']:.3f}")
        print(f"   ROC-AUC: {result['metrics']['roc_auc']:.3f}")
        print(f"   Training Time: {result['training_time_seconds']:.1f}s")
        
        if 'model_path' in result:
            print(f"\n💾 Model saved: {result['model_path']}")
    else:
        print("Invalid choice.")

if __name__ == '__main__':
    main()
