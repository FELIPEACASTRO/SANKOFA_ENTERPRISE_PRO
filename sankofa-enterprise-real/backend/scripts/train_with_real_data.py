#!/usr/bin/env python3
"""
Script para treinar o modelo de fraude com dados reais do Kaggle
Dataset: Credit Card Fraud Detection (MLG-ULB)
"""

import os
import sys
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_dataset():
    """Baixa o dataset do Kaggle (mirror do GitHub)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_DIR / "creditcard_kaggle.csv"
    
    if dataset_path.exists():
        logger.info(f"Dataset já existe em {dataset_path}")
        return dataset_path
    
    logger.info("Baixando dataset de fraude de cartão de crédito...")
    logger.info("Fonte: Kaggle Credit Card Fraud Detection (MLG-ULB)")
    
    try:
        urllib.request.urlretrieve(DATASET_URL, dataset_path)
        logger.info(f"Dataset baixado com sucesso: {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"Erro ao baixar dataset: {e}")
        raise


def load_and_prepare_data(dataset_path):
    """Carrega e prepara os dados para treinamento"""
    logger.info("Carregando dataset...")
    df = pd.read_csv(dataset_path)
    
    logger.info(f"Dataset carregado: {len(df)} transações")
    logger.info(f"Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    feature_names = list(X.columns)
    logger.info(f"Features: {len(feature_names)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Treino: {len(X_train)} amostras")
    logger.info(f"Teste: {len(X_test)} amostras")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_ensemble_model(X_train, y_train):
    """Treina o modelo ensemble com calibração"""
    logger.info("Iniciando treinamento do ensemble...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info("Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    logger.info("Treinando Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    
    logger.info("Treinando Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    
    logger.info("Calibrando probabilidades...")
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    rf_calibrated.fit(X_train_scaled, y_train)
    
    return {
        'scaler': scaler,
        'random_forest': rf_calibrated,
        'gradient_boosting': gb,
        'logistic_regression': lr,
        'feature_names': list(X_train.columns)
    }


def evaluate_model(models, X_test, y_test):
    """Avalia o modelo no conjunto de teste"""
    logger.info("Avaliando modelo...")
    
    scaler = models['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    rf_proba = models['random_forest'].predict_proba(X_test_scaled)[:, 1]
    gb_proba = models['gradient_boosting'].predict_proba(X_test_scaled)[:, 1]
    lr_proba = models['logistic_regression'].predict_proba(X_test_scaled)[:, 1]
    
    ensemble_proba = (rf_proba * 0.4 + gb_proba * 0.35 + lr_proba * 0.25)
    
    threshold = 0.35
    y_pred = (ensemble_proba >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, ensemble_proba)
    
    logger.info("\n" + "="*50)
    logger.info("MÉTRICAS DO MODELO")
    logger.info("="*50)
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info("="*50)
    
    logger.info("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude']))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'threshold': threshold
    }


def save_model(models, metrics, version="1.1.0"):
    """Salva o modelo treinado"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'scaler': models['scaler'],
        'models': {
            'random_forest': models['random_forest'],
            'gradient_boosting': models['gradient_boosting'],
            'logistic_regression': models['logistic_regression'],
        },
        'weights': {
            'random_forest': 0.4,
            'gradient_boosting': 0.35,
            'logistic_regression': 0.25
        },
        'feature_names': models['feature_names'],
        'threshold': metrics['threshold'],
        'metrics': metrics,
        'version': version,
        'training_data': 'kaggle_creditcard_fraud',
        'training_samples': 284807
    }
    
    model_path = MODELS_DIR / f"fraud_engine_real_v{version}.joblib"
    joblib.dump(model_data, model_path)
    logger.info(f"Modelo salvo em: {model_path}")
    
    symlink_path = MODELS_DIR / "fraud_engine_v1.0.0.joblib"
    if symlink_path.exists():
        symlink_path.unlink()
    
    import shutil
    shutil.copy(model_path, symlink_path)
    logger.info(f"Modelo copiado para: {symlink_path}")
    
    return model_path


def main():
    """Função principal"""
    logger.info("="*60)
    logger.info("TREINAMENTO COM DADOS REAIS - KAGGLE CREDIT CARD FRAUD")
    logger.info("="*60)
    
    try:
        dataset_path = download_dataset()
        
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(dataset_path)
        
        models = train_ensemble_model(X_train, y_train)
        
        metrics = evaluate_model(models, X_test, y_test)
        
        model_path = save_model(models, metrics)
        
        logger.info("\n" + "="*60)
        logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("="*60)
        logger.info(f"Modelo: {model_path}")
        logger.info(f"Recall: {metrics['recall']:.1%} (detecção de fraudes)")
        logger.info(f"Precision: {metrics['precision']:.1%} (precisão)")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
