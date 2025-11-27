#!/usr/bin/env python3
"""
Script rápido para treinar com amostra do dataset real do Kaggle
Usa undersampling para balancear classes e acelerar treinamento
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_and_balance_data():
    """Carrega dados e aplica undersampling para balanceamento"""
    dataset_path = DATA_DIR / "creditcard_kaggle.csv"
    
    if not dataset_path.exists():
        logger.error(f"Dataset não encontrado: {dataset_path}")
        logger.info("Execute primeiro: python scripts/train_with_real_data.py")
        return None
    
    logger.info("Carregando dataset...")
    df = pd.read_csv(dataset_path)
    
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    logger.info(f"Total: {len(df)} | Fraudes: {len(fraud)} | Legítimas: {len(legit)}")
    
    np.random.seed(42)
    legit_sample = legit.sample(n=len(fraud) * 10, random_state=42)  # 10:1 ratio
    
    balanced_df = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=42)
    logger.info(f"Dataset balanceado: {len(balanced_df)} amostras")
    
    X = balanced_df.drop(['Class'], axis=1)
    y = balanced_df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def train_ensemble(X_train, y_train):
    """Treina ensemble rápido"""
    logger.info("Treinando ensemble...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    logger.info("  Random Forest (50 árvores)...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y_train)
    
    logger.info("  Gradient Boosting (30 iterações)...")
    gb = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.15,
        random_state=42
    )
    gb.fit(X_scaled, y_train)
    
    logger.info("  Logistic Regression...")
    lr = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_scaled, y_train)
    
    return {
        'scaler': scaler,
        'random_forest': rf,
        'gradient_boosting': gb,
        'logistic_regression': lr,
        'feature_names': list(X_train.columns)
    }


def evaluate_and_save(models, X_test, y_test):
    """Avalia e salva o modelo"""
    X_scaled = models['scaler'].transform(X_test)
    
    rf_proba = models['random_forest'].predict_proba(X_scaled)[:, 1]
    gb_proba = models['gradient_boosting'].predict_proba(X_scaled)[:, 1]
    lr_proba = models['logistic_regression'].predict_proba(X_scaled)[:, 1]
    
    ensemble_proba = (rf_proba * 0.4 + gb_proba * 0.35 + lr_proba * 0.25)
    
    threshold = 0.35
    y_pred = (ensemble_proba >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, ensemble_proba)
    
    logger.info("\n" + "="*50)
    logger.info("MÉTRICAS DO MODELO (Dados Reais Kaggle)")
    logger.info("="*50)
    logger.info(f"Precision: {precision:.1%}")
    logger.info(f"Recall: {recall:.1%}")
    logger.info(f"F1-Score: {f1:.1%}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info("="*50)
    
    print("\n" + classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude']))
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'scaler': models['scaler'],
        'models': {
            'random_forest': models['random_forest'],
            'gradient_boosting': models['gradient_boosting'],
            'logistic_regression': models['logistic_regression'],
        },
        'weights': {'random_forest': 0.4, 'gradient_boosting': 0.35, 'logistic_regression': 0.25},
        'feature_names': models['feature_names'],
        'threshold': threshold,
        'metrics': {'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc},
        'version': '1.1.0',
        'training_data': 'kaggle_creditcard_fraud_balanced',
    }
    
    model_path = MODELS_DIR / "fraud_engine_v1.0.0.joblib"
    backup_path = MODELS_DIR / "fraud_engine_v1.0.0.backup.joblib"
    
    if model_path.exists():
        import shutil
        shutil.copy(model_path, backup_path)
        logger.info(f"Backup criado: {backup_path}")
    
    joblib.dump(model_data, model_path)
    logger.info(f"Modelo salvo: {model_path}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}


def main():
    logger.info("="*60)
    logger.info("TREINAMENTO RÁPIDO COM DADOS REAIS (KAGGLE)")
    logger.info("="*60)
    
    data = load_and_balance_data()
    if data is None:
        return False
    
    X_train, X_test, y_train, y_test = data
    
    models = train_ensemble(X_train, y_train)
    
    metrics = evaluate_and_save(models, X_test, y_test)
    
    logger.info("\nTREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info(f"Recall: {metrics['recall']:.1%} | Precision: {metrics['precision']:.1%}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
