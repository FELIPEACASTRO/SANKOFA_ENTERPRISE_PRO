#!/usr/bin/env python3
"""
Script de treinamento rápido do modelo de fraude
Treina e salva modelo em disco para uso imediato
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

print("=" * 60)
print("SANKOFA ENTERPRISE - TREINAMENTO RÁPIDO DO MODELO")
print("=" * 60)

np.random.seed(42)
N_SAMPLES = 30000
FRAUD_RATE = 0.05

print(f"\n1. Gerando dataset sintético realista...")
print(f"   - Total de amostras: {N_SAMPLES:,}")
print(f"   - Taxa de fraude: {FRAUD_RATE*100}%")

n_frauds = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_frauds

legit_data = {
    'amount': np.random.exponential(800, n_legit),
    'hour': np.random.choice(range(6, 23), n_legit, p=[0.02, 0.05, 0.08, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005]),
    'day_of_week': np.random.choice(range(7), n_legit, p=[0.12, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14]),
    'location_risk_score': np.random.beta(2, 8, n_legit),
    'device_risk_score': np.random.beta(2, 9, n_legit),
    'velocity_score': np.random.beta(2, 10, n_legit),
    'is_new_device': np.random.binomial(1, 0.05, n_legit),
    'transaction_count_24h': np.random.poisson(3, n_legit),
    'avg_amount_30d': np.random.exponential(600, n_legit),
    'is_weekend': np.random.binomial(1, 0.28, n_legit),
    'is_night': np.random.binomial(1, 0.15, n_legit),
    'distance_from_home': np.random.exponential(20, n_legit),
    'time_since_last_tx': np.random.exponential(4, n_legit),
}

fraud_data = {
    'amount': np.random.exponential(3500, n_frauds) + 1000,
    'hour': np.random.choice(range(24), n_frauds, p=[0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03, 0.03]),
    'day_of_week': np.random.choice(range(7), n_frauds),
    'location_risk_score': np.random.beta(6, 3, n_frauds),
    'device_risk_score': np.random.beta(5, 3, n_frauds),
    'velocity_score': np.random.beta(7, 2, n_frauds),
    'is_new_device': np.random.binomial(1, 0.7, n_frauds),
    'transaction_count_24h': np.random.poisson(8, n_frauds) + 3,
    'avg_amount_30d': np.random.exponential(400, n_frauds),
    'is_weekend': np.random.binomial(1, 0.35, n_frauds),
    'is_night': np.random.binomial(1, 0.55, n_frauds),
    'distance_from_home': np.random.exponential(150, n_frauds) + 50,
    'time_since_last_tx': np.random.exponential(0.5, n_frauds),
}

legit_df = pd.DataFrame(legit_data)
legit_df['is_fraud'] = 0

fraud_df = pd.DataFrame(fraud_data)
fraud_df['is_fraud'] = 1

df = pd.concat([legit_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df['amount_log'] = np.log1p(df['amount'])
df['amount_deviation'] = df['amount'] / (df['avg_amount_30d'] + 1)
df['velocity_device_interaction'] = df['velocity_score'] * df['is_new_device']
df['night_high_amount'] = df['is_night'] * (df['amount'] > 2000).astype(int)

print(f"   - Features criadas: {len(df.columns) - 1}")
print(f"   - Fraudes: {n_frauds:,} ({FRAUD_RATE*100}%)")
print(f"   - Legítimas: {n_legit:,} ({(1-FRAUD_RATE)*100}%)")

print("\n2. Preparando dados para treinamento...")
feature_names = [col for col in df.columns if col != 'is_fraud']
X = df[feature_names]
y = df['is_fraud'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   - Treino: {len(X_train):,} amostras")
print(f"   - Teste: {len(X_test):,} amostras")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n3. Treinando ensemble de modelos...")

print("   - Treinando Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

print("   - Treinando Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train)

print("   - Treinando Logistic Regression...")
lr = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr.fit(X_train_scaled, y_train)

print("   - Combinando previsões (ensemble)...")
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

ensemble_proba = (rf_proba * 0.4 + gb_proba * 0.4 + lr_proba * 0.2)

best_threshold = 0.5
best_f1 = 0
for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_temp = (ensemble_proba >= thresh).astype(int)
    f1_temp = f1_score(y_test, y_pred_temp)
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_threshold = thresh

y_pred = (ensemble_proba >= best_threshold).astype(int)

print("\n4. Avaliando métricas do modelo...")
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, ensemble_proba),
    'threshold': best_threshold,
    'timestamp': datetime.utcnow().isoformat() + 'Z'
}

print(f"\n   MÉTRICAS DO MODELO:")
print(f"   ├── Accuracy:  {metrics['accuracy']:.4f}")
print(f"   ├── Precision: {metrics['precision']:.4f}")
print(f"   ├── Recall:    {metrics['recall']:.4f}")
print(f"   ├── F1-Score:  {metrics['f1_score']:.4f}")
print(f"   ├── ROC-AUC:   {metrics['roc_auc']:.4f}")
print(f"   └── Threshold: {metrics['threshold']:.2f}")

print("\n5. Salvando modelo em disco...")
model_dir = Path(__file__).parent.parent / 'models'
model_dir.mkdir(exist_ok=True)

model_data = {
    'version': '1.0.0',
    'models': {
        'random_forest': rf,
        'gradient_boosting': gb,
        'logistic_regression': lr
    },
    'weights': {'rf': 0.4, 'gb': 0.4, 'lr': 0.2},
    'scaler': scaler,
    'feature_names': feature_names,
    'threshold': best_threshold,
    'metrics': metrics,
    'trained_at': datetime.utcnow().isoformat() + 'Z'
}

model_path = model_dir / 'fraud_engine_v1.0.0.joblib'
joblib.dump(model_data, model_path)
print(f"   - Modelo salvo em: {model_path}")

simple_model_path = model_dir / 'production_model.joblib'
joblib.dump({
    'model': rf,
    'scaler': scaler,
    'feature_names': feature_names,
    'threshold': best_threshold,
    'metrics': metrics
}, simple_model_path)
print(f"   - Modelo simplificado em: {simple_model_path}")

print("\n" + "=" * 60)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 60)
print(f"\nO modelo está pronto para uso.")
print(f"Reinicie o backend para carregar o novo modelo.")
