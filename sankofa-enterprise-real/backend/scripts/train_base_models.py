"""
Script para treinar e salvar todos os modelos base do ensemble.
Treina: Random Forest, Gradient Boosting, Extra Trees, MLP, Isolation Forest

Este script garante que todos os modelos necessários estejam treinados e salvos.
"""
import sys
import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_training_data(n_samples: int = 20000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """Gera dados sinteticos para treinamento"""
    np.random.seed(42)

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

        if is_fraud:
            # Padroes de fraude
            hour = np.random.choice([1, 2, 3, 4, 5, 22, 23])
            amount = np.random.exponential(5000) + 1000
            is_new_device = int(np.random.random() < 0.8)
            is_new_recipient = int(np.random.random() < 0.9)
            velocity_score = np.random.uniform(0.6, 1.0)
            transaction_count_24h = np.random.randint(5, 20)
            lat += np.random.normal(0, 0.5)
            lon += np.random.normal(0, 0.5)
        else:
            # Transacoes normais
            hour = np.random.choice(range(6, 23))
            amount = np.random.exponential(500) + 50
            is_new_device = int(np.random.random() < 0.1)
            is_new_recipient = int(np.random.random() < 0.3)
            velocity_score = np.random.uniform(0.0, 0.4)
            transaction_count_24h = np.random.randint(0, 5)

        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour >= 22 else 0
        is_high_amount = 1 if amount > 5000 else 0

        data.append({
            'amount': round(amount, 2),
            'log_amount': np.log1p(amount),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'is_high_amount': is_high_amount,
            'is_new_device': is_new_device,
            'is_new_recipient': is_new_recipient,
            'velocity_score': velocity_score,
            'transaction_count_24h': transaction_count_24h,
            'is_fraud': int(is_fraud),
        })

    return pd.DataFrame(data)


def train_models(model_path: Path) -> dict:
    """Treina todos os modelos base"""
    logger.info("=" * 60)
    logger.info("TREINANDO MODELOS BASE DO ENSEMBLE")
    logger.info("=" * 60)

    # Gerar dados
    logger.info("Gerando dados de treinamento...")
    data = generate_training_data(n_samples=20000, fraud_rate=0.03)
    logger.info(f"Dataset: {len(data)} transacoes, {data['is_fraud'].sum()} fraudes ({data['is_fraud'].mean():.2%})")

    # Preparar features
    feature_cols = [
        'amount', 'log_amount', 'hour', 'day_of_week', 'is_weekend',
        'is_night', 'is_high_amount'
    ]

    X = data[feature_cols].values
    y = data['is_fraud'].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Modelos a treinar
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'extra_trees_gnn': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            solver='adam', max_iter=200, random_state=42
        ),
        'isolation_forest': IsolationForest(
            n_estimators=100, contamination=0.03,
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    ensemble_aucs = {}

    for name, model in models.items():
        logger.info(f"\nTreinando {name}...")

        if name == 'isolation_forest':
            # Isolation Forest e diferente - treinar apenas com dados normais
            X_train_normal = X_train_scaled[y_train == 0]
            model.fit(X_train_normal)

            # Predizer anomalias (-1 = anomalia, 1 = normal)
            y_pred = model.predict(X_val_scaled)
            y_pred_binary = (y_pred == -1).astype(int)

            # Score para AUC
            y_scores = -model.score_samples(X_val_scaled)  # Maior score = mais anomalo
            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_binary = model.predict(X_val_scaled)
            y_scores = model.predict_proba(X_val_scaled)[:, 1]

        # Calcular metricas
        auc = roc_auc_score(y_val, y_scores)
        precision = precision_score(y_val, y_pred_binary, zero_division=0)
        recall = recall_score(y_val, y_pred_binary, zero_division=0)
        f1 = f1_score(y_val, y_pred_binary, zero_division=0)

        logger.info(f"  AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Salvar modelo
        model_file = model_path / f"{name}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  Salvo: {model_file}")

        results[name] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        ensemble_aucs[name] = auc

    # Salvar scaler
    scaler_file = model_path / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"\nScaler salvo: {scaler_file}")

    # Calcular pesos do ensemble baseado no AUC
    total_auc = sum(ensemble_aucs.values())
    weights = {name: auc / total_auc for name, auc in ensemble_aucs.items()}

    # Calcular ensemble AUC (media ponderada das predicoes)
    ensemble_scores = np.zeros(len(y_val))
    for name, model in models.items():
        if name == 'isolation_forest':
            scores = -model.score_samples(X_val_scaled)
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = model.predict_proba(X_val_scaled)[:, 1]
        ensemble_scores += scores * weights[name]

    ensemble_auc = roc_auc_score(y_val, ensemble_scores)

    # Salvar configuracao do ensemble
    config = {
        'weights': weights,
        'ensemble_auc': ensemble_auc,
        'individual_aucs': ensemble_aucs,
        'trained_at': datetime.now().isoformat(),
        'feature_cols': feature_cols,
    }

    config_file = model_path / "ensemble_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"\nEnsemble config salvo: {config_file}")

    # Resumo
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DO TREINAMENTO")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(f"  {name}: AUC={metrics['auc']:.4f}")
    logger.info(f"  ENSEMBLE: AUC={ensemble_auc:.4f}")
    logger.info("=" * 60)

    return results


def main():
    """Executa o treinamento"""
    model_path = Path(__file__).parent.parent / "models" / "production"
    model_path.mkdir(parents=True, exist_ok=True)

    results = train_models(model_path)

    logger.info("\nTREINAMENTO CONCLUIDO COM SUCESSO!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
