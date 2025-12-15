"""
Sankofa Enterprise Pro - CatBoost Model Training Script
Treina o modelo CatBoost com dados sintéticos realistas para detecção de fraude

Features incluídas:
- latitude/longitude (geolocalização)
- merchant_category (categoria do comerciante)
- Todas as features categóricas e numéricas do modelo
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime

# Adicionar path do backend
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ml_engine.catboost_model import CatBoostFraudModel, CATBOOST_AVAILABLE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Constantes para geração de dados realistas do Brasil
BRAZIL_COORDS = {
    "São Paulo": {"lat": -23.5505, "lon": -46.6333},
    "Rio de Janeiro": {"lat": -22.9068, "lon": -43.1729},
    "Brasília": {"lat": -15.7801, "lon": -47.9292},
    "Salvador": {"lat": -12.9714, "lon": -38.5014},
    "Fortaleza": {"lat": -3.7172, "lon": -38.5433},
    "Belo Horizonte": {"lat": -19.9167, "lon": -43.9345},
    "Manaus": {"lat": -3.1190, "lon": -60.0217},
    "Curitiba": {"lat": -25.4290, "lon": -49.2671},
    "Recife": {"lat": -8.0476, "lon": -34.8770},
    "Porto Alegre": {"lat": -30.0346, "lon": -51.2177},
}

# Localizações suspeitas (cidades de fronteira ou áreas de risco)
SUSPICIOUS_COORDS = {
    "Foz do Iguaçu": {"lat": -25.5163, "lon": -54.5854},
    "Ciudad del Este": {"lat": -25.5167, "lon": -54.6167},  # Paraguai
    "Ponta Porã": {"lat": -22.5267, "lon": -55.7256},
    "Tabatinga": {"lat": -4.2533, "lon": -69.9386},
    "Oiapoque": {"lat": 3.8406, "lon": -51.8331},
}

MERCHANT_CATEGORIES = [
    "supermercado",
    "posto_combustivel",
    "restaurante",
    "farmacia",
    "transferencia_pf",
    "transferencia_pj",
    "pagamento_servicos",
    "varejo_geral",
    "eletronicos",
    "joalheria",
    "casino_online",
    "crypto_exchange",
    "viagem_turismo",
]

# Categorias de alto risco para fraude
HIGH_RISK_CATEGORIES = ["crypto_exchange", "casino_online", "joalheria", "eletronicos"]

CHANNELS = ["pix", "web", "mobile", "atm", "ted", "doc"]
DEVICE_TYPES = ["android", "ios", "windows", "mac", "linux", "unknown"]
PIX_KEY_TYPES = ["cpf", "email", "telefone", "aleatoria", "cnpj"]
ESTADOS = ["SP", "RJ", "MG", "BA", "RS", "PR", "SC", "PE", "CE", "DF", "GO", "PA", "AM"]
BANCOS = ["001", "033", "104", "237", "341", "356", "389", "422", "745", "756"]


def generate_synthetic_fraud_data(n_samples: int = 50000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """
    Gera dados sintéticos realistas para treinamento do modelo CatBoost

    Padrões de fraude simulados:
    1. Transações de alto valor em horários noturnos
    2. Localizações suspeitas (fronteiras)
    3. Categorias de alto risco (crypto, casino)
    4. Velocidade alta de transações
    5. Dispositivos novos com valores altos
    6. Combinações suspeitas de features
    """
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    print(f"Gerando {n_samples} transações ({n_fraud} fraudes, {n_legit} legítimas)...")

    # ========== TRANSAÇÕES LEGÍTIMAS ==========
    legit_data = generate_legitimate_transactions(n_legit)
    legit_labels = np.zeros(n_legit)

    # ========== TRANSAÇÕES FRAUDULENTAS ==========
    fraud_data = generate_fraudulent_transactions(n_fraud)
    fraud_labels = np.ones(n_fraud)

    # Combinar datasets
    df = pd.concat([
        pd.DataFrame(legit_data),
        pd.DataFrame(fraud_data)
    ], ignore_index=True)

    df["is_fraud"] = np.concatenate([legit_labels, fraud_labels])

    # Shuffle mantendo seed para reprodutibilidade
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset gerado: {len(df)} registros")
    print(f"Distribuição de fraudes: {df['is_fraud'].mean()*100:.2f}%")

    return df


def generate_legitimate_transactions(n: int) -> dict:
    """Gera transações legítimas com padrões normais"""

    # Selecionar cidades normais
    cities = list(BRAZIL_COORDS.keys())
    selected_cities = np.random.choice(cities, n)

    data = {
        # Features numéricas
        "amount": np.random.exponential(500, n),  # Média R$500
        "hour": np.random.choice(range(24), n, p=get_normal_hour_distribution()),
        "velocity_score": np.random.uniform(0, 0.3, n),  # Baixa velocidade
        "device_risk_score": np.random.uniform(0, 0.2, n),
        "location_risk_score": np.random.uniform(0, 0.2, n),
        "is_new_device": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "is_new_location": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "transactions_last_1h": np.random.poisson(1, n),
        "transactions_last_24h": np.random.poisson(5, n),
        "avg_amount_30d": np.random.exponential(600, n),
        "account_age_days": np.random.exponential(365, n) + 30,  # Contas mais antigas

        # Geolocalização - cidades normais
        "latitude": [BRAZIL_COORDS[c]["lat"] + np.random.normal(0, 0.1) for c in selected_cities],
        "longitude": [BRAZIL_COORDS[c]["lon"] + np.random.normal(0, 0.1) for c in selected_cities],

        # Features categóricas
        "canal": np.random.choice(CHANNELS, n, p=[0.4, 0.25, 0.2, 0.05, 0.05, 0.05]),
        "tipo_transacao": np.random.choice(["compra", "transferencia", "pagamento", "saque"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "estado": np.random.choice(ESTADOS, n),
        "pais": np.array(["BR"] * n),
        "device_type": np.random.choice(DEVICE_TYPES, n, p=[0.45, 0.35, 0.1, 0.05, 0.03, 0.02]),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(
            MERCHANT_CATEGORIES, n,
            p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.05, 0.08, 0.05, 0.03, 0.01, 0.01, 0.01, 0.01]
        ),
    }

    # Adicionar day_of_week baseado em distribuição normal (mais transações em dias úteis)
    data["day_of_week"] = np.random.choice(range(7), n, p=[0.18, 0.18, 0.18, 0.18, 0.15, 0.07, 0.06])
    data["is_night"] = ((data["hour"] >= 0) & (data["hour"] <= 5) | (data["hour"] >= 23)).astype(int)
    data["is_weekend"] = (np.array(data["day_of_week"]) >= 5).astype(int)

    return data


def generate_fraudulent_transactions(n: int) -> dict:
    """
    Gera transações fraudulentas com padrões suspeitos

    Distribui fraudes em diferentes padrões:
    - 30% alto valor em horário noturno
    - 20% localização suspeita (fronteiras)
    - 20% categoria de alto risco
    - 15% dispositivo novo + valor alto
    - 15% velocidade alta de transações
    """

    n_night_high = int(n * 0.30)
    n_suspicious_location = int(n * 0.20)
    n_high_risk_category = int(n * 0.20)
    n_new_device_high = int(n * 0.15)
    n_high_velocity = n - n_night_high - n_suspicious_location - n_high_risk_category - n_new_device_high

    all_data = []

    # Padrão 1: Alto valor em horário noturno
    all_data.append(generate_night_high_value_fraud(n_night_high))

    # Padrão 2: Localização suspeita
    all_data.append(generate_suspicious_location_fraud(n_suspicious_location))

    # Padrão 3: Categoria de alto risco
    all_data.append(generate_high_risk_category_fraud(n_high_risk_category))

    # Padrão 4: Dispositivo novo com valor alto
    all_data.append(generate_new_device_fraud(n_new_device_high))

    # Padrão 5: Alta velocidade de transações
    all_data.append(generate_high_velocity_fraud(n_high_velocity))

    # Combinar todos os padrões
    combined = {}
    for key in all_data[0].keys():
        combined[key] = np.concatenate([d[key] for d in all_data])

    return combined


def generate_night_high_value_fraud(n: int) -> dict:
    """Fraudes com alto valor em horário noturno"""
    cities = list(BRAZIL_COORDS.keys())
    selected_cities = np.random.choice(cities, n)

    return {
        "amount": np.random.exponential(8000, n) + 5000,  # R$5000-20000+
        "hour": np.random.choice([0, 1, 2, 3, 4, 22, 23], n),  # Horário noturno
        "velocity_score": np.random.uniform(0.3, 0.7, n),
        "device_risk_score": np.random.uniform(0.3, 0.6, n),
        "location_risk_score": np.random.uniform(0.2, 0.5, n),
        "is_new_device": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "is_new_location": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "transactions_last_1h": np.random.poisson(3, n),
        "transactions_last_24h": np.random.poisson(8, n),
        "avg_amount_30d": np.random.exponential(400, n),
        "account_age_days": np.random.exponential(60, n) + 5,
        "latitude": [BRAZIL_COORDS[c]["lat"] + np.random.normal(0, 0.1) for c in selected_cities],
        "longitude": [BRAZIL_COORDS[c]["lon"] + np.random.normal(0, 0.1) for c in selected_cities],
        "canal": np.random.choice(CHANNELS, n, p=[0.5, 0.2, 0.2, 0.03, 0.04, 0.03]),
        "tipo_transacao": np.random.choice(["transferencia", "compra", "pagamento", "saque"], n, p=[0.6, 0.2, 0.1, 0.1]),
        "estado": np.random.choice(ESTADOS, n),
        "pais": np.array(["BR"] * n),
        "device_type": np.random.choice(DEVICE_TYPES, n),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n, p=[0.2, 0.15, 0.15, 0.45, 0.05]),  # Mais chaves aleatórias
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n),
        "day_of_week": np.random.choice(range(7), n),
        "is_night": np.ones(n, dtype=int),
        "is_weekend": np.random.choice([0, 1], n, p=[0.5, 0.5]),
    }


def generate_suspicious_location_fraud(n: int) -> dict:
    """Fraudes em localizações suspeitas (fronteiras)"""
    suspicious_cities = list(SUSPICIOUS_COORDS.keys())
    selected_cities = np.random.choice(suspicious_cities, n)

    return {
        "amount": np.random.exponential(3000, n) + 1000,
        "hour": np.random.choice(range(24), n),
        "velocity_score": np.random.uniform(0.4, 0.8, n),
        "device_risk_score": np.random.uniform(0.4, 0.7, n),
        "location_risk_score": np.random.uniform(0.7, 1.0, n),  # Alto risco de localização
        "is_new_device": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "is_new_location": np.ones(n, dtype=int),  # Sempre nova localização
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "transactions_last_1h": np.random.poisson(2, n),
        "transactions_last_24h": np.random.poisson(6, n),
        "avg_amount_30d": np.random.exponential(500, n),
        "account_age_days": np.random.exponential(90, n) + 10,
        "latitude": [SUSPICIOUS_COORDS[c]["lat"] + np.random.normal(0, 0.05) for c in selected_cities],
        "longitude": [SUSPICIOUS_COORDS[c]["lon"] + np.random.normal(0, 0.05) for c in selected_cities],
        "canal": np.random.choice(CHANNELS, n),
        "tipo_transacao": np.random.choice(["transferencia", "compra", "pagamento", "saque"], n, p=[0.7, 0.1, 0.1, 0.1]),
        "estado": np.random.choice(["PR", "MS", "AM", "AP", "RR"], n),  # Estados de fronteira
        "pais": np.random.choice(["BR", "PY", "AR", "UY"], n, p=[0.6, 0.2, 0.1, 0.1]),
        "device_type": np.random.choice(DEVICE_TYPES, n),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n),
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(["transferencia_pf", "transferencia_pj", "crypto_exchange"], n),
        "day_of_week": np.random.choice(range(7), n),
        "is_night": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "is_weekend": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    }


def generate_high_risk_category_fraud(n: int) -> dict:
    """Fraudes em categorias de alto risco"""
    cities = list(BRAZIL_COORDS.keys())
    selected_cities = np.random.choice(cities, n)

    return {
        "amount": np.random.exponential(5000, n) + 2000,
        "hour": np.random.choice(range(24), n),
        "velocity_score": np.random.uniform(0.3, 0.6, n),
        "device_risk_score": np.random.uniform(0.3, 0.6, n),
        "location_risk_score": np.random.uniform(0.2, 0.5, n),
        "is_new_device": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "is_new_location": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "transactions_last_1h": np.random.poisson(2, n),
        "transactions_last_24h": np.random.poisson(5, n),
        "avg_amount_30d": np.random.exponential(600, n),
        "account_age_days": np.random.exponential(120, n) + 15,
        "latitude": [BRAZIL_COORDS[c]["lat"] + np.random.normal(0, 0.1) for c in selected_cities],
        "longitude": [BRAZIL_COORDS[c]["lon"] + np.random.normal(0, 0.1) for c in selected_cities],
        "canal": np.random.choice(CHANNELS, n),
        "tipo_transacao": np.random.choice(["compra", "transferencia", "pagamento"], n),
        "estado": np.random.choice(ESTADOS, n),
        "pais": np.array(["BR"] * n),
        "device_type": np.random.choice(DEVICE_TYPES, n),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n),
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(HIGH_RISK_CATEGORIES, n),  # Categorias de alto risco
        "day_of_week": np.random.choice(range(7), n),
        "is_night": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "is_weekend": np.random.choice([0, 1], n, p=[0.5, 0.5]),
    }


def generate_new_device_fraud(n: int) -> dict:
    """Fraudes com dispositivo novo e valor alto"""
    cities = list(BRAZIL_COORDS.keys())
    selected_cities = np.random.choice(cities, n)

    return {
        "amount": np.random.exponential(6000, n) + 3000,
        "hour": np.random.choice(range(24), n),
        "velocity_score": np.random.uniform(0.2, 0.5, n),
        "device_risk_score": np.random.uniform(0.6, 0.9, n),  # Alto risco de dispositivo
        "location_risk_score": np.random.uniform(0.3, 0.6, n),
        "is_new_device": np.ones(n, dtype=int),  # Sempre dispositivo novo
        "is_new_location": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "transactions_last_1h": np.random.poisson(1, n),
        "transactions_last_24h": np.random.poisson(3, n),
        "avg_amount_30d": np.random.exponential(400, n),
        "account_age_days": np.random.exponential(200, n) + 20,
        "latitude": [BRAZIL_COORDS[c]["lat"] + np.random.normal(0, 0.1) for c in selected_cities],
        "longitude": [BRAZIL_COORDS[c]["lon"] + np.random.normal(0, 0.1) for c in selected_cities],
        "canal": np.random.choice(CHANNELS, n, p=[0.3, 0.3, 0.3, 0.03, 0.04, 0.03]),
        "tipo_transacao": np.random.choice(["transferencia", "compra", "pagamento", "saque"], n),
        "estado": np.random.choice(ESTADOS, n),
        "pais": np.array(["BR"] * n),
        "device_type": np.random.choice(["android", "ios", "unknown"], n, p=[0.4, 0.3, 0.3]),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n),
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n),
        "day_of_week": np.random.choice(range(7), n),
        "is_night": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "is_weekend": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    }


def generate_high_velocity_fraud(n: int) -> dict:
    """Fraudes com alta velocidade de transações"""
    cities = list(BRAZIL_COORDS.keys())
    selected_cities = np.random.choice(cities, n)

    return {
        "amount": np.random.exponential(2000, n) + 500,
        "hour": np.random.choice(range(24), n),
        "velocity_score": np.random.uniform(0.7, 1.0, n),  # Alta velocidade
        "device_risk_score": np.random.uniform(0.3, 0.6, n),
        "location_risk_score": np.random.uniform(0.2, 0.5, n),
        "is_new_device": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "is_new_location": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "is_new_receiver": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "transactions_last_1h": np.random.poisson(8, n) + 5,  # Muitas transações
        "transactions_last_24h": np.random.poisson(20, n) + 10,
        "avg_amount_30d": np.random.exponential(500, n),
        "account_age_days": np.random.exponential(150, n) + 10,
        "latitude": [BRAZIL_COORDS[c]["lat"] + np.random.normal(0, 0.1) for c in selected_cities],
        "longitude": [BRAZIL_COORDS[c]["lon"] + np.random.normal(0, 0.1) for c in selected_cities],
        "canal": np.random.choice(CHANNELS, n),
        "tipo_transacao": np.random.choice(["transferencia", "compra", "pagamento", "saque"], n, p=[0.5, 0.2, 0.2, 0.1]),
        "estado": np.random.choice(ESTADOS, n),
        "pais": np.array(["BR"] * n),
        "device_type": np.random.choice(DEVICE_TYPES, n),
        "pix_key_type": np.random.choice(PIX_KEY_TYPES, n),
        "banco_recebedor": np.random.choice(BANCOS, n),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n),
        "day_of_week": np.random.choice(range(7), n),
        "is_night": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "is_weekend": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    }


def get_normal_hour_distribution() -> list:
    """Retorna distribuição normal de horários de transação"""
    return [
        0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5h (madrugada)
        0.05, 0.07, 0.08, 0.09, 0.09, 0.08,  # 6-11h (manhã)
        0.07, 0.07, 0.07, 0.06, 0.05, 0.04,  # 12-17h (tarde)
        0.03, 0.02, 0.02, 0.01, 0.01, 0.00   # 18-23h (noite)
    ]


def train_and_evaluate_catboost(df: pd.DataFrame) -> dict:
    """Treina e avalia o modelo CatBoost"""

    if not CATBOOST_AVAILABLE:
        print("ERRO: CatBoost não está instalado!")
        print("Execute: pip install catboost")
        sys.exit(1)

    # Separar features e labels
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].values

    # Split temporal (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split de validação (do treino, 20% para validação)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nSplit dos dados:")
    print(f"  Treino: {len(X_train)} ({y_train.mean()*100:.2f}% fraudes)")
    print(f"  Validação: {len(X_val)} ({y_val.mean()*100:.2f}% fraudes)")
    print(f"  Teste: {len(X_test)} ({y_test.mean()*100:.2f}% fraudes)")

    # Inicializar modelo
    print("\nInicializando CatBoost...")
    model = CatBoostFraudModel({
        "model_path": str(backend_path / "models" / "production")
    })

    # Treinar
    print("Treinando modelo...")
    train_metrics = model.train(X_train, y_train, X_val, y_val)

    print(f"\nMétricas de Treino:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Avaliar no teste
    print("\nAvaliando no conjunto de teste...")
    predictions = model.predict(X_test)
    y_pred = np.array([p.is_fraud for p in predictions])
    y_pred_proba = np.array([p.fraud_probability for p in predictions])

    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nMétricas de Teste:")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legítima", "Fraude"]))

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    # Feature importance
    print("\nTop 10 Features mais importantes:")
    importance = model.get_feature_importance()
    for i, (feature, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feature}: {imp:.4f}")

    # Salvar modelo
    model_path = backend_path / "models" / "production" / "catboost_fraud_model.cbm"
    print(f"\nSalvando modelo em {model_path}...")
    model.save(str(model_path))

    # Salvar métricas
    metrics = {
        "trained_at": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "fraud_rate": float(y.mean()),
        "train_metrics": train_metrics,
        "test_auc": test_auc,
        "feature_importance": {k: float(v) for k, v in list(importance.items())[:20]},
        "features_used": model.feature_names,
        "categorical_features": model.categorical_features,
    }

    metrics_path = backend_path / "models" / "production" / "catboost_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas salvas em {metrics_path}")

    return metrics


def main():
    """Função principal"""
    print("=" * 60)
    print("SANKOFA ENTERPRISE PRO - CatBoost Model Training")
    print("=" * 60)

    # Gerar dados sintéticos
    df = generate_synthetic_fraud_data(n_samples=50000, fraud_rate=0.03)

    # Treinar e avaliar
    metrics = train_and_evaluate_catboost(df)

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"\nO modelo CatBoost agora pode usar os seguintes campos:")
    print("  - latitude/longitude (geolocalização)")
    print("  - merchant_category (categoria do comerciante)")
    print("  - location_risk_score")
    print("  - device_risk_score")
    print("  - E todas as outras features categóricas e numéricas")
    print(f"\nROC-AUC no teste: {metrics['test_auc']:.4f}")

    return metrics


if __name__ == "__main__":
    main()
