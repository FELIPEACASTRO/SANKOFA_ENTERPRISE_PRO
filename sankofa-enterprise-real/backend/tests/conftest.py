"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def small_fraud_dataset():
    """
    Fixture com dataset pequeno (100 samples) para testes rápidos
    """
    np.random.seed(42)
    n_samples = 100  # Reduzido para velocidade
    
    X = pd.DataFrame({
        'amount': np.random.lognormal(5, 2, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'merchant_risk': np.random.uniform(0, 1, n_samples),
        'user_history': np.random.uniform(0, 100, n_samples),
        'velocity_1h': np.random.randint(0, 10, n_samples)
    })
    
    # 15% fraude (correlacionado)
    y = ((X['amount'] > X['amount'].quantile(0.85)) & 
         (X['merchant_risk'] > 0.7)).astype(int)
    
    return X, y.values


@pytest.fixture(scope="function")
def fraud_engine():
    """Fixture que retorna engine limpo"""
    from ml_engine.production_fraud_engine import ProductionFraudEngine
    return ProductionFraudEngine()


@pytest.fixture(scope="function")
def trained_engine(small_fraud_dataset, fraud_engine):
    """Fixture que retorna engine já treinado"""
    X, y = small_fraud_dataset
    fraud_engine.fit(X, y)
    return fraud_engine, X, y
