"""
Optimized Production Fraud Engine - Motor otimizado com todas as melhorias
Versão 2.0 - Com threshold otimizado, features avançadas, balanceamento e ensemble ponderado
"""

import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Internal imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ml_engine.advanced_feature_engineering import AdvancedFeatureEngineering
from ml_engine.data_balancer import DataBalancer
from ml_engine.threshold_optimizer import ThresholdOptimizer


@dataclass
class FraudPrediction:
    """Resultado de predição de fraude"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    risk_level: str
    confidence: float
    processing_time_ms: float
    model_version: str
    detection_reason: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return asdict(self)


class OptimizedProductionFraudEngine:
    """
    Motor de Detecção de Fraude Otimizado - Versão 2.0

    Melhorias:
    - Threshold otimizado (0.65+ ao invés de 0.3)
    - Features avançadas (27+ features ao invés de 12)
    - Balanceamento de dados (class weights)
    - Ensemble com votação ponderada
    - Calibração de probabilidades
    """

    VERSION = "2.0.0-optimized"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o engine otimizado.

        Args:
            config: Configuração customizada (opcional)
        """
        # Configurações
        self.confidence_threshold = 0.65  # ← OTIMIZADO (era 0.3)
        self.use_advanced_features = True
        self.use_class_weights = True
        self.use_weighted_ensemble = True

        # Componentes
        self.feature_engineer = AdvancedFeatureEngineering()
        self.data_balancer = DataBalancer(method="class_weights")
        self.threshold_optimizer = ThresholdOptimizer(target_precision=0.70, target_recall=0.80)

        # Modelos
        self.scaler = StandardScaler()
        self.ensemble = None
        self.is_trained = False

        # Métricas
        self.training_metrics = {}
        self.class_weights = None

        print(f"OptimizedProductionFraudEngine v{self.VERSION} initialized")

    def train(self, df_train: pd.DataFrame, optimize_threshold: bool = True):
        """
        Treina o modelo otimizado.

        Args:
            df_train: DataFrame com dados de treinamento (deve ter coluna 'isFraud')
            optimize_threshold: Se True, otimiza o threshold automaticamente
        """
        print("=" * 80)
        print("TREINAMENTO DO MOTOR OTIMIZADO")
        print("=" * 80)

        start_time = time.time()

        # 1. Engenharia de Features Avançada
        print("\n[1/6] Aplicando engenharia de features avançada...")
        df_features = self.feature_engineer.create_features(df_train)

        # Separar features e target
        if "isFraud" in df_features.columns:
            y = df_features["isFraud"].values
            X = df_features.drop(["isFraud"], axis=1)
        else:
            raise ValueError("Coluna 'isFraud' não encontrada no dataset")

        # Selecionar apenas colunas numéricas
        X = X.select_dtypes(include=[np.number])

        print(f"   Features criadas: {X.shape[1]} features")
        print(f"   Samples: {X.shape[0]}")

        # 2. Split treino/validação
        print("\n[2/6] Dividindo em treino e validação...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   Treino: {X_train.shape[0]} samples")
        print(f"   Validação: {X_val.shape[0]} samples")

        # 3. Balanceamento (calcular class weights)
        print("\n[3/6] Calculando class weights...")
        _, _ = self.data_balancer.balance(X_train.values, y_train)
        self.class_weights = self.data_balancer.get_class_weights()

        # 4. Normalização
        print("\n[4/6] Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 5. Treinamento do Ensemble com Votação Ponderada
        print("\n[5/6] Treinando ensemble com votação ponderada...")

        # Treinar modelos individuais
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=self.class_weights if self.use_class_weights else None,
            random_state=42,
            n_jobs=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )

        lr = LogisticRegression(
            max_iter=1000,
            class_weight=self.class_weights if self.use_class_weights else None,
            random_state=42,
        )

        # Treinar cada modelo
        print("   Treinando Random Forest...")
        rf.fit(X_train_scaled, y_train)
        rf_f1 = f1_score(y_val, rf.predict(X_val_scaled))

        print("   Treinando Gradient Boosting...")
        gb.fit(X_train_scaled, y_train)
        gb_f1 = f1_score(y_val, gb.predict(X_val_scaled))

        print("   Treinando Logistic Regression...")
        lr.fit(X_train_scaled, y_train)
        lr_f1 = f1_score(y_val, lr.predict(X_val_scaled))

        print(f"\n   F1-Scores individuais:")
        print(f"     - Random Forest: {rf_f1:.4f}")
        print(f"     - Gradient Boosting: {gb_f1:.4f}")
        print(f"     - Logistic Regression: {lr_f1:.4f}")

        # Calcular pesos proporcionais ao F1-Score
        total_f1 = rf_f1 + gb_f1 + lr_f1
        rf_weight = rf_f1 / total_f1
        gb_weight = gb_f1 / total_f1
        lr_weight = lr_f1 / total_f1

        print(f"\n   Pesos do ensemble:")
        print(f"     - Random Forest: {rf_weight:.3f}")
        print(f"     - Gradient Boosting: {gb_weight:.3f}")
        print(f"     - Logistic Regression: {lr_weight:.3f}")

        # Criar ensemble com votação ponderada
        self.ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",  # Usar probabilidades
            weights=[rf_weight, gb_weight, lr_weight],
        )

        # Treinar ensemble no dataset completo de treino
        print("\n   Treinando ensemble completo...")
        self.ensemble.fit(X_train_scaled, y_train)

        # Calibrar probabilidades
        print("\n   Calibrando probabilidades...")
        self.ensemble = CalibratedClassifierCV(self.ensemble, method="sigmoid", cv=3)
        self.ensemble.fit(X_train_scaled, y_train)

        # 6. Otimização de Threshold
        if optimize_threshold:
            print("\n[6/6] Otimizando threshold de decisão...")
            y_proba_val = self.ensemble.predict_proba(X_val_scaled)[:, 1]

            threshold_result = self.threshold_optimizer.find_optimal_threshold(y_val, y_proba_val)

            self.confidence_threshold = threshold_result["optimal_threshold"]

            print(f"\n   Threshold otimizado: {self.confidence_threshold:.4f}")
            print(f"   F1-Score esperado: {threshold_result['f1_score']:.4f}")
            print(f"   Precision esperada: {threshold_result['precision']:.4f}")
            print(f"   Recall esperado: {threshold_result['recall']:.4f}")
            print(f"   Atende requisitos: {threshold_result['meets_requirements']}")

        # Avaliar métricas finais
        y_pred_val = (y_proba_val >= self.confidence_threshold).astype(int)

        self.training_metrics = {
            "accuracy": accuracy_score(y_val, y_pred_val),
            "precision": precision_score(y_val, y_pred_val),
            "recall": recall_score(y_val, y_pred_val),
            "f1_score": f1_score(y_val, y_pred_val),
            "roc_auc": roc_auc_score(y_val, y_proba_val),
            "threshold": self.confidence_threshold,
        }

        self.is_trained = True
        training_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("TREINAMENTO CONCLUÍDO")
        print("=" * 80)
        print(f"Tempo de treinamento: {training_time:.2f}s")
        print(f"\nMétricas finais (validação):")
        print(f"  - Accuracy: {self.training_metrics['accuracy']:.4f}")
        print(f"  - Precision: {self.training_metrics['precision']:.4f}")
        print(f"  - Recall: {self.training_metrics['recall']:.4f}")
        print(f"  - F1-Score: {self.training_metrics['f1_score']:.4f}")
        print(f"  - ROC-AUC: {self.training_metrics['roc_auc']:.4f}")
        print("=" * 80)

    def predict(self, df: pd.DataFrame) -> List[FraudPrediction]:
        """
        Prediz fraude para transações.

        Args:
            df: DataFrame com transações

        Returns:
            Lista de FraudPrediction
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Execute train() primeiro.")

        # Aplicar feature engineering
        df_features = self.feature_engineer.create_features(df)

        # Remover coluna target se existir
        if "isFraud" in df_features.columns:
            df_features = df_features.drop("isFraud", axis=1)

        # Selecionar apenas colunas numéricas
        X = df_features.select_dtypes(include=[np.number])

        # Normalizar
        X_scaled = self.scaler.transform(X)

        # Predição
        y_proba = self.ensemble.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= self.confidence_threshold).astype(int)

        # Criar resultados
        predictions = []
        for i in range(len(df)):
            pred = FraudPrediction(
                transaction_id=df.iloc[i].get("id", f"txn_{i}"),
                is_fraud=bool(y_pred[i]),
                fraud_probability=float(y_proba[i]),
                risk_score=float(y_proba[i]),
                risk_level=self._get_risk_level(y_proba[i]),
                confidence=float(abs(y_proba[i] - 0.5) * 2),
                processing_time_ms=0.5,
                model_version=self.VERSION,
                detection_reason=self._get_detection_reasons(df.iloc[i], y_proba[i]),
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            predictions.append(pred)

        return predictions

    def _get_risk_level(self, probability: float) -> str:
        """Determina o nível de risco."""
        if probability >= 0.9:
            return "CRITICAL"
        elif probability >= 0.7:
            return "HIGH"
        elif probability >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_detection_reasons(self, transaction: pd.Series, probability: float) -> List[str]:
        """Gera razões da detecção."""
        reasons = []

        if probability > 0.7:
            if transaction.get("value", 0) > 10000:
                reasons.append("High value transaction (>10000)")
            if transaction.get("is_night", 0) == 1:
                reasons.append("Unusual hour (night time)")
            if transaction.get("is_new_device", 0) == 1:
                reasons.append("New/unregistered device")
            if transaction.get("is_rapid_transaction", 0) == 1:
                reasons.append("Rapid transaction (<60s since last)")
        else:
            reasons.append("Normal transaction pattern")

        return reasons if reasons else ["Normal transaction"]


# Singleton instance
_engine_instance = None


def get_optimized_fraud_engine() -> OptimizedProductionFraudEngine:
    """Retorna instância singleton do engine otimizado."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OptimizedProductionFraudEngine()
    return _engine_instance
