"""
Sankofa Enterprise Pro - CatBoost Model Integration
Motor de detecção de fraude usando CatBoost com suporte a features categóricas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path
from datetime import datetime

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    Pool = None

logger = logging.getLogger(__name__)


@dataclass
class CatBoostPrediction:
    """Resultado de predição do CatBoost"""
    fraud_probability: float
    is_fraud: bool
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]]
    processing_time_ms: float


class CatBoostFraudModel:
    """
    Modelo CatBoost para detecção de fraude
    
    Vantagens do CatBoost:
    - Suporte nativo a features categóricas
    - Ordenação de categorias otimizada
    - Menos overfitting em dados desbalanceados
    - GPU acceleration disponível
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        self.config = config or {}
        self.model = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.is_trained = False
        self.model_path = Path(self.config.get("model_path", "./models"))
        self.threshold = self.config.get("threshold", 0.5)
        
        self._initialize_model()
        logger.info(f"CatBoost Fraud Model initialized v{self.VERSION}")
    
    def _initialize_model(self):
        """Inicializa o modelo CatBoost com parâmetros otimizados para fraude"""
        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            border_count=128,
            class_weights=[1, 10],
            auto_class_weights='Balanced',
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
            task_type='CPU',
            thread_count=-1,
        )
    
    def get_categorical_features(self) -> List[str]:
        """Retorna lista de features categóricas para fraude"""
        return [
            'canal',
            'tipo_transacao', 
            'estado',
            'pais',
            'device_type',
            'pix_key_type',
            'banco_recebedor',
            'merchant_category',
            'day_of_week',
            'hour_bucket',
        ]
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Pré-processa dados para CatBoost
        
        Args:
            df: DataFrame com features
            fit: Se True, ajusta feature names
            
        Returns:
            DataFrame processado e índices de features categóricas
        """
        df = df.copy()
        
        numeric_features = [
            'amount', 'valor', 'latitude', 'longitude',
            'velocity_score', 'device_risk_score', 'location_risk_score',
            'hour', 'day_of_week', 'is_night', 'is_weekend',
            'is_new_device', 'is_new_location', 'is_new_receiver',
            'transactions_last_1h', 'transactions_last_24h',
            'avg_amount_30d', 'account_age_days',
        ]
        
        categorical_features = self.get_categorical_features()
        
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'unknown'
            df[col] = df[col].fillna('unknown').astype(str)
        
        if 'hour' in df.columns:
            df['hour_bucket'] = pd.cut(
                df['hour'], 
                bins=[-1, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening']
            ).astype(str)
        
        all_features = [f for f in numeric_features + categorical_features if f in df.columns]
        
        if fit:
            self.feature_names = all_features
            self.categorical_features = [f for f in categorical_features if f in df.columns]
        
        cat_indices = [all_features.index(f) for f in self.categorical_features if f in all_features]
        
        return df[all_features], cat_indices
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Treina o modelo CatBoost
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Métricas de treinamento
        """
        X_processed, cat_indices = self.preprocess_data(X_train, fit=True)
        
        train_pool = Pool(
            data=X_processed,
            label=y_train,
            cat_features=cat_indices
        )
        
        eval_pool = None
        if X_val is not None and y_val is not None:
            X_val_processed, _ = self.preprocess_data(X_val, fit=False)
            eval_pool = Pool(
                data=X_val_processed,
                label=y_val,
                cat_features=cat_indices
            )
        
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=eval_pool is not None,
            plot=False
        )
        
        self.is_trained = True
        
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, zero_division=0),
            'recall': recall_score(y_train, y_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_pred_proba) if len(np.unique(y_train)) > 1 else 0.5,
        }
        
        logger.info(f"CatBoost trained - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> List[CatBoostPrediction]:
        """
        Faz predições com o modelo
        
        Args:
            X: DataFrame com features
            
        Returns:
            Lista de predições
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        import time
        start_time = time.time()
        
        X_processed, _ = self.preprocess_data(X, fit=False)
        
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        processing_time = (time.time() - start_time) * 1000
        
        predictions = []
        for i, prob in enumerate(probabilities):
            predictions.append(CatBoostPrediction(
                fraud_probability=float(prob),
                is_fraud=prob >= self.threshold,
                feature_importance=feature_importance,
                shap_values=None,
                processing_time_ms=processing_time / len(probabilities)
            ))
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades para integração com ensemble"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        X_processed, _ = self.preprocess_data(X, fit=False)
        return self.model.predict_proba(X_processed)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calcula valores SHAP para explicabilidade"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        X_processed, cat_indices = self.preprocess_data(X, fit=False)
        pool = Pool(data=X_processed, cat_features=cat_indices)
        
        return self.model.get_feature_importance(pool, type='ShapValues')
    
    def save(self, filepath: Optional[str] = None):
        """Salva o modelo"""
        if filepath is None:
            filepath = self.model_path / "catboost_fraud_model.cbm"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(filepath))
        
        metadata = {
            'version': self.VERSION,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'threshold': self.threshold,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(metadata, str(filepath) + '.meta')
        
        logger.info(f"CatBoost model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """Carrega o modelo"""
        if filepath is None:
            filepath = self.model_path / "catboost_fraud_model.cbm"
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model.load_model(str(filepath))
        
        meta_path = str(filepath) + '.meta'
        if Path(meta_path).exists():
            metadata = joblib.load(meta_path)
            self.feature_names = metadata.get('feature_names', [])
            self.categorical_features = metadata.get('categorical_features', [])
            self.threshold = metadata.get('threshold', 0.5)
        
        self.is_trained = True
        logger.info(f"CatBoost model loaded from {filepath}")


class CatBoostEnsembleIntegration:
    """Integração do CatBoost com o ensemble existente"""
    
    def __init__(self, catboost_model: CatBoostFraudModel, weight: float = 0.25):
        self.catboost = catboost_model
        self.weight = weight
    
    def get_ensemble_prediction(
        self,
        X: pd.DataFrame,
        other_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combina predição do CatBoost com outros modelos
        
        Args:
            X: Features
            other_predictions: Dict com predições de outros modelos
            
        Returns:
            Predição combinada
        """
        if not self.catboost.is_trained:
            total_weight = sum(1 for _ in other_predictions.values())
            combined = sum(pred for pred in other_predictions.values()) / total_weight
            return combined
        
        catboost_proba = self.catboost.predict_proba(X)[:, 1]
        
        weights = {'catboost': self.weight}
        remaining_weight = 1 - self.weight
        other_weight = remaining_weight / len(other_predictions)
        
        combined = catboost_proba * self.weight
        for name, pred in other_predictions.items():
            weights[name] = other_weight
            combined += pred * other_weight
        
        return combined


def create_catboost_model(config: Optional[Dict[str, Any]] = None) -> CatBoostFraudModel:
    """Factory function para criar modelo CatBoost"""
    if not CATBOOST_AVAILABLE:
        logger.warning("CatBoost not available, returning None")
        return None
    
    return CatBoostFraudModel(config)
