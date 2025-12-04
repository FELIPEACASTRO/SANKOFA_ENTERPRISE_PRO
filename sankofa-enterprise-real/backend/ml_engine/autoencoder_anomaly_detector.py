"""
Sankofa Enterprise Pro - Autoencoder Anomaly Detector
Detecção de anomalias não supervisionada baseada em reconstrução
Referência: MoE Paper (arXiv:2504.03750) + FinSafeNet (Nature 2024)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class AnomalyResult:
    """Resultado da detecção de anomalia"""
    transaction_id: str
    reconstruction_error: float
    anomaly_score: float
    is_anomaly: bool
    percentile_rank: float
    feature_contributions: Dict[str, float]
    anomaly_type: str
    confidence: float
    timestamp: str


@dataclass
class AutoencoderMetrics:
    """Métricas do Autoencoder"""
    mean_reconstruction_error: float
    std_reconstruction_error: float
    threshold: float
    anomaly_rate: float
    model_version: str


class AutoencoderAnomalyDetector:
    """
    Detector de Anomalias baseado em Autoencoder
    
    Arquitetura:
    - Encoder: Comprime features para representação latente
    - Decoder: Reconstrói features originais
    - Anomalias = Alto erro de reconstrução
    
    Features:
    - Detecção não supervisionada (sem labels)
    - Identifica fraudes inéditas
    - Feature importance via gradientes
    - Threshold adaptativo por percentil
    
    Baseado em:
    - MoE Paper (arXiv:2504.03750): Autoencoder para anomalias
    - FinSafeNet (Nature 2024): Deep learning para transações
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        encoding_dim: int = 16,
        hidden_layers: List[int] = None,
        anomaly_percentile: float = 95.0,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Inicializa o Autoencoder
        
        Args:
            encoding_dim: Dimensão do espaço latente
            hidden_layers: Camadas ocultas do encoder
            anomaly_percentile: Percentil para threshold de anomalia
            learning_rate: Taxa de aprendizado
            epochs: Épocas de treinamento
            batch_size: Tamanho do batch
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.anomaly_percentile = anomaly_percentile
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.feature_names: List[str] = []
        self.training_errors: List[float] = []
        self.is_fitted = False
        
        logger.info(f"AutoencoderAnomalyDetector initialized (v{self.VERSION})")
    
    def _build_model(self, input_dim: int) -> None:
        """Constrói arquitetura do autoencoder"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using simplified model")
            self._build_simple_model(input_dim)
            return
        
        input_layer = keras.layers.Input(shape=(input_dim,))
        
        x = input_layer
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        x = encoded
        for units in reversed(self.hidden_layers):
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
        
        decoded = keras.layers.Dense(input_dim, activation='linear', name='output')(x)
        
        self.model = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        logger.info(f"Autoencoder built: {input_dim} -> {self.hidden_layers} -> {self.encoding_dim}")
    
    def _build_simple_model(self, input_dim: int) -> None:
        """Modelo simplificado sem TensorFlow (usando PCA)"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Neither TensorFlow nor sklearn available")
        
        self.pca_model = PCA(n_components=min(self.encoding_dim, input_dim))
        self.use_simple = True
        logger.info(f"Using PCA-based simple autoencoder: {input_dim} -> {self.encoding_dim}")
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepara features para o modelo"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_cols = ['transaction_id', 'is_fraud', 'fraud_probability', 'label']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if not numeric_cols:
            raise ValueError("No numeric features found in data")
        
        self.feature_names = numeric_cols
        X = data[numeric_cols].fillna(0).values
        
        return X
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina o autoencoder com dados normais
        
        Args:
            data: DataFrame com transações (idealmente não-fraudulentas)
            
        Returns:
            Métricas de treinamento
        """
        logger.info(f"Training autoencoder on {len(data)} samples")
        
        X = self._prepare_features(data)
        X_scaled = self.scaler.fit_transform(X)
        
        input_dim = X_scaled.shape[1]
        
        if hasattr(self, 'use_simple') and self.use_simple:
            self.pca_model.fit(X_scaled)
            X_reconstructed = self.pca_model.inverse_transform(
                self.pca_model.transform(X_scaled)
            )
            reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        else:
            self._build_model(input_dim)
            
            history = self.model.fit(
                X_scaled, X_scaled,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            X_reconstructed = self.model.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        self.training_errors = reconstruction_errors.tolist()
        self.threshold = np.percentile(reconstruction_errors, self.anomaly_percentile)
        
        self.is_fitted = True
        
        metrics = AutoencoderMetrics(
            mean_reconstruction_error=float(np.mean(reconstruction_errors)),
            std_reconstruction_error=float(np.std(reconstruction_errors)),
            threshold=float(self.threshold),
            anomaly_rate=float(np.mean(reconstruction_errors > self.threshold)),
            model_version=self.VERSION
        )
        
        logger.info(f"Autoencoder trained. Threshold: {self.threshold:.6f}")
        
        return {
            'mean_error': metrics.mean_reconstruction_error,
            'std_error': metrics.std_reconstruction_error,
            'threshold': metrics.threshold,
            'anomaly_rate': metrics.anomaly_rate,
            'features_count': len(self.feature_names)
        }
    
    def detect_anomaly(self, transaction: Dict[str, Any]) -> AnomalyResult:
        """
        Detecta se uma transação é anômala
        
        Args:
            transaction: Dados da transação
            
        Returns:
            Resultado da detecção de anomalia
        """
        if not self.is_fitted:
            return self._create_default_result(transaction)
        
        start_time = datetime.now()
        
        df = pd.DataFrame([transaction])
        
        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features:
            return self._create_default_result(transaction)
        
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0
        
        X = df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self, 'use_simple') and self.use_simple:
            X_reconstructed = self.pca_model.inverse_transform(
                self.pca_model.transform(X_scaled)
            )
        else:
            X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        feature_errors = (X_scaled[0] - X_reconstructed[0]) ** 2
        reconstruction_error = float(np.mean(feature_errors))
        
        percentile_rank = float(
            np.mean([1 if e < reconstruction_error else 0 for e in self.training_errors]) * 100
        )
        
        is_anomaly = reconstruction_error > self.threshold
        
        if reconstruction_error > 0:
            normalized_error = reconstruction_error / (self.threshold + 1e-10)
            anomaly_score = min(1.0, normalized_error)
        else:
            anomaly_score = 0.0
        
        feature_contributions = {}
        if len(feature_errors) == len(self.feature_names):
            total_error = sum(feature_errors) + 1e-10
            for i, fname in enumerate(self.feature_names):
                feature_contributions[fname] = float(feature_errors[i] / total_error)
        
        top_contributors = sorted(
            feature_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        if is_anomaly:
            if percentile_rank > 99:
                anomaly_type = "extreme_outlier"
            elif any('amount' in f.lower() for f, _ in top_contributors[:2]):
                anomaly_type = "unusual_amount"
            elif any('velocity' in f.lower() or 'frequency' in f.lower() for f, _ in top_contributors[:2]):
                anomaly_type = "unusual_velocity"
            elif any('time' in f.lower() or 'hour' in f.lower() for f, _ in top_contributors[:2]):
                anomaly_type = "unusual_timing"
            else:
                anomaly_type = "behavioral_anomaly"
        else:
            anomaly_type = "normal"
        
        confidence = min(1.0, abs(reconstruction_error - self.threshold) / (self.threshold + 1e-10))
        
        transaction_id = transaction.get('transaction_id', f"TXN_{hashlib.md5(str(transaction).encode()).hexdigest()[:8]}")
        
        return AnomalyResult(
            transaction_id=transaction_id,
            reconstruction_error=reconstruction_error,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            percentile_rank=percentile_rank,
            feature_contributions=dict(top_contributors),
            anomaly_type=anomaly_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_default_result(self, transaction: Dict[str, Any]) -> AnomalyResult:
        """Cria resultado padrão quando modelo não está treinado"""
        transaction_id = transaction.get('transaction_id', 'UNKNOWN')
        return AnomalyResult(
            transaction_id=transaction_id,
            reconstruction_error=0.0,
            anomaly_score=0.0,
            is_anomaly=False,
            percentile_rank=50.0,
            feature_contributions={},
            anomaly_type="not_analyzed",
            confidence=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    def batch_detect(self, transactions: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """
        Detecta anomalias em lote
        
        Args:
            transactions: Lista de transações
            
        Returns:
            Lista de resultados de anomalia
        """
        return [self.detect_anomaly(t) for t in transactions]
    
    def get_latent_representation(self, transaction: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Obtém representação latente (encoding) da transação
        
        Args:
            transaction: Dados da transação
            
        Returns:
            Vetor de representação latente
        """
        if not self.is_fitted:
            return None
        
        df = pd.DataFrame([transaction])
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0
        
        X = df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self, 'use_simple') and self.use_simple:
            return self.pca_model.transform(X_scaled)[0]
        else:
            return self.encoder.predict(X_scaled, verbose=0)[0]
    
    def update_threshold(self, new_percentile: float) -> float:
        """
        Atualiza threshold de anomalia
        
        Args:
            new_percentile: Novo percentil (0-100)
            
        Returns:
            Novo threshold
        """
        if not self.training_errors:
            return self.threshold or 0.0
        
        self.anomaly_percentile = new_percentile
        self.threshold = np.percentile(self.training_errors, new_percentile)
        
        logger.info(f"Threshold updated: {self.threshold:.6f} (percentile {new_percentile})")
        
        return float(self.threshold)
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do detector"""
        return {
            "version": self.VERSION,
            "is_fitted": self.is_fitted,
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "anomaly_percentile": self.anomaly_percentile,
            "threshold": self.threshold,
            "features_count": len(self.feature_names),
            "training_samples": len(self.training_errors),
            "tensorflow_available": TF_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE
        }


_autoencoder_instance = None

def get_autoencoder_detector() -> AutoencoderAnomalyDetector:
    """Obtém instância singleton do detector"""
    global _autoencoder_instance
    if _autoencoder_instance is None:
        _autoencoder_instance = AutoencoderAnomalyDetector()
    return _autoencoder_instance
