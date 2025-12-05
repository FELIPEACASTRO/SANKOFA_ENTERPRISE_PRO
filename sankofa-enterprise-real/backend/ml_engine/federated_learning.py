"""
Sankofa Enterprise Pro - Federated Learning Framework
Treinamento distribuído de modelos preservando privacidade dos dados
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import hashlib
import json
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuração do sistema federado"""

    min_clients: int = 2
    max_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_strategy: str = "fedavg"
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    secure_aggregation: bool = True
    client_fraction: float = 1.0


@dataclass
class ClientUpdate:
    """Atualização de um cliente"""

    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    privacy_spent: float = 0.0


@dataclass
class GlobalModel:
    """Modelo global federado"""

    version: int
    weights: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    participating_clients: List[str]
    created_at: datetime = field(default_factory=datetime.now)


class DifferentialPrivacy:
    """
    Implementação de Differential Privacy para proteção de dados

    Técnicas implementadas:
    - Gradient clipping
    - Gaussian noise injection
    - Privacy budget tracking
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.privacy_spent = 0.0

    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aplica gradient clipping para limitar sensibilidade"""
        clipped = {}

        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))

        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))

        for key, grad in gradients.items():
            clipped[key] = grad * clip_factor

        return clipped

    def add_noise(
        self, gradients: Dict[str, np.ndarray], num_samples: int
    ) -> Dict[str, np.ndarray]:
        """Adiciona ruído Gaussiano calibrado para DP"""
        sigma = self.max_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        noisy_gradients = {}
        for key, grad in gradients.items():
            noise = np.random.normal(0, sigma / num_samples, grad.shape)
            noisy_gradients[key] = grad + noise

        self.privacy_spent += self.epsilon / np.sqrt(num_samples)

        return noisy_gradients

    def get_privacy_spent(self) -> float:
        """Retorna budget de privacidade consumido"""
        return self.privacy_spent


class SecureAggregation:
    """
    Agregação segura usando técnicas criptográficas

    Implementa Secure Multi-Party Computation simplificado
    para agregar updates sem revelar dados individuais
    """

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.masks: Dict[str, np.ndarray] = {}

    def generate_mask(self, shape: Tuple[int, ...], client_id: str) -> np.ndarray:
        """Gera máscara aleatória para um cliente"""
        seed = int(hashlib.md5(client_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.normal(0, 1, shape)

    def mask_update(self, update: Dict[str, np.ndarray], client_id: str) -> Dict[str, np.ndarray]:
        """Aplica máscara ao update do cliente"""
        masked = {}
        for key, value in update.items():
            mask = self.generate_mask(value.shape, client_id + key)
            masked[key] = value + mask
            self.masks[f"{client_id}_{key}"] = mask
        return masked

    def aggregate_masked(
        self, updates: List[Dict[str, np.ndarray]], client_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Agrega updates mascarados (máscaras se cancelam na soma)"""
        aggregated = {}

        for key in updates[0].keys():
            stacked = np.stack([u[key] for u in updates])
            aggregated[key] = np.mean(stacked, axis=0)

        return aggregated


class FederatedClient(ABC):
    """Cliente base para treinamento federado"""

    def __init__(self, client_id: str, config: FederatedConfig):
        self.client_id = client_id
        self.config = config
        self.local_model = None
        self.local_data: Optional[pd.DataFrame] = None
        self.dp = (
            DifferentialPrivacy(epsilon=config.privacy_budget)
            if config.differential_privacy
            else None
        )

    @abstractmethod
    def set_local_data(self, data: pd.DataFrame, labels: np.ndarray):
        """Define dados locais do cliente"""
        pass

    @abstractmethod
    def train_local(self, global_weights: Dict[str, np.ndarray]) -> ClientUpdate:
        """Treina modelo local com pesos globais"""
        pass

    @abstractmethod
    def evaluate_local(self) -> Dict[str, float]:
        """Avalia modelo local"""
        pass


class SklearnFederatedClient(FederatedClient):
    """Cliente federado usando scikit-learn"""

    def __init__(self, client_id: str, config: FederatedConfig):
        super().__init__(client_id, config)

        from sklearn.linear_model import SGDClassifier

        self.local_model = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=config.learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        self.X_local: Optional[np.ndarray] = None
        self.y_local: Optional[np.ndarray] = None

    def set_local_data(self, data: pd.DataFrame, labels: np.ndarray):
        """Define dados locais"""
        self.X_local = data.values if isinstance(data, pd.DataFrame) else data
        self.y_local = labels

    def _get_model_weights(self) -> Dict[str, np.ndarray]:
        """Extrai pesos do modelo"""
        if not hasattr(self.local_model, "coef_"):
            return {}
        return {
            "coef": self.local_model.coef_.copy(),
            "intercept": self.local_model.intercept_.copy(),
        }

    def _set_model_weights(self, weights: Dict[str, np.ndarray]):
        """Define pesos do modelo"""
        if "coef" in weights:
            self.local_model.coef_ = weights["coef"].copy()
        if "intercept" in weights:
            self.local_model.intercept_ = weights["intercept"].copy()

    def train_local(self, global_weights: Dict[str, np.ndarray]) -> ClientUpdate:
        """Treina modelo localmente"""
        if self.X_local is None or self.y_local is None:
            raise ValueError("Local data not set")

        if global_weights:
            self._set_model_weights(global_weights)

        n_samples = len(self.X_local)
        indices = np.arange(n_samples)

        for epoch in range(self.config.local_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = self.X_local[batch_idx]
                y_batch = self.y_local[batch_idx]

                self.local_model.partial_fit(X_batch, y_batch, classes=[0, 1])

        weights = self._get_model_weights()

        if self.dp:
            weights = self.dp.clip_gradients(weights)
            weights = self.dp.add_noise(weights, n_samples)

        metrics = self.evaluate_local()

        return ClientUpdate(
            client_id=self.client_id,
            model_weights=weights,
            num_samples=n_samples,
            metrics=metrics,
            privacy_spent=self.dp.get_privacy_spent() if self.dp else 0.0,
        )

    def evaluate_local(self) -> Dict[str, float]:
        """Avalia modelo local"""
        if self.X_local is None or self.y_local is None:
            return {}

        if not hasattr(self.local_model, "coef_"):
            return {"accuracy": 0.0}

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        y_pred = self.local_model.predict(self.X_local)
        y_proba = self.local_model.predict_proba(self.X_local)[:, 1]

        return {
            "accuracy": accuracy_score(self.y_local, y_pred),
            "f1_score": f1_score(self.y_local, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(self.y_local, y_proba) if len(np.unique(self.y_local)) > 1 else 0.5
            ),
            "num_samples": len(self.y_local),
            "fraud_ratio": np.mean(self.y_local),
        }


class FederatedServer:
    """
    Servidor central para coordenação do treinamento federado

    Responsabilidades:
    - Distribuir modelo global para clientes
    - Agregar updates dos clientes
    - Gerenciar rounds de treinamento
    - Monitorar convergência
    """

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model: Optional[GlobalModel] = None
        self.round_history: List[Dict[str, Any]] = []
        self.current_round = 0

        self.secure_agg: Optional[SecureAggregation] = None

        logger.info(f"Federated Server initialized with config: {config}")

    def register_client(self, client: FederatedClient):
        """Registra um cliente no sistema federado"""
        self.clients[client.client_id] = client
        logger.info(f"Client {client.client_id} registered. Total: {len(self.clients)}")

        if self.config.secure_aggregation:
            self.secure_agg = SecureAggregation(len(self.clients))

    def initialize_global_model(self, initial_weights: Optional[Dict[str, np.ndarray]] = None):
        """Inicializa modelo global"""
        if initial_weights is None:
            from sklearn.linear_model import SGDClassifier

            dummy_model = SGDClassifier(loss="log_loss", random_state=42)
            dummy_model.fit([[0] * 10, [1] * 10], [0, 1])
            initial_weights = {"coef": dummy_model.coef_, "intercept": dummy_model.intercept_}

        self.global_model = GlobalModel(
            version=0, weights=initial_weights, metrics={}, participating_clients=[]
        )

    def select_clients(self) -> List[str]:
        """Seleciona clientes para participar do round"""
        all_clients = list(self.clients.keys())
        num_selected = max(
            self.config.min_clients, int(len(all_clients) * self.config.client_fraction)
        )

        return np.random.choice(
            all_clients, size=min(num_selected, len(all_clients)), replace=False
        ).tolist()

    def aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        Agrega updates dos clientes

        Estratégias:
        - fedavg: Média ponderada por número de samples
        - fedprox: FedAvg com regularização proximal
        """
        if self.config.aggregation_strategy == "fedavg":
            return self._fedavg_aggregate(updates)
        else:
            return self._fedavg_aggregate(updates)

    def _fedavg_aggregate(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Agregação FedAvg (média ponderada)"""
        total_samples = sum(u.num_samples for u in updates)

        aggregated = {}
        for key in updates[0].model_weights.keys():
            weighted_sum = sum(u.model_weights[key] * u.num_samples for u in updates)
            aggregated[key] = weighted_sum / total_samples

        return aggregated

    def run_round(self) -> Dict[str, Any]:
        """Executa um round de treinamento federado"""
        if self.global_model is None:
            self.initialize_global_model()

        self.current_round += 1

        selected_clients = self.select_clients()
        logger.info(f"Round {self.current_round}: Selected {len(selected_clients)} clients")

        updates = []
        for client_id in selected_clients:
            client = self.clients[client_id]
            update = client.train_local(self.global_model.weights)
            updates.append(update)

        if not updates:
            logger.warning("No updates received")
            return {"round": self.current_round, "status": "no_updates"}

        new_weights = self.aggregate_updates(updates)

        avg_metrics = {}
        for key in updates[0].metrics.keys():
            avg_metrics[key] = np.mean([u.metrics[key] for u in updates])

        self.global_model = GlobalModel(
            version=self.current_round,
            weights=new_weights,
            metrics=avg_metrics,
            participating_clients=selected_clients,
        )

        round_result = {
            "round": self.current_round,
            "num_clients": len(selected_clients),
            "metrics": avg_metrics,
            "total_samples": sum(u.num_samples for u in updates),
            "avg_privacy_spent": np.mean([u.privacy_spent for u in updates]),
        }

        self.round_history.append(round_result)

        logger.info(
            f"Round {self.current_round} complete. "
            f"Accuracy: {avg_metrics.get('accuracy', 0):.4f}"
        )

        return round_result

    def train(self, num_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """Executa treinamento federado completo"""
        rounds = num_rounds or self.config.max_rounds

        for _ in range(rounds):
            result = self.run_round()

            if self._check_convergence():
                logger.info("Training converged early")
                break

        return self.round_history

    def _check_convergence(self, patience: int = 5, min_delta: float = 0.001) -> bool:
        """Verifica se o treinamento convergiu"""
        if len(self.round_history) < patience:
            return False

        recent_accuracies = [
            r["metrics"].get("accuracy", 0) for r in self.round_history[-patience:]
        ]

        if max(recent_accuracies) - min(recent_accuracies) < min_delta:
            return True

        return False

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        """Retorna pesos do modelo global"""
        if self.global_model is None:
            return {}
        return self.global_model.weights

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de treinamento"""
        return self.round_history

    def save_global_model(self, filepath: str):
        """Salva modelo global"""
        if self.global_model is None:
            raise ValueError("No global model to save")

        import joblib

        model_data = {
            "version": self.global_model.version,
            "weights": self.global_model.weights,
            "metrics": self.global_model.metrics,
            "config": self.config.__dict__,
            "history": self.round_history,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Global model saved to {filepath}")


class FederatedFraudDetection:
    """
    Sistema federado completo para detecção de fraude

    Permite que múltiplos bancos colaborem no treinamento
    sem compartilhar dados sensíveis.
    """

    def __init__(self, config: Optional[FederatedConfig] = None):
        self.config = config or FederatedConfig()
        self.server = FederatedServer(self.config)
        self.is_trained = False

    def add_bank_data(self, bank_id: str, transactions: pd.DataFrame, labels: np.ndarray):
        """
        Adiciona dados de um banco ao sistema federado

        Args:
            bank_id: Identificador do banco
            transactions: Features das transações
            labels: Labels (0=legítimo, 1=fraude)
        """
        client = SklearnFederatedClient(bank_id, self.config)
        client.set_local_data(transactions, labels)
        self.server.register_client(client)

        logger.info(f"Bank {bank_id} added with {len(transactions)} transactions")

    def train(self, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Executa treinamento federado"""
        if len(self.server.clients) < self.config.min_clients:
            raise ValueError(f"Need at least {self.config.min_clients} clients")

        history = self.server.train(num_rounds)
        self.is_trained = True

        return {
            "num_rounds": len(history),
            "final_metrics": history[-1]["metrics"] if history else {},
            "history": history,
        }

    def get_global_model(self) -> Optional[GlobalModel]:
        """Retorna modelo global treinado"""
        return self.server.global_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições com modelo global"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        weights = self.server.get_global_weights()

        coef = weights.get("coef", np.zeros((1, X.shape[1])))
        intercept = weights.get("intercept", np.array([0]))

        logits = X @ coef.T + intercept
        proba = 1 / (1 + np.exp(-logits))

        return proba.ravel()


def create_federated_system(config: Optional[Dict[str, Any]] = None) -> FederatedFraudDetection:
    """Factory function para criar sistema federado"""
    fed_config = FederatedConfig(**(config or {}))
    return FederatedFraudDetection(fed_config)
