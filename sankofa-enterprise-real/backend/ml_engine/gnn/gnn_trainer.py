"""
Sankofa Enterprise Pro - GNN Trainer
Treinador para Graph Neural Networks

Baseado em:
- PyTorch training best practices
- Weighted loss for imbalanced data
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@dataclass
class TrainerConfig:
    """Configuração do treinador"""
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 0.001
    pos_weight: float = 10.0  # Peso para classe positiva (fraude)
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    clip_grad_norm: float = 1.0
    eval_every: int = 1
    save_best: bool = True
    device: str = "auto"


class GNNTrainer:
    """
    Treinador para GNN de detecção de fraude

    Features:
    - Weighted loss para dados desbalanceados
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Logging de métricas
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None
    ):
        self.config = config or TrainerConfig()
        self.model = model

        # Determinar device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Loss function com peso para classe positiva
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.pos_weight]).to(self.device)
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Estado de treinamento
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_f1": [],
            "learning_rate": []
        }

        logger.info(f"GNNTrainer v{self.VERSION} initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Config: {self.config}")

    def _create_scheduler(self):
        """Criar learning rate scheduler"""

        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        return None

    def train(
        self,
        train_data: List,
        val_data: Optional[List] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Treinar o modelo

        Args:
            train_data: Lista de Data objects ou DataLoader
            val_data: Dados de validação (opcional)
            callbacks: Lista de callbacks (opcional)

        Returns:
            Histórico de treinamento
        """
        logger.info(f"Starting training for {self.config.epochs} epochs...")

        # Criar DataLoader se necessário
        if HAS_PYG and not isinstance(train_data, DataLoader):
            train_loader = PyGDataLoader(
                train_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        else:
            train_loader = train_data

        val_loader = None
        if val_data is not None:
            if HAS_PYG and not isinstance(val_data, DataLoader):
                val_loader = PyGDataLoader(
                    val_data,
                    batch_size=self.config.batch_size,
                    shuffle=False
                )
            else:
                val_loader = val_data

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Warmup
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Treinar uma época
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validar
            if val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self._validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_auc"].append(val_metrics["auc"])
                self.history["val_f1"].append(val_metrics["f1"])

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"Val AUC: {val_metrics['auc']:.4f} - "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )

                # Early stopping check
                if self._check_early_stopping(val_metrics["auc"]):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                # Scheduler step
                if self.scheduler and self.config.scheduler == "plateau":
                    self.scheduler.step(val_metrics["auc"])
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}"
                )

            # Scheduler step (para outros schedulers)
            if self.scheduler and self.config.scheduler != "plateau" and epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_loss)

        return {
            "history": self.history,
            "best_metric": self.best_metric,
            "final_epoch": self.current_epoch
        }

    def _train_epoch(self, data_loader) -> float:
        """Treinar uma época"""

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            # Mover para device
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            if HAS_PYG and hasattr(batch, 'edge_index'):
                out = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=getattr(batch, 'edge_attr', None)
                )
                y = batch.y
            else:
                # Fallback
                out = self.model(batch)
                y = batch

            # Loss
            if out.dim() > 1:
                out = out.squeeze(-1)
            if y.dim() > 1:
                y = y.squeeze(-1)

            loss = self.criterion(out, y.float())

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )

            # Update
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate(self, data_loader) -> Dict[str, float]:
        """Validar o modelo"""

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)

                # Forward
                if HAS_PYG and hasattr(batch, 'edge_index'):
                    out = self.model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=getattr(batch, 'edge_attr', None)
                    )
                    y = batch.y
                else:
                    out = self.model(batch)
                    y = batch

                if out.dim() > 1:
                    out = out.squeeze(-1)
                if y.dim() > 1:
                    y = y.squeeze(-1)

                loss = self.criterion(out, y.float())
                total_loss += loss.item()

                # Coletar predições
                probs = torch.sigmoid(out)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Calcular métricas
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "loss": total_loss / max(len(data_loader), 1),
            "auc": self._calculate_auc(all_labels, all_preds),
            "f1": self._calculate_f1(all_labels, all_preds)
        }

        return metrics

    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular AUC-ROC"""
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) > 1:
                return roc_auc_score(y_true, y_pred)
        except ImportError:
            pass

        # Fallback: cálculo manual simplificado
        return 0.5

    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Calcular F1-score"""
        try:
            from sklearn.metrics import f1_score
            y_pred_binary = (y_pred >= threshold).astype(int)
            return f1_score(y_true, y_pred_binary, zero_division=0)
        except ImportError:
            pass

        # Fallback
        return 0.0

    def _check_early_stopping(self, current_metric: float) -> bool:
        """Verificar early stopping"""

        if current_metric > self.best_metric + self.config.min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0

            if self.config.save_best:
                self._save_checkpoint("best_model.pt")

            return False

        self.patience_counter += 1
        return self.patience_counter >= self.config.patience

    def _save_checkpoint(self, filename: str):
        """Salvar checkpoint do modelo"""

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "config": self.config
        }

        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename: str):
        """Carregar checkpoint"""

        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]

        logger.info(f"Checkpoint loaded: {filename}")

    def predict(self, data) -> np.ndarray:
        """Fazer predições"""

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            if isinstance(data, list):
                for batch in data:
                    if hasattr(batch, 'to'):
                        batch = batch.to(self.device)

                    if HAS_PYG and hasattr(batch, 'edge_index'):
                        out = self.model(
                            x=batch.x,
                            edge_index=batch.edge_index,
                            edge_attr=getattr(batch, 'edge_attr', None)
                        )
                    else:
                        out = self.model(batch)

                    probs = torch.sigmoid(out).cpu().numpy()
                    all_preds.extend(probs.flatten())
            else:
                if hasattr(data, 'to'):
                    data = data.to(self.device)

                if HAS_PYG and hasattr(data, 'edge_index'):
                    out = self.model(
                        x=data.x,
                        edge_index=data.edge_index,
                        edge_attr=getattr(data, 'edge_attr', None)
                    )
                else:
                    out = self.model(data)

                all_preds = torch.sigmoid(out).cpu().numpy().flatten()

        return np.array(all_preds)


def create_trainer(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None
) -> GNNTrainer:
    """Factory function para criar GNNTrainer"""
    if config:
        trainer_config = TrainerConfig(**config)
    else:
        trainer_config = TrainerConfig()
    return GNNTrainer(model, trainer_config)
