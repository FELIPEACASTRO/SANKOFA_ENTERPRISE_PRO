"""
Sankofa Enterprise Pro - Temporal Graph Network
GNN para grafos dinâmicos temporais

Baseado em:
- TGN (Temporal Graph Networks) - Twitter Research
- TGAT (Temporal Graph Attention)
- DyRep (Dynamic Representation Learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Verificar se PyTorch Geometric está disponível
try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@dataclass
class TGNConfig:
    """Configuração do Temporal Graph Network"""
    node_dim: int = 64
    edge_dim: int = 16
    time_dim: int = 64
    memory_dim: int = 100
    embedding_dim: int = 100
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    aggregator: str = "last"  # last, mean, attention
    memory_updater: str = "gru"  # gru, rnn


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network para grafos dinâmicos

    Componentes:
    1. Memory Module: Mantém estado de cada nó
    2. Message Function: Gera mensagens de eventos
    3. Message Aggregator: Agrega mensagens recebidas
    4. Memory Updater: Atualiza memória com novas mensagens
    5. Embedding Module: Gera embeddings finais
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[TGNConfig] = None):
        super().__init__()

        self.config = config or TGNConfig()

        if not HAS_PYG:
            self._init_fallback()
            return

        self._init_modules()

        logger.info(f"TemporalGraphNetwork v{self.VERSION} initialized")

    def _init_fallback(self):
        """Modo fallback sem PyG"""
        self.fallback_mode = True
        self.fallback_model = nn.Sequential(
            nn.Linear(self.config.node_dim + self.config.time_dim, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, 1),
            nn.Sigmoid()
        )

    def _init_modules(self):
        """Inicializar módulos do TGN"""
        self.fallback_mode = False

        # Memory: estado persistente de cada nó
        self.memory = None  # Inicializado no reset
        self.last_update = None  # Timestamp da última atualização

        # Time encoder
        self.time_encoder = TimeEncoderTGN(self.config.time_dim)

        # Message function
        self.message_function = MessageFunction(
            memory_dim=self.config.memory_dim,
            edge_dim=self.config.edge_dim,
            time_dim=self.config.time_dim,
            output_dim=self.config.memory_dim
        )

        # Message aggregator
        if self.config.aggregator == "attention":
            self.message_aggregator = AttentionAggregator(self.config.memory_dim)
        else:
            self.message_aggregator = LastAggregator()

        # Memory updater
        if self.config.memory_updater == "gru":
            self.memory_updater = GRUMemoryUpdater(
                memory_dim=self.config.memory_dim,
                message_dim=self.config.memory_dim
            )
        else:
            self.memory_updater = RNNMemoryUpdater(
                memory_dim=self.config.memory_dim,
                message_dim=self.config.memory_dim
            )

        # Embedding module (Graph Attention)
        self.embedding_module = TemporalAttention(
            node_dim=self.config.memory_dim,
            edge_dim=self.config.edge_dim,
            time_dim=self.config.time_dim,
            output_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )

        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.embedding_dim // 2, 1)
        )

    def reset_memory(self, num_nodes: int, device: torch.device = None):
        """Resetar memória para novo batch"""
        device = device or torch.device('cpu')

        self.memory = torch.zeros(num_nodes, self.config.memory_dim, device=device)
        self.last_update = torch.zeros(num_nodes, device=device)

    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass para um batch de eventos

        Args:
            source_nodes: [batch_size] - nós de origem
            target_nodes: [batch_size] - nós de destino
            timestamps: [batch_size] - timestamps dos eventos
            edge_features: [batch_size, edge_dim] - features das arestas
            node_features: [num_nodes, node_dim] - features dos nós

        Returns:
            [batch_size, 1] - probabilidade de fraude
        """
        if self.fallback_mode:
            # Fallback simples
            x = node_features[source_nodes] if node_features is not None else torch.randn(len(source_nodes), self.config.node_dim)
            t = self.time_encoder(timestamps) if hasattr(self, 'time_encoder') else torch.zeros(len(timestamps), self.config.time_dim)
            return self.fallback_model(torch.cat([x, t], dim=-1))

        batch_size = len(source_nodes)
        device = source_nodes.device

        # Garantir memória inicializada
        if self.memory is None:
            num_nodes = max(source_nodes.max(), target_nodes.max()).item() + 1
            self.reset_memory(num_nodes, device)

        # 1. Obter memórias atuais
        source_memory = self.memory[source_nodes]
        target_memory = self.memory[target_nodes]

        # 2. Calcular time encoding
        source_time_delta = timestamps - self.last_update[source_nodes]
        target_time_delta = timestamps - self.last_update[target_nodes]

        source_time_enc = self.time_encoder(source_time_delta)
        target_time_enc = self.time_encoder(target_time_delta)

        # 3. Gerar mensagens
        source_messages = self.message_function(
            source_memory, target_memory, edge_features, source_time_enc
        )
        target_messages = self.message_function(
            target_memory, source_memory, edge_features, target_time_enc
        )

        # 4. Agregar mensagens (para nós com múltiplos eventos)
        # Simplificado: usar última mensagem
        aggregated_source = source_messages
        aggregated_target = target_messages

        # 5. Atualizar memória
        new_source_memory = self.memory_updater(source_memory, aggregated_source)
        new_target_memory = self.memory_updater(target_memory, aggregated_target)

        # 6. Gerar embeddings
        source_embedding = self.embedding_module(
            new_source_memory, source_time_enc
        )

        # 7. Classificar
        logits = self.classifier(source_embedding)

        # 8. Atualizar memória (detach para não backprop através do tempo)
        with torch.no_grad():
            self.memory[source_nodes] = new_source_memory.detach()
            self.memory[target_nodes] = new_target_memory.detach()
            self.last_update[source_nodes] = timestamps
            self.last_update[target_nodes] = timestamps

        return torch.sigmoid(logits)

    def get_embeddings(
        self,
        node_ids: torch.Tensor,
        current_time: float
    ) -> torch.Tensor:
        """Obter embeddings atuais para nós específicos"""

        if self.memory is None:
            return torch.zeros(len(node_ids), self.config.embedding_dim)

        node_memory = self.memory[node_ids]
        time_delta = current_time - self.last_update[node_ids]
        time_enc = self.time_encoder(time_delta)

        return self.embedding_module(node_memory, time_enc)


class TimeEncoderTGN(nn.Module):
    """Time encoder usando funções de base"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)

        # Inicializar com frequências variadas
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] - time deltas

        Returns:
            [batch_size, dim]
        """
        t = t.unsqueeze(-1).float()  # [batch_size, 1]
        return torch.cos(self.w(t))


class MessageFunction(nn.Module):
    """Função de mensagem para gerar mensagens de eventos"""

    def __init__(
        self,
        memory_dim: int,
        edge_dim: int,
        time_dim: int,
        output_dim: int
    ):
        super().__init__()

        input_dim = memory_dim * 2 + time_dim
        if edge_dim > 0:
            input_dim += edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        source_memory: torch.Tensor,
        target_memory: torch.Tensor,
        edge_features: Optional[torch.Tensor],
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """Gerar mensagem"""

        inputs = [source_memory, target_memory, time_encoding]
        if edge_features is not None:
            inputs.append(edge_features)

        return self.mlp(torch.cat(inputs, dim=-1))


class LastAggregator(nn.Module):
    """Agregador que usa última mensagem"""

    def forward(
        self,
        messages: torch.Tensor,
        indices: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        return messages  # Já é a última


class AttentionAggregator(nn.Module):
    """Agregador com atenção"""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(
        self,
        messages: torch.Tensor,
        indices: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        # Atenção sobre mensagens
        weights = F.softmax(self.attention(messages), dim=0)
        return weights * messages


class GRUMemoryUpdater(nn.Module):
    """Atualizador de memória baseado em GRU"""

    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def forward(
        self,
        memory: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        return self.gru(message, memory)


class RNNMemoryUpdater(nn.Module):
    """Atualizador de memória baseado em RNN"""

    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.rnn = nn.RNNCell(message_dim, memory_dim)

    def forward(
        self,
        memory: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        return self.rnn(message, memory)


class TemporalAttention(nn.Module):
    """Módulo de atenção temporal para embeddings"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        output_dim: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=node_dim if i == 0 else output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )

        self.output_proj = nn.Linear(node_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        node_memory: torch.Tensor,
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_memory: [batch_size, memory_dim]
            time_encoding: [batch_size, time_dim]

        Returns:
            [batch_size, output_dim]
        """
        # Adicionar dimensão de sequência
        x = node_memory.unsqueeze(1)  # [batch, 1, dim]

        # Passar por camadas de atenção (self-attention)
        for layer in self.layers:
            attn_out, _ = layer(x, x, x)
            x = x + attn_out

        # Projetar para output
        x = self.output_proj(x.squeeze(1))
        x = self.norm(x)

        return x


def create_tgn(config: Optional[Dict[str, Any]] = None) -> TemporalGraphNetwork:
    """Factory function para criar TGN"""
    if config:
        tgn_config = TGNConfig(**config)
    else:
        tgn_config = TGNConfig()
    return TemporalGraphNetwork(tgn_config)
