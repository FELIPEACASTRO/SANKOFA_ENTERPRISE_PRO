"""
Sankofa Enterprise Pro - Fraud GNN
Graph Neural Network principal para detecção de fraude

Baseado em:
- ATLAS (Capital One)
- HOGRL (IJCAI 2024)
- GAT (Graph Attention Networks)
- GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Verificar se PyTorch Geometric está disponível
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, SAGEConv, GCNConv, TransformerConv
    from torch_geometric.nn import global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning("PyTorch Geometric not installed. GNN features will be limited.")


@dataclass
class FraudGNNConfig:
    """Configuração do GNN"""
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.3
    use_edge_features: bool = True
    edge_dim: int = 16
    aggregation: str = "mean"  # mean, max, attention
    residual: bool = True
    layer_norm: bool = True
    use_temporal: bool = True


class FraudGNN(nn.Module):
    """
    Graph Neural Network Avançado para Detecção de Fraude

    Arquitetura:
    - Multi-head Graph Attention (GAT)
    - Residual connections
    - Layer normalization
    - Edge features
    - Temporal encoding
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[FraudGNNConfig] = None):
        super().__init__()

        self.config = config or FraudGNNConfig()

        if not HAS_PYG:
            logger.error("PyTorch Geometric required for FraudGNN")
            self._init_fallback()
            return

        self._init_layers()

        logger.info(f"FraudGNN v{self.VERSION} initialized")
        logger.info(f"Config: {self.config}")

    def _init_fallback(self):
        """Inicializar modelo fallback sem PyG"""
        self.fallback_mode = True
        self.fallback_model = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim),
            nn.Sigmoid()
        )

    def _init_layers(self):
        """Inicializar camadas do GNN"""
        self.fallback_mode = False

        # Projeção de entrada
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

        # Temporal encoding (se habilitado)
        if self.config.use_temporal:
            self.time_encoder = TimeEncoder(self.config.hidden_dim)

        # Edge feature projection (se habilitado)
        if self.config.use_edge_features:
            self.edge_encoder = nn.Linear(self.config.edge_dim, self.config.hidden_dim)

        # Camadas GAT
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(self.config.num_layers):
            in_channels = self.config.hidden_dim
            out_channels = self.config.hidden_dim

            # Usar TransformerConv para edge features
            if self.config.use_edge_features:
                layer = TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels // self.config.num_heads,
                    heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    edge_dim=self.config.hidden_dim if i == 0 else None,
                    concat=True
                )
            else:
                layer = GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels // self.config.num_heads,
                    heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    concat=True
                )

            self.gat_layers.append(layer)

            if self.config.layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_channels))

        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim)
        )

        # Pooling attention (se usar attention aggregation)
        if self.config.aggregation == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 1),
                nn.Softmax(dim=0)
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            timestamps: Node timestamps [num_nodes]
            batch: Batch assignment [num_nodes]

        Returns:
            Fraud probabilities [num_nodes, 1] or [batch_size, 1]
        """
        if self.fallback_mode:
            return self.fallback_model(x)

        # 1. Projeção de entrada
        h = self.input_proj(x)

        # 2. Adicionar encoding temporal
        if self.config.use_temporal and timestamps is not None:
            time_enc = self.time_encoder(timestamps)
            h = h + time_enc

        # 3. Encode edge features
        edge_enc = None
        if self.config.use_edge_features and edge_attr is not None:
            edge_enc = self.edge_encoder(edge_attr)

        # 4. Message passing com GAT
        for i, gat_layer in enumerate(self.gat_layers):
            h_prev = h

            # Aplicar GAT
            if self.config.use_edge_features and edge_enc is not None and i == 0:
                h = gat_layer(h, edge_index, edge_attr=edge_enc)
            else:
                h = gat_layer(h, edge_index)

            # Activation
            h = F.elu(h)

            # Residual connection
            if self.config.residual and h_prev.shape == h.shape:
                h = h + h_prev

            # Layer norm
            if self.config.layer_norm:
                h = self.layer_norms[i](h)

            # Dropout
            h = F.dropout(h, p=self.config.dropout, training=self.training)

        # 5. Classificação
        out = self.classifier(h)

        # 6. Sigmoid para probabilidade
        out = torch.sigmoid(out)

        return out

    def forward_batch(
        self,
        data_list: List['Data']
    ) -> torch.Tensor:
        """
        Forward para batch de grafos

        Args:
            data_list: Lista de objetos Data do PyG

        Returns:
            Fraud probabilities [total_nodes, 1]
        """
        if self.fallback_mode or not HAS_PYG:
            # Concatenar features e processar
            x = torch.cat([d.x for d in data_list], dim=0)
            return self.fallback_model(x)

        # Criar batch
        batch = Batch.from_data_list(data_list)

        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=getattr(batch, 'edge_attr', None),
            timestamps=getattr(batch, 'timestamps', None),
            batch=batch.batch
        )

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Obter embeddings dos nós (antes da classificação)"""

        if self.fallback_mode:
            h = x
            for layer in list(self.fallback_model.children())[:-2]:
                h = layer(h)
            return h

        h = self.input_proj(x)

        for i, gat_layer in enumerate(self.gat_layers):
            h_prev = h
            h = gat_layer(h, edge_index)
            h = F.elu(h)

            if self.config.residual and h_prev.shape == h.shape:
                h = h + h_prev

            if self.config.layer_norm:
                h = self.layer_norms[i](h)

        return h


class TimeEncoder(nn.Module):
    """Encoder temporal usando funções sin/cos"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Frequências para encoding
        self.frequencies = nn.Parameter(
            torch.randn(hidden_dim // 2) * 0.1,
            requires_grad=True
        )

        # Projeção final
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps

        Args:
            timestamps: [num_nodes] - timestamps normalizados

        Returns:
            [num_nodes, hidden_dim] - encoding temporal
        """
        # Normalizar timestamps se necessário
        if timestamps.max() > 1:
            timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)

        # Reshape para broadcasting
        t = timestamps.unsqueeze(-1)  # [N, 1]
        freqs = self.frequencies.unsqueeze(0)  # [1, hidden_dim//2]

        # Sin/cos encoding
        angles = t * freqs * 2 * np.pi
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        # Concatenar
        enc = torch.cat([sin_enc, cos_enc], dim=-1)

        return self.proj(enc)


class GraphAttentionLayer(nn.Module):
    """
    Camada de Graph Attention customizada

    Implementa atenção multi-head com edge features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: Optional[int] = None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # Transformações lineares
        self.W = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, out_channels * heads)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.edge_encoder = None
            self.att_edge = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.att_edge is not None:
            nn.init.xavier_uniform_(self.att_edge)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [N, in_channels]
            edge_index: [2, E]
            edge_attr: [E, edge_dim] (opcional)

        Returns:
            [N, heads * out_channels]
        """
        N = x.size(0)

        # Transformar features
        x = self.W(x).view(N, self.heads, self.out_channels)

        # Calcular attention scores
        src, dst = edge_index
        alpha_src = (x[src] * self.att_src).sum(dim=-1)
        alpha_dst = (x[dst] * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst

        # Adicionar edge features se disponíveis
        if edge_attr is not None and self.edge_encoder is not None:
            edge_emb = self.edge_encoder(edge_attr).view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_emb * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        # Softmax por nó destino
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Agregar mensagens
        out = torch.zeros(N, self.heads, self.out_channels, device=x.device)

        for h in range(self.heads):
            # Softmax local
            alpha_h = alpha[:, h]
            alpha_softmax = torch.zeros(N, device=x.device)

            for i in range(edge_index.size(1)):
                d = dst[i]
                alpha_softmax[d] = torch.logsumexp(
                    torch.stack([alpha_softmax[d], alpha_h[i]]), dim=0
                )

            # Agregar
            for i in range(edge_index.size(1)):
                s, d = src[i], dst[i]
                weight = torch.exp(alpha_h[i] - alpha_softmax[d])
                out[d, h] += weight * x[s, h]

        return out.view(N, -1)


def create_fraud_gnn(config: Optional[Dict[str, Any]] = None) -> FraudGNN:
    """Factory function para criar FraudGNN"""
    if config:
        gnn_config = FraudGNNConfig(**config)
    else:
        gnn_config = FraudGNNConfig()
    return FraudGNN(gnn_config)
