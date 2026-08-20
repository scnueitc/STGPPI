import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool,
)
from torch_geometric.utils import softmax


def orthogonal_init(module, gain=1.0):
    
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidueEmbedding(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(21, embed_dim)

    def forward(self, aa_types):
        return self.embedding(aa_types)


class PPIAffinityModelEdgeAttr(nn.Module):
    
    def __init__(self, cont_feat_dim=3, aa_embed_dim=32, hidden_dim=64,
                 num_layers=3, dropout=0.2, conv_type='gine',
                 pooling='mean_max', heads=4, edge_attr_dim=9):
        super().__init__()
        assert conv_type in ('gine', 'gat', 'transformer'), f"未知 conv_type: {conv_type}"
        if conv_type in ('gat', 'transformer'):
            assert hidden_dim % heads == 0, "hidden_dim 必须能被 heads 整除"
        self.conv_type = conv_type
        self.heads = heads
        out_channels = hidden_dim // heads

        self.aa_embed = ResidueEmbedding(aa_embed_dim)
        self.input_proj = nn.Linear(cont_feat_dim + aa_embed_dim, hidden_dim)

        
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_dim)

       
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == 'gine':
                nn_ = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(nn_, edge_dim=hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(
                    GATConv(hidden_dim, out_channels, heads=heads,
                            edge_dim=hidden_dim, dropout=dropout, add_self_loops=True)
                )
            elif conv_type == 'transformer':
                self.convs.append(
                    TransformerConv(hidden_dim, out_channels, heads=heads,
                                    edge_dim=hidden_dim, dropout=dropout)
                )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

       
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        
        self.pooling = pooling
        pool_dim = hidden_dim * (2 if pooling == 'mean_max' else 1)

        
        self.regressor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

       
        orthogonal_init(self.input_proj)
        orthogonal_init(self.edge_encoder)

    def _conv_forward(self, conv, h, edge_index, edge_attr):
        if self.conv_type == 'gine':
            return conv(h, edge_index, edge_attr)
        return conv(h, edge_index, edge_attr)

    def forward(self, data):
        
        aa_emb = self.aa_embed(data.aa_types)
        h = torch.cat([data.x, aa_emb], dim=1)
        h = self.input_proj(h)

        
        edge_attr = self.edge_encoder(data.edge_attr)

        
        for conv, bn in zip(self.convs, self.bns):
            identity = h
            out = self._conv_forward(conv, h, data.edge_index, edge_attr)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=0.2, training=self.training)
            h = out + identity

       
        att_scores = self.attention_gate(h)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), device=h.device, dtype=torch.long)
        att_weights = softmax(att_scores, batch)
        weighted_h = h * att_weights

        if self.pooling == "mean_max":
            graph_vec = torch.cat([
                global_mean_pool(weighted_h, batch),
                global_max_pool(weighted_h, batch),
            ], dim=-1)
        else:
            graph_vec = global_mean_pool(weighted_h, batch)

       
        out = self.regressor(graph_vec)
        return out.squeeze(-1)


def build_model(conv_type='gine', **kwargs):
  
