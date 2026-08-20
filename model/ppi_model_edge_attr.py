import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, TransformerConv,
    global_mean_pool, global_max_pool,
)
from torch_geometric.utils import softmax


def orthogonal_init(module, gain=1.0):
    """对 nn.Linear 做正交初始化（避免逐层随机初始化的方差漂移）。

    使用 torch.nn.init.orthogonal_（内部 QR 分解），对任意形状都不会零填充。
    """
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
    """支持 edge_attr 的 PPI 亲和力模型。

    通过 conv_type 在 GINEConv / GATConv / TransformerConv 三种支持边属性的
    图卷积之间切换，与 MutiChainPPIGBuilder_all_power_optimized.py 产出的
    Data(x, aa_types, edge_index, edge_attr, y) 直接配套。

    边属性流程：原始 9 维 edge_attr -> edge_encoder(Linear) -> hidden_dim
    -> 各卷积层以 edge_dim=hidden_dim 消费。
    """

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

        # 边属性编码器：把原始 edge_attr(9) 投影到 hidden_dim
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_dim)

        # 构建卷积层
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

        # 注意力门控（图级池化权重）
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 图级池化
        self.pooling = pooling
        pool_dim = hidden_dim * (2 if pooling == 'mean_max' else 1)

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 仅对输入投影与边编码器做正交初始化（卷积层用 PyG 默认 glorot）
        orthogonal_init(self.input_proj)
        orthogonal_init(self.edge_encoder)

    def _conv_forward(self, conv, h, edge_index, edge_attr):
        if self.conv_type == 'gine':
            return conv(h, edge_index, edge_attr)
        return conv(h, edge_index, edge_attr)

    def forward(self, data):
        # 1. 节点特征拼接与初始投影
        aa_emb = self.aa_embed(data.aa_types)
        h = torch.cat([data.x, aa_emb], dim=1)
        h = self.input_proj(h)

        # 2. 边属性编码（投影到 hidden_dim）
        edge_attr = self.edge_encoder(data.edge_attr)

        # 3. 残差图卷积堆叠（统一包装：conv -> BN -> ReLU -> dropout -> +残差）
        for conv, bn in zip(self.convs, self.bns):
            identity = h
            out = self._conv_forward(conv, h, data.edge_index, edge_attr)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=0.2, training=self.training)
            h = out + identity

        # 4. 注意力加权池化
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

        # 5. 回归输出
        out = self.regressor(graph_vec)
        return out.squeeze(-1)


def build_model(conv_type='gine', **kwargs):
    """工厂函数：快速构造指定类型的模型。"""