# Model from "Query-Based Adaptive Aggregation for Multi-Dataset Joint Training
#             Toward Universal Visual Place Recognition"
#         - https://arxiv.org/abs/2507.03831
# Parts of this code are from https://github.com/xjh19972/QAA

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

MODEL_REPOS = {
    1024: "xjh19972/QAA-1024",
    2048: "xjh19972/QAA-2048",
    4096: "xjh19972/QAA-4096",
    8192: "xjh19972/QAA-8192",
}

DINOV2_ARCHS = {
    'dinov2_vits14': 384, 'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536,
}


class QuerySelfAttn(nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, self_attn_flag=True, self_attn_out_norm=True):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn_flag = self_attn_flag
        self.self_attn_out_norm = self_attn_out_norm
        if self_attn_flag:
            self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)

    def forward(self):
        q = self.queries
        if self.self_attn_flag:
            q = q + self.self_attn(q, q, q)[0]
        if self.self_attn_out_norm:
            q = self.norm_q(q)
        return q


class QueryCrossAttn(nn.Module):
    def __init__(self, in_dim, output_dim, nheads=8, arch="conv", skip="none", out_norm=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)
        self.out_norm = out_norm
        self.conv = nn.Conv1d(in_dim, output_dim, 1)
        self.norm2_out = nn.LayerNorm(output_dim)

    def forward(self, x, q):
        x_flatten = x.flatten(2).permute(0, 2, 1)
        out, _ = self.cross_attn(q, x_flatten, x_flatten)
        out = self.norm_out(out)
        out = self.conv(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.out_norm:
            out = self.norm2_out(out)
        return out.permute(0, 2, 1)


class QAAAggregator(nn.Module):
    def __init__(self, num_channels=768, num_clusters=64, cluster_dim=128, token_dim=0,
                 num_queries=256, feature_nheads=16, score_nheads=16,
                 self_attn_out_norm=True, out_norm=False, score_norm="none",
                 attn_arch="conv", skip_connection="none", **kwargs):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.score_norm = score_norm
        if token_dim != 0:
            self.token_features = nn.Sequential(
                nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim))
        self.queries_feature = QuerySelfAttn(cluster_dim, num_queries, nheads=feature_nheads,
                                             self_attn_flag=True, self_attn_out_norm=self_attn_out_norm)
        self.queries_score = QuerySelfAttn(num_channels, num_queries, nheads=score_nheads,
                                           self_attn_flag=True, self_attn_out_norm=self_attn_out_norm)
        self.score = QueryCrossAttn(num_channels, num_clusters, nheads=score_nheads,
                                    arch=attn_arch, skip=skip_connection, out_norm=out_norm)
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x, t = x
        f = self.queries_feature().permute(0, 2, 1).repeat(x.shape[0], 1, 1)
        q = self.queries_score().repeat(x.shape[0], 1, 1)
        p = self.score(x, q)
        if self.token_dim != 0:
            t = self.token_features(t)
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        vlad = F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        if self.token_dim != 0:
            vlad = torch.cat([F.normalize(t, p=2, dim=-1), vlad], dim=-1)
        return F.normalize(vlad, p=2, dim=-1)


class DINOv2Backbone(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', num_trainable_blocks=2,
                 norm_layer=False, return_token=False, **kwargs):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.norm_layer = norm_layer
        self.domain_prompt = "none"

    def forward(self, x, domain_idx=None):
        B, _, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks:
            x = blk(x)
        if self.norm_layer:
            x = self.model.norm(x)
        t = x[:, 0]
        f = x[:, 1:].reshape(B, H // 14, W // 14, self.num_channels).permute(0, 3, 1, 2)
        return f, t


class QAA(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_config_dict):
        super().__init__()
        self.model_config = model_config_dict
        self.decorrelation = 'none'
        self.backbone = DINOv2Backbone(
            model_name=model_config_dict['backbone_arch'],
            **model_config_dict['backbone_config'])
        self.aggregator = QAAAggregator(**model_config_dict['agg_config'])

    def forward(self, x):
        return self.aggregator(self.backbone(x))


def get_qaa(descriptors_dimension=8192):
    if descriptors_dimension not in MODEL_REPOS:
        raise ValueError(f"descriptors_dimension must be one of {list(MODEL_REPOS.keys())}")
    model = QAA.from_pretrained(MODEL_REPOS[descriptors_dimension])
    model.eval()
    return model
