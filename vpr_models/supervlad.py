# Model from "SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition" 
#         - https://proceedings.neurips.cc/paper_files/paper/2024/file/0b135d408253205ba501d55c6539bfc7-Paper-Conference.pdf
# Parts of this code are from https://github.com/Lu-Feng/SuperVLAD

import os
import gdown
from typing import Literal
from collections import OrderedDict
import math, os

import torch
from torch import nn
import torch.nn.functional as F

MODELS_INFO = {
    "SuperVLAD": (
        "https://drive.google.com/file/d/1wRkUO4E8s5hNRNNIWcuA8RUvlGob3Tbf/view",
        4, 1, False
    ),
    "SuperVLAD-CrossImage": (
        "https://drive.google.com/file/d/1yomnWGTJko6nf3F2Ju6RWsLhP2EG82tL/view",
        4, 1, True
    ),
    "1-ClusterVLAD": (
        "https://drive.google.com/file/d/1pQcJx9n2-keAh9TttssZkz6D0vjpFWU6/view",
        1, 2, False
    ),
}


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    
class SuperVLAD(nn.Module):
    """SuperVLAD layer implementation"""

    def __init__(self, clusters_num=4, ghost_clusters_num=1, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        clusters_num += ghost_clusters_num
        self.clusters_num = clusters_num
        self.ghost_clusters_num = ghost_clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        # self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = vlad[:,:-self.ghost_clusters_num,:]
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad


class SuperVLADModel(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, supervlad_clusters=4, crossimage_encoder=False, ghost_clusters=1):
        super().__init__()
        self.arch_name = "Dinov2"
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
        self.backbone_features_dim = 768

        self.crossimage_encoder = crossimage_encoder
        self.aggregation = SuperVLAD(clusters_num=supervlad_clusters, ghost_clusters_num=ghost_clusters, dim=self.backbone_features_dim, work_with_tokens=False)

        if self.crossimage_encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Cross-image encoder

    def forward(self, x):
        x = self.backbone(x, is_training=True)

        B,P,D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P-1))
        #x0 = x[:, 0]
        x1 = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 
        x = self.aggregation(x1)
        
        if self.crossimage_encoder:
            x = self.encoder(x.view(B,-1,D)).view(B,-1)

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

def get_supervlad(model_name: Literal['SuperVLAD', 'SuperVLAD-CrossImage', "1-ClusterVLAD"] = "SuperVLAD", descriptors_dimension: int = 3072):
    if model_name not in MODELS_INFO:
        raise ValueError(f"model should be one of {list(MODELS_INFO.keys())}")
    try:
        descriptors_dimension = int(descriptors_dimension)
    except:
        raise ValueError(f"descriptors_dimension should be an integer, not a {type(descriptors_dimension)}")

    drive_url, sv_clusters, gh_clusters, has_ci_enc = MODELS_INFO[model_name]
    model = SuperVLADModel(supervlad_clusters=sv_clusters, ghost_clusters=gh_clusters, crossimage_encoder=has_ci_enc)

    file_path = f"trained_models/supervlad/{model_name}.pth"
    if not os.path.exists(file_path):
        os.makedirs("trained_models/supervlad", exist_ok=True)
        gdown.download(url=drive_url, output=file_path, fuzzy=True) 
    state_dict = torch.load(file_path, weights_only=False)['model_state_dict']
    
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)

    model = model.eval()

    return model
