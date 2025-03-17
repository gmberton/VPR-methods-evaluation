# Model from "DINO-Mix: Enhancing Visual Place Recognition with Foundational Vision Model and Feature Mixing"
# - https://arxiv.org/abs/2311.00230
# Parts of this code are from https://github.com/GaoShuang98/DINO-Mix


from typing import Literal

import numpy as np

import torch
from torch import nn
from .mixvpr import MixVPR

CHECKPOINT_URL = "https://github.com/GaoShuang98/DINO-Mix/releases/download/v1.0.0/dinov2_vitb14_mix.ckpt"

DEFAULT_AGG_CONFIG = {
    'in_channels': 768,
    'in_h': 16,
    'in_w': 16,
    'out_channels': 1024,
    'mix_depth': 2,
    'mlp_ratio': 1,
    'out_rows': 4
}

_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2(nn.Module):
    """
    Extract features from an intermediate layer in Dino-v2
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39, use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model: The DINO-v2 model to use
            - layer: The layer to extract features from
            - use_cls: If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs: If True, the descriptors are normalized
            - device: PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layer1 = layer1

        self.pretrained = pretrained
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name

        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            if self.layer1 > 39:
                print('Please confirm the correctness of the layer! The highest block layer of vitg14 is 39 layers')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            if self.layer1 > 23:
                print('Please confirm the correctness of the layer! The highest block layer of vitl14 is 23 layers')
                exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            if self.layer1 > 11:
                print('Please confirm the correctness of the layer! The highest block layer of VITB14 is 11 layers')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            if self.layer1 > 11:
                print('Please confirm the correctness of the layer! The highest block layer of vits14 is 11 layers')
                exit()
        else:
            print(f'The model name definition is incorrect, please check model_name:{self.dino_model}')

        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)

        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()

    def forward(self, x):

        x = self.dino_model.forward_features(x)
        x = x['x_norm_patchtokens']
        bs, f, c = x.shape
        x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)
        return x.permute(0, 3, 1, 2)


class DinoMixModel(torch.nn.Module):
    """
    VPR Model with a backbone and an aggregator.

    Args:
        backbone_arch (str): Architecture of the backbone.
        pretrained (bool): Whether to use a pretrained backbone.
        layer1 (int): Layer index for backbone.
        use_cls (bool): Whether to use classification token.
        norm_descs (bool): Whether to normalize descriptors.
        mixvpr_config (dict): Configuration for the aggregator.
    """

    def __init__(self,
                 backbone_arch: _DINO_V2_MODELS = 'dinov2_vitb14',
                 pretrained=True,
                 layer1=7,
                 use_cls=False,
                 norm_descs=True,
                 mixvpr_config={},

                 ):
        super().__init__()

        self.backbone = DinoV2(model_name=backbone_arch, layer1=layer1, use_cls=use_cls,
                               norm_descs=norm_descs, pretrained=pretrained)
        self.aggregator = MixVPR(**mixvpr_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def get_dino_mix(pretrained=True):
    model = DinoMixModel(
        backbone_arch='dinov2_vitb14',
        pretrained=pretrained,
        layer1=7,
        use_cls=False,
        norm_descs=True,
        mixvpr_config=DEFAULT_AGG_CONFIG,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])

    model.eval()

    return model
