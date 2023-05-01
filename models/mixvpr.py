# Model from "MixVPR: Feature Mixing for Visual Place Recognition" - https://arxiv.org/abs/2303.02190
# Parts of this code are from https://github.com/amaralibey/MixVPR

import os
import gdown
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


MODELS_INFO = {
    128: ("https://drive.google.com/file/d/1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz/view",
          "resnet50_MixVPR_128_channels(64)_rows(2)",
          64, 2),
    512: ("https://drive.google.com/file/d/1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj/view",
          "resnet50_MixVPR_512_channels(256)_rows(2)",
          256, 2),
    4096: ("https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view",
          "resnet50_MixVPR_4096_channels(1024)_rows(4)",
          1024, 4),
}


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()
        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.layer4 = nn.Identity()
        out_channels = 2048
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x1):
        x = self.model.conv1(x1)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


class MixVPRModel(torch.nn.Module):
    def __init__(self, agg_config={}):
        super().__init__()
        self.backbone = ResNet()
        self.aggregator = MixVPR(**agg_config)

    def forward(self, x):
        x = transforms.Resize([320, 320])(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def get_mixvpr(descriptors_dimension):
    url, filename, out_channels, out_rows = MODELS_INFO[descriptors_dimension]
    model_config = {'in_channels': 1024, 'in_h': 20, 'in_w': 20, 'out_channels': out_channels,
                    'mix_depth': 4, 'mlp_ratio': 1, 'out_rows': out_rows}
    model = MixVPRModel(agg_config=model_config)
    file_path = f"trained_models/mixvpr/{filename}"
    if not os.path.exists(file_path):
        os.makedirs("trained_models/mixvpr", exist_ok=True)
        gdown.download(url=url, output=file_path, fuzzy=True)
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    model = model.eval()
    
    return model

