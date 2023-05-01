# Model from "GSV-Cities: Toward Appropriate Supervised Visual Place Recognition" - https://arxiv.org/abs/2210.10239
# Parts of this code are from https://github.com/amaralibey/gsv-cities

import os
import gdown
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


MODELS_INFO = {
    512: "resnet50_ConvAP_128_2x2.ckpt",
    2048: "resnet50_ConvAP_512_2x2.ckpt",
    4096: "resnet50_ConvAP_1024_2x2.ckpt",
    8192: "resnet50_ConvAP_2048_2x2.ckpt"
}
URL = "https://drive.google.com/drive/folders/1VYPw9uGD11NgiGFgfWueLt3noJYOIuhL"


class ConvAP(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """

    def __init__(self, in_channels, out_channels=512, s1=2, s2=2):
        super(ConvAP, self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x):
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x


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
            ConvAP(in_dim=hw, mlp_ratio=mlp_ratio)
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
        x = self.model.layer4(x)
        return x


class ConvAPModel(torch.nn.Module):
    def __init__(self, agg_config={}):
        super().__init__()
        self.backbone = ResNet()
        self.aggregator = ConvAP(**agg_config)

    def forward(self, x):
        x = transforms.Resize([320, 320])(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def get_convap(descriptors_dimension):
    filename = MODELS_INFO[descriptors_dimension]
    file_path = f"trained_models/convap/{filename}"
    if not os.path.exists(file_path):
        os.makedirs("trained_models/convap", exist_ok=True)
        gdown.download_folder(URL, output="trained_models/convap", use_cookies=False)

    model_config = {'in_channels': 2048, 'out_channels': descriptors_dimension // 4, 's1': 2, 's2': 2}
    model = ConvAPModel(agg_config=model_config)
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)
    model = model.eval()

    return model

