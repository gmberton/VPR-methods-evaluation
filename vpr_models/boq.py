# Model from "BoQ: A Place is Worth a Bag of Learnable Queries" - https://arxiv.org/abs/2405.07364
# Parts of this code are from https://github.com/amaralibey/Bag-of-Queries


from typing import Literal

import torch
import torchvision
from torch import nn

DINOV2 = "Dinov2"
RESNET50 = 'ResNet50'

AVAILABLE_BACKBONES = {
    f"{RESNET50}": [16384],
    f"{DINOV2}": [12288],
}

MODEL_URLS = {
    f"{RESNET50}_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    f"{DINOV2}_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
}


class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
            self,
            backbone_name="resnet50",
            pretrained=True,
            unfreeze_n_blocks=1,
            crop_last_block=True,
    ):
        """Class representing the resnet backbone used in the pipeline.

        Args:
            backbone_name (str): The architecture of the resnet backbone to instantiate.
            pretrained (bool): Whether the model is pretrained or not.
            unfreeze_n_blocks (int): The number of residual blocks to unfreeze (starting from the end).
            crop_last_block (bool): Whether to crop the last residual block.

        Raises:
            ValueError: if the backbone_name corresponds to an unknown architecture.
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!"
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        all_layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]

        if crop_last_block:
            all_layers.remove(resnet.layer4)
        nb_layers = len(all_layers)

        # Check if the number of unfrozen blocks is valid
        assert (
                isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Split the resnet into frozen and unfrozen parts
            self.frozen_layers = nn.Sequential(*all_layers[:nb_layers - unfreeze_n_blocks])
            self.unfrozen_layers = nn.Sequential(*all_layers[nb_layers - unfreeze_n_blocks:])

            # this is helful to make PyTorch count the right number of trainable params
            # because it doesn't detect the torch.no_grad() context manager at init time
            self.frozen_layers.requires_grad_(False)
        else:
            # If the model is not pretrained, we keep all layers trainable
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0
            self.frozen_layers = nn.Identity()
            self.unfrozen_layers = nn.Sequential(*all_layers)

        # Calculate the output channels from the last conv layer of the model
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = all_layers[-1][-1].conv2.out_channels
        else:
            self.out_channels = all_layers[-1][-1].conv3.out_channels

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)

        x = self.unfrozen_layers(x)
        return x


class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]

    def __init__(
            self,
            backbone_name="dinov2_vitb14",
            unfreeze_n_blocks=2,
            reshape_output=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output

        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"

        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)

        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False

        # unfreeze the last few blocks
        for block in self.dino.blocks[-unfreeze_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # remove the output norm layer of dino
        self.dino.norm = nn.Identity()  # remove the normalization layer

        self.out_channels = self.dino.embed_dim

    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks:]:
            x = blk(x)

        x = x[:, 1:]  # remove the [CLS] token

        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape  # or C = self.embed_dim
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
                                                        batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # the following two lines are used during training.
        # for stability purposes
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim // 64) for _ in range(num_layers)])

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns


class VPRModel(torch.nn.Module):
    def __init__(self,
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        x = self.backbone(x)
        x, _ = self.aggregator(x)
        return x


def get_boq(backbone: Literal['ResNet50', 'Dinov2',] = "ResNet50", descriptors_dimension: int = 12288):
    if backbone not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        descriptors_dimension = int(descriptors_dimension)
    except:
        raise ValueError(f"descriptors_dimension should be an integer, not a {type(descriptors_dimension)}")
    if descriptors_dimension not in AVAILABLE_BACKBONES[backbone]:
        raise ValueError(f"descriptors_dimension should be one of {AVAILABLE_BACKBONES[backbone]}")

    if DINOV2 in backbone:
        # load the backbone
        backbone_model = DinoV2()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone_model.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=descriptors_dimension // 384,  # 32 for dinov2
        )

    elif RESNET50 in backbone:
        backbone_model = ResNet(
            backbone_name=backbone.lower(),
            crop_last_block=True,
        )
        aggregator = BoQ(
            in_channels=backbone_model.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=512,
            num_queries=64,
            num_layers=2,
            row_dim=descriptors_dimension // 512,  # 32 for resnet
        )

    model = VPRModel(
        backbone=backbone_model,
        aggregator=aggregator
    )

    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_URLS[f"{backbone}_{descriptors_dimension}"],
            map_location=torch.device('cpu'),

        ), strict=False
    )
    model.eval()
    return model
