# Code from Optimal Transport Aggregation for Visual Place Recognition https://arxiv.org/abs/2311.15937

import torch
import torchvision.transforms as transforms


class SaladWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
    def forward(self, images):
        b, c, h, w = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)

class SaladIndoorWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("Enrico-Chiavassa/Indoor-VPR", "get_trained_model",
                                    method="salad", backbone="Dinov2", fc_output_dim=8448)
    def forward(self, images):
        b, c, h, w = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)