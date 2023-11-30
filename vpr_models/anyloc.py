# Code from AnyLoc https://arxiv.org/abs/2308.00688

import torch
import torchvision.transforms as transforms


class AnyLocWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device="cuda")
    def forward(self, images):
        b, c, h, w = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)

