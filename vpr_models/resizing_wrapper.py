
import torch
import torchvision.transforms as transforms


class ResizingWrapper(torch.nn.Module):
    """Wrapper to be used for any model that doesn't accept images of arbitrary
    resolution, like SALAD, CricaVPR, AnyLoc"""
    def __init__(self, model, resize_type="multiples_of_14"):
        """All these models rely on DINO-v2 which needs h, w to be multiple of 14"""
        super().__init__()
        self.model = model
        self.resize_type = resize_type
    
    def forward(self, images):
        if self.resize_type == "multiples_of_14":
            b, c, h, w = images.shape
            # DINO wants height and width as multiple of 14, therefore resize them
            # to the nearest multiple of 14
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            images = transforms.functional.resize(images, [h, w], antialias=True)
        return self.model(images)

