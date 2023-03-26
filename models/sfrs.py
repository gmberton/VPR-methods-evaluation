
import torch
import torchvision.transforms as tfm

from models import utils


class SFRSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

        self.un_normalize = utils.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.normalize = tfm.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])

    def forward(self, images):
        images = self.normalize(self.un_normalize(images))
        descriptors = self.net(images)
        return descriptors

