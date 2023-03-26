
import torch

from models import sfrs
from models import convap
from models import mixvpr
from models import netvlad


def get_model(method_name):
    if method_name == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        descriptors_dimension = 2048
    if method_name == "sfrs":
        model = sfrs.SFRSModel()
        descriptors_dimension = 4096
    if method_name == "netvlad":
        model = netvlad.NetVLAD()
        descriptors_dimension = 4096
    if method_name == "mixvpr":
        model = mixvpr.get_mixvpr()
        descriptors_dimension = 512
    if method_name == "convap":
        model = convap.get_convap()
        descriptors_dimension = 2048

    return model, descriptors_dimension

