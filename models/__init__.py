
import torch

from models import sfrs
from models import convap
from models import mixvpr
from models import netvlad


def get_model(method_name, backbone=None, descriptors_dimension=None):
    if method_name == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    if method_name == "sfrs":
        model = sfrs.SFRSModel()
    if method_name == "netvlad":
        model = netvlad.NetVLAD(descriptors_dimension=descriptors_dimension)
    if method_name == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
    if method_name == "convap":
        model = convap.get_convap(descriptors_dimension=descriptors_dimension)

    return model, descriptors_dimension

