
import torch

from models import sfrs
from models import convap
from models import mixvpr
from models import netvlad


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "sfrs":
        model = sfrs.SFRSModel()
    elif method == "netvlad":
        model = netvlad.NetVLAD(descriptors_dimension=descriptors_dimension)
    elif method == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    elif method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
    elif method == "convap":
        model = convap.get_convap(descriptors_dimension=descriptors_dimension)
    elif method == "eigenplaces":
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    
    return model

