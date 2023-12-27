
import torch

from vpr_models import sfrs, apgem, salad, anyloc, convap, mixvpr, netvlad


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "sfrs":
        model = sfrs.SFRSModel()
    elif method == "apgem":
        model = apgem.GeM()
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
    elif method == "anyloc":
        model = anyloc.AnyLocWrapper()
    elif method == "salad":
        model = salad.SaladWrapper()
    
    return model

