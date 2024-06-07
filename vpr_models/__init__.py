
import torch

from vpr_models import sfrs, apgem, convap, mixvpr, netvlad

from vpr_models.resizing_wrapper import ResizingWrapper

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
    elif method == "eigenplaces-indoor":
        model = torch.hub.load("Enrico-Chiavassa/Indoor-VPR", "get_trained_model",
                               backbone=backbone, fc_output_dim=descriptors_dimension)
    elif method == "anyloc":
        # model = anyloc.AnyLocWrapper()
        model = ResizingWrapper(torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device="cuda"))
    elif method == "salad":
        model = ResizingWrapper(torch.hub.load("serizba/salad", "dinov2_salad"))
    elif method == "salad-indoor":
        model = ResizingWrapper(torch.hub.load("Enrico-Chiavassa/Indoor-VPR", "get_trained_model",
                                method="salad", backbone="Dinov2", fc_output_dim=8448))
    
    return model

