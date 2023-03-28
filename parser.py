
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance (in meters) for a prediction to be considered a positive")
    parser.add_argument("--method", type=str, default="cosplace",
                        choices=["netvlad", "sfrs", "cosplace", "convap", "mixvpr"],
                        help="_")
    parser.add_argument("--backbone", type=str, default=None,
                        help="_")
    parser.add_argument("--descriptors_dimension", type=int, default=None,
                        help="_")
    parser.add_argument("--dataset_folder", type=str, default="/home/gabriele/vg_datasets/retrieval/st_lucia/images/test", #required=True,
                        help="_")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="_")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="_")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="_")
    parser.add_argument("--device", type=str, default="cuda",
                        help="_")
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="_")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="_")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="_")
    
    args = parser.parse_args()
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
    
    if args.method == "netvlad":
        if args.backbone not in [None, "VGG16"]:
            raise ValueError("When using NetVLAD the backbone must be None or VGG16")
        if args.descriptors_dimension not in [None, 4096, 32768]:
            raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 4096
        
    elif args.method == "sfrs":
        if args.backbone not in [None, "VGG16"]:
            raise ValueError("When using SFRS the backbone must be None or VGG16")
        if args.descriptors_dimension not in [None, 4096]:
            raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 4096
    
    elif args.method == "cosplace":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone == "VGG16" and args.descriptors_dimension not in [64, 128, 256, 512]:
            raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
        if args.backbone == "ResNet18" and args.descriptors_dimension not in [32, 64, 128, 256, 512]:
            raise ValueError("When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]")
        if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
            raise ValueError(f"When using CosPlace with {args.backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]")
    
    elif args.method == "convap":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if args.descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
            raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]")
    
    elif args.method == "mixvpr":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if args.descriptors_dimension not in [None, 128, 512, 4096]:
            raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")
    
    return args

