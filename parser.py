import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--positive_dist_threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosplace",
        choices=[
            "netvlad",
            "apgem",
            "sfrs",
            "cosplace",
            "convap",
            "mixvpr",
            "eigenplaces",
            "eigenplaces-indoor",
            "anyloc-urban",
            "anyloc-indoor",
            "anyloc-aerial",
            "anyloc-structured",
            "anyloc-unstructured",
            "anyloc-global",
            "salad",
            "salad-indoor",
            "cricavpr",
            "clique-mining",
        ],
        help="_",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
        help="_",
    )
    parser.add_argument("--descriptors_dimension", type=int, default=None, help="_")
    parser.add_argument("--database_folder", type=str, required=True, help="path/to/database")
    parser.add_argument("--queries_folder", type=str, required=True, help="path/to/queries")
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="set to 1 if database images may have different resolution"
    )
    parser.add_argument(
        "--log_dir", type=str, default="default", help="experiment name, output logs will be saved under logs/log_dir"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument(
        "--recall_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    parser.add_argument(
        "--num_preds_to_save", type=int, default=0, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
    )
    args = parser.parse_args()

    args.use_labels = not args.no_labels

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
            raise ValueError(
                "When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]"
            )
        if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
        ]:
            raise ValueError(
                f"When using CosPlace with {args.backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]"
            )

    elif args.method == "convap":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if args.descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
            raise ValueError(
                "When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]"
            )

    elif args.method == "mixvpr":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if args.descriptors_dimension not in [None, 128, 512, 4096]:
            raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")

    elif args.method == "eigenplaces":
        if args.backbone is None:
            args.backbone = "ResNet50"
        if args.descriptors_dimension is None:
            args.descriptors_dimension = 512
        if args.backbone == "VGG16" and args.descriptors_dimension not in [512]:
            raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
        if args.backbone == "ResNet18" and args.descriptors_dimension not in [256, 512]:
            raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
        if args.backbone in ["ResNet50", "ResNet101", "ResNet152"] and args.descriptors_dimension not in [
            128,
            256,
            512,
            2048,
        ]:
            raise ValueError(
                f"When using EigenPlaces with {args.backbone} the descriptors_dimension must be in [128, 256, 512, 2048]"
            )

    elif args.method == "eigenplaces-indoor":
        args.backbone = "ResNet50"
        args.descriptors_dimension = 2048

    elif args.method == "apgem":
        args.backbone = "Resnet101"
        args.descriptors_dimension = 2048

    elif args.method.startswith("anyloc"):
        args.backbone = "DINOv2"
        args.descriptors_dimension = 49152

    elif args.method == "salad":
        args.backbone = "DINOv2"
        args.descriptors_dimension = 8448

    elif args.method == "clique-mining":
        args.backbone = "DINOv2"
        args.descriptors_dimension = 8448

    elif args.method == "salad-indoor":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 8448

    elif args.method == "cricavpr":
        args.backbone = "Dinov2"
        args.descriptors_dimension = 10752

    if args.image_size and len(args.image_size) > 2:
        raise ValueError(
            f"The --image_size parameter can only take up to 2 values, but has received {len(args.image_size)}."
        )

    return args
