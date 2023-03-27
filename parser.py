
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance (in meters) for a prediction to be considered a positive")
    parser.add_argument("--method_name", type=str, default="cosplace",
                        choices=["cosplace", "netvlad", "sfrs", "convap", "mixvpr"],
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
    parser.add_argument("--num_preds_to_save", type=int, default=3,
                        help="_")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="_")
    
    args = parser.parse_args()
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")

    return args

