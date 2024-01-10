
import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import vpr_models
import parser
import commons
import visualizations
from test_dataset import TestDataset

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(output_folder, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}")
logging.info(f"The outputs are being saved in {output_folder}")

model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
model = model.eval().to(args.device)

test_ds = TestDataset(args.database_folder, args.queries_folder,
                      positive_dist_threshold=args.positive_dist_threshold,
                      resize = args.image_resolution)
logging.info(f"Testing on {test_ds}")

with torch.inference_mode():
    logging.debug("Extracting database descriptors for evaluation/testing")
    database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
    all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
    for images, indices in tqdm(database_dataloader, ncols=100):
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors
        
    logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
    queries_subset_ds = Subset(test_ds,
                                list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=1)
    for images, indices in tqdm(queries_dataloader, ncols=100):
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

queries_descriptors = all_descriptors[test_ds.database_num:]
database_descriptors = all_descriptors[:test_ds.database_num]

# Use a kNN to find predictions
faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
faiss_index.add(database_descriptors)
del database_descriptors, all_descriptors

logging.debug("Calculating recalls")
_, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

# For each query, check if the predictions are correct
positives_per_query = test_ds.get_positives()
recalls = np.zeros(len(args.recall_values))
for query_index, preds in enumerate(predictions):
    for i, n in enumerate(args.recall_values):
        if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
            recalls[i:] += 1
            break

# Divide by queries_num and multiply by 100, so the recalls are in percentages
recalls = recalls / test_ds.queries_num * 100
recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
logging.info(recalls_str)

# Save visualizations of predictions
if args.num_preds_to_save != 0:
    logging.info("Saving final predictions")
    # For each query save num_preds_to_save predictions
    visualizations.save_preds(predictions[:, :args.num_preds_to_save], test_ds,
                              output_folder, args.save_only_wrong_preds)

