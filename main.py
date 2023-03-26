
import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import models
import parser
import commons
from test_dataset import TestDataset

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(output_folder, stdout="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

model, descriptors_dimension = models.get_model(args.method_name)
model = model.eval().to(args.device)

test_ds = TestDataset(args.dataset_folder, positive_dist_threshold=args.positive_dist_threshold)

# with torch.inference_mode():
#     logging.debug("Extracting database descriptors for evaluation/testing")
#     database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
#     database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
#                                      batch_size=args.batch_size, pin_memory=(args.device == "cuda"))
#     all_descriptors = np.empty((len(test_ds), descriptors_dimension), dtype="float32")
#     for images, indices in tqdm(database_dataloader, ncols=100):
#         descriptors = model(images.to(args.device))
#         descriptors = descriptors.cpu().numpy()
#         all_descriptors[indices.numpy(), :] = descriptors
#
#     logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
#     queries_subset_ds = Subset(test_ds,
#                                list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)))
#     queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
#                                     batch_size=1, pin_memory=(args.device == "cuda"))
#     for images, indices in tqdm(queries_dataloader, ncols=100):
#         descriptors = model(images.to(args.device))
#         descriptors = descriptors.cpu().numpy()
#         all_descriptors[indices.numpy(), :] = descriptors
#
# queries_descriptors = all_descriptors[test_ds.database_num:]
# database_descriptors = all_descriptors[:test_ds.database_num]
#
# # Use a kNN to find predictions
# faiss_index = faiss.IndexFlatL2(descriptors_dimension)
# faiss_index.add(database_descriptors)
# del database_descriptors, all_descriptors
#
# logging.debug("Calculating recalls")
# _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))
#
# # For each query, check if the predictions are correct
# positives_per_query = test_ds.get_positives()
# recalls = np.zeros(len(args.recall_values))
# for query_index, preds in enumerate(predictions):
#     for i, n in enumerate(args.recall_values):
#         if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
#             recalls[i:] += 1
#             break
#
# # Divide by queries_num and multiply by 100, so the recalls are in percentages
# recalls = recalls / test_ds.queries_num * 100
# recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
# logging.info(recalls_str)



FC_OUTPUT_DIM = descriptors_dimension
RECALL_VALUES = args.recall_values
eval_ds = test_ds
model = model.eval()
with torch.no_grad():
    database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=4,
                                     batch_size=16, pin_memory=True)
    all_descriptors = np.ones((len(eval_ds), FC_OUTPUT_DIM), dtype="float32")
    for images, indices in tqdm(database_dataloader, ncols=100):
        descriptors = model(images.cuda())
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

    queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=4,
                                    batch_size=16, pin_memory=True)
    for images, indices in tqdm(queries_dataloader, ncols=100):
        descriptors = model(images.cuda())
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

queries_descriptors = all_descriptors[eval_ds.database_num:]
database_descriptors = all_descriptors[:eval_ds.database_num]

# Use a kNN to find predictions
faiss_index = faiss.IndexFlatL2(FC_OUTPUT_DIM)
faiss_index.add(database_descriptors)

_, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
torch.save(torch.from_numpy(predictions), "a_predictions.torch")

#### For each query, check if the predictions are correct
positives_per_query = eval_ds.get_positives()
recalls = np.zeros(len(RECALL_VALUES))
for query_index, preds in enumerate(predictions):
    for i, n in enumerate(RECALL_VALUES):
        if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
            recalls[i:] += 1
            break

# Divide by queries_num and multiply by 100, so the recalls are in percentages
recalls = recalls / eval_ds.queries_num * 100
recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
logging.info(recalls_str)

