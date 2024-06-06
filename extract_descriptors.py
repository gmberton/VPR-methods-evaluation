
import argparse
import faiss
import logging
import numpy as np
import sys
import torch

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import commons
import parser
import save
from test_dataset import ExtractDescriptorsDataset
import vpr_models


args = parser.parse_arguments("extract")
if args.remove_timestamp:
    log_dir = Path("logs") / args.log_dir
else:
    start_time = datetime.now()
    log_dir = Path("logs") / args.log_dir / start_time.strftime('%Y-%m-%d_%H-%M-%S')
commons.setup_logging(log_dir, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}")
logging.info(f"The logs are being saved in {log_dir}")

model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
model = model.eval().to(args.device)

test_ds = ExtractDescriptorsDataset(args.database_folder, image_size=args.image_size)

logging.info(f"Extracting the descriptors of {test_ds}")

with torch.inference_mode():
    database_dataloader = DataLoader(dataset=test_ds, num_workers=args.num_workers,
                                    batch_size=args.batch_size)
    all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
    for images, indices in tqdm(database_dataloader):
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

if not args.output_dir:
    args.output_dir = log_dir
else:
    args.output_dir = Path(args.output_dir)

logging.info(f"Saving the descriptors of {test_ds} in {args.output_dir}")
save.save_descriptors(args, test_ds=test_ds, descriptors=all_descriptors)
    