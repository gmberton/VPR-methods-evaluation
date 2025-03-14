import os
from glob import glob

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [p for p in images_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any images")
    return images_paths


class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()

        self.database_paths = read_images_paths(database_folder)
        self.queries_paths = read_images_paths(queries_folder)

        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        self.num_database = len(self.database_paths)
        self.num_queries = len(self.queries_paths)

        if use_labels:
            # Read UTM coordinates, which must be contained within the paths
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            try:
                # This is just a sanity check
                image_path = self.database_paths[0]
                utm_east = float(image_path.split("@")[1])
                utm_north = float(image_path.split("@")[2])
            except:
                raise ValueError(
                    "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                    f"but it is {image_path}, which does not contain the UTM coordinates."
                )

            self.database_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
            ).astype(float)
            self.queries_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
            ).astype(float)

            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.positives_per_query = knn.radius_neighbors(
                self.queries_utms, radius=positive_dist_threshold, return_distance=False
            )

        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

    def get_positives(self):
        return self.positives_per_query
