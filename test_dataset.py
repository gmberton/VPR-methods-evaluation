
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="images/test/database",
                 queries_folder="images/test/queries", positive_dist_threshold=25):
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
        self.dataset_folder = dataset_folder
        self.dataset_name = os.path.basename(dataset_folder)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        # Read paths for database and queries from all_images_paths.txt
        images_paths_file = f"{dataset_folder}/all_images_paths.txt"
        if not os.path.exists(images_paths_file):
            raise FileNotFoundError(f"File {images_paths_file} does not exist. This file should contain "
                                    "the paths of all images, to avoid having to glob() them every "
                                    "time, which is expensive for the largest (i.e. realistic) datasets")
        with open(images_paths_file, "r") as file:
            all_images_paths = file.read().splitlines()
        self.database_paths = [path for path in all_images_paths if path.startswith(database_folder)]
        self.queries_paths = [path for path in all_images_paths if path.startswith(queries_folder)]
        
        if not os.path.exists(self.dataset_folder + "/" + self.database_paths[0]):
            raise FileNotFoundError(f"Database image with path {self.database_paths[0]} "
                                    f"does not exist within {self.dataset_folder}. It is likely "
                                    f"that the content of {images_paths_file} is wrong.")
        
        if not os.path.exists(self.dataset_folder + "/" + self.queries_paths[0]):
            raise FileNotFoundError(f"Query images with path {self.queries_paths[0]} "
                                    f"does not exist within {self.dataset_folder}. It is likely "
                                    f"that the content of {images_paths_file} is wrong.")
        
        # Read UTM coordinates, which should be contained within the paths
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        try:
            # This is just a sanity check
            image_path = self.database_paths[0]
            utm_east = float(image_path.split("@")[1])
            utm_north = float(image_path.split("@")[2])
        except:
            raise ValueError("The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                             f"but it is {image_path}, which does not contain the UTM coordinates.")
        
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)
        
        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, index):
        image_path = self.dataset_folder + "/" + self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #queries: {self.queries_num}; #database: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query

