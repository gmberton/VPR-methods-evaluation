# Part of this code are taken from https://github.com/cvg/Hierarchical-Localization
# and https://github.com/naver/deep-image-retrieval

import os
import sys
from pathlib import Path
from zipfile import ZipFile

import gdown
import sklearn
import torch

sys.path.append(str(Path("third_party/deep-image-retrieval")))
os.environ["DB_ROOT"] = ""  # required by dirtorch

from dirtorch.extract_features import load_model  # noqa: E402
from dirtorch.utils import common  # noqa: E402

# The DIR model checkpoints (pickle files) include sklearn.decomposition.pca,
# which has been deprecated in sklearn v0.24
# and must be explicitly imported with `from sklearn.decomposition import PCA`.
# This is a hacky workaround to maintain forward compatibility.
sys.modules["sklearn.decomposition.pca"] = sklearn.decomposition._pca


class GeM(torch.nn.Module):
    """This is the trained model from NAVER labs, often referred to as AP-GeM,
    from the paper "Learning with Average Precision: Training Image Retrieval
    with a Listwise Loss"
    """

    def __init__(self):
        super().__init__()
        self.conf = {
            "model_name": "Resnet-101-AP-GeM",
            "whiten_name": "Landmarks_clean",
            "whiten_params": {
                "whitenp": 0.25,
                "whitenv": None,
                "whitenm": 1.0,
            },
            "pooling": "gem",
            "gemp": 3,
        }
        dir_models = {
            "Resnet-101-AP-GeM": "https://docs.google.com/uc?export=download&id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy",
        }
        checkpoint = Path(torch.hub.get_dir(), "dirtorch", self.conf["model_name"] + ".pt")
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True, parents=True)
            link = dir_models[self.conf["model_name"]]
            gdown.download(str(link), str(checkpoint) + ".zip", quiet=False)
            zf = ZipFile(str(checkpoint) + ".zip", "r")
            zf.extractall(checkpoint.parent)
            zf.close()
            os.remove(str(checkpoint) + ".zip")

        self.net = load_model(checkpoint, False)  # first load on CPU
        if self.conf["whiten_name"]:
            assert self.conf["whiten_name"] in self.net.pca

    def forward(self, image):
        descs = self.net(image)
        if len(descs.shape) == 1:
            # The model squeezes the descriptors if batch size is 1
            descs = descs.unsqueeze(0)

        if self.conf["whiten_name"]:
            whitened_descs = []
            for desc in descs:
                # For how the PCA is implemented, it takes only one descriptor
                # at a time
                desc = desc.unsqueeze(0)  # batch dimension
                pca = self.net.pca[self.conf["whiten_name"]]
                desc = common.whiten_features(desc.cpu().numpy(), pca, **self.conf["whiten_params"])
                desc = torch.from_numpy(desc)
                assert len(desc) == 1
                desc = desc[0]
                whitened_descs.append(desc)
            descs = torch.stack(whitened_descs)

        return descs
