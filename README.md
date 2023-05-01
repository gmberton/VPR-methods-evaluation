
# VPR-methods-evaluation

This repo is used to easily evaluate pre-trained Visual Place Recognition methods.
A number of trained models are supported (e.g. NetVLAD, SFRS, CosPlace, MixVPR...) and it uses the weights released by the respective authors.

## How to use

The code is designed to be readily used with our [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) repo, so that using a few simple commands you can download a dataset and test any model on it.

```
mkdir VPR-codebase
cd vpr-codebase

git clone https://github.com/gmberton/VPR-datasets-downloader
cd VPR-datasets-downloader
python3 download_svox.py

cd ..

git clone https://github.com/gmberton/VPR-methods-evaluation
cd VPR-methods-evaluation
python3 main.py --database_folder ../VPR-datasets-downloader/datasets/st_lucia/images/test/database --queries_folder ../VPR-datasets-downloader/datasets/st_lucia/images/test/queries
```
