
# VPR-methods-evaluation

This repo is used to easily evaluate pre-trained Visual Place Recognition methods, and is released as part of the [ICCV 2023 EigenPlaces paper](https://openaccess.thecvf.com/content/ICCV2023/html/Berton_EigenPlaces_Training_Viewpoint_Robust_Models_for_Visual_Place_Recognition_ICCV_2023_paper.html).
All train state-of-the-art models since 2016 are supported (namely NetVLAD, SFRS, Conv-AP, CosPlace, MixVPR, EigenPlaces, AnyLoc and SALAD) and it uses the weights released by the respective authors.

To see the results of these methods on a large number (14!) of VPR datasets check out the EigenPlaces paper.

[[ICCV 2023 Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Berton_EigenPlaces_Training_Viewpoint_Robust_Models_for_Visual_Place_Recognition_ICCV_2023_paper.html)] [[ArXiv](https://arxiv.org/abs/2308.10832)] [[BibTex](https://github.com/gmberton/EigenPlaces#cite)]

## How to use

The code is designed to be readily used with our [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) repo, so that using a few simple commands you can download a dataset and test any model on it. The VPR-datasets-downloader code allows you to download multiple VPR datasets that are automatically formatted in the same format as used by this repo.

```
mkdir VPR-codebase
cd vpr-codebase

git clone https://github.com/gmberton/VPR-datasets-downloader
cd VPR-datasets-downloader
python3 download_st_lucia.py

cd ..

git clone https://github.com/gmberton/VPR-methods-evaluation
cd VPR-methods-evaluation
python3 main.py --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
    --database_folder=../VPR-datasets-downloader/datasets/st_lucia/images/test/database \
    --queries_folder=../VPR-datasets-downloader/datasets/st_lucia/images/test/queries
```
This should produce this as output `R@1: 98.8, R@5: 99.7, R@10: 99.9, R@20: 100.0`, which will be saved in a log file under `./logs/`

You can easily change the paths for different datasets, and you can use any of the following methods: NetVLAD, SFRS, CosPlace, Conv-AP, MixVPR, EigenPlaces, AnyLoc, SALAD.
Note that each method has weights only for certain architectures. For example NetVLAD only has weights for VGG16 with descriptors_dimension 32768 and 4069 (with PCA).


### Visualize predictions

Predictions can be easily visualized through the `num_preds_to_save` parameter. For example running this

```
python3 main.py --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
    --num_preds_to_save=3 --exp_name=cosplace_on_stlucia \
    --database_folder=../VPR-datasets-downloader/datasets/st_lucia/images/test/database \
    --queries_folder=../VPR-datasets-downloader/datasets/st_lucia/images/test/queries
```
will generate under the path `./logs/cosplace_on_stlucia/*/preds` images such as

<p float="left">
  <img src="https://raw.githubusercontent.com/gmberton/VPR-methods-evaluation/master/images/pred.jpg"  height="200"/>
</p>

Given that saving predictions for each query might take long, you can also pass the parameter `--save_only_wrong_preds` which will save only predictions for wrongly predicted queries (i.e. where the first prediction is wrong).


## Acknowledgements

If you use this repository please cite our paper
```
@inproceedings{Berton_2023_EigenPlaces,
  title={EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition},
  author={Berton, Gabriele and Trivigno, Gabriele and Caputo, Barbara and Masone, Carlo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023},
  month={October},
  pages={11080-11090}
}
```

Kudos to the authors of [NetVLAD](https://github.com/Relja/netvlad), [SFRS](https://github.com/yxgeee/OpenIBL), [CosPlace](https://github.com/gmberton/CosPlace), [Conv-AP](https://github.com/amaralibey/gsv-cities), [MixVPR](https://github.com/amaralibey/mixVPR), [EigenPlaces](https://github.com/gmberton/EigenPlaces), [AnyLoc](https://github.com/AnyLoc/AnyLoc), [SALAD](https://github.com/serizba/salad) for open sourcing their models' weights. The code for each model has been taken from their respective repositories, excpet for the code for NetVLAD which has been taken from [hloc](https://github.com/cvg/Hierarchical-Localization).
Make sure to cite them if you use each model's code.
