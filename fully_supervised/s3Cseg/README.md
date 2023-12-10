# CNN-based Semantic Segmentation of the Seafloor in Side-Scan Sonar Data

This repository contains PyTorch implementations of the following CNN-based encoder-decoder architectures for semantic segmentation of the seafloor in side-scan sonar images:

- [RT-Seg](https://www.mdpi.com/1424-8220/19/9/1985): A Real-Time Semantic Segmentation Network for Side-Scan Sonar Images by Wang Q, Wu M, Yu F, Feng C, Li K, Zhu Y, Rigall E and He B.
- [ECNet](https://www.mdpi.com/1424-8220/19/9/2009): Efficient Convolutional Networks for Side Scan Sonar Image Segmentation by Wu M, Wang Q, Rigall E, Li K, Zhu W, He B and Yan T.
- [DcNet](https://link.springer.com/article/10.1007/s11802-021-4668-5): Dilated Convolutional Neural Networks for Side-Scan Sonar Image Semantic Segmentation by Zhao X, Qin R, Zhang Q, Yu F, Wang Q, and He B.

These architectures were used as baselines for comparison with the ViT-based architecture proposed in *"A convolutional vision transformer for semantic segmentation of side-scan sonar data" published in Ocean Engineering, Volume 86, part 2, 15 October 2023, DOI: [10.1016/j.oceaneng.2023.115647](https://www.sciencedirect.com/science/article/pii/S0029801823020310).*

## Getting Started

### Prerequisites

The file [requirements.txt](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/requirements.txt) contains the necessary Python packages for this project. To install, run:
```
pip install -r requirements.txt
```

All models were trained on an NVIDIA A100 Tensor Core GPU operating on Ubuntu 22.04.2 with Python 3.9.17 and PyTorch 2.0.0+cu120, and evaluated on an NVIDIA Jetson AGX Orin Developer Kit running Jetpack 5.1.1 with Python 3.8.10 and PyTorch 2.0.0+nv23.5.

<!-- The **dataset** used for training is available for download via [this link](https://zenodo.org/records/xxxx). -->

### Training

The file [main.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/main.py) contains the main training loop. It takes the following arguments:
```
--wandb_entity		WandB entity.
--wandb_project		WandB project name.
--wandb_api_key		WandB api key.
--data_dir			Path to training data.
--out_dir			Path to save logs, checkpoints and models.
--arch				Type of architecture to train.
[--config_file]		Path to configuration file.
[--load_checkpoint]	Path to checkpoint to resume training from.
[--load_weights]	Path to pretrained weights.
[--seed]			Random seed.
[--num_workers]		Number of data loading workers per GPU.
[--batch_size]		Number of distinct images loaded per GPU.
```

The arguments in brackets are optional. Further details on WandB specific arguments can be found in [Weights & Biases documentation](https://docs.wandb.ai/guides/track/environment-variables). Currently there are three architectures to choose from: *dcnet*, *ecnet* and *rtseg*. The default configurations of these architectures can be found in the file [configs/models.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/configs/models.py). Modifications to the configurations of these architectures and to the default training hyperparameters can be optionally done via a yaml file; see [config.yaml](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/config.yaml) for an example. The file [configs/base.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/configs/base.py), on the other hand, contains all the base configuration parameters.

To train a *rtseg* model with user-specified configurations contained in config.yaml, run:
```
python3 main.py --wandb_entity <wandb-user-name> --wandb_project <wandb-project-name> --wandb_api_key <wandb-api-key> --data_dir /path/to/sss/dataset --out_dir /path/to/out/dir --config_file /path/to/config.yaml --batch_size 64 --arch rtseg
```

### Evaluation

The file [eval.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/fully_supervised/s3Cseg/eval.py) contains qualitative, quantitative and runtime performance evaluation metrics for semantic segmentation. It takes the following arguments:
```
--data_dir			Path to dataset.
--model_path		Path to trained model.
--mode			    Evaluation mode.
[--config_file]		Path to configuration file.
[--cmap_file]		Path to color map file.
[--out_dir]			Path to save evaluation report.
[--arch]			Type of architecture.
[--device]			Device to compute runtime statistics for.
[--batch_size]		Number of distinct images per batch.
```

The arguments in brackets are optional. Mode can be set to either *quantitative*, *qualitative* or *runtime*. Device can be set to either *cpu* or *cuda*.

To evaluate runtime performance of a *dcnet* model with user-specified configurations contained in config.yaml, run:
```
python3 eval.py --arch dcnet --model_path /path/to/trained/model.pth --data_dir /path/to/sss/test/dataset --out_dir /path/to/out/dir --config_file /path/to/config.yaml --cmap_file /path/to/cmap.csv --batch_size 1 --device cpu --mode runtime
```

## Pretrained Models

Coming soon . . .

## Citation

If you find this repository useful, please consider giving us a star :star:

```
@article{rajani2023s3Tseg,
    title = {A convolutional vision transformer for semantic segmentation of side-scan sonar data},
    author = {Hayat Rajani and Nuno Gracias and Rafael Garcia},
    journal = {Ocean Engineering},
    volume = {286},
    pages = {115647},
    year = {2023},
    issn = {0029-8018},
    doi = {https://doi.org/10.1016/j.oceaneng.2023.115647},
    url = {https://www.sciencedirect.com/science/article/pii/S0029801823020310},
}
```

### Acknowledgement
Our implementation is built upon [[EsViT](https://github.com/microsoft/esvit)] [[Timm](https://github.com/huggingface/pytorch-image-models)]

This work was supported by the DeeperSense project, funded by the European Union’s Horizon 2020 Research and Innovation programme under grant agreement no. [101016958](https://cordis.europa.eu/project/id/101016958).

### Related Projects
[[s3Tseg](https://github.com/CIRS-Girona/s3Tseg)] [[w-s3Tseg](https://github.com/CIRS-Girona/w-s3Tseg)]