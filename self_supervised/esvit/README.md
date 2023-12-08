# EsViT for Side-Scan Sonar Data

This is a modified PyTorch implementation of [EsViT](https://github.com/microsoft/esvit), a non-contrastive region-level matching pretext task for self-supervised pre-training of Vision Transformers (ViTs). The modifications are primarily aimed to facilitate self-supervised pretraining on side-scan sonar images for the architecture proposed in *"A convolutional vision transformer for semantic segmentation of side-scan sonar data" published in Ocean Engineering, Volume 86, part 2, 15 October 2023, DOI: [10.1016/j.oceaneng.2023.115647](https://www.sciencedirect.com/science/article/pii/S0029801823020310).*

## Getting Started

### Prerequisites

The file [requirements.txt](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/self_supervised/esvit/requirements.txt) contains the necessary Python packages for this project. To install, run:
```
pip install -r requirements.txt
```

All models were trained on an NVIDIA A100 Tensor Core GPU operating on Ubuntu 22.04.2 with Python 3.9.17 and PyTorch 2.0.0+cu120, and evaluated on an NVIDIA Jetson AGX Orin Developer Kit running Jetpack 5.1.1 with Python 3.8.10 and PyTorch 2.0.0+nv23.5.

The **dataset** used for training is available for download via [this link](https://zenodo.org/records/10209445).

### Training

The file [main.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/self_supervised/esvit/main.py) contains the main training loop. It takes the following arguments:
```
--wandb_entity		WandB entity.
--wandb_project		WandB project name.
--wandb_api_key		WandB api key.
--data_dir			Path to training data.
--out_dir			Path to save logs, checkpoints and models.
--arch				Name of architecture to train
[--config_file]		Path to configuration file.
[--load_checkpoint]	Path to checkpoint to resume training from.
[--load_weights]	Path to pretrained weights.
[--seed]			Random seed.
[--num_workers]		Number of data loading workers per GPU.
[--batch_size]		Number of distinct images loaded per GPU.
[--use_fp16]		Whether or not to use half precision for training
```

The arguments in brackets are optional. Further details on WandB specific arguments can be found in [Weights & Biases documentation](https://docs.wandb.ai/guides/track/environment-variables). Currently there are six architectures to choose from: *cswin_mini*, *lsda_mini*, *sima_mini*, *sima_tiny*, *sima_micro* and *sima_nano*. The default configurations of these architectures can be found in the file [configs/models.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/self_supervised/esvit/configs/models.py). Modifications to the configurations of these architectures and to the default training hyperparameters can be optionally done via a yaml file; see [config.yaml](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/self_supervised/esvit/config.yaml) for an example. The file [configs/base.py](https://github.com/DeeperSense/deepersense-seafloorscan/blob/main/self_supervised/esvit/configs/base.py), on the other hand, contains all the base configuration parameters.

To train a *sima_tiny* model on a single node with 2 GPUs with user-specified configurations contained in config.yaml, run:
```
torchrun --nproc_per_node=1 --master_port=1234 main.py --wandb_entity <wandb-user-name> --wandb_project <wandb-project-name> --wandb_api_key <wandb-api-key> --data_dir /path/to/sss/dataset --out_dir /path/to/out/dir --config_file /path/to/config.yaml --batch_size 256 --arch sima_mini
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
Our implementation is built upon [[EsViT](https://github.com/microsoft/esvit)] [[Timm](https://github.com/huggingface/pytorch-image-models)] [[CrossFormer](https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py)] [[CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py)]

This work was supported by the DeeperSense project, funded by the European Union’s Horizon 2020 Research and Innovation programme under grant agreement no. [101016958](https://cordis.europa.eu/project/id/101016958).

### Related Projects
[[s3Tseg](https://github.com/CIRS-Girona/s3Tseg)] [[w-s3Tseg](https://github.com/CIRS-Girona/w-s3Tseg)]