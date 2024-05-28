# Fast-DDPM

Official PyTorch implementation of: 

[Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation](https://arxiv.org/abs/2405.14802) 

We propose Fast-DDPM, a simple yet effective approach that improves training speed, sampling speed, and generation quality of diffusion models simultaneously. Fast-DDPM trains and samples using only 10 time steps, reducing the training time to 0.2x and the sampling time to 0.01x compared to DDPM.

<p align="center">
  <img src="Overview.png" alt="DDPM vs. Fast-DDPM" width="750">
</p>

The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Hongxu Jiang (hongxu.jiang@medicine.ufl.edu).

## Requirements
* Python==3.10.6
* torch==1.12.1
* torchvision==0.15.2
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* scikit-image
* medpy
* pillow
* scipy
* `pip install -r requirements.txt`

## Publicly available Dataset
- Prostate-MRI-US-Biopsy dataset
- LDCT-and-Projection-data dataset
- BraTS 2018 dataset
- The processed dataset can be accessed here: https://drive.google.com/file/d/1kF0g8fMR5XPQ2FTbutfTQ-hwG_mTqerx/view?usp=drive_link.

## Usage
### 1. Git clone or download the codes.

### 2. Prepare data
* Please download our processed dataset or download from the official websites.
* After downloading, extract the file and put it into folder "data/". The directory structure should be as follows:

```bash
├── configs
│
├── data
│	├── LD_FD_CT_train
│	├── LD_FD_CT_test
│	├── PMUB-train
│	├── PMUB-test
│	├── Brats_train
│	└── Brats_test
│
├── datasets
│
├── functions
│
├── models
│
└── runners

```

* 


### 3. Training/Sampling a Fast-DDPM model
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The total number of time steps is defaulted as 1000 in the paper, so the number of involved time steps for Fast-DDPM should be less than 1000 as an integer.
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --scheduler_type {SAMPLING STRATEGY} --timesteps {STEPS}
```
```
python fast_ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --scheduler_type {SAMPLING STRATEGY} --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `SAMPLING STRATEGY` controls the scheduler sampling strategy proposed in the paper (either uniform or non-uniform).
- `STEPS` controls how many timesteps used in the training and inference process. It should be an integer less than 1000 for Fast-DDPM, which is 10 by default.


### 4. Training/Sampling a DDPM model
* Please make sure that the hyperparameters such as scheduler type and timesteps are consistent between training and sampling.
* The total number of time steps is defaulted as 1000 in the paper, so the number of time steps for DDPM is defaulted as 1000.
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --timesteps {STEPS}
```
```
python ddpm_main.py --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS}
```

where 
- `DATASET_NAME` should be selected among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `STEPS` controls how many timesteps used in the training and inference process. It should be 1000 in the setting of this paper.


## References
* The code is mainly adapted from [DDIM](https://github.com/ermongroup/ddim).


## Citations
If you use our code or dataset, please cite our paper as below:
```bibtex
@article{jiang2024fast,
  title={Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation},
  author={Jiang, Hongxu and Imran, Muhammad and Ma, Linhai and Zhang, Teng and Zhou, Yuyin and Liang, Muxuan and Gong, Kuang and Shao, Wei},
  journal={arXiv preprint arXiv:2405.14802},
  year={2024}
}
```
