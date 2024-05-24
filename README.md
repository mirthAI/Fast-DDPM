# Fast-DDPM

Official PyTorch implementation of: 

Fast-DDPM: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation (Currently under review)

Fast-DDPM is a simple yet effective approach which is capable of improving training speed, sampling speed, and generation quality simultaneously. Fast-DDPM trains and samples using only 10 time steps, reducing the training time to 0.2$\times$ and the sampling time to 0.01$\times$ compared to DDPM.

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
- The processed dataset can be accessed here: https://drive.google.com/file/d/1u9OWKxWCNGLadNEaiho3uUFGuGL2KXl9/view?usp=drive_link.

## Usage
### 1. Git clone or download the codes.

### 2. Prepare data
* Please download our processed dataset or download from the official websites.
* After downloading, extract the file and put it into folder "data/". The directory structure should be as follows:

```bash
.
├── configs
│
├── data
│	├── LD_FD_CT_train
│	├── LD_FD_CT_test
│	├── PMUB-train
│	├── PMUB-test
│	├── train2D
│	└── val2D
│
├── datasets
│
├── functions
│
├── models
│
└── runners

```


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
- `DATASET_NAME` should be among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `SAMPLING STRATEGY` controls the scheduler sampling strategy proposed in the paper(either uniform or non-uniform).
- `STEPS` controls how many timesteps used in the training and inference process. It should be an integer less than 1000 for Fast-DDPM.


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
- `DATASET_NAME` should be among `LDFDCT` for image denoising task, `BRATS` for image-to-image translation task and `PMUB` for multi image super-resolution task.
- `STEPS` controls how many timesteps used in the training and inference process. It should be 1000 in the setting of this paper.


## References
* The code is mainly adapted from [DDIM](https://github.com/ermongroup/ddim).


## Citations
If you use our code or dataset, please cite our paper as below:
