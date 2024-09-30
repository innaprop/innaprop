# Image Classification with INNAprop

This repository contains the implementation of Sections 3.1 and 3.2 from our recent paper, enabling you to replicate our image classification results.

## Repository Overview

There are several directories in this repo:
* [optimizers/](optimizers) source code for the optimizers.
* [results/](results) image classification results.


## Getting Started

### 1. Reproduce heatmaps
To reproduce heatmaps, run the following commands:

```
python compute_best_lr_inna.py --network vgg11 --dataset cifar10 --optimizer innaprop --epochs 20 --wd 0.01
python compute_best_lr_inna.py --network vgg11 --dataset cifar10 --optimizer innaprop --epochs 200 --wd 0.01
```

 

### 2. Reproduce Figure 2
Run the command below with the specified optimizer and arguments:

```
python train_with_best_args.py --network vgg11 --dataset cifar10 --optimizer INNAprop --alpha 0.1 --beta 0.9 --lr 1e-3 --wd 1e-2
```

### 3. Reproduce Food101 Finetuning
To replicate the finetuning experiment on Food101:

```
python finetune_with_best_args.py --network resnet18 --dataset food101 --optimizer AdamW --lr 1e-4 --wd 0.0
```

### 4. Reproduce ImageNet Experiments:
To run the ImageNet experiments:
```
torchrun --nproc_per_node=4 train_classification.py --model resnet18 --opt INNAprop  \
--alpha 0.1 --beta 0.9 \
--lr 1e-3 --lr-scheduler=cosineannealinglr --lr-min 1e-5 --weight-decay 0.01 \
--output-dir results/train/imagenet/resnet18/INNAprop_alpha_0.1_beta_0.9_lr_0.001
```

## Acknowledgement

The training code is is adapted from the [PyTorch](https://github.com/pytorch/vision/blob/main/references/classification/train.py) repository. 