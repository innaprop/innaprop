import os
import sys
import argparse
import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
from timm.models import VisionTransformer

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import pandas as pd
import csv
import math
import numpy as np
import random
import time

from models_scratch import *
from data_utils import *
from train import *
from optimizers import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

def build_data(dataset):
    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "cifar100":
        trainloader, testloader = get_cifar100_loaders()
    elif dataset== "mnist":
        trainloader, testloader = get_mnist_loaders()
        
    else: 
        raise ValueError("Invalid dataset name.")
    return trainloader, testloader

def build_model(network, dataset, device):
    if dataset == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    
    if network == "vgg11":
        net = VGG("VGG11", num_classes=num_classes)
    elif network == "vgg16":
        net = VGG("VGG16", num_classes=num_classes)
    elif network == "resnet18":
        net = resnet18(norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif network == "resnet34":
        net = resnet34(norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif network == "densenet121":
        net = densenet121(norm_layer=nn.BatchNorm2d, num_classes=num_classes)
        net.head = nn.Linear(net.head.in_features, 10)
    elif network == "lenet":
        net = LeNet5(num_classes=num_classes)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)
    return net

def create_optimizer(optimizer, lr, alpha, beta, weight_decay, net):
    if optimizer == 'SGDM':
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer== 'AdamW':
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer== 'AdamWFree':
        opt = AdamWScheduleFree(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer== 'SGDFree':
        opt = SGDScheduleFree(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer== 'DINAdam':
        opt = DINAdam(net.parameters(), alpha=alpha, beta=beta, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'INNAprop':
        opt = INNAprop(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    elif optimizer == 'INNAprop_v2':
        opt = INNAprop_v2(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    elif optimizer == 'INNAdam':
        opt = INNAdam(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer.')
    return opt
        
def adjust_alpha_cosine(optimizer, epoch, max_epochs, start_alpha=0.1, final_alpha=1.0, min_alpha=0.1, warmup_epochs=10):
    if epoch < warmup_epochs:
        cos_inner = math.pi * epoch / warmup_epochs
        new_alpha = start_alpha + (min_alpha - start_alpha) * (1 - math.cos(cos_inner)) / 2
    elif epoch <= max_epochs:
        adjusted_epoch = epoch - warmup_epochs
        max_adjusted_epoch = max_epochs - warmup_epochs
        cos_inner = math.pi * adjusted_epoch / max_adjusted_epoch
        new_alpha = min_alpha + (final_alpha - min_alpha) * (1 - math.cos(cos_inner)) / 2
    elif epoch > max_epochs :
        new_alpha = final_alpha
    
    for param_group in optimizer.param_groups:
        param_group['alpha'] = new_alpha
        print('alpha', new_alpha)
        
def adjust_beta_cosine(optimizer, epoch, max_epochs, start_beta=1.0, final_beta=0.5, min_beta=0.8, warmup_epochs=10):
    if epoch < warmup_epochs:
        cos_inner = math.pi * epoch / warmup_epochs
        new_beta = start_beta + (min_beta - start_beta) * (1 - math.cos(cos_inner)) / 2
    elif epoch <= max_epochs:
        adjusted_epoch = epoch - warmup_epochs
        max_adjusted_epoch = max_epochs - warmup_epochs
        cos_inner = math.pi * adjusted_epoch / max_adjusted_epoch
        new_beta = min_beta + (final_beta - min_beta) * (1 - math.cos(cos_inner)) / 2
    elif epoch > max_epochs :
        new_beta = final_beta
    
    for param_group in optimizer.param_groups:
        param_group['beta'] = new_beta
        print('beta', new_beta)
        
def main(args):
    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    nb_experiment = args["nb_experiment"]
    alpha = args['alpha']
    beta = args['beta']
    lr = args['lr']
    grad_clip = args['grad_clip']
    weight_decay = args['wd']
    lr_scheduler = args['lr_scheduler']
    alpha_scheduler = args['alpha_scheduler']
    beta_scheduler = args['beta_scheduler']
    seed = args['seed']
        
    
    if optimizer.startswith('INNA') or optimizer.startswith('DIN'): 
        outdir = f"results/train/{dataset}/{network}/{optimizer}_alpha_{alpha}_beta_{beta}_wd_{weight_decay}"
    else:
        outdir = f"results/train/{dataset}/{network}/{optimizer}_wd_{weight_decay}"
    os.makedirs(outdir, exist_ok=True)
    
    current_time = time.localtime()
    date_string = time.strftime("%Y-%m-%d_%H-%M", current_time)
    filename = os.path.join(outdir, f"results_{date_string}.csv")

    fieldnames = ["run_id", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "optimizer", "lr"]

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    print(f"File: {filename}")

    device = "cuda" if torch.cuda.is_available() else "cpu"        
    trainloader, testloader =  build_data(dataset)
    

    for k in tqdm(range(nb_experiment), desc="run_loop", leave=False):
        set_seed(seed + k)
        net = build_model(network, dataset, device)
        opt = create_optimizer(optimizer, lr, alpha, beta, weight_decay, net)
        scheduler = CosineAnnealingLR(opt, T_max = epochs, last_epoch=-1)
        best_acc = 0
        
        
        for epoch in tqdm(range(epochs), desc="epoch_loop", leave=False):
            if alpha_scheduler:
                adjust_alpha_cosine(opt, epoch, 200, start_alpha=alpha, final_alpha= 1.0, min_alpha=alpha, warmup_epochs=20)
            if beta_scheduler:
                adjust_beta_cosine(opt, epoch, 200, start_beta=beta, final_beta=2.0, min_beta=beta, warmup_epochs=20)
            
            train_loss, train_acc = train(net, opt, trainloader, grad_clip)
            test_loss, test_acc = test(net, testloader)
            best_acc = max(best_acc, test_acc)
            
            if lr_scheduler:
                scheduler.step()
                                            
            with open(filename, mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if optimizer.startswith('INNA') or optimizer.startswith('DIN'):
                    if alpha_scheduler or beta_scheduler:
                        writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": best_acc, "optimizer": f"{optimizer}, ($\\alpha$, $\\beta$)= ({alpha},{beta}) + schedulers", "lr": opt.param_groups[0]['lr']})
                    else: 
                        writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": best_acc, "optimizer": f"{optimizer}, ($\\alpha$, $\\beta$)= ({alpha},{beta})", "lr": opt.param_groups[0]['lr']})
                else:
                     writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": best_acc, "optimizer": f"{optimizer}", "lr": opt.param_groups[0]['lr']})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Classic Optimizer Experiment with best args")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--nb_experiment", type=int, default=1, help="number of independent runs for each configuration")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--wd', default=0.01, type=float, help='weight decay for optimizers')
    parser.add_argument("--grad_clip", type=float, default=0.0, help="grad_clip")
    parser.add_argument("--lr_scheduler", type=boolean_string, default="True", help="lr scheduler")
    parser.add_argument("--alpha_scheduler", type=boolean_string, default="False", help="alpha scheduler")
    parser.add_argument("--beta_scheduler", type=boolean_string, default="False", help="beta scheduler")
    parser.add_argument("--seed", type=int, default=5000, help="seed")


    args = vars(parser.parse_args())
    main(args)
