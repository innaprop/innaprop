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

def build_data(dataset, size):
    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders(size)
    elif dataset == "cifar100":
        trainloader, testloader = get_cifar100_loaders(size)
    elif dataset == "food101":
        trainloader, testloader = get_food101_loaders()
        
    else: 
        raise ValueError("Invalid dataset name.")
    return trainloader, testloader

def build_model(network, dataset, device):
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "food101":
        num_classes = 101
    else:
        num_classes = 10
    
    if network == "vgg11":
        net = models.vgg11_bn(weights='IMAGENET1K_V1', dropout=0.2)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_features, num_classes)
    elif network == "resnet18":
        net = models.resnet18(weights='IMAGENET1K_V1')
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    return net

def create_optimizer(optimizer, lr, alpha, beta, weight_decay, net):
    if optimizer == 'SGD':
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer== 'AdamW':
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer== 'RMSProp':
        opt = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer== 'DINadam':
        opt = DINAdam(net.parameters(), alpha=alpha, beta=beta, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'INNA':
        opt = INNA(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    elif optimizer == 'INNAprop':
        opt = INNAprop(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer.')
    return opt


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
    scheduling = args['scheduling']
    seed = args['seed']
        
    if optimizer.startswith('INNA') or optimizer.startswith('DIN'):
        outdir = f"results/finetune/{dataset}/{network}/{optimizer}_alpha_{alpha}_beta_{beta}"
    else:
        outdir = f"results/finetune/{dataset}/{network}/{optimizer}"
    os.makedirs(outdir, exist_ok=True)
    
    current_time = time.localtime()
    date_string = time.strftime("%Y-%m-%d_%H-%M", current_time)
    filename = os.path.join(outdir, f"results_{date_string}.csv")

    fieldnames = ["run_id", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "optimizer"]

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    print(f"File: {filename}")

    device = "cuda" if torch.cuda.is_available() else "cpu"        
    size = 224
    trainloader, testloader =  build_data(dataset, size)
    

    for k in tqdm(range(nb_experiment), desc="run_loop", leave=False):
        set_seed(seed + k)
        net = build_model(network, dataset, device)
        opt = create_optimizer(optimizer, lr, alpha, beta, weight_decay, net)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max = epochs, last_epoch=-1, eta_min=lr/10)
        best_acc = 0
        for epoch in tqdm(range(epochs), desc="epoch_loop", leave=False):
            
            train_loss, train_acc = train(net, opt, trainloader, grad_clip)
            test_loss, test_acc = test(net, opt, testloader, trainloader)
            best_acc = max(best_acc, test_acc)
            if scheduling:
                scheduler.step()
                
            with open(filename, mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if optimizer.startswith('INNA') or optimizer == 'DINadam':
                    writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": best_acc, "optimizer": f"{optimizer}, ($\\alpha$, $\\beta$)= ({alpha},{beta})"})
                else:
                     writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": best_acc, "optimizer": f"{optimizer}"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Classic Optimizer Experiment with best args")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--nb_experiment", type=int, default=3, help="number of independent runs for each configuration")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--wd', default=0.0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument("--grad_clip", type=float, default=0.0, help="grad_clip")
    parser.add_argument("--scheduling", type=boolean_string, default="True", help="scheduler")
    parser.add_argument("--seed", type=int, default=5000, help="seed")


    args = vars(parser.parse_args())
    main(args)
