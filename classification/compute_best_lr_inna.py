import os
import argparse

import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.backends.cudnn as cudnn

import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.trial import TrialState
import joblib

from models_scratch import *
from data_utils import *
from train import *
from optimizers import *

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def init(network, dataset):
    if dataset == "cifar100":
        num_classes = 100
    else:
        num_classes = 10

    if network == "nin":
        net= NiN(num_classes=num_classes)
    elif network == "vgg11":
        net = VGG("VGG11", num_classes=num_classes)
    elif network == "vgg16":
        net = VGG("VGG16", num_classes=num_classes)
    elif network == "resnet18":
        net = resnet18(norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif network == "resnet34":
        net = resnet34(norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif network == "densenet121":
        net = densenet121(num_classes=num_classes)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)
    return net

def adjust_learning_rate(optimizer, epoch, lr, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr / (1 + epoch)**power
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def objective(trial):
    global optimizer
    global dataset
    global network
    global epochs
    global alpha
    global beta
    global wd
    
    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "cifar100":
        trainloader, testloader = get_cifar100_loaders()
    else: 
        raise ValueError("Invalid dataset name.")
   
    net = init(network, dataset)
    lr = trial.suggest_categorical("lr", [1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [wd])
    if optimizer == 'inna':
        opt = INNA(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    elif optimizer == 'innaprop':
        opt = INNAprop(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    elif optimizer == 'dinadam':
        opt = DINAdam(net.parameters(), lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer.')
    
    scheduler = CosineAnnealingLR(opt, T_max = 200, last_epoch=-1)
    best_loss = 1e6
    best_acc = 0

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train(net, opt, trainloader) 
        test_loss, test_acc = test(net, testloader) 
        best_loss = min(best_loss, train_loss)
        best_acc = max(best_acc, test_acc)
        scheduler.step()
         
    return best_loss, best_acc, train_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lr search for INNA")
    parser.add_argument("--optimizer", type=str, default="inna")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--trials", type=int, default=144, help="number of trials")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1,0.5,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.1,0.5, 0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    
    args = vars(parser.parse_args())
    optimizer = args["optimizer"]
    network = args["network"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    n_trials = args["trials"]
    alphas = args["alphas"]
    betas = args["betas"]
    wd = args["wd"]
    
    outdir = f"results/best_learning_rates/{dataset}/{network}/{optimizer}_alpha_{alphas}_beta_{betas}_epochs_{epochs}_wd_{wd}"
    os.makedirs(outdir, exist_ok=True)
    current_time = time.localtime()
    date_string = time.strftime("%Y-%m-%d_%H-%M", current_time)
    file_name = f"best_lr_{date_string}.csv"
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    df = pd.DataFrame(columns=["alpha", "beta", "lr", "weight_decay", "train_loss", "test_acc", "train_acc"])
    set_seed(5000)
    
    for alpha in alphas:
        for beta in betas:
            param_grid = {
            'lr': [1e-3],
            "weight_decay": [wd]
            }
            sampler = optuna.samplers.GridSampler(param_grid)
            study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize", "maximize"], pruner=optuna.pruners.NopPruner())    
            study.optimize(objective, n_trials=n_trials)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  alpha: ", alpha, "beta: ", beta)
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))
            
            for trial in complete_trials:
                df = df.append({
                    "alpha": alpha, "beta": beta,
                    "lr": trial.params["lr"], 
                    "weight_decay": trial.params["weight_decay"], 
                    "train_loss": trial.values[0],
                    "test_acc": trial.values[1], 
                    "train_acc": trial.values[2]}, ignore_index=True)
                
            df = df.sort_values(by=['train_loss', 'test_acc', 'train_acc'], ascending=[True, False, False])
                    
            csv_path = os.path.join(outdir, file_name)
            df.to_csv(csv_path, index=False)

            study_path = os.path.join(outdir, f"study_{alpha}_{beta}.pkl")
            joblib.dump(study, study_path)
                
                    