import os
import argparse

import pandas as pd
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

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

def init(network, dataset):
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
        net = densenet121(num_classes=num_classes)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)
    return net

def objective(trial):
    global optimizer
    global dataset
    global epochs

    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "cifar100":
        trainloader, testloader = get_cifar100_loaders()
    else: 
        raise ValueError("Invalid dataset name.")

    net = init(network, dataset)
    lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2, 1e-1, 0.0])
    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
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
         
    return best_loss, best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lr search for Optimizers")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--trials", type=int, default=25, help="number of trials")
    parser.add_argument("--optimizer", type=str, default="adam")

    args = vars(parser.parse_args())
    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    n_trials = args["trials"]

    outdir = f"results/best_learning_rates/{dataset}/{network}/{optimizer}"
    os.makedirs(outdir, exist_ok=True)

    current_time = time.localtime()
    date_string = time.strftime("%Y-%m-%d_%H-%M", current_time)
    file_name = f"best_lr_{date_string}.csv"
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    df = pd.DataFrame(columns=["lr", "weight_decay", "train_loss", "test_acc"])


    param_grid = {'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                 "weight_decay": [1e-4, 1e-3, 1e-2, 1e-1, 0.0]}
    
    sampler = optuna.samplers.GridSampler(param_grid)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize"], pruner=optuna.pruners.NopPruner())    
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    for trial in complete_trials:
        df = df.append({
            "lr": trial.params["lr"], 
            "weight_decay": trial.params["weight_decay"], 
            "train_loss": trial.values[0],
            "test_acc": trial.values[1]}, ignore_index=True)
        
    df = df.sort_values(by=['test_acc', 'train_loss'], ascending=[False, True])
        
    study_path = os.path.join(outdir, "study.pkl")
    joblib.dump(study, study_path)

    csv_path = os.path.join(outdir, file_name)
    df.to_csv(csv_path)
