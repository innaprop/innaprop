import torch
import torch.nn as nn
from tqdm import tqdm
import itertools


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(net, optimizer, trainloader, grad_clip=0.0):
    criterion = nn.CrossEntropyLoss()
    net.train()
    #optimizer.train()
    
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, desc="batch_loop", leave=False) as pbar:
        for inputs, targets in pbar:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(train_loss=train_loss/len(trainloader), train_accuracy= 100*correct / (total))

    return train_loss / len(trainloader), 100*correct / total


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(), tqdm(testloader, desc="batch_loop", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(test_loss=test_loss / len(testloader), test_accuracy=100*correct / (total))

    return test_loss / len(testloader), 100*correct / total

