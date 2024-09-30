import torchvision
import torchvision.transforms as transforms
import torch
import os 
import sys

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def get_cifar10_train_loader(batch_size=256, size=32):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        #transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)

def get_cifar10_test_loader(size=32):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

def get_cifar10_loaders(batch_size=256, size=32):
    trainloader = get_cifar10_train_loader(batch_size, size)
    testloader = get_cifar10_test_loader(size)
    return trainloader, testloader

def get_cifar100_train_loader(batch_size=256, size=32):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        #transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)

def get_cifar100_test_loader(size=32):
    transform = transforms.Compose([
        #transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

def get_cifar100_loaders(batch_size=256, size=32):
    trainloader = get_cifar100_train_loader(batch_size, size)
    testloader = get_cifar100_test_loader(size)
    return trainloader, testloader

def get_mnist_loaders(batch_size=128):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader
    
def get_food101_loaders(batch_size=256):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Assuming the dataset is split into train and test folders
    trainset = torchvision.datasets.Food101(root=data_dir, split='train',transform=transform)
    testset = torchvision.datasets.Food101(root=data_dir, split='test', transform=transform)

    # DataLoader instances for train and test sets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

    return trainloader, testloader