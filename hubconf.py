import torch

def resnet18_cifar10(return_transform=True):
    model = torch.load('detectron/cifar10.pt')
    if return_transform:
        return model, torch.load('detectron/cifar10_input_transform.pt')
    return model
