dependencies = ['torch']
import torch
import os
from os.path import join

dirname = os.path.dirname(__file__)

def resnet18_cifar10(return_transform=True):
    """
    Loads a torchvision ResNet18 model trained on CIFAR10
    args: return_transform=True, returns the default input transform for this model 
        ```
        model = resnet18_cifar10(return_transform=False)
        model, tf = resnet18_cifar10(return_transform=True)
        ```
    """
    model = torch.load(join(dirname, 'detectron/cifar10.pt'))
    if return_transform:
        return model, torch.load(join(dirname, 'detectron/cifar10_input_transform.pt'))
    return model
