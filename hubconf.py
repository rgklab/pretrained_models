dependencies = ['torch']
import torch
import pickle
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

def uci_heart():
    """
    Loads a featurized version of the UCI heart dataset, 
    as well as an XGB boost model trained on the Cleveland split.

    `model, data = uci_heart()`

    Data format:
    {
        'Cleveland': {'train': (data, labels), 'test': (data, labels), 'val': (data, labels)}
        'Hungary': (data, labels), 
        'Switzerland': (data, labels),
        'VA Long Beach': (data, labels)
    }

    The model is trained using default parameters
    {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist',
    } as well as num_boost_round=100.

    The test auc on cleveland is 0.809

    Data source:
    https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
    """
    data = pickle.load(open(join(dirname, 'detectron/uci_heart_features.pkl'), 'rb'))
    model = pickle.load(open(join(dirname, 'detectron/uci_xgb_cleveland.pkl'), 'rb'))
    return model, data
