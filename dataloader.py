import numpy as np
from PIL import Image

import torchvision
from torchvision import transforms
import torch

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from data_augmentation import get_data_augmentation_for_cifar10, TransformTwice

cifar10_mean = (0.4913, 0.4821, 0.4465)
cifar10_std = (0.2470, 0.2434, 0.2615)

cifar10_base_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, 
                             std=cifar10_std)
    ]
)
    
    
def load_cifar10_for_contrastive_learning(root='data/', s=0.5, batch_size=256, download=True, num_workers=0, seed=0):
    cifar10 = CIFAR10(root, train=True, transform=TransformTwice(get_data_augmentation_for_cifar10(32, s=s)), download=download)
    cifar10_train_dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return cifar10_train_dataloader


def load_cifar10_for_linear_evaluation(root='data/', batch_size=128, download=True, num_workers=0, seed=0):
    cifar10_train = CIFAR10(root, train=True, transform=cifar10_base_transforms, download=download)
    cifar10_test = CIFAR10(root, train=False, transform=cifar10_base_transforms, download=download)
    
    # these dataloader are not shuffled, becuase they are only used once for computing representation
    cifar10_train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # do not shuffle, because this dataloader will be used only for
    cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return cifar10_train_dataloader, cifar10_test_dataloader

