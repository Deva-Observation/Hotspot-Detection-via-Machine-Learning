import os
import re
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def import_train_dataset(BATCH_SIZE):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    path = r'.\dataset\iccad_official\iccad3\train'
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = ImageFolder(root=path, transform=trans)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return train_loader


def import_test_dataset(BATCH_SIZE):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    path = r'.\dataset\iccad_official\iccad3\test'
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_set = ImageFolder(root=path, transform=trans)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return test_loader
    
def import_valid_dataset(BATCH_SIZE):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    path = r'.\dataset\iccad_official\iccad3\valid'
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    valid_set = ImageFolder(root=path, transform=trans)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return valid_loader

import_train_dataset(64)