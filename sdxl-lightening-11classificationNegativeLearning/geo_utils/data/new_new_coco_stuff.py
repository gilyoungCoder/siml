
import torch
from torch.utils.data import Dataset

import random
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

import torchvision.transforms.v2 as transforms
import torchvision

import os

class NewNewCocoStuffDataset(Dataset):
    def __init__(self, dataset_path, indices, additional_dataset_paths=None, excluded_dataset_paths=None, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.dataset_list = sorted(os.listdir(self.dataset_path))
        self.dataset_list = [os.path.join(self.dataset_path, self.dataset_list[i]) for i in indices]

        if additional_dataset_paths is not None:
            for additional_dataset_path in additional_dataset_paths:
                if os.path.isdir(additional_dataset_path):
                    newly_added_dataset = sorted(os.listdir(additional_dataset_path))
                    newly_added_dataset = [os.path.join(additional_dataset_path, x) for x in newly_added_dataset]
                    self.dataset_list += newly_added_dataset
                    # self.dataset_list += sorted(os.listdir(additional_dataset_path))
                elif os.path.isfile(additional_dataset_path) and ".jpg" in additional_dataset_path:
                    self.dataset_list.append(additional_dataset_path)
                else:
                    raise ValueError(f"additional_dataset_path is not a valid path: {additional_dataset_path}")
                    
        
        if excluded_dataset_paths is not None:
            for excluded_dataset_path in excluded_dataset_paths:
                self.dataset_list = [x for x in self.dataset_list if x not in sorted(os.listdir(excluded_dataset_path))]


        # Sanity check
        for dataset in self.dataset_list:
            if not os.path.isfile(dataset):
                raise ValueError(f"dataset path is not a valid path: {dataset}")

        
    def __getitem__(self, idx):
        # img = Image.open(os.path.join(self.dataset_path, self.dataset_list[idx])).convert('RGB')
        img = Image.open(self.dataset_list[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.join(self.dataset_path, self.dataset_list[idx])
       
    def __len__(self):
        return len(self.dataset_list)