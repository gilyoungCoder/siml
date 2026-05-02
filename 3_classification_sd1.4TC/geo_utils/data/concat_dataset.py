from torch.utils.data import Dataset
import torch

from PIL import Image
import json
import numpy as np

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors

import os

class ConcatDataset(Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in self.datasets])
    
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class SimpleDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_real=True, verbose=False):
        self.data_dir = data_dir
        if isinstance(data_dir, str):
            self.data_list = os.listdir(data_dir)
            self.data_list = [os.path.join(self.data_dir, x) for x in self.data_list]
        elif isinstance(data_dir, list):
            self.data_list = []
            for data_dir_elem in data_dir:
                data_list = os.listdir(data_dir_elem)
                self.data_list += [os.path.join(data_dir_elem, x) for x in data_list]
        
        self.transform = transform
        self.is_real = is_real
        self.verbose = verbose

    def __getitem__(self, idx):
        # sample = self.data[idx]
        # sample_path = os.path.join(self.data_dir, self.data_list[idx])
        sample_path = self.data_list[idx]
        sample = Image.open(sample_path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)

        return_value = [sample]

        if self.is_real:
            return_value.append(torch.ones((1,)))
        else:
            return_value.append(torch.zeros((1,)))
        

        if self.verbose:
            return_value.append(self.data_list[idx])

        return tuple(return_value)

    def __len__(self):
        return len(self.data_list)