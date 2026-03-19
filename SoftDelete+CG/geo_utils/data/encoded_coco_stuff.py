import torch
from torch.utils.data import Dataset

from PIL import Image

from .base_dataset import BaseDataset

import os
import numpy as np
import yaml

class EncodedCocoStuffDataset(Dataset):
    def __init__(self, dataset_path, indices, additional_dataset_paths=None, excluded_dataset_paths=None, transform=None, epoch=1):
        self.dataset_path = dataset_path
        self.transform = transform
        self.real_image_path_list = sorted(os.listdir(self.dataset_path))
        self.real_image_path_list = [os.path.join(self.dataset_path, self.real_image_path_list[i]) for i in indices] # This is for excluded dataset path

        # Prepare encoded dataset
        self.encoded_image_path_list = []

        if additional_dataset_paths is not None:
            for additional_dataset_path in additional_dataset_paths:
                if os.path.isdir(additional_dataset_path):
                    newly_added_dataset = sorted(os.listdir(additional_dataset_path))
                    newly_added_dataset = [os.path.join(additional_dataset_path, x) for x in newly_added_dataset]
                    self.real_image_path_list += newly_added_dataset
                elif os.path.isfile(additional_dataset_path) and ".npy" in additional_dataset_path:
                    # self.real_image_path_list.append(additional_dataset_path)
                    self.encoded_image_path_list.append(additional_dataset_path)
                elif os.path.isfile(additional_dataset_path) and ".jpg" in additional_dataset_path:
                    # pass
                    self.real_image_path_list.append(additional_dataset_path)
                else:
                    raise ValueError(f"additional_dataset_path is not a valid path: {additional_dataset_path}")
        
        print("real image path list: ", len(self.real_image_path_list))
        print("encoded image path list: ", len(self.encoded_image_path_list))
        
        if excluded_dataset_paths is not None:
            for excluded_dataset_path in excluded_dataset_paths:
                # excluded_dataset_path_name = excluded_dataset_path.split(".")[-1]
                exclude_dataset_name_list = [ x.split(".")[0] for x in os.listdir(excluded_dataset_path) ]
                self.real_image_path_list = [ x for x in self.real_image_path_list if x not in exclude_dataset_name_list ]

        # Sanity check
        for dataset in self.real_image_path_list:
            if not os.path.isfile(dataset):
                raise ValueError(f"dataset path is not a valid path: {dataset}")

        # extract real image name
        self.image_name_list = [x.split(".")[0] for x in self.real_image_path_list]
        self.image_name_list = [x.split("/")[-1] for x in self.image_name_list]

        # encoded_dataset_path = "/".join(self.dataset_path.split("/")[:-1])
        encoded_dataset_path = "/".join(self.dataset_path.split("/")[:-2])
        no_flip_encoded_dataset_path = encoded_dataset_path + "/no_flip"
        flip_encoded_dataset_path = encoded_dataset_path + "/flip"
        for image_name in self.image_name_list:
            no_flip_image_path = os.path.join(no_flip_encoded_dataset_path, image_name + ".npy")
            flip_image_path = os.path.join(flip_encoded_dataset_path, image_name + ".npy")
            self.encoded_image_path_list.append(no_flip_image_path)
            self.encoded_image_path_list.append(flip_image_path)

        self.encoded_image_path_list = self.encoded_image_path_list * epoch

        

    def __getitem__(self, idx):
        # if self.transform:
        #     img = self.transform(img)
        # return img, os.path.join(self.dataset_path, self.real_image_path_list[idx])
        encoded_img = np.load(self.encoded_image_path_list[idx])
        encoded_img = torch.from_numpy(encoded_img).float()
        return encoded_img, self.encoded_image_path_list[idx]
       
    def __len__(self):
        return len(self.encoded_image_path_list)
        # return len(self.encoded_image_path_list) * 2

class EncodedCocoStuffManualDataset(Dataset):
    def __init__(self, dataset_path, manual_data_file, transform=None, epoch=1):
        self.dataset_path = dataset_path
        self.transform = transform

        print("Get into EncodedCocoStuffManualDataset")
        print("manual_data_file: ", manual_data_file)

        assert type(manual_data_file) == str, "manual_data_file should be a string"

        if ".txt" in manual_data_file:
            with open(manual_data_file, "r") as f:
                self.real_image_path_list = f.readlines()
                self.real_image_path_list = [x.strip() for x in self.real_image_path_list]
        elif ".yaml" in manual_data_file:
            with open(manual_data_file, "r") as f:
                self.real_image_path_list = yaml.load(f, Loader=yaml.FullLoader)
                self.real_image_path_list = [x for x in self.real_image_path_list.keys()]

        # extract real image name
        self.image_name_list = [x.split(".")[0] for x in self.real_image_path_list]
        self.image_name_list = [x.split("/")[-1] for x in self.image_name_list]

        # Prepare encoded dataset
        self.encoded_image_path_list = []

        # encoded_dataset_path = "/".join(self.dataset_path.split("/")[:-1])
        encoded_dataset_path = "/".join(self.dataset_path.split("/")[:-2])
        no_flip_encoded_dataset_path = encoded_dataset_path + "/no_flip"
        flip_encoded_dataset_path = encoded_dataset_path + "/flip"
        for image_name in self.image_name_list:
            no_flip_image_path = os.path.join(no_flip_encoded_dataset_path, image_name + ".npy")
            flip_image_path = os.path.join(flip_encoded_dataset_path, image_name + ".npy")
            self.encoded_image_path_list.append(no_flip_image_path)
            self.encoded_image_path_list.append(flip_image_path)
    
    def __getitem__(self, idx):
        # if self.transform:
        #     img = self.transform(img)
        # return img, os.path.join(self.dataset_path, self.real_image_path_list[idx])
        encoded_img = np.load(self.encoded_image_path_list[idx])
        encoded_img = torch.from_numpy(encoded_img).float()
        return encoded_img, self.encoded_image_path_list[idx]
       
    def __len__(self):
        return len(self.encoded_image_path_list)