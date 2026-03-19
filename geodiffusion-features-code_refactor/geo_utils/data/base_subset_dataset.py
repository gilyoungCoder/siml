from torch.utils.data import Dataset
import torch

from PIL import Image
import json
import numpy as np

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors

import os
from functools import reduce 

import multiprocessing
import parmap

from .base_dataset import BaseDataset

def p_is_img_path_valid(idx, img_prefix):
    img_file = idx["file_name"]
    flags = map(lambda a: a.startswith(img_file), os.listdir(img_prefix))
    total_flag = reduce(lambda acc, cur: acc or cur, flags, False) 
    return total_flag

# class BaseSubsetDataset(Dataset):
class BaseSubsetDataset(BaseDataset):
    def __init__(self, dataset_config, is_main_process=True, subset_list=None):
        """
        Dataset_config:
            type: 'COCOStuffDataset'
            ann_dir=data_root + 'data/train/annotations',
            img_prefix=data_root + 'data/train/real_images',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(256, 256), keep_ratio=False, backend='pillow'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=8),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        """
        super().__init__(dataset_config, is_main_process)
        
        if subset_list is not None:
            new_ann_images = []
            for subset in subset_list:
                subset_name = subset.split("/")[-1]
                subset_name = subset_name.split(".")[0]
                # print(subset_name)
                for ann_image in self.ann_images:
                    if subset_name in ann_image["file_name"]:
                        new_ann_images.append(ann_image)
                        break
            self.ann_images = new_ann_images
        print("BaseSubsetDataset: Loading annotations complete")
        print("Corresponding images: {}".format(len(self.ann_images)))

        # self.pipeline = self.initialize_pipeline(dataset_config['pipeline'])
        # self.load_classes()
