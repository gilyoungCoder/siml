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

def p_is_img_path_valid(idx, img_prefix):
    img_file = idx["file_name"]
    flags = map(lambda a: a.startswith(img_file), os.listdir(img_prefix))
    total_flag = reduce(lambda acc, cur: acc or cur, flags, False) 
    return total_flag

class BaseDataset(Dataset):
    def __init__(self, dataset_config, is_main_process=True):
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
        self.dataset_config = dataset_config
        self.img_prefix = dataset_config['img_prefix']
        self.ann_images = []
        self.ann_annotations = []
        self.ann_categories = []
        self.is_main_process = is_main_process

        print("BaseDataset: Loading annotations from {}".format(dataset_config["ann_file"]))

        if dataset_config["ann_file"].endswith(".json"):
            loaded_ann_obj = self.load_ann_obj_from_ann_file(dataset_config["ann_file"])
            if loaded_ann_obj is not None:
                self.ann_images += loaded_ann_obj[0]
                self.ann_annotations += loaded_ann_obj[1]
                self.ann_categories += loaded_ann_obj[2]
        else:
            for ann_file in os.listdir(dataset_config['ann_file']):
                if ann_file.endswith(".json"):
                    loaded_ann_obj = self.load_ann_obj_from_ann_file(dataset_config['ann_file'] + "/" + ann_file)
                    if loaded_ann_obj is not None:
                        self.ann_images += loaded_ann_obj[0]
                        self.ann_annotations += loaded_ann_obj[1]
                        self.ann_categories += loaded_ann_obj[2]
        print("BaseDataset: Loading annotations complete")

        self.remove_segmentation_from_annotations()
        print("BaseDataset: Removed segmentation from annotations")

        self.pipeline = self.initialize_pipeline(dataset_config['pipeline'])
        self.load_classes()
    
    def remove_segmentation_from_annotations(self):
        for ann in self.ann_annotations:
            if "segmentation" in ann:
                del ann["segmentation"]

    def load_ann_obj_from_ann_file(self, ann_file_path):
        with open(ann_file_path) as f:
            ann_obj = json.load(f)
            return ann_obj["images"], ann_obj["annotations"], ann_obj["categories"] 

    def is_img_path_valid(self, ann_file_obj):
        images_obj = ann_file_obj["images"]
        
        if torch.cuda.is_available():
            process_number = multiprocessing.cpu_count() // 4 // torch.cuda.device_count()
        else: # Assue we use TPU in this case.
            process_number = 32
        result = parmap.map(p_is_img_path_valid, images_obj, self.img_prefix, pm_pbar=True, pm_processes=process_number)
        return reduce(lambda acc, cur: acc and cur, result, True)

    def get_anns_from_idx(self, idx):
        image_obj = self.ann_images[idx]
        image_id = image_obj["id"]
        ann = filter(lambda a: a["image_id"] == image_id, self.ann_annotations)
        return ann

    def get_gt_labels_from_idx(self, idx):
        ann_obj = self.get_anns_from_idx(idx)
        if ann_obj is None:
            return None
        return [ann["category_id"] for ann in ann_obj]

    def load_classes(self):
        self.CLASSES = {}
        for category in self.ann_categories:
            self.CLASSES[category["id"]] = category["name"]

    def get_classes_from_idx(self, idx):
        gt_labels = self.get_gt_labels_from_idx(idx)
        if gt_labels is None:
            return None
        return [self.ann_categories[gt_label]["name"] for gt_label in gt_labels]
    
    def get_img_meta_from_idx(self, idx):
        img_metas = self.ann_images[idx]
        return img_metas

    def load_img_from_idx(self, idx):
        img_metadata = self.get_img_meta_from_idx(idx)
        img_path = self.img_prefix + img_metadata["file_name"]
        img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        return img

    
    def get_bbox_from_idx(self, idx):
        img_meta = self.get_img_meta_from_idx(idx)
        anns = self.get_anns_from_idx(idx)
        if anns is None:
            return None
        bbox = [ann["bbox"] for ann in anns]
        width, height = img_meta["width"], img_meta["height"]
        bbox = torchvision.tv_tensors.BoundingBoxes(bbox, format="XYWH", canvas_size=(height, width))
        if bbox.size(0) == 0:
            return None
        return bbox

    def get_caption_from_idx(self, idx):
        anns = self.get_anns_from_idx(idx)
        if anns is None:
            return None
        return [ann.get("caption", None) for ann in anns]

    def initialize_pipeline(self, pipeline_list):
        pipeline_function_list = []
        pipeline_function_list.append(transforms.ToDtype(torch.float32))
        for pipeline_elem in pipeline_list:
            pipeline_type = pipeline_elem["type"]
            if pipeline_type == 'LoadImageFromFile':
                pass

            elif pipeline_type == 'LoadAnnotations':
                pass

            elif pipeline_type == 'Resize':
                pipeline_function_list.append(transforms.Resize(pipeline_elem["img_scale"]))

            elif pipeline_type == 'RandomFlip':
                pipeline_function_list.append(transforms.RandomHorizontalFlip(pipeline_elem["flip_ratio"]))

            elif pipeline_type == 'Normalize':
                pipeline_function_list.append(transforms.Normalize(pipeline_elem["mean"], pipeline_elem["std"]))

            elif pipeline_type == 'Pad':
                pass # Because resize makes the image size fixed

            elif pipeline_type == 'DefaultFormatBundle':
                pass
            elif pipeline_type == 'Collect':
                pass
            else:
                raise ValueError("Pipeline type not recognized: {}".format(pipeline_type))
        return transforms.Compose(pipeline_function_list)
    
    def apply_pipeline(self, *tensor):
        return self.pipeline(*tensor)

    def __len__(self):
        # return len(self.ann_files)
        return len(self.ann_images)
