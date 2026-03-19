
import torch
from torch.utils.data import Dataset

import random
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset
from .base_subset_dataset import BaseSubsetDataset

import torchvision.transforms.v2 as transforms
import torchvision

class NewCOCOStuffDataset(BaseDataset):
    def __init__(self, prompt_version='v1', num_bucket_per_side=None, 
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0., blip_finetune=True, tokenizer=None, is_main_process=True, 
                 **kwargs):
        # super().__init__(**kwargs)
        self.prompt_version = prompt_version
        self.no_sections = num_bucket_per_side
        print('Using prompt version: {}, num_bucket_per_side: {}'.format(prompt_version, num_bucket_per_side))
        
        self.FEAT_SIZE = [each // 8 for each in kwargs['pipeline'][2].img_scale][::-1]
        print('Using feature size: {}'.format(self.FEAT_SIZE))
        
        self.foreground_loss_mode = foreground_loss_mode
        self.foreground_loss_weight = foreground_loss_weight
        self.foreground_loss_norm = foreground_loss_norm
        print('Using foreground_loss_mode: {}, foreground_loss_weight: {}, foreground_loss_norm: {}'.format(foreground_loss_mode, foreground_loss_weight, foreground_loss_norm))
        
        self.uncond_prob = uncond_prob
        print('Using unconditional generation probability: {}'.format(uncond_prob))
        
        self.tokenizer = tokenizer
        self.blip_finetune = blip_finetune
        super().__init__(kwargs, is_main_process)

        self.instance_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', ]

        self.bbox_transform = transforms.Compose([
            transforms.ClampBoundingBoxes(),
        ])
    
        
    def __getitem__(self, idx):
        """
        Returns a data item: {pixel_values: tensor of (3, H, W),  text: string}
        """
        # First, retrieve the image path from the config file.
        # Second, load the image from the path.
        # Third, load the annotations from the path.
        # Fourth, apply the pipeline to the image and annotations.

        img = self.load_img_from_idx(idx)
        img_meta = self.get_img_meta_from_idx(idx)
        bboxes = self.get_bbox_from_idx(idx)
        captions = self.get_caption_from_idx(idx)
        # assert captions
        # print("original bbox", bbox)
        gt_labels = self.get_gt_labels_from_idx(idx)

        bboxes = self.bbox_transform(bboxes)

        if bboxes.shape[1] == 0:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

        # Check the size of bbox and remove the bbox whose size is less than 2% of the image area.
        sanity_checked_bboxes_idx = []
        for bbox_idx in range(len(bboxes)):
            bbox = bboxes[bbox_idx]
            bbox_area = bbox[2] * bbox[3]
            img_area = img.shape[1] * img.shape[2]

            if bbox_area >= 0.02 * img_area:
                sanity_checked_bboxes_idx.append(bbox_idx)

        # print("Len of sanity checked bboxes ", len(sanity_checked_bboxes_idx))
        # if len(sanity_checked_bboxes_idx) < 3 or len(sanity_checked_bboxes_idx) > 8:
        # if len(sanity_checked_bboxes_idx) > 8:
        #     # idx = random.randint(0, len(self) - 1)
        #     # return self.__getitem__(idx)
        #     sanity_checked_bboxes_idx = sanity_checked_bboxes_idx[:8]
        if len(sanity_checked_bboxes_idx) == 0:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

        img, bboxes = self.apply_pipeline(img, bboxes)

        # Convert bboxes format [left top x, left top y, width, height] to [left top x, left top y, right bottom x, right bottom y]
        bboxes = transforms.ConvertBoundingBoxFormat("xyxy")(bboxes)

        if bboxes is None:
            example = {}
            example["pixel_values"] = img.data
            example["text"] = "A scene captured by a camera of a drone"
            if self.foreground_loss_mode is not None:
                bbox_mask = torch.zeros(self.FEAT_SIZE).float()
                bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), 0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)
                example["bbox_mask"] = bbox_mask
            # example["bbox"] = np.array(coords)
            example["bbox"] = None
            # For debug
            example["img_metadata"] = img_meta
            return example
        
        labels = [self.CLASSES[each].split('.')[-1] for each in gt_labels]
        
        coords = []
        if self.prompt_version == 'v1':
            img_shape = torch.tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]]) # data["img"].shape = (N, H, W)
            bboxes = bboxes / img_shape
            
            # random shuffle bbox annotations
            # index = list(range(len(labels)))
            index = list(sanity_checked_bboxes_idx)
            random.shuffle(index)
            # index = index[:22] # 9+3*22+2=77
            index = index[:18] # 9+3*22+2=77
            
            # generate bbox mask and text prompt
            # constant: background -> 0, foreground -> self.foreground_loss_weight
            # area:     background -> 0, foreground -> 1 / area ^ self.foreground_loss_weight (for area, smaller weight, larger variance with respect to areas)
            objs = []
            instance_objs = []
            bbox_mask = torch.zeros(self.FEAT_SIZE).float() # [H, W]
            bbox_binary_mask = torch.zeros(self.FEAT_SIZE).float() # [H, W]
            for each in index:
                label = labels[each]
                bbox = bboxes[each]
                caption = captions[each]
                
                # generate bbox mask
                FEAT_SIZE = torch.tensor([self.FEAT_SIZE[1], self.FEAT_SIZE[0], self.FEAT_SIZE[1], self.FEAT_SIZE[0]])
                coord = torch.round(bbox * FEAT_SIZE).int().tolist()
                coords.append(coord)
                if label in ['person']:
                    bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 2 * 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2) if self.foreground_loss_mode == 'constant' else \
                                                                    1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                else:
                    bbox_mask[coord[1]: coord[3], coord[0]: coord[2]] = self.foreground_loss_weight * 1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), 0.2) if self.foreground_loss_mode == 'constant' else \
                                                                    1 / torch.pow(torch.tensor((coord[3] - coord[1] + 1) * (coord[2] - coord[0] + 1)), self.foreground_loss_weight)
                
                if label in self.instance_labels:
                    bbox_binary_mask[coord[1]: coord[3], coord[0]: coord[2]] = 1
                # generate text prompt
                bbox = self.token_pair_from_bbox(bbox.tolist())

                
                if self.blip_finetune and caption is not None:
                    objs.append(" ".join([caption, bbox]))
                else:
                    # objs.append(' '.join([self.class2text[label], bbox]))
                    objs.append(' '.join([label, bbox]))
                
                if label in self.instance_labels:
                    instance_objs.append(' '.join([label, bbox]))
            
            bbox_mask[bbox_mask == 0] = 1 * 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), 0.2) if self.foreground_loss_mode == 'constant' else 1 / torch.pow(torch.tensor(self.FEAT_SIZE[0] * self.FEAT_SIZE[1]), self.foreground_loss_weight)
            bbox_mask = bbox_mask / torch.sum(bbox_mask) * self.FEAT_SIZE[0] * self.FEAT_SIZE[1] if self.foreground_loss_norm else bbox_mask
            
            # text = "A scene captured by a camera of a drone, containing "
            text = "A image with "
            if self.uncond_prob > 0 and len(objs) == 0:
                # text = 'A scene captured by a camera of a drone, containing no person' if random.random() > self.uncond_prob else ""
                text = text + 'no person' if random.random() > self.uncond_prob else ""
            elif self.uncond_prob > 0 and len(objs) > 0:
                # text = 'A scene captured by a camera of a drone, containing ' + ' '.join(objs) if random.random() > self.uncond_prob else ""
                text = text + ' '.join(objs) if random.random() > self.uncond_prob else ""
            elif self.uncond_prob == 0 and len(objs) > 0:
                text = text + ' '.join(objs)
            else:
                # text = "A scene captured by a camera of a drone"
                text = "A image with "

            instance_text = "A image with " + ' '.join(instance_objs)

        else:
            raise NotImplementedError("Prompt version {} is not supported!".format(self.prompt_version))
        
        example = {}
        example["pixel_values"] = img.data
        example["text"] = text
        example["instance_text"] = instance_text
        if self.tokenizer is not None:
            example["input_instance_ids"] = torch.tensor(self.tokenizer([instance_text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True).input_ids)
            example["input_ids"] = torch.tensor(self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True).input_ids)
        if self.foreground_loss_mode is not None:
            example["bbox_mask"] = bbox_mask
        example["bbox"] = np.array(coords)
        example["bbox_binary_mask"] = bbox_binary_mask
        # For debug
        example["img_metadata"] = img_meta

        return example

    # code borrowed from https://github.com/CompVis/taming-transformers
    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections[0] - 1)))
        y_discrete = int(round(y * (self.no_sections[1] - 1)))
        return "<l{}>".format(y_discrete * self.no_sections[0] + x_discrete)

    def token_pair_from_bbox(self, bbox):
        return self.tokenize_coordinates(bbox[0], bbox[1]) + ' ' + self.tokenize_coordinates(bbox[2], bbox[3])


class NewCOCOStuffSubsetDataset(NewCOCOStuffDataset, BaseSubsetDataset):
    def __init__(self, prompt_version='v1', num_bucket_per_side=None, 
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0., blip_finetune=True, tokenizer=None, is_main_process=True, 
                 **kwargs):
        subset_list = kwargs.pop('subset_list', None)
        NewCOCOStuffDataset.__init__(self, prompt_version=prompt_version, num_bucket_per_side=num_bucket_per_side,
                                    foreground_loss_mode=foreground_loss_mode, foreground_loss_weight=foreground_loss_weight,
                                    foreground_loss_norm=foreground_loss_norm, feat_size=feat_size, uncond_prob=uncond_prob,
                                    blip_finetune=blip_finetune, tokenizer=tokenizer, is_main_process=is_main_process, **kwargs)
        BaseSubsetDataset.__init__(self, kwargs, is_main_process=is_main_process, subset_list=subset_list)


class AugmentedNewCOCOStuffSubsetDataset(NewCOCOStuffDataset, BaseSubsetDataset):
    def __init__(self, prompt_version='v1', num_bucket_per_side=None, 
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0., blip_finetune=True, tokenizer=None, is_main_process=True, 
                 **kwargs):
        # super().__init__(prompt_version=prompt_version, num_bucket_per_side=num_bucket_per_side,
        #                  foreground_loss_mode=foreground_loss_mode, foreground_loss_weight=foreground_loss_weight,
        #                  foreground_loss_norm=foreground_loss_norm, feat_size=feat_size, uncond_prob=uncond_prob,
        #                  blip_finetune=blip_finetune, tokenizer=tokenizer, is_main_process=is_main_process, **kwargs)
        subset_list = kwargs.pop('subset_list', None)
        NewCOCOStuffDataset.__init__(self, prompt_version=prompt_version, num_bucket_per_side=num_bucket_per_side,
                                    foreground_loss_mode=foreground_loss_mode, foreground_loss_weight=foreground_loss_weight,
                                    foreground_loss_norm=foreground_loss_norm, feat_size=feat_size, uncond_prob=uncond_prob,
                                    blip_finetune=blip_finetune, tokenizer=tokenizer, is_main_process=is_main_process, **kwargs)
        augmented_ann_images = []
        augmented_list = kwargs.pop('augmented_list', None)
        print("AugmentedNewCOCOStuffSubsetDataset: Loading annotations from augmented_list")
        if augmented_list is not None:
            for subset in augmented_list:
                subset_name = subset.split("/")[-1]
                subset_name = subset_name.split(".")[0]
                for ann_image in self.ann_images:
                    if subset_name in ann_image["file_name"]:
                        new_ann_image = ann_image.copy()
                        new_ann_image["file_name"] = subset
                        augmented_ann_images.append(new_ann_image)
                        break
        print("AugmentedNewCOCOStuffSubsetDataset: Loading augmented annotations complete")
        print("Corresponding images: {}".format(len(augmented_ann_images)))
        BaseSubsetDataset.__init__(self, kwargs, is_main_process=is_main_process, subset_list=subset_list)
        self.num_real_data = len(self.ann_images)
        self.ann_images += augmented_ann_images

        print("AugmentedNewCOCOStuffSubsetDataset: Loading total annotations complete")
        print("Corresponding images: {}".format(len(self.ann_images)))

    def load_img_from_idx(self, idx):
        img_metadata = self.get_img_meta_from_idx(idx)
        img_path = self.img_prefix + img_metadata["file_name"] if idx < self.num_real_data else img_metadata["file_name"]
        img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        return img



class DummyMissingPersonDataset(Dataset):
    # CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
            #    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    # ]
    CLASSES = [None, 'person']

    def __init__(self, prompt_version='v1', num_bucket_per_side=None, 
                 foreground_loss_mode=None, foreground_loss_weight=1.0, foreground_loss_norm=False, feat_size=64,
                 uncond_prob=0.,
                 **kwargs):
        # super().__init__(**kwargs)
        super().__init__()
        self.prompt_version = prompt_version
        self.no_sections = num_bucket_per_side
        print('Using prompt version: {}, num_bucket_per_side: {}'.format(prompt_version, num_bucket_per_side))
        
        self.FEAT_SIZE = [each // 8 for each in kwargs['pipeline'][2].img_scale][::-1]
        print('Using feature size: {}'.format(self.FEAT_SIZE))
        
        self.foreground_loss_mode = foreground_loss_mode
        self.foreground_loss_weight = foreground_loss_weight
        self.foreground_loss_norm = foreground_loss_norm
        print('Using foreground_loss_mode: {}, foreground_loss_weight: {}, foreground_loss_norm: {}'.format(foreground_loss_mode, foreground_loss_weight, foreground_loss_norm))
        
        self.uncond_prob = uncond_prob
        print('Using unconditional generation probability: {}'.format(uncond_prob))
        
        self.class2text = {
            'person': 'person'
        }

        
    def __getitem__(self, idx):
        """
        Returns a data item: {pixel_values: tensor of (3, H, W),  text: string}
        """
        # First, retrieve the image path from the config file.
        # Second, load the image from the path.
        # Third, load the annotations from the path.
        # Fourth, apply the pipeline to the image and annotations.

        
        example = {}
        example["pixel_values"] = torch.zeros(3, 512, 512)
        example["text"] = "simple text"
        if self.foreground_loss_mode is not None:
            example["bbox_mask"] = torch.zeros(64, 64)
        example["bbox"] = torch.zeros(1, 4, dtype=torch.int32)
        # For debug
        example["img_metadata"] = torch.zeros(1)

        return example
    
    def __len__(self):
        return 3000


if __name__ == "__main__":
    from mmengine.config import Config
    # config = Config.fromfile('/mnt/home/jeongjun/layout_diffusion/GeoDiffusion/configs/data/missing_person_256x256.py')
    # config = Config.fromfile('/home/djfelrl11/geodiffusion/configs/data/missing_person_512x512.py')
    # config = Config.fromfile('/mnt/home/jeongjun/layout_diffusion/GeoDiffusion/configs/data/missing_person_512x512.py')
    config = Config.fromfile('/home/djfelrl11/geodiffusion/configs/data/missing_person_512x512.py')
    dataset_args = dict(
        prompt_version="v1", 
        num_bucket_per_side=[256, 256],
        # num_bucket_per_side=[512, 512],
        foreground_loss_mode="constant", 
        foreground_loss_weight=1.0,
        foreground_loss_norm=True,
        feat_size=64,
    )
    dataset_args_train = dict(
        uncond_prob=0.1,
    )
    config.data.val.update(dataset_args)
    # config.data.train.update(dataset_args_train)
    # config.data.train.remove("type")
    # del config.data.train["type"]
    # dataset = MissingPersonDataset(**config.data.train)
    dataset = NewCOCOStuffDataset(**config.data.val)
    data = dataset[0]

    def save_normalized_image(img, bboxes=None, path="."):
        img = img - img.min()
        img = img / img.max()
        img = (img * 255).astype(np.uint8)
        img = img[::8, : :8]
        img = overwrite_bbox_on_the_image(img, bboxes)
        Image.fromarray(img).save(path)
    
    def overwrite_bbox_on_the_image(img, bboxes):
        img = img.copy()
        for bbox in bboxes:
            bbox = bbox.astype(int)
            if bbox[2] >= img.shape[1]:
                bbox[2] = img.shape[1] - 1
            if bbox[3] >= img.shape[0]:
                bbox[3] = img.shape[0] - 1
            img[bbox[1]:bbox[3], bbox[0]] = (255, 255, 255)
            img[bbox[1]:bbox[3], bbox[2]] = (255, 255, 255)
            img[bbox[1], bbox[0]:bbox[2]] = (255, 255, 255)
            img[bbox[3], bbox[0]:bbox[2]] = (255, 255, 255)
        return img
    save_normalized_image(data["pixel_values"].numpy().transpose(1, 2, 0), data["bbox"], "test1.png")
    print(data["img_metadata"]["file_name"])
    breakpoint()