import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from typing import List, Union, Optional
from collections import OrderedDict
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lpips

from tqdm import tqdm
from evaluate.utils import (
    get_transform, get_img_files, match_files, gather_img_tensors, read_prompt_to_ids,
    ImagePathDataset
)

# desired size of the output image -- disregarding high-frequency details
# following the paper
IMSIZE = 64
loss_fn_alex = lpips.LPIPS(net='alex').eval()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--img_path', help='path to generated images to be evaluated', type=str, required=True)
    parser.add_argument('--ref_path', help='path to reference images (the original ones)', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to txt prompts (separated by newline), If not provided, compare pairs of two images and return the average lpips score.', type=str, required=False, default=None)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', help='device to run on', type=str, required=False, default=None)
    parser.add_argument('--batch_size', help='batch size', type=int, required=False, default=50)
    parser.add_argument('--num_workers', help='number of workers', type=int, required=False, default=None)

    args = parser.parse_args()
    return args


@torch.no_grad()
def calc_lpips_from_files(
    img_files: Union[str, List[str]], 
    ref_files: Union[str, List[str]], 
    model: nn.Module,
    device: Optional[torch.device]=None,
    reduce: str='mean', 
    **kwargs
) -> Union[float, List[float]]:
    """
    Calculates LPIPS score between two images from image file lists
    """
    if isinstance(img_files, str):
        img_files = [img_files]
    if isinstance(ref_files, str):
        ref_files = [ref_files]
    assert len(img_files) == len(ref_files), 'Number of images in img_files and ref_files should be same'

    if device is not None and isinstance(device, str):
        device = torch.device(device)
    
    batch_size = kwargs.get('batch_size', 1)
    if batch_size > len(img_files):
        batch_size = len(img_files)
    num_workers = kwargs.get('num_workers', 0)

    transform = get_transform(IMSIZE, normalize=True, center_crop=True)
    img_loader = DataLoader(ImagePathDataset(img_files, transform=transform), batch_size=batch_size, num_workers=num_workers)
    ref_loader = DataLoader(ImagePathDataset(ref_files, transform=transform), batch_size=batch_size, num_workers=num_workers)

    if device is not None:
        model.to(device)
    
    lpips_scores = []
    disable_tqdm = kwargs.get('disable_tqdm', False)
    tbar = tqdm(zip(img_loader, ref_loader), total=len(img_loader), disable=disable_tqdm, leave=False)
    for img_batch, ref_batch in tbar:
        if device is not None:
            img_batch = img_batch.to(device)
            ref_batch = ref_batch.to(device)
        l = loss_fn_alex(img_batch, ref_batch).squeeze()
        if l.ndim == 0:
            # squeeze() may make it a scalar
            lpips_scores.append(l.item())
        else:
            lpips_scores.extend(l.tolist())

    model.cpu()
    
    if reduce == 'mean':
        return float(np.mean(lpips_scores))
    elif reduce == 'sum':
        return float(np.sum(lpips_scores))
    else:
        return lpips_scores


@torch.no_grad()
def calc_lpips_from_tensors(img_tensors: Union[torch.Tensor, List[torch.Tensor]], ref_tensors: Union[torch.Tensor, List[torch.Tensor]], reduce: str='mean') -> Union[float, List[float]]:
    """
    Calculates LPIPS score between two images from tensors
    """
    img_tensors = gather_img_tensors(img_tensors)
    ref_tensors = gather_img_tensors(ref_tensors)
    
    assert len(img_tensors) == len(ref_tensors), 'Number of images in img_tensors and ref_tensors should be same'

    lpips_scores = []
    for img, ref in tqdm(zip(img_tensors, ref_tensors)):
        l = loss_fn_alex(ref, img)
        lpips_scores.append(l.item())
    
    if reduce == 'mean':
        return float(np.mean(lpips_scores))
    elif reduce == 'sum':
        return float(np.sum(lpips_scores))
    else:
        return lpips_scores


def main():

    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = None

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers
    
    img_files = get_img_files(args.img_path)
    ref_files = get_img_files(args.ref_path)

    if len(img_files) == 0:
        raise ValueError('No images found in img_path')
    elif len(ref_files) == 0:
        raise ValueError('No images found in ref_path')
    elif len(img_files) != len(ref_files):
        print('Number of images in img_path and ref_path are different. Matching files...')
        img_files, ref_files = match_files(img_files, ref_files)
    
    assert len(img_files) == len(ref_files), 'Number of images in img_path and ref_path should be same'
    print(f'Number of images: {len(img_files)}')

    if args.prompts_path is None:
        prompts = [""] * len(img_files)
        prompt_to_ids = read_prompt_to_ids(prompts=prompts)
        # same NULL prompts for all images
    else:    
        prompt_to_ids = read_prompt_to_ids(path=args.prompts_path)
    
    # Calculate LPIPS score for each prompt
    lpips_scores = []
    for prompt, indices in prompt_to_ids.items():
        _img_files = [img_files[idx] for idx in indices]
        _ref_files = [ref_files[idx] for idx in indices]
        assert len(img_files) == len(ref_files), f'Number of images in img_files and ref_files should be same for {prompt}'
        lpips_score = calc_lpips_from_files(
            _img_files, 
            _ref_files, 
            model=loss_fn_alex, 
            device=device,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )
        lpips_scores.append({
            "prompt": prompt,
            "score": lpips_score,
            "length": len(indices),
        })
        print(f'LPIPS score: {lpips_score} for prompt: {prompt}')
        
    df = pd.DataFrame.from_dict(lpips_scores)
    print(df)
    if args.save_path is not None:
        df.to_csv(args.save_path)
    

if __name__=='__main__':
    main()

