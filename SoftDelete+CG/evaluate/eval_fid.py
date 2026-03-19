from typing import Union, List, Optional
import os
import pathlib
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_frechet_distance, calculate_activation_statistics,
    adaptive_avg_pool2d
)
from evaluate.utils import (
    get_img_files, match_files, read_prompt_to_ids,
    ImagePathDataset, get_transform
)

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--num_workers', type=int,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--save_stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                            'The first path is used as input and the second as output.'))
    
    parser.add_argument('--img_path', type=str, required=False, default=None,)
    parser.add_argument('--ref_path', type=str, required=False, default=None,)
    parser.add_argument('--prompts_path', type=str, required=False, default=None,)
    parser.add_argument('--save_path', type=str, required=False, default=None,)
    parser.add_argument('--match_files', action='store_true', help='Match files in img_path and ref_path')

    args = parser.parse_args()

    return args


IMG_SIZE = 299


def get_activations(
    files: List[str], 
    model: nn.Module, 
    dims: int=2048,
    device: Optional[torch.device]=None,
    **kwargs,
) -> np.ndarray:
    """Calculates the activations of the pool_3 layer for all images.

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    batch_size = kwargs.get('batch_size', 50)
    if batch_size > len(files):
        batch_size = len(files)
    num_workers = kwargs.get('num_workers', 0)

    transform = get_transform(size=IMG_SIZE, center_crop=True)
    loader = DataLoader(ImagePathDataset(files, transform=transform), batch_size=batch_size, num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(loader):
        if device is not None:
            batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
    files: List[str], 
    model: nn.Module, 
    batch_size=50, 
    dims=2048,
    device=None, 
    num_workers: int=1,
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    # save stats at the directory of the first file
    save_file = pathlib.Path(files[0]).parent / 'stats.npz'

    if save_file is not None and os.path.exists(save_file):
        print(f'Loading saved statistics from {save_file}')
        f = np.load(save_file)
        mu, sigma = f['mu'][:], f['sigma'][:]

    else:
        print(f'Calculating statistics for {len(files)} files')
        act = get_activations(files, model, dims=dims, device=device, batch_size=batch_size, num_workers=num_workers)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        
        print(f'Saving statistics to {save_file}')
        try:
            np.savez_compressed(save_file, mu=mu, sigma=sigma)
        except:
            print(f'Failed to save statistics to {save_file}')

    return mu, sigma


def calc_fid_from_files(
    img_files: Union[str, List[str]], 
    ref_files: Union[str, List[str]], 
    model: nn.Module, 
    device: Optional[torch.device]=None,
    **kwargs
) -> float:
    """
    Calculates FID score(s) between two images from image file lists
    """
    if isinstance(img_files, str):
        img_files = [img_files]
    if isinstance(ref_files, str):
        ref_files = [ref_files]
    assert len(img_files) == len(ref_files), 'Number of images in img_files and ref_files should be same'

    batch_size = kwargs.get('batch_size', 50)
    num_workers = kwargs.get('num_workers', 4)
    dims = kwargs.get('dims', 2048)

    m1, s1 = calculate_activation_statistics(
        img_files, model, batch_size, dims, device, num_workers,
    )
    m2, s2 = calculate_activation_statistics(
        ref_files, model, batch_size, dims, device, num_workers,
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


def main():
    args = parse_args()

    if args.device is None:
        device = None
    else:
        device = torch.device(args.device)

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

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx])
    if device is not None:
        model.to(device)

    img_files = get_img_files(args.img_path)
    ref_files = get_img_files(args.ref_path)

    if len(img_files) == 0:
        raise ValueError('No images found in img_path')
    elif len(ref_files) == 0:
        raise ValueError('No images found in ref_path')
    elif len(img_files) != len(ref_files):
        print('Number of images in img_path and ref_path are different. Matching files...')
        if args.match_files:
            img_files, ref_files = match_files(img_files, ref_files)
    
    assert len(img_files) == len(ref_files), 'Number of images in img_path and ref_path should be same'
    print(f'Number of images: {len(img_files)}')

    # Modified
    if args.prompts_path is None:
        prompts = [""] * len(img_files) # NULL prompts
        prompt_to_ids = read_prompt_to_ids(prompts=prompts)
    else:
        prompt_to_ids = read_prompt_to_ids(path=args.prompts_path)
    
    fid_scores = []
    for prompt, indices in prompt_to_ids.items():
        _img_files = [img_files[idx] for idx in indices]
        _ref_files = [ref_files[idx] for idx in indices]
        assert len(img_files) == len(ref_files), f'Number of images in img_files and ref_files should be same for {prompt}'
        
        fid_value = calc_fid_from_files(
            _img_files, 
            _ref_files, 
            model=model, 
            device=device,
            batch_size=args.batch_size, 
            num_workers=num_workers, 
            dims=args.dims,
        )

        fid_scores.append({
            'prompt': prompt,
            'score': float(fid_value),
            'length': len(indices)
        })
        print(f'FID score: {fid_value} for prompt: {prompt}')

    df = pd.DataFrame.from_dict(fid_scores)
    print(df)
    if args.save_path is not None:
        df.to_csv(args.save_path)


if __name__ == '__main__':
    main()
