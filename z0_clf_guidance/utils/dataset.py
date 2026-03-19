import os
import random
from PIL import Image
from torch.utils.data import Dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _scan_dir(d):
    return sorted([
        os.path.join(d, f) for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])


def _scan_dirs(dirs):
    all_paths = []
    if isinstance(dirs, str):
        return _scan_dir(dirs)
    for d in dirs:
        all_paths.extend(_scan_dir(d))
    return all_paths


class ThreeClassFolderDataset(Dataset):
    """
    3-class dataset for z0 classifier training.
      0: non-people (benign images without people)
      1: non-nude (people, clothed)
      2: nude

    Each class corresponds to one or more directories.
    Supports optional class balancing via undersampling.
    """

    def __init__(self, benign_dir, person_dir, nude_dir,
                 transform=None, balance=True, seed=42):
        self.transform = transform
        self.paths = []
        self.labels = []

        benign_paths = _scan_dirs(benign_dir)
        person_paths = _scan_dirs(person_dir)
        nude_paths = _scan_dirs(nude_dir)

        if balance:
            rng = random.Random(seed)
            min_size = min(len(benign_paths), len(person_paths), len(nude_paths))
            benign_paths = rng.sample(benign_paths, min_size)
            person_paths = rng.sample(person_paths, min_size)
            nude_paths = rng.sample(nude_paths, min_size)

        for p in benign_paths:
            self.paths.append(p)
            self.labels.append(0)
        for p in person_paths:
            self.paths.append(p)
            self.labels.append(1)
        for p in nude_paths:
            self.paths.append(p)
            self.labels.append(2)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}


class FourClassFolderDataset(Dataset):
    """
    4-class dataset for z0 classifier training.
      0: benign (no people)
      1: safe_clothed (person with clothes)
      2: harm_nude (nudity)
      3: harm_color (color artifacts / distortions)

    Each class corresponds to one or more directories.
    Supports optional class balancing via undersampling.
    """

    def __init__(self, benign_dir, person_dir, nude_dir, harm_color_dir,
                 transform=None, balance=True, seed=42):
        self.transform = transform
        self.paths = []
        self.labels = []

        benign_paths = _scan_dirs(benign_dir)
        person_paths = _scan_dirs(person_dir)
        nude_paths = _scan_dirs(nude_dir)
        color_paths = _scan_dirs(harm_color_dir)

        if balance:
            rng = random.Random(seed)
            min_size = min(len(benign_paths), len(person_paths),
                           len(nude_paths), len(color_paths))
            benign_paths = rng.sample(benign_paths, min_size)
            person_paths = rng.sample(person_paths, min_size)
            nude_paths = rng.sample(nude_paths, min_size)
            color_paths = rng.sample(color_paths, min_size)

        for p in benign_paths:
            self.paths.append(p)
            self.labels.append(0)
        for p in person_paths:
            self.paths.append(p)
            self.labels.append(1)
        for p in nude_paths:
            self.paths.append(p)
            self.labels.append(2)
        for p in color_paths:
            self.paths.append(p)
            self.labels.append(3)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}
