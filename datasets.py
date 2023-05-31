import csv
import os
import random
import json
from functools import partial
from glob import glob
from math import ceil
from typing import List, Union, Optional, Callable, Tuple, Any

import numpy as np
from scipy.io import loadmat
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg, check_integrity, download_file_from_google_drive

from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation, KRandomResizedCrop
from torchvision import transforms as T
from torchvision.transforms import functional as FT
import torch.nn.functional as F
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, ImageNet, Caltech101, Caltech256, Flowers102, \
    Food101, DTD, OxfordIIITPet, StanfordCars, FGVCAircraft, VisionDataset

import kornia.augmentation as K

class ImageList(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('splits/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]


class FacesInTheWild300W(Dataset):
    """Adapted from https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/main/test_datasets.py"""
    def __init__(self, root, split, mode='indoor_outdoor', transform=None, loader=default_loader, download=False, shots=None):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.loader = loader
        images = []
        keypoints = []
        if 'indoor' in mode:
            print('Loading indoor images')
            images += glob(os.path.join(self.root, '01_Indoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '01_Indoor', '*.pts'))
        if 'outdoor' in mode:
            print('Loading outdoor images')
            images += glob(os.path.join(self.root, '02_Outdoor', '*.png'))
            keypoints += glob(os.path.join(self.root, '02_Outdoor', '*.pts'))
        images = list(sorted(images))[0:len(images) - 1]
        keypoints = list(sorted(keypoints))

        split_path = os.path.join(self.root, f'{mode}_{split}.npy')
        # while not os.path.exists(split_path):
        self.generate_dataset_splits(len(images), shots=shots)
        split_idxs = np.load(split_path)
        print(split, split_path, max(split_idxs), len(images), len(keypoints))
        self.images = [images[i] for i in split_idxs]
        self.keypoints = [keypoints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.3, 0.3, 0.4], shots=None):
        np.random.seed(0)
        print(split_sizes)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        if shots is None:
            split1, split2 = int(l * split_sizes[0]), int(l * sum(split_sizes[:2]))
            train_idx = idxs[:split1]
            valid_idx = idxs[split1:split2]
            test_idx = idxs[split2:]
        else:
            split1, split2 = int(l * split_sizes[0]), int(l * sum(split_sizes[:2]))
            print("fs", shots, split2, split1)
            shot_split = int(l * shots)
            train_idx = idxs[:shot_split // 2]
            valid_idx = idxs[shot_split // 2:shot_split]
            test_idx = idxs[shot_split:]
        # print(max(train_idx), max(valid_idx), max(test_idx))
        print("Generated train")
        np.save(os.path.join(self.root, f'{self.mode}_train'), train_idx)
        np.save(os.path.join(self.root, f'{self.mode}_valid'), valid_idx)
        print("Generated train and val")
        np.save(os.path.join(self.root, f'{self.mode}_test'), test_idx)
        print("Generated test")

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        keypoint = open(self.keypoints[index], 'r').readlines()
        keypoint = keypoint[3:-1]
        keypoint = [s.strip().split(' ') for s in keypoint]
        keypoint = torch.tensor([(float(x), float(y)) for x, y in keypoint])
        bbox_x1, bbox_x2 = keypoint[:, 0].min().item(), keypoint[:, 0].max().item()
        bbox_y1, bbox_y2 = keypoint[:, 1].min().item(), keypoint[:, 1].max().item()
        bbox_width = ceil(bbox_x2 - bbox_x1)
        bbox_height = ceil(bbox_y2 - bbox_y1)
        bbox_length = max(bbox_width, bbox_height)

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=bbox_length, width=bbox_length)
        keypoint = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y in keypoint])

        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        new_h, new_w = image.shape[1:]

        keypoint = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in keypoint])
        keypoint = keypoint.flatten()
        keypoint = F.normalize(keypoint, dim=0)
        return image, keypoint

    def __len__(self):
        return len(self.images)


CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA(VisionDataset):
    """Adapted from https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/main/test_datasets.py"""
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            shots: int = None
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        # if download:
        #    self.download()

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        if shots is None:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            self.identity = identity.data[mask]
            self.bbox = bbox.data[mask]
            self.landmarks_align = landmarks_align.data[mask]
            self.attr = attr.data[mask]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header
        else:
            self.filename = [self.filename[i] for i, m in enumerate(mask) if m]
            l_shot = int(shots * len(self.filename))
            self.filename = self.filename[:l_shot]
            self.identity = identity.data[mask][:l_shot]
            self.bbox = bbox.data[mask][:l_shot]
            self.landmarks_align = landmarks_align.data[mask][:l_shot]
            self.attr = attr.data[mask][:l_shot]
            # map from {-1, 1} to {0, 1}
            self.attr = torch.div(self.attr + 1, 2).to(int)
            self.attr_names = attr.header

        print()

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            print("CHeck integrity", filename, ext)
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        print("FOLDER", os.path.join(self.root, self.base_folder))
        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        w, h = X.width, X.height
        min_side = min(w, h)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)
        new_w, new_h = X.shape[1:]

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        # transform the landmarks
        new_target = torch.zeros_like(target)
        if 'landmarks' in self.target_type:
            for i in range(int(len(target) / 2)):
                new_target[i * 2] = ((target[i * 2] - ((w - min_side) / 2)) / min_side) * new_w
                new_target[i * 2 + 1] = ((target[i * 2 + 1] - ((h - min_side) / 2)) / min_side) * new_h

        return X, new_target.float()

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class LeedsSportsPose(Dataset):
    def __init__(self, root, split, transform=None, loader=default_loader, download=False, shots=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.loader = loader
        images = glob(os.path.join(self.root, 'images', '*.jpg'))
        images = sorted(images)
        joints = loadmat(os.path.join(self.root, 'joints.mat'))['joints']
        joints = np.array(
            [[(joints[0, j, i], joints[1, j, i], joints[2, j, i]) for j in range(joints.shape[1])] for i in
             range(joints.shape[2])])

        split_path = os.path.join(self.root, f'{split}.npy')
        # while not os.path.exists(split_path):
        self.generate_dataset_splits(len(images), shots=shots)
        split_idxs = np.load(split_path)
        self.images = [images[i] for i in split_idxs]
        self.joints = [joints[i] for i in split_idxs]

    def generate_dataset_splits(self, l, split_sizes=[0.6, 0.4], shots=None):
        np.random.seed(0)
        assert sum(split_sizes) == 1
        idxs = np.arange(l)
        np.random.shuffle(idxs)
        print("shots", shots)
        if shots is None:
            split1 = int(l * split_sizes[0])
            train_idx = idxs[:split1]
            test_idx = idxs[split1:]
        else:
            print("shots", shots)
            split1 = int(l * shots)
            train_idx = idxs[:split1]
            test_idx = idxs[split1:]
        print(max(train_idx), max(test_idx))
        np.save(os.path.join(self.root, 'train'), train_idx)
        np.save(os.path.join(self.root, 'test'), test_idx)

    def __getitem__(self, index):
        # get image in original resolution
        path = self.images[index]
        image = self.loader(path)
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        joints = self.joints[index]

        bbox_x1 = int((w - min_side) / 2) if w >= min_side else 0
        bbox_x2 = bbox_x1 + min_side
        bbox_y1 = int((h - min_side) / 2) if h >= min_side else 0
        bbox_y2 = bbox_y1 + min_side

        image = FT.crop(image, top=bbox_y1, left=bbox_x1, height=min_side, width=min_side)
        joints = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y, _ in joints])

        h, w = image.height, image.width
        min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        new_h, new_w = image.shape[1:]

        joints = torch.tensor([[
            ((x - ((w - min_side) / 2)) / min_side) * new_w,
            ((y - ((h - min_side) / 2)) / min_side) * new_h,
        ] for x, y in joints])
        joints = joints.flatten()

        return image, joints

    def __len__(self):
        return len(self.images)

class SUN397(ImageList):
    def __init__(self, root, split, transform=None):
        # some files exists only in /storage/shared/datasets/mimit67_indoor_scenes/indoorCVPR_09/images_train_test/Images/
        root = os.path.join(root, "SUN397")
        with open(os.path.join(root, 'ClassName.txt')) as f:
            classes = [line.strip() for line in f]

        with open(os.path.join(root, f'{split}_01.txt')) as f:
            samples = []
            for line in f:
                path = line.strip()
                for y, cls in enumerate(classes):
                    if path.startswith(cls+'/'):
                        samples.append((os.path.join(root, path[1:]), y))
                        break
        super().__init__(samples, transform)

def load_pretrain_datasets(dataset='cifar10',
                           datadir='/data',
                           color_aug='default'):

    if dataset == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        train_transform = MultiView(RandomResizedCrop(224, scale=(0.2, 1.0)))
        test_transform = T.Compose([T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = ImageNet100(datadir, split='train', transform=train_transform)
        valset   = ImageNet100(datadir, split='train', transform=test_transform)
        testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        if color_aug == 'default':
            s = 1
        elif color_aug == 'strong':
            s = 2.
        elif color_aug == 'weak':
            s = 0.5
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                           K.RandomGrayscale(p=0.2*s),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                           K.RandomGrayscale(p=0.2*s),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform, download=True)
        valset   = STL10(datadir, split='train',           transform=test_transform, download=True)
        testset  = STL10(datadir, split='test',            transform=test_transform, download=True)

    elif dataset == 'stl10_rot':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))
        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           RandomRotation(p=0.5),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    elif dataset == 'stl10_sol':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform, download=True)
        valset   = STL10(datadir, split='train',           transform=test_transform, download=True)
        testset  = STL10(datadir, split='test',            transform=test_transform, download=True)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(train=trainset,
                val=valset,
                test=testset,
                t1=t1, t2=t2)

def load_datasets(dataset='cifar10',
                  datadir='/data',
                  pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        trainval = ImageNet100(datadir, split='train', transform=transform)
        n_trainval = len(trainval)
        train, val = random_split(trainval, [int(n_trainval * 0.9), int(n_trainval * 0.1)], generator=generator(42))
        test     = ImageNet100(datadir, split='val', transform=transform)
        num_classes = 100

    elif dataset == 'food101':
        trainval   = Food101(root=datadir, split='train', transform=transform, download=True)
        train, val = random_split(trainval, [68175, 7575], generator=generator(42))
        test       = Food101(root=datadir, split='test',  transform=transform, download=True)
        num_classes = 101

    elif dataset == 'cifar10':
        trainval   = CIFAR10(root=datadir, train=True,  transform=transform, download=True)
        train, val = random_split(trainval, [45000, 5000], generator=generator(43))
        test       = CIFAR10(root=datadir, train=False, transform=transform, download=True)
        num_classes = 10

    elif dataset == 'cifar100':
        trainval   = CIFAR100(root=datadir, train=True,  transform=transform, download=True)
        train, val = random_split(trainval, [45000, 5000], generator=generator(44))
        test       = CIFAR100(root=datadir, train=False, transform=transform, download=True)
        num_classes = 100

    elif dataset == 'sun397':
        trn_indices, val_indices = torch.load('splits/sun397.pth')
        trainval = SUN397(root=datadir, split='Training', transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = SUN397(root=datadir, split='Testing',  transform=transform)
        num_classes = 397

    elif dataset == 'dtd':
        train    = DTD(root=datadir, split='train', transform=transform, download=True)
        val      = DTD(root=datadir, split='val',   transform=transform, download=True)
        trainval = ConcatDataset([train, val])
        test     = DTD(root=datadir, split='test',  transform=transform, download=True)
        num_classes = 47

    elif dataset == 'pets':
        trainval   = OxfordIIITPet(root=datadir, split='trainval', transform=transform, download=True)
        train, val = random_split(trainval, [2940, 740], generator=generator(49))
        test       = OxfordIIITPet(root=datadir, split='test',     transform=transform, download=True)
        num_classes = 37

    elif dataset == 'caltech101':
        transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        train    = Subset(D, trn_indices)
        val      = Subset(D, val_indices)
        trainval = ConcatDataset([train, val])
        test     = Subset(D, tst_indices)
        num_classes = 101

    elif dataset == 'flowers':
        train = Flowers102(datadir, split="train", transform=transform, download=True)
        val = Flowers102(datadir, split="val", transform=transform, download=True)
        test = Flowers102(datadir, split="test", transform=transform, download=True)

        trainval = ConcatDataset([train, val])
        num_classes = 102

    elif dataset in ['flowers-5shot', 'flowers-10shot']:
        if dataset == 'flowers-5shot':
            n = 5
        else:
            n = 10

        if pretrain_data == 'imagenet100':
            transform = T.Compose([T.Resize(256, interpolation=Image.BICUBIC),
                                   T.CenterCrop(224),
                                   T.ToTensor(),
                                   T.Normalize(mean, std)])

            train_transform = T.Compose([
                                   T.Resize(256, interpolation=Image.BICUBIC),
                                   T.TenCrop(224),
                                   T.Lambda(
                                       lambda crops:  T.Normalize(mean, std)(
                                           torch.stack([
                                              T.ToTensor()(crop) for crop in crops
                                           ]).float()
                                       )
                                   ),
                                ])
        else:
            train_transform = transform

        train = Flowers102(datadir, split="train", transform=train_transform, download=True)
        val = Flowers102(datadir, split="val", transform=transform, download=True)

        trainval = Flowers102(datadir, split="train", transform=train_transform, download=True)
        trainval._image_files += val._image_files
        trainval._labels += val._labels

        test = Flowers102(datadir, split="test", transform=transform, download=True)

        ####   make train 5-shot
        train_indices = defaultdict(list)
        for i, y in enumerate(train._labels):
            train_indices[y].append(i)

        train_indices = sum([random.sample(train_indices[y], n) for y in train_indices.keys()], [])
        train = Subset(trainval, train_indices)
        ####

        ####   make trainval 5-shot
        trainval_indices = defaultdict(list)
        for i, y in enumerate(trainval._labels):
            trainval_indices[y].append(i)

        trainval_indices = sum([random.sample(trainval_indices[y], n) for y in trainval_indices.keys()], [])
        trainval = Subset(trainval, trainval_indices)
        ####

        num_classes = 102

    elif dataset == 'stl10':
        trainval   = STL10(root=datadir, split='train', transform=transform, download=True)
        test       = STL10(root=datadir, split='test',  transform=transform, download=True)
        train, val = random_split(trainval, [4500, 500], generator=generator(50))
        num_classes = 10


    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        train, val = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif dataset == 'cub200':
        if pretrain_data == 'imagenet100':
            transform = T.Compose([T.Resize(256, interpolation=Image.BICUBIC),
                                   T.CenterCrop(224),
                                   T.ToTensor(),
                                   T.Normalize(mean, std)])

            train_transform = T.Compose([
                                   T.Resize(256, interpolation=Image.BICUBIC),
                                   T.TenCrop(224),
                                   T.Lambda(
                                       lambda crops:  T.Normalize(mean, std)(
                                           torch.stack([
                                              T.ToTensor()(crop) for crop in crops
                                           ]).float()
                                       )
                                   ),
                                ])
        else:
            train_transform = transform

        trn_indices, val_indices = torch.load('splits/cub200.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=train_transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 200

    elif dataset == 'cars':
        trainval = StanfordCars(datadir, "train", transform=transform, download=True)
        test =  StanfordCars(datadir, "test", transform=transform, download=True)
        train, val = random_split(trainval, [7000, 1144], generator=generator(51))
        num_classes = 196

    elif dataset == 'aircraft':
        trainval = FGVCAircraft(datadir, "trainval", transform=transform, download=True)
        train = FGVCAircraft(datadir, "train", transform=transform, download=True)
        val = FGVCAircraft(datadir, "val", transform=transform, download=True)

        test =  FGVCAircraft(datadir, "test", transform=transform, download=True)
        num_classes = 100

    elif dataset == "celeba":
        train = CelebA(datadir, split="train", target_type="landmarks", transform=transform, download=False)
        val = CelebA(datadir, split="valid", target_type="landmarks", transform=transform, download=False)
        test  = CelebA(datadir, split="valid", target_type="landmarks", transform=transform, download=False)
        trainval = ConcatDataset([train, val])
        num_classes = 10

    elif dataset == "300w":
        train = FacesInTheWild300W(datadir, split="train", transform=transform, download=False)
        val = FacesInTheWild300W(datadir, split="valid", transform=transform, download=False)
        test = FacesInTheWild300W(datadir, split="test", transform=transform, download=False)
        trainval = ConcatDataset([train, val])
        num_classes = 136

    elif dataset == "lspose":
        train = val = trainval = LeedsSportsPose(datadir, split="train", transform=transform, download=False)
        test = LeedsSportsPose(datadir, split="train", transform=transform, download=False)
        num_classes = 28

    # elif dataset == 'dog':
    #     not metnioned in the paper?
    #     trn_indices, val_indices = torch.load('splits/dog.pth')
    #     trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
    #     train    = Subset(trainval, trn_indices)
    #     val      = Subset(trainval, val_indices)
    #     test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
    #     num_classes = 120



    return dict(trainval=trainval,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)


def load_fewshot_datasets(dataset='cifar10',
                          datadir='/data',
                          pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    if dataset == 'cub200':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        test.samples = train.samples + test.samples
        test.targets = train.targets + test.targets

    elif dataset == 'fc100':
        train = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)

    elif dataset == 'plant_disease':
        test = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        # test  = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        # test.samples = train.samples + test.samples
        # test.targets = train.targets + test.targets

    else:
        raise NotImplementedError(dataset)

    return dict(test=test)

def load_datasets_for_augm_interpolation(
        dataset='cifar10',
        datadir='/data',
        pretrain_data="stl10",
        augmentation='colorjitter'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        test_transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224,),
                T.ToTensor(),
                                    ])
        standard_transforms = dict(
            #flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            #grayscale=K.RandomGrayscale(p=1),
            blur= GaussianBlur(23, [0.1, 2.0], p=1),
            identity=T.Compose([]),
        )

        transforms = dict()

        for (k,v) in standard_transforms.items():
            assert k in ["color", "blur", "identity"]
            modified = v
            if k == "color":
                for i in range(1, 17):
                    modified.brightness = (i/8) * v.brightness
                    modified.contrast = (i/8) * v.contrast
                    modified.saturation = (i/8) * v.saturation
                    modified.hue = (i/8) * v.hue
                    transforms[f'color_{i}'] = T.Compose([modified, T.Normalize(mean, std)])
            elif k == "blur":
                for i in range(1, 17):
                    #modified.sigma = (i/8) * v.sigma
                    for j in range(len(modified.sigma)):
                        modified.sigma[j] = (i/8) * v.sigma[j]
                    transforms[f'blur_{i}'] = T.Compose([modified, T.Normalize(mean, std)])
            elif k=="identity":
                transforms[k] = v
        # else TODO

        #transforms = {
        #    k: T.Compose([v, T.Normalize(mean, std)])
        #    for (k,v) in transforms.items()
        #}

        # testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        s = 1

        test_transform = T.Compose([
            T.Resize(96),
            T.CenterCrop(96),
            T.ToTensor(),
                                    ])

        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur=GaussianBlur(9, (0.1, 2.0), p=1),
            crop=K.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
            normalize=T.Normalize(mean, std),
        )
    else:
        raise NotImplementedError(pretrain_data)
        # testset  = STL10(datadir, split='test',            transform=test_transform, download=True)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        test     = ImageNet100(datadir, split='val', transform=test_transform)

    elif dataset == 'food101':
        test       = Food101(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'cifar10':
        test       = CIFAR10(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        test       = CIFAR100(root=datadir, train=False, transform=test_transform, download=True)

    elif dataset == 'sun397':
        test     = SUN397(root=datadir, split='Testing',  transform=test_transform)

    elif dataset == 'dtd':
        test     = DTD(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'pets':
        test       = OxfordIIITPet(root=datadir, split='test',     transform=test_transform, download=True)

    elif dataset == 'caltech101':
        test_transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=test_transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        test     = Subset(D, tst_indices)

    elif dataset == 'flowers':
        test = Flowers102(datadir, split="test", transform=test_transform, download=True)

    elif dataset == 'stl10':
        test = STL10(root=datadir, split='test',  transform=test_transform, download=True)

    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cub200':
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=test_transform)

    elif dataset == 'cars':
        test =  StanfordCars(datadir, "test", transform=test_transform, download=True)

    elif dataset == 'aircraft':
        test =  FGVCAircraft(datadir, "test", transform=test_transform, download=True)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(
        test=test,
        transforms=transforms
    )


def load_datasets_for_cosine_sim(
        dataset='cifar10',
        datadir='/data',
        pretrain_data="stl10",
        color_aug='default'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        test_transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224,),
                T.ToTensor(),
                                    ])
        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur= GaussianBlur(23, (0.1, 2.0), p=1),
            # crop=RandomResizedCrop(224, scale=(0.2, 1.0)),
            identity=nn.Sequential(),
            # normalize=T.Normalize(mean, std), #TODO bug?
        )

        transforms = {
            k: nn.Sequential(v, T.Normalize(mean, std))
            for (k,v) in transforms.items()
        }

        # testset  = ImageNet100(datadir, split='val', transform=test_transform)

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        s = 1

        test_transform = T.Compose([
            T.Resize(96),
            T.CenterCrop(96),
            T.ToTensor(),
                                    ])

        transforms = dict(
            flip=K.RandomHorizontalFlip(p=1),
            color=ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
            grayscale=K.RandomGrayscale(p=1),
            blur=GaussianBlur(9, (0.1, 2.0), p=1),
            crop=K.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
            normalize=T.Normalize(mean, std),
        )
    else:
        raise NotImplementedError(pretrain_data)
        # testset  = STL10(datadir, split='test',            transform=test_transform, download=True)
    if dataset == 'imagenet100':
        """
        https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        """
        test, test_no_transform = [ImageNet100(datadir, split='val', transform=t) for t in [test_transform, None]]

    elif dataset == 'food101':
        test, test_no_transform = [Food101(root=datadir, split='test',  transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'cifar10':
        test, test_no_transform = [CIFAR10(root=datadir, train=False, transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'cifar100':
        test, test_no_transform = [CIFAR100(root=datadir, train=False, transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'sun397':
        test, test_no_transform = [SUN397(root=datadir, split='Testing',  transform=t) for t in [test_transform, None]]

    elif dataset == 'dtd':
        test, test_no_transform = [DTD(root=datadir, split='test',  transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'pets':
        test, test_no_transform = [OxfordIIITPet(root=datadir, split='test',     transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'caltech101':
        test_transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D_no_transform = Caltech101(datadir, transform=T.Lambda(lambda img: img.convert('RGB')), download=True)
        D = Caltech101(datadir, transform=test_transform, download=True)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        test     = Subset(D, tst_indices)
        test_no_transform = Subset(D_no_transform, tst_indices)

    elif dataset == 'flowers':
        test, test_no_transform = [Flowers102(datadir, split="test", transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'stl10':
        test, test_no_transform = [STL10(root=datadir, split='test',  transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'mit67':
        """
        https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019
        """
        test, test_no_transform = [ImageFolder(os.path.join(datadir, 'test'),  transform=t) for t in [test_transform, None]]

    elif dataset == 'cub200':
        test, test_no_transform = [ImageFolder(os.path.join(datadir, 'test'),  transform=t) for t in [test_transform, None]]

    elif dataset == 'cars':
        test, test_no_transform = [ StanfordCars(datadir, "test", transform=t, download=True) for t in [test_transform, None]]

    elif dataset == 'aircraft':
        test, test_no_transform = [ FGVCAircraft(datadir, "test", transform=t, download=True) for t in [test_transform, None]]

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(
        test=test,
        test_no_transform=test_no_transform,
        transforms=transforms
    )