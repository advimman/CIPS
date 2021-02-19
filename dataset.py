__all__ = ['MultiScaleDataset',
           'ImageDataset'
           ]

from io import BytesIO
import math

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

import tensor_transforms as tt


class MultiScaleDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False, crop_size=64, integer_values=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop_size = crop_size
        self.integer_values = integer_values
        self.n = resolution // crop_size
        self.log_size = int(math.log(self.n, 2))
        self.crop = tt.RandomCrop(crop_size)
        self.crop_resolution = tt.RandomCrop(resolution)
        self.to_crop = to_crop
        self.coords = tt.convert_to_coord_format(1, resolution, resolution, integer_values=self.integer_values)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img).unsqueeze(0)

        if self.to_crop:
            img = self.crop_resolution(img)

        stack = torch.cat([img, self.coords], 1)
        del img

        data[0] = self.crop(stack).squeeze(0)
        stack = stack.squeeze(0)

        stack_strided = None
        for ls in range(self.log_size, 0, -1):
            n = 2 ** ls
            bias = self.resolution - n*self.crop_size + n
            bw = np.random.randint(bias)
            bh = np.random.randint(bias)
            stack_strided = stack[:, bw::n, bh::n]
            if stack_strided.size(2) != self.crop or stack_strided.size(1) != self.crop:
                data[ls] = self.crop(stack_strided.unsqueeze(0)).squeeze(0)
            else:
                data[ls] = stack_strided

        del stack
        del stack_strided

        return data


class ImageDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop = tt.RandomCrop(resolution)
        self.to_crop = to_crop

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        if self.to_crop:
            img = self.crop(img.unsqueeze(0)).squeeze(0)

        return img
