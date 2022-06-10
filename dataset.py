from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset

import cv2
import numpy as np
import torch
from skimage import io
import torchvision.datasets as datasets
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from scipy import special
import os


class SatelliteDataset(Dataset):

    def __init__(self, rootdir, clip=True, seed=0, **kwargs):
        from os import listdir
        from os.path import isfile, join

        self.rootdir = rootdir
        self.img_list = [f for f in listdir(self.rootdir)]
        #self.length = len(self.img_list)
        assert (seed + 1) * len(self) - 1 <= 2**32 - 1

    def __len__(self):
        return len(self.img_list)
        #return self.length

    def __getitem__(self, index):
        img_name = os.path.join(self.rootdir, self.img_list[index])
        image = io.imread(img_name)
        if image.shape[0] + image.shape[1] > 512 or image.shape[0] + image.shape[1] < 512:
            print(image.shape)
            print(img_name)
        image = image / 255.
        image *= 2.0
        image -= 1.0
        #print("inside get_item:")
        #print(np.min(image), np.max(image))
        image = image.transpose(2,0,1) 
        return image.astype('float32')


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

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
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
