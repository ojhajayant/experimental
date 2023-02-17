#!/usr/bin/env python
"""
preprocess_albumentations.py: This contains the data-pre-processing routine
implemented using albumentations library.
https://github.com/albumentations-team/albumentations
"""
from __future__ import print_function

import sys
import cv2

import random

random.seed(0)

import numpy as np
import torch
from albumentations import Compose, RandomCrop, HorizontalFlip, Normalize
from albumentations import CoarseDropout, PadIfNeeded, ShiftScaleRotate, RandomBrightness, RandomContrast
from albumentations.augmentations.transforms import HueSaturationValue
from albumentations.augmentations.crops.transforms import RandomSizedCrop
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import datasets

import cfg

sys.path.append('./')
from cfg import get_args

args = get_args()
file_path = args.data

class AddPatchGaussian():
    def __init__(self, patch_size: (int, int), max_scale: float, randomize_patch_size: bool, randomize_scale: bool,
                 **kwargs):
        """
        Args:
        - patch_size: size of patch(h,w) tuple. if -1, it means all image
        - max_scale: max scale size. this value should be in [1, 0]
        - randomize_patch_size: whether randomize patch size or not
        - randomize_scale: whether randomize scale or not
        """
        assert (patch_size[0] >= 1) or (patch_size[1] >= 1) or (patch_size[0] == -1) or (patch_size[1] == -1)
        assert 0.0 <= max_scale <= 1.0

        self.patch_size = patch_size
        self.max_scale = max_scale
        self.randomize_patch_size = randomize_patch_size
        self.randomize_scale = randomize_scale

    def __call__(self, x: torch.tensor):
        c, w, h = x.shape[-3:]

        assert c == 3
        assert h >= 1 and w >= 1
        # assert h == w
        patch_size = []
        # randomize scale and patch_size
        scale = random.uniform(0, 1) * self.max_scale if self.randomize_scale else self.max_scale
        patch_size.append(
            random.randrange(1, self.patch_size[0] + 1) if self.randomize_patch_size else self.patch_size[0])
        patch_size.append(
            random.randrange(1, self.patch_size[1] + 1) if self.randomize_patch_size else self.patch_size[1])
        gaussian = torch.normal(mean=0.0, std=scale, size=(c, w, h))
        gaussian_image = torch.clamp(x + gaussian, 0.0, 1.0)
        mask = self._get_patch_mask((w, h), tuple(patch_size)).repeat(c, 1, 1)
        patch_gaussian = torch.where(mask == True, gaussian_image, x)

        return patch_gaussian
    
    def _get_patch_mask(self, im_size: (int, int), window_size: (int, int)):
        """
        Args:
        - im_size: size of image
        - window_size: size of window. if -1, return full size mask
        """
        # assert im_size >= 1
        # assert (1 <= window_size) or (window_size == -1)

        # if window_size == -1, return all True mask.
        if window_size == -1:
            return torch.ones(im_size[0], im_size[1], dtype=torch.bool)

        mask = torch.zeros(im_size[0], im_size[1], dtype=torch.bool)  # all elements are False

        # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
        window_center_h = random.randrange(0, im_size[1]) if window_size[1] % 2 == 1 else random.randrange(0, im_size[
            1] + 1)
        window_center_w = random.randrange(0, im_size[0]) if window_size[0] % 2 == 1 else random.randrange(0, im_size[
            0] + 1)

        for idx_h in range(window_size[0]):
            for idx_w in range(window_size[1]):
                h = window_center_h - math.floor(window_size[0] / 2) + idx_h
                w = window_center_w - math.floor(window_size[1] / 2) + idx_w

                if (0 <= h < im_size[0]) and (0 <= w < im_size[1]):
                    mask[h, w] = True

        return mask
    
    
class album_Compose:
    def __init__(self,
                 img_size,
                 train=True,
                 mean=[0.49139968, 0.48215841, 0.44653091],
                 std=[0.24703223, 0.24348513, 0.26158784]
                 ):
        self.img_size = img_size
        if train:
            self.albumentations_transform = Compose([
                PadIfNeeded(min_height= img_size[0] + img_size[0] // 4,
                            min_width= img_size[1] + img_size[1] // 4,
                            border_mode=cv2.BORDER_WRAP,
                            always_apply=True, p=1.0),
                RandomSizedCrop((img_size[0],img_size[1]), img_size[0],img_size[1],
                                always_apply=True,
                                p=1.0),
                HorizontalFlip(p=0.5),
#                 ShiftScaleRotate(shift_limit=0.1, 
#                                  scale_limit=0.2,
#                                  rotate_limit=10,
#                                  border_mode=cv2.BORDER_WRAP),
                CoarseDropout(max_holes=1, max_height=img_size[0] // 4,
                              max_width=img_size[1] // 4,
                              min_height=img_size[0] // 4,
                              min_width=img_size[1] // 4,
                              always_apply=False, p=0.65,
                              fill_value=tuple([x * 255.0 for x in mean])),
#                 HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=1, p=0.5),
#                 RandomBrightness(limit=0.1, p=0.5),
#                 RandomContrast(limit=0.07, p=0.5),
                Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(),

            ])
        else:
            self.albumentations_transform = Compose([
                Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(),

            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        if args.cmd == 'train':
            img = AddPatchGaussian(patch_size=((self.img_size[0] * 15) // 16, self.img_size[1] // 2), max_scale=0.79,
                                   randomize_patch_size=False,
                                   randomize_scale=False)(img)
        # if train:
            # img = AddPatchGaussian(patch_size=img_size // 2, max_scale=0.79, randomize_patch_size=True,
                                  #  randomize_scale=True)(img)
        return img


def preprocess_data_albumentations(mean_tuple, std_tuple, img_size):
    """
    Used for pre-processing the data
    when for args.use_albumentations True
    """
    # Train Phase transformations
    global args
    # tensor_args1 = dict(always_apply=True, p=1.0)
    tensor_args2 = dict(num_classes=1, sigmoid=True, normalize=None)
    norm_args = dict(mean=mean_tuple, std=std_tuple, max_pixel_value=255.0,
                     always_apply=False, p=1.0)
    print("************")
    train_transforms = album_Compose(img_size, train=True, mean=mean_tuple,
                                     std=std_tuple)

    # Test Phase transformations
    test_transforms = album_Compose(img_size, train=False, mean=mean_tuple,
                                    std=std_tuple)

    train_kwargs = dict(train=True, download=True, transform=train_transforms)
    test_kwargs = dict(train=False, download=True, transform=test_transforms)
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(file_path, **train_kwargs)
        test_dataset = datasets.CIFAR10(file_path, **test_kwargs)
    elif args.dataset == 'MNIST':
        train_dataset = datasets.MNIST(file_path, **train_kwargs)
        test_dataset = datasets.MNIST(file_path, **test_kwargs)

    print("CUDA Available?", args.cuda)

    # For reproducibility
    torch.manual_seed(args.SEED)

    if args.cuda:
        torch.cuda.manual_seed(args.SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size,
                           num_workers=4,
                           pin_memory=True) if args.cuda else \
        dict(shuffle=True, batch_size=args.batch_size)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader
