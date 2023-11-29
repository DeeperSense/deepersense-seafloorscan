# Modified by Hayat Rajani (hayat.rajani@udg.edu)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from DINO
https://github.com/facebookresearch/dino/blob/main/main_dino.py
"""


import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import random


class RandomRotation:
    """Rotate the image by an angle randomly sampled from a discrete set
    as opposed to torchvision.transforms.RandomRotation which randomly
    samples the angle from a range.
    Args:
        degrees: Discrete set of angles (in degrees).
    """

    def __init__(self, degrees=range(-90,90)):
        self.degrees = degrees
    
    def __call__(self, x):
        degree = random.choice(self.degrees)
        return F.rotate(x, degree)
    

class DataAugmentation(object):
    """Defines a set of data transformations for global and local crops of the
    input image under the standard multicrop scheme.
    Args:
        global_crops_size: Expected output size of the global crop.
        global_crops_scale: Scale range of the cropped image before resizing,
            relative to the area of the original image.
        local_crops_size: Expected output size of the local crop.
        local_crops_scale: Scale range of the cropped image before resizing,
            relative to the area of the original image.
        local_crops_num: Number of small local views to generate. Set to zero
            to disable multi-crop training.
        mean: Dataset mean.
        std: Dataset standard deviation.
    """

    def __init__(self, global_crops_size, global_crops_scale,
                local_crops_size, local_crops_scale, local_crops_num,
                mean, std):
        rotate = RandomRotation((0,90,180,270))
        flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        color_jitter = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8)
        blur = transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=0.5)
        sharp = transforms.RandomAdjustSharpness(2, p=0.5)

        # transformations for global crops
        self.global_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            sharp,
            blur,
            normalize,
            rotate,
        ])

        # transformations for small local crops
        self.local_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip,
            color_jitter,
            sharp,
            blur,
            normalize,
            rotate,
        ])
        self.local_crops_num = local_crops_num

    def __call__(self, image):
        crops = []
        crops.append(self.global_aug(image))
        crops.append(self.global_aug(image))
        for _ in range(self.local_crops_num):
            crops.append(self.local_aug(image))
        return crops
