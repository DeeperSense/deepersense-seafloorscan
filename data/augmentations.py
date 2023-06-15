"""Defines wrappers around several torchvision transforms to appropriately
augment, when applicable, the pseudo masks along with the corresponding
input images.
"""

import numpy
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):        
        if isinstance(sample, tuple):
            return (F.normalize(sample[0], self.mean, self.std), sample[1], sample[2])
        else:
            return F.normalize(sample, self.mean, self.std)


class ToTensor():   
    def __call__(self, sample):
        if isinstance(sample, tuple):
            return (F.to_tensor(sample[0]), sample[1], sample[2])
        else:
            return F.to_tensor(sample)


class RandomHorizontalFlip():
    def __call__(self, sample):
        r = random.random()
        if r < 0.5:
            if isinstance(sample, tuple):
                return (F.hflip(sample[0]),
                    numpy.fliplr(sample[1]), numpy.fliplr(sample[2]))
            else:
                return F.hflip(sample)
        else:
            return sample


class RandomVerticalFlip():
    def __call__(self, sample):
        r = random.random()
        if r < 0.5:
            if isinstance(sample, tuple):
                return (F.vflip(sample[0]), 
                    numpy.flipud(sample[1]), numpy.flipud(sample[2]))
            else:
                return F.vflip(sample)
        else:
            return sample


class RandomRotate():
    def __call__(self, sample):
        degree = numpy.random.choice((-180, -90, 0, 90, 180))
        if isinstance(sample, tuple):
            return (sample[0].rotate(degree), 
                numpy.rot90(sample[1], k=degree//90), numpy.rot90(sample[2], k=degree//90))
        else:
            return sample.rotate(degree)


class ColorJitter():
    def __init__(self, p=1):
        self.transform = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=p)
    def __call__(self, sample):
        if isinstance(sample, tuple):
            return (self.transform(sample[0]), sample[1], sample[2])
        else:
            return self.transform(sample)


class GaussianBlur():
    def __init__(self, p=1):
        self.transform = transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=p)
    def __call__(self, sample):
        if isinstance(sample, tuple):
            return (self.transform(sample[0]), sample[1], sample[2])
        else:
            return self.transform(sample)


class Sharpen():
    def __init__(self, p=1):
        self.transform = transforms.RandomAdjustSharpness(2, p=p)
    def __call__(self, sample):
        if isinstance(sample, tuple):
            return (self.transform(sample[0]), sample[1], sample[2])
        else:
            return self.transform(sample)
