"""Defines wrappers around several torchvision transforms to appropriately
augment, when applicable, the pseudo masks along with the corresponding
input images.
"""

import numpy
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from enum import Enum


class TransformType(Enum):
    PHOTOMETRIC = 0
    GEOMETRIC = 1


class Normalize():
    TYPE = TransformType.PHOTOMETRIC

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample, getParam=False):
        if isinstance(sample, tuple):
            sample = (F.normalize(sample[0], self.mean, self.std), sample[1], sample[2])
        else:
            sample = F.normalize(sample, self.mean, self.std)
        return (sample, None) if getParam else sample


class ToTensor():
    TYPE = TransformType.PHOTOMETRIC

    def __call__(self, sample, getParam=False):
        if isinstance(sample, tuple):
            sample = (F.to_tensor(sample[0]), sample[1], sample[2])
        else:
            sample = F.to_tensor(sample)
        return (sample, None) if getParam else sample


class RandomHorizontalFlip():
    TYPE = TransformType.GEOMETRIC

    def __call__(self, sample, p=None, getParam=False):
        p = p or random.random()
        if p < 0.5:
            if isinstance(sample, tuple):
                sample = (F.hflip(sample[0]), numpy.fliplr(sample[1]), numpy.fliplr(sample[2]))
            elif isinstance(sample, numpy.ndarray):     # for pseudomask reversion
                sample = numpy.fliplr(sample)
            else:
                sample = F.hflip(sample)
        return (sample, p) if getParam else sample


class RandomVerticalFlip():
    TYPE = TransformType.GEOMETRIC

    def __call__(self, sample, p=None, getParam=False):
        p = p or random.random()
        if p < 0.5:
            if isinstance(sample, tuple):
                sample = (F.vflip(sample[0]), numpy.flipud(sample[1]), numpy.flipud(sample[2]))
            elif isinstance(sample, numpy.ndarray):     # for pseudomask reversion
                sample = numpy.flipud(sample)
            else:
                sample = F.vflip(sample)
        return (sample, p) if getParam else sample


class RandomRotate():
    TYPE = TransformType.GEOMETRIC

    def __call__(self, sample, degree=None, getParam=False):
        degree = degree or numpy.random.choice((-180, -90, 0, 90, 180))
        if isinstance(sample, tuple):
            sample = (sample[0].rotate(degree), 
                numpy.rot90(sample[1], k=degree//90), numpy.rot90(sample[2], k=degree//90))
        elif isinstance(sample, numpy.ndarray):         # for pseudomask reversion
            sample = numpy.rot90(sample, k=degree//90)
        else:
            sample = sample.rotate(degree)
        return (sample, -degree) if getParam else sample


class ColorJitter():
    TYPE = TransformType.PHOTOMETRIC

    def __init__(self, p=1):
        self.transform = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=p)

    def __call__(self, sample, getParam=False):
        if isinstance(sample, tuple):
            sample = (self.transform(sample[0]), sample[1], sample[2])
        else:
            sample = self.transform(sample)
        return (sample, None) if getParam else sample


class GaussianBlur():
    TYPE = TransformType.PHOTOMETRIC

    def __init__(self, p=0.5):
        self.transform = transforms.RandomApply([
            transforms.GaussianBlur(3)
        ], p=p)

    def __call__(self, sample, getParam=False):
        if isinstance(sample, tuple):
            sample = (self.transform(sample[0]), sample[1], sample[2])
        else:
            sample = self.transform(sample)
        return (sample, None) if getParam else sample


class Sharpen():
    TYPE = TransformType.PHOTOMETRIC

    def __init__(self, p=0.5):
        self.transform = transforms.RandomAdjustSharpness(2, p=p)

    def __call__(self, sample, getParam=False):
        if isinstance(sample, tuple):
            sample = (self.transform(sample[0]), sample[1], sample[2])
        else:
            sample = self.transform(sample)
        return (sample, None) if getParam else sample


# TODO: getParams for photometric transformations
