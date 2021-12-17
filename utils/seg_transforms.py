import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
"""
Segmentation transforms for training and testing
# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
"""
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        # print("these are ur transforms -> ", self.transforms)
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)

class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class Rescale(object):
    def __init__(self,output_sz):
        self.output_sz = output_sz
        
    def __call__(self, image, *args):
        return (F.resize(image, self.output_sz),) + (F.resize(args[0], self.output_sz),)


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + (F.to_tensor(args[0]),)