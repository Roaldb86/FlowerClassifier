import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
from PIL import Image
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import seaborn as sns
import json

# Define means and std as global variables for easy maintanance
means = [0.485, 0.456, 0.406]
std =  [0.229, 0.224, 0.225]

def transform_data(scenario, resize=224 , crop=224, rotation=30):
    """
    Transforms data into tensorobject and crops, resizes and normalizes data.

    Input params:
    -------------------------------------
    type:        train or test
    resize:      default = 224
    crop:        default = 224
    rotation:    default = 30
    means, std:  for normalization
    -------------------------------------
    returns transform object
    -------------------------------------
    """
    global means, std

    if scenario == 'train':
        transform = transforms.Compose([transforms.RandomRotation(rotation),
                                        transforms.RandomResizedCrop(resize),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])
    elif scenario == 'test':
        transform = transforms.Compose([transforms.Resize(resize),
                                        transforms.CenterCrop(crop),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])
    else:
        print("please specify test or train transform")

    return transform


def load_data(dir, transformer, batch_size=32, shuffle=True):
    """
    Loads data by pytorch dataloader

    Input:
    ---------------------------------------------------------------------
    dir:         directory of data
    transformer: use transform_data function to generate a transform
    batch_size:  default = 32
    shuffle:     defines if data will be shuffled randomly. default=True
    ---------------------------------------------------------------------
    returns dataloader
    """

    data = datasets.ImageFolder(dir, transform=transformer)
    return torch.utils.data.DataLoader(data, batch_size, shuffle)

def process_image(image_path):
    """Function for image processing, takes imagepath as input, returns processed image """
    # Call global varibales
    global means, std
    # Open image with PIL
    image = Image.open(image_path)

    # Resize the image
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    # Crop the image
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin,
                      top_margin))
    # Normalize
    image = np.array(image)/255
    image = (image - means)/std

    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))

    return image

def imshow(image, ax=None, title=None):
    """ Show image in matplotlib, takes image and optional axis and title arguments"""

    # Global variables
    global means, std

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Add title if function is called with title argument
    if title:
        plt.title(title)

    # Undo preprocessing
    image = std * image + means

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
