import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

# Import scripts with functions and utilities
import functions
import utilities

def get_input_args():
    """Creates 6 arguments for the train.py script to load from CLI """
    parser = argparse.ArgumentParser(description="Imageclassifier model trainer")
    parser.add_argument("data_directory", help="<directory for input data>", type=str)
    parser.add_argument("--save_dir", help="<directory for saving checkpoint>", type=str, default="/checkpoint")
    parser.add_argument("--arch", help="<model architeure>", type=str, default="vgg19")
    parser.add_argument("--learning_rate", help="<Learning rate for training the model>", type=float , default=0.001)
    parser.add_argument("--hidden_units", help="<number of units in hidden layer>", type=int , default=512)
    parser.add_argument("--epochs", help="<number of epochs", type=int , default=5)
    parser.add_argument("--gpu", help="<use gpu or not", type=bool , default=True)

    args = parser.parse_args()
    return args


def main():

    # Get CLI arguments
    args = get_input_args()
    
    # Prep data
    train_transform = utilities.transform_data('train')
    test_transform = utilities.transform_data('test')
    # Dataloaders
    trainloader = utilities.load_data(args.data_directory + '/' + 'train', train_transform)
    validationloader = utilities.load_data(args.data_directory + '/' + 'valid', test_transform)

    # Setup and train model
    model, optimizer, criterion = functions.model_setup(args.arch, args.hidden_units, args.learning_rate)
    trained_model = functions.train_model(optimizer, criterion, model, trainloader, validationloader, args.gpu, args.epochs)

    # Save the model
    functions.save_checkpoint(trained_model, args.save_dir)

if __name__ == '__main__':

    main()
