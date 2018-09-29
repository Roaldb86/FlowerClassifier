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



def model_setup(structure='vgg19', hidden_layer1 = 512,lr = 0.001):
    """
    Input parameters:
    ------------------------------------------------------------
    Structure:     'vgg19' (default), 'densenet121' or 'alexnet'
    Dropout:       default is 0.5
    learning_rate: default is 0.001
    ------------------------------------------------------------
    Returns model, optimizer and criterion
    ------------------------------------------------------------
    """

    structures = {"vgg19":25088,
                  "densenet121" : 1024,
                  "alexnet" : 9216 }

    # Download correct architecture
    if structure == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Invalid model, try 'densenet121', 'alexnet' or leave blank for vgg19")


    # Freeze parameters in the pretrained model
    for param in model.parameters():
        param.requires_grad = False

        # Setup custom classifier
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(0.5)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 200)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(200,150)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(150,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.cuda()

        return model , optimizer ,criterion

def check_perfomance(loader, model, criterion, device, optimizer):
    """
    Validation function, takes dataloader, model, criterion, device and optimizer as
    arguments and returns test_loss and accuracy
    """
    test_loss = 0
    accuracy = 0
    model.eval()
    # Loop over images and labels in dataloader
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Calculate loss
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Accuracy
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def train_model(optimizer, criterion, model, trainloader, validationloader, epochs=5):
    """
    Training a pytorch model. Takes optimizer, criterion, model and epochs(default=5)
    as input. Returns a trained model
    """

    # Set Device to cuda if availble else cpu
    # And initiate some internal variabels
    device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')
    model.to(device)
    print_every = 5
    steps = 0
    running_loss = 0

    for e in range(epochs):

        model.train()

        for inputs, labels in trainloader:

            steps += 1
            # Move input and label to current device. CUDA if available, else CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Feed-forward
            output = model.forward(inputs)
            loss = criterion(output, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()



            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                optimizer.zero_grad()
                with torch.no_grad():

                    test_loss, accuracy = check_perfomance(validationloader, model, criterion, device, optimizer)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Device : {}..".format(device),
                      "Test_loss : {}..".format(test_loss/len(validationloader)),
                      "Test_Accuracy : {}..".format(accuracy/len(validationloader)))

                running_loss = 0

    return model
def process_image(image_path):
    """Takes image path as input and returns a processed image"""

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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std

    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))

    return image


def save_checkpoint(model, save_dir):
    """Saves the model state and architecture"""

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
                  'hidden_units' : 512,
                  'arch' : 'vgg19',
                  'optimizer' : optimizer.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'state_dict' :  model.state_dict()

                 }

    torch.save(checkpoint, save_dir + "/" + 'checkpoint.pth')


def load_checkpoint(filepath):
    """ Loads checkpoint of model. Takes filepath as input and returns loaded model."""
    
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    model,_,_ = model_setup(arch, hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(image_path, model, cat_to_name, gpu, k=5 ):
    """
    Prediction function. Takes, images_path, model, category_label, gpu(True/False), k=number 
    of probable flowers as input. Returns a list of flowername, flowerlabels, and probabilities
    """
    # Process image
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    if gpu == True:
        device = 'cuda:0'
    else:
        device = 'cpu'
    model.to(device)
    
               
    img = process_image(image_path)

    # transform numpy to tensor object
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probabilities
    probs = torch.exp(model.forward(model_input.to(device)))


    # Highest probabilities, 5 by default
    top_probs, top_labs = probs.topk(k)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0]
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = []
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labels, top_flowers


