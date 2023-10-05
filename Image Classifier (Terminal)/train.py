# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 

# Imports here
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import json
from torchvision import datasets, models, transforms
from torch import nn, optim
from collections import OrderedDict 

# arg parse
import argparse
# Data directory (exp: flowers) is should be declared on the input
# Example Input: python train.py flowers 
parser = argparse.ArgumentParser(description="Train Models")
parser.add_argument("dir_data", action="store", type=str)
# only for directory not files name
parser.add_argument("--save_dir", dest="directory_name", action="store", type=str, default="saved_models")
# for the architecture (user can choose one of the two models: vgg16, vgg13 )
parser.add_argument("--arch", dest="arch", action="store", type=str,  default="vgg16")
# set hyperparameters 
parser.add_argument("--learning_rate", dest="learn_rate", type=float, default="0.001")
parser.add_argument("--hidden_units", dest="hidden_units", type=int, default="512")
parser.add_argument("--epochs", dest="epochs", type=int, default=10)
# train with gpu
parser.add_argument("--gpu", action="store_true")
result_parser = parser.parse_args()

data_dir = "ImageClassifier/" + result_parser.dir_data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
# Training sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Validation and Testing sets
test_and_validate_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# TODO: Load the datasets with ImageFolder
# For Train data
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

# For Test data
test_data = datasets.ImageFolder(test_dir, transform=test_and_validate_transforms)

# For Validation data
valid_data = datasets.ImageFolder(valid_dir, transform=test_and_validate_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
# For Train data
trainloaders = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

# For Test data
testloaders = torch.utils.data.DataLoader(test_data, batch_size=16)

# For Validation data
validloaders = torch.utils.data.DataLoader(valid_data, batch_size=16)
    
# TODO: Build and train your network
# figure it out the model architecture, I'm using vgg16
if result_parser.arch == "vgg16":
    model = models.vgg16(pretrained=True)
elif result_parser.arch == "vgg13":
    model = models.vgg13(pretrained=True)

# Freezing parameters
for param in model.parameters():
    param.requires_grad = False

# create new classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 2048)),
    ('relu1', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(2048, result_parser.hidden_units)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(result_parser.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

# put the new classifier on the model
model.classifier = classifier
criterion = nn.NLLLoss()

# make only the classifier parameters trained
optimizer = optim.Adam(model.classifier.parameters(), lr=result_parser.learn_rate)

# use GPU if available
if result_parser.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# train the network
model.to(device);

epochs = result_parser.epochs
running_loss = 0
train_step = 0
print_every = 50

for e in range(epochs):
    for inputs, labels in trainloaders:
        running_loss = 0
        train_step += 1
        # move inputs and labels tensors to default device
        inputs, labels = inputs.to(device), labels.to(device)
        # zero out the gradients of parameters
        optimizer.zero_grad()
        # forward pass
        train_ps = model.forward(inputs)
        # calculate the loss
        loss = criterion(train_ps, labels)
        # compute gradients of paramters 
        loss.backward()
        # update model's parameters
        optimizer.step()
        # added current batch's loss
        running_loss += loss.item()
        
        if train_step % print_every == 0:
            valid_loss = 0
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    valid_ps = model.forward(inputs)
                    loss = criterion(valid_ps, labels)
                    valid_loss += loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(valid_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"Epoch no: {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validate loss: {valid_loss/len(validloaders):.3f}.. "
                  f"Validate accuracy: {valid_accuracy/len(validloaders):.3f}")
            model.train()


# TODO: Save the checkpoint
checkpoint = {'input_size': 25088, 
              'output_size': 102, 
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'classifier': model.classifier
             }

# checkpoint will be saved outside ImageClassifier's Folder
os.mkdir(result_parser.directory_name)
torch.save(checkpoint, result_parser.directory_name + '/checkpoint.pth')

