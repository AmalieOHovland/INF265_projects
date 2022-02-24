# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:48:52 2022

@author: kvern
"""

#%% Import stuff

from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch
from torch import nn
from datetime import datetime

device = (torch.device('cpu'))
seed = 123
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)


#%% Load and preprocess CIFAR-10 dataset

def load_cifar(train_val_split=0.9, 
               data_path='./data/', 
               preprocessor=None, 
               seed=123, 
               keep_labels=['plane', 'bird']):
    '''
    Parameters
    ----------
    train_val_split : double
        Split ratio between train data and validation data.
    data_path : str
        Path where data is stored. Data is downloaded if is does not already excist
    preprocessor : torchvision.transforms.transforms.Compose
        Preprocessor for preprocessing data. Default used if none is provided
    seed : int
        Random seed used in random operations
    keep_labels : list
        List of image labels we want from cifar10 dataset

    Returns
    -------
    data_train : torch.utils.data.dataset.Subset
        Training data
    data_val : torch.utils.data.dataset.Subset
        Validation data
    data_test : torch.utils.data.dataset.Subset
        Testing data

    '''
    
    # Define preprocessor if not already given
    if preprocessor is None:
        preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468),
                                (0.2470, 0.2435, 0.2616))])
    
    # Load training and validation data
    data_train_val = datasets.CIFAR10(
        data_path,       
        train=True,      
        download=True,
        transform=preprocessor)
    
    # Split training and testing data
    n_train = int(len(data_train_val)*train_val_split)
    n_val =  len(data_train_val) - n_train

    data_train, data_val = random_split(
        data_train_val, 
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(seed))
    
    # Load testing data
    data_test = datasets.CIFAR10(
        data_path,       
        train=False,      
        download=True,
        transform=preprocessor)

    # Identify which labels too keep
    labels = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 
               'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    final_labels = list(map(labels.get, keep_labels))
    
    label_map = {}
    for i, label in enumerate(final_labels):
        label_map[label] = i
    
    # Shave off datasets, only keeping the wanted labels
    data_train = [(img, label_map[label]) for img, label in data_train if label in final_labels]
    data_val = [(img, label_map[label]) for img, label in data_val if label in final_labels]
    data_test = [(img, label_map[label]) for img, label in data_test if label in final_labels]
    
    # Print data set sizes for sanity check
    print("Size of the train dataset:        ", len(data_train))
    print("Size of the validation dataset:   ", len(data_val))
    print("Size of the test dataset:         ", len(data_test))
    
    return data_train, data_val, data_test



#%% Define neural network

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Number of layers in network
        self.L = 4
        
        # Initialize Zs and As to dictionary
        # Z[l] = W[l]A[l-1] + b[l]
        # A[l] = g[Z[l]]
        self.z = {i:None for i in range(1, self.L+1)}
        self.a = {i:None for i in range(self.L+1)}
        
        '''
        Create fully connected (fc) layers
        Layers:
            n[l0] = 3072
            n[l1] = 512
            n[l2] = 128
            n[l3] = 32
            n[l4] = 2
        '''
        self.fc = nn.ModuleDict({str(i):None for i in range(1, self.L+1)})
        self.fc['1'] = nn.Linear(in_features=3072, out_features=512)
        self.fc['2'] = nn.Linear(in_features=512, out_features=128)
        self.fc['3'] = nn.Linear(in_features=128, out_features=32)
        self.fc['4'] = nn.Linear(in_features=32, out_features=2)
        
        
    def forward(self, x):
        
        # Input layer
        self.a[0] = torch.flatten(x, 1)
        
        # First layer
        self.z[1] = self.fc['1'](self.a[0])
        self.a[1] = torch.relu(self.z[1])
        
        # Second layer
        self.z[2] = self.fc['2'](self.a[1])
        self.a[2] = torch.relu(self.z[2])
        
        # Third layer 
        self.z[3] = self.fc['3'](self.a[2])
        self.a[3] = torch.relu(self.z[3])
        
        # Fourth layer (output layer)
        self.z[4] = self.fc['4'](self.a[3])
        self.a[4] = self.z[4]
        
        return self.a[4]
    
    
#%%

def train(n_epochs, optimizer, model, loss_fn, train_loader):
    print(" --------- Using Pytorch's SGD ---------")
    
    n_batches = len(train_loader)
    
    losses_train = []

    for epoch in range(1, n_epochs+1):
        
        epoch_loss = 0
        
        # For each batch in train_loader
        for batch in train_loader:
            # Split batch into  
            inputs, labels = batch
            
            # Zero gradient for every batch
            model.zero_grad()
            
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            
            # Compute gradient
            loss.backward()  
            
            # Adjust parameters
            optimizer.step()
            
            # Add loss from current batch
            epoch_loss += loss.item()
            
        losses_train.append(epoch_loss / n_batches)
        
        if epoch == 1 or epoch % 5 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.5f}'.format(
                datetime.now().time(), epoch, epoch_loss/n_batches))
            
    return losses_train


def train_manual_update(n_epochs, lr, weight_decay, model, loss_fn, train_loader):
    print(" --------- Using Manual Update ---------")
    
    n_batches = len(train_loader)
    
    losses_train = []

    for epoch in range(1, n_epochs+1):
        
        epoch_loss = 0
        
        # For each batch in train_loader
        for batch in train_loader:
            # Split batch into  
            inputs, labels = batch
            
            # Zero gradient for every batch
            model.zero_grad()
            
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            
            # Compute gradient
            loss.backward()  
            
            # Adjust parameters
            for p in model.parameters():
                p.data = (1 - lr*weight_decay)*p.data - lr*p.grad
            
            # Add loss from current batch
            epoch_loss += loss.item()
            
        losses_train.append(epoch_loss / n_batches)
        
        if epoch == 1 or epoch % 5 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.5f}'.format(
                datetime.now().time(), epoch, epoch_loss/n_batches))
            
    return losses_train

#%% Testing both trainers to see if results are the same

data_train, data_val, data_test = load_cifar()
n_epochs = 10
batch_size = 64
lr = 1e-1
weight_decay = 0.1
momentum = 0.9
loss_fn = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False)

torch.manual_seed(seed)
model = MyMLP()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader)

torch.manual_seed(seed)
model = MyMLP()
losses_train_manual = train_manual_update(n_epochs, lr, weight_decay, model, loss_fn, train_loader)


    





























