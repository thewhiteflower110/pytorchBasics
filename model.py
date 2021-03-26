#import libraries
import torch
from torch.utils import data
from torchvision import datasets,transforms
import numpy as np
from torch import nn
import torch.nn.functional as f
from PIL import Image
import matplotlib.pyplot as plt
from torch import optim
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import pickle

class net(nn.Module): #why do we use this? inheritance
  def __init__(self):
    super().__init__() #why do we use this?
    #inherits all th properties of nn.module, whoch provides lots of methods for easy use
    self.hidden1=nn.Linear(784,128) #creates xW+b for 784 i/p and 128 o/p
    self.hidden2=nn.Linear(128,64)
    self.output=nn.Linear(64,10)
  
  def forward():
    x=F.relu(self.hidden1(x))
    x=F.relu(self.hidden2(x))
    x=F.softmax(self.output(x),dim=1) #rows-> batch columns->softmax

'''
######## 2nd way to declare model

from collections import OrderedDict
input_size=784
hidden_size=[128,64]
output_size=10

model=nn.Sequential(OrderedDict([
                     ('fc1',nn.Linear(input_size,hidden_size[0])),
                     ('relu1',nn.ReLU()),
                     ('fc2',nn.Linear(hidden_size[0],hidden_size[1])),
                    ('relu2',nn.ReLU()),
                      ('fc3',nn.Linear(hidden_size[1],output_size)),
                      ('softmax',nn.Softmax(dim=1))]))

######## 3rd way to declare model
input_size=784
hidden_size=[128,64]
output_size=10

model=nn.Sequential(
                     nn.Linear(input_size,hidden_size[0]),
                     nn.ReLU(),
                    nn.Linear(hidden_size[0],hidden_size[1]),
                    nn.ReLU(),
                      nn.Linear(hidden_size[1],output_size),
                     nn.Softmax(dim=1))
'''